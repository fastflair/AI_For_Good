from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf

from dna_atomic import (
    AtomicStructure,
    AtomicStructureError,
    atomic_density_projection,
    atomic_edge_target,
    build_parametric_atomic_dna,
    estimate_helical_pitch_A,
    load_structure,
    write_pdb,
)
from dna_cymatics import (
    STEP_PARAMS,
    build_dna_structure,
    clean_sequence,
    dna_summary,
    fft_components,
    hz_to_note,
    image_registration_metrics,
    image_metrics,
    polar_harmonic_spectrum,
    rank_circular_membrane_modes,
    create_physical_drive_audio,
    create_musical_audio_from_modes,
)
from dna_sources import (
    DEFAULT_GENE_QUERY,
    DEFAULT_UPSTREAM_BP,
    HOXB1A_201_SEQUENCE,
    retrieve_gene_sequences,
    write_sequence_bundle,
)

SAMPLE_DNA = HOXB1A_201_SEQUENCE
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "dna_cymatics_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------- plotting helpers ---------------------------

def _plotly_coarse_3d(dna):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=dna.strand1[:, 0], y=dna.strand1[:, 1], z=dna.strand1[:, 2],
        mode="lines", name="Coarse strand 1", line=dict(width=5),
    ))
    fig.add_trace(go.Scatter3d(
        x=dna.strand2[:, 0], y=dna.strand2[:, 1], z=dna.strand2[:, 2],
        mode="lines", name="Coarse strand 2", line=dict(width=5),
    ))
    # Render only a subset of base-pair cross-links for performance.
    stride = max(1, len(dna.basepair_edges) // 350)
    for i in range(0, len(dna.basepair_edges), stride):
        edge = dna.basepair_edges[i]
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines",
            showlegend=False, line=dict(width=2), hovertext=[f"bp {i+1}", f"bp {i+1}"],
        ))
    fig.update_layout(
        title="Sequence-dependent coarse-grained reconstruction",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0), height=620,
    )
    return fig


def _plotly_atomic_3d(structure: AtomicStructure, max_points: int = 16000):
    n = structure.n_atoms
    if n == 0:
        return go.Figure()
    if n <= max_points:
        idx = np.arange(n)
    else:
        # Deterministic stratified sampling preserves both ends of a long DNA fiber.
        idx = np.linspace(0, n - 1, max_points, dtype=int)
    p = structure.atoms[idx]
    elements = [structure.elements[i] for i in idx]
    colors = {"H": "lightgray", "C": "gray", "N": "royalblue", "O": "red", "P": "orange", "S": "yellow"}
    fig = go.Figure()
    for element in sorted(set(elements)):
        mask = np.array([e == element for e in elements])
        if not mask.any():
            continue
        q = p[mask]
        fig.add_trace(go.Scatter3d(
            x=q[:, 0], y=q[:, 1], z=q[:, 2],
            mode="markers", name=element,
            marker=dict(size=2.6 if element != "H" else 1.7, color=colors.get(element, "white"), opacity=0.75),
        ))
    fig.update_layout(
        title=f"{structure.source} — {structure.dna_form} — {n:,} atoms",
        scene=dict(xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0), height=620,
    )
    return fig


def _image_plot(arr, title, cmap="magma", xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
    ax.imshow(arr, origin="lower", cmap=cmap, aspect="equal")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def _difference_plot(a, b):
    return _image_plot(np.abs(a - b), "Absolute target / measured difference", cmap="inferno")


def _sequence_choices_from_bundle(bundle):
    choices = list(bundle.sequence_catalog().keys())
    choices.sort(key=lambda x: (0 if x.startswith("hoxb1a-201") else 1 if "mature mRNA" in x else 2))
    return choices


def _retrieval_table(bundle):
    rows = [
        {
            "sequence_class": "Genomic DNA",
            "name": bundle.symbol,
            "identifier": bundle.genomic_accession,
            "length_nt": len(bundle.genomic_sequence),
            "assembly": bundle.assembly,
            "coordinates": f"{bundle.genomic_accession}:{bundle.start}-{bundle.end}",
            "source": "NCBI",
        },
        {
            "sequence_class": "Promoter / upstream",
            "name": bundle.symbol,
            "identifier": bundle.genomic_accession,
            "length_nt": len(bundle.promoter_sequence),
            "assembly": bundle.assembly,
            "coordinates": f"{bundle.genomic_accession}:{bundle.promoter_start}-{bundle.promoter_end}",
            "source": "NCBI",
        },
    ]
    for t in bundle.transcripts:
        rows.append({
            "sequence_class": "Mature mRNA/cDNA",
            "name": t.name,
            "identifier": t.ensembl_id or ";".join(t.ncbi_accessions),
            "length_nt": len(t.cDNA),
            "assembly": "transcript",
            "coordinates": f"{t.start}-{t.end}" if t.start and t.end else "",
            "source": "Ensembl" if t.ensembl_id else "NCBI RefSeq",
        })
        rows.append({
            "sequence_class": "Protein-coding CDS",
            "name": t.name,
            "identifier": t.ensembl_id or ";".join(t.ncbi_accessions),
            "length_nt": len(t.cds),
            "assembly": "transcript",
            "coordinates": f"{t.start}-{t.end}" if t.start and t.end else "",
            "source": "Ensembl" if t.ensembl_id else "NCBI RefSeq",
        })
    return pd.DataFrame(rows)


# --------------------------- sequence retrieval ---------------------------

def fetch_gene_action(query, upstream_bp):
    try:
        bundle = retrieve_gene_sequences(query, int(upstream_bp))
        choices = _sequence_choices_from_bundle(bundle)
        default_choice = next((c for c in choices if c.startswith("hoxb1a-201") and "mRNA/cDNA" in c), choices[0])
        catalog = bundle.sequence_catalog()
        strand = "plus" if bundle.strand == 1 else "minus"
        status_lines = [
            f"### Retrieved **{bundle.symbol}**",
            f"Assembly: **{bundle.assembly}** ({bundle.assembly_accession}) · chromosome **{bundle.chromosome}** · "
            f"coordinates **{bundle.start:,}–{bundle.end:,}** · strand **{strand}**",
            f"Genomic sequence: **{len(bundle.genomic_sequence):,} nt** · upstream: **{len(bundle.promoter_sequence):,} nt**",
            "",
            f"[ZFIN gene]({bundle.sources['zfin_gene']}) · [NCBI Gene]({bundle.sources['ncbi_gene']}) · "
            f"[Ensembl gene]({bundle.sources['ensembl_gene']})",
        ]
        for item in bundle.source_status:
            status_lines.append(f"- **{item['source']}** — {item['status']}: {item['detail']}")
        run_dir = OUTPUT_ROOT / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        zip_path = run_dir / f"{bundle.symbol}_{bundle.assembly}_sequences.zip"
        write_sequence_bundle(bundle, zip_path)
        return (
            "\n".join(status_lines),
            _retrieval_table(bundle),
            gr.update(choices=choices, value=default_choice),
            catalog,
            catalog[default_choice],
            str(zip_path),
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def load_retrieved_sequence(choice, catalog):
    if not choice or not catalog:
        return gr.update()
    if choice not in catalog:
        raise gr.Error("Selected sequence is no longer available. Re-fetch the gene package.")
    return catalog[choice]


# --------------------------- main modeling pipeline ---------------------------

def _make_atomic_structure(sequence: str, source: str, dna_form: str, pdb_file: str | None):
    coarse = build_dna_structure(sequence, model="sequence-dependent")
    warning = ""
    if source == "Uploaded PDB/mmCIF":
        if not pdb_file:
            raise AtomicStructureError("Upload a PDB or mmCIF structure first.")
        return load_structure(pdb_file), coarse, warning
    if source == "Internal sequence-derived atom-site model":
        atomic = build_parametric_atomic_dna(coarse, dna_form=dna_form)
        warning = (
            "Using the built-in parametric heavy-atom geometry model. It uses standard nucleic-acid atom names "
            "and sequence-dependent B-DNA twist/rise, but it is not force-field minimized or experimentally refined."
        )
        return atomic, coarse, warning
    raise AtomicStructureError(f"Unsupported structure source: {source}")


def run_pipeline(
    sequence, dna_model, structure_source, dna_form, pdb_file,
    projection_mode, atom_weighting, projection_size, atomic_blur_A, helical_pitch_A, auto_pitch,
    target_transform, membrane_radius_mm, membrane_speed, max_angular_mode, max_radial_mode, top_modes,
    target_type, musical_quantization, physical_duration_per_mode_s, musical_duration_s, tempo_bpm,
):
    try:
        seq = clean_sequence(sequence)
        coarse = build_dna_structure(seq, model=dna_model)
        atomic, coarse_for_plot, structure_warning = _make_atomic_structure(seq, structure_source, dna_form, pdb_file)
        effective_pitch_A = estimate_helical_pitch_A(coarse.step_df, default_A=float(helical_pitch_A)) if bool(auto_pitch) else float(helical_pitch_A)
        literal_proj = atomic_density_projection(
            atomic, mode="Axial atomic density", size=int(projection_size), blur_A=float(atomic_blur_A),
            point_weighting=atom_weighting, helical_pitch_A=effective_pitch_A,
        )
        if projection_mode == "Axial atomic density":
            selected_proj = literal_proj
        else:
            selected_proj = atomic_density_projection(
                atomic, mode=projection_mode, size=int(projection_size), blur_A=float(atomic_blur_A),
                point_weighting=atom_weighting, helical_pitch_A=effective_pitch_A,
            )
        target = selected_proj.image
        if target_transform == "Edge / nodal geometry":
            target = atomic_edge_target(target, blur_sigma_px=1.0)
        modes, mode_recon = rank_circular_membrane_modes(
            target,
            radius_m=float(membrane_radius_mm) / 1000.0,
            wave_speed_m_s=float(membrane_speed),
            max_angular_mode=int(max_angular_mode),
            max_radial_mode=int(max_radial_mode),
            top_modes=int(top_modes),
            target_type=target_type,
        )

        physical_audio, physical_events = create_physical_drive_audio(
            modes, duration_per_mode_s=float(physical_duration_per_mode_s), use_exact_frequencies=True
        )
        best_mode_audio, best_mode_events = create_physical_drive_audio(
            modes.head(1), duration_per_mode_s=max(float(physical_duration_per_mode_s), 3.0), use_exact_frequencies=True
        )
        musical_audio, musical_events = create_musical_audio_from_modes(
            modes, duration_s=float(musical_duration_s), tempo_bpm=float(tempo_bpm),
            quantization=musical_quantization,
        )

        spectrum_components, spectrum = fft_components(
            target, canvas_width_m=2.0 * float(membrane_radius_mm) / 1000.0, top_n=min(24, int(top_modes)),
        )
        harmonics = polar_harmonic_spectrum(target, max_m=32)
        summary = dna_summary(coarse)

        run_dir = OUTPUT_ROOT / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        literal_png = run_dir / "atomic_axial_projection.png"
        target_png = run_dir / "cymatics_target.png"
        spectrum_png = run_dir / "spatial_spectrum.png"
        modes_png = run_dir / "circular_mode_reconstruction.png"
        atomic_pdb = run_dir / "atomic_structure.pdb"
        physical_wav = run_dir / "physical_drive_all_candidates.wav"
        best_mode_wav = run_dir / "physical_drive_best_mode.wav"
        musical_wav = run_dir / "dna_musical_sonification.wav"
        mode_csv = run_dir / "circular_modes.csv"
        harmonic_csv = run_dir / "polar_harmonics.csv"
        spectrum_csv = run_dir / "spatial_frequency_components.csv"
        json_path = run_dir / "analysis.json"
        bundle_path = run_dir / "dna_cymatics_analysis.zip"

        sf.write(physical_wav, physical_audio, 44_100, subtype="PCM_16")
        sf.write(best_mode_wav, best_mode_audio, 44_100, subtype="PCM_16")
        sf.write(musical_wav, musical_audio, 44_100, subtype="PCM_16")
        modes.to_csv(mode_csv, index=False)
        harmonics.drop(columns=["complex_amplitude"], errors="ignore").to_csv(harmonic_csv, index=False)
        pd.DataFrame(spectrum_components).to_csv(spectrum_csv, index=False)
        write_pdb(atomic, atomic_pdb)

        for path, arr, title, cmap in [
            (literal_png, literal_proj.image, "Literal axial atomic-density projection", "magma"),
            (target_png, target, f"Cymatics target — {projection_mode} + {target_transform}", "magma"),
            (spectrum_png, spectrum, "Spatial Fourier spectrum", "viridis"),
            (modes_png, mode_recon, "Circular membrane mode-family reconstruction (exploratory)", "magma"),
        ]:
            fig = _image_plot(arr, title, cmap=cmap)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        payload = {
            "software": {"version": "0.6", "application": "DNA-Cymatics"},
            "sequence": {"length_bp": len(seq), "sha256": __import__('hashlib').sha256(seq.encode()).hexdigest(), "model": dna_model},
            "structure": {
                "source": atomic.source, "dna_form": atomic.dna_form, "n_atoms": atomic.n_atoms,
                "metadata": atomic.metadata, "warning": structure_warning,
            },
            "projection": {
                "literal_mode": "Axial atomic density",
                "selected_mode": projection_mode,
                "target_transform": target_transform,
                "point_weighting": atom_weighting,
                "helical_pitch_A": effective_pitch_A,
                "width_A": selected_proj.width_A,
                "height_A": selected_proj.height_A,
                "element_counts": selected_proj.element_counts,
            },
            "resonator": {
                "model": "ideal circular tension-dominated membrane",
                "radius_m": float(membrane_radius_mm) / 1000.0,
                "wave_speed_m_s": float(membrane_speed),
                "note": "Experimental metal plates require calibration/FEM; this is not a universal Chladni frequency model.",
            },
            "modes": modes.to_dict(orient="records"),
            "physical_events": physical_events,
            "best_mode_events": best_mode_events,
            "musical_events": musical_events,
            "summary": summary,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in [literal_png, target_png, spectrum_png, modes_png, atomic_pdb, physical_wav, best_mode_wav, musical_wav, mode_csv, harmonic_csv, spectrum_csv, json_path]:
                z.write(p, p.name)

        note_lines = [
            f"**{len(seq):,} bp** · GC **{summary['GC_percent']:.1f}%** · estimated B-style turns **{summary['estimated_turns']:.2f}**",
            f"3D structure: **{atomic.source}** · **{atomic.n_atoms:,} modeled heavy-atom sites**",
            f"2D reference: **literal axial atomic density** · selected target source: **{projection_mode}** · target transform: **{target_transform}**",
            f"Atomic weighting: **{atom_weighting}** · helical pitch used for folding: **{effective_pitch_A:.3f} Å**",
            f"Circular membrane: radius **{float(membrane_radius_mm):.1f} mm**, wave speed **{float(membrane_speed):.1f} m/s**",
        ]
        if structure_warning:
            note_lines.append(f"⚠️ {structure_warning}")
        note_lines += [
            "",
            "**Physical-drive interpretation:** the exact resonances are emitted one mode at a time. "
            "The BEST SINGLE MODE WAV is the primary physical test signal. The musical WAV is a separate sonification and is not expected to generate a static Chladni pattern.",
        ]

        return (
            _plotly_atomic_3d(atomic),
            _image_plot(literal_proj.image, "Literal axial atomic-density projection", cmap="magma", xlabel="Å", ylabel="Å"),
            _image_plot(target, f"Cymatics target — {projection_mode} + {target_transform}", cmap="magma", xlabel="normalized X", ylabel="normalized Y"),
            _image_plot(spectrum, "Spatial Fourier spectrum", cmap="viridis"),
            _image_plot(mode_recon, "Circular membrane mode reconstruction", cmap="magma"),
            modes,
            "\n\n".join(note_lines),
            str(physical_wav),
            str(best_mode_wav),
            str(musical_wav),
            str(bundle_path),
            target,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


# --------------------------- experimental verification ---------------------------

def verify_cymatics(dna_reference, measured_image, threshold):
    if dna_reference is None:
        raise gr.Error("Generate a DNA target first.")
    if measured_image is None:
        raise gr.Error("Upload a measured cymatics image first.")
    try:
        measured = np.asarray(measured_image)
        if measured.ndim == 3:
            measured = np.mean(measured[..., :3], axis=2)
        result = image_registration_metrics(np.asarray(dna_reference, dtype=float), measured, threshold=float(threshold))
        aligned = result.pop("aligned_image")
        md = "### Experimental verification\n\n" + "\n".join(f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in result.items())
        md += (
            "\n\nThe registration is an image-processing step; it does not prove frequency causality. "
            "For a defensible physical validation, record the resonator geometry, boundary condition, actuator location, "
            "drive frequency, drive amplitude, and the camera image at the same time."
        )
        return _difference_plot(np.asarray(dna_reference), aligned), md
    except Exception as exc:
        raise gr.Error(str(exc)) from exc



with gr.Blocks(title="DNA → Cymatics → Music") as demo:
    gr.Markdown(
        """
# DNA → 3D Atomic Geometry → 2D Molecular Pattern → Resonant Modes → Music

This research application now separates three different things that should not be conflated:

**(1) molecular structure**, **(2) a 2D target pattern derived from molecular coordinates**, and **(3) a physical resonator response**.

The canonical sequence is **zebrafish hoxb1a-201**, 1,507 nt. The default sequence-only structure path is a **pure-Python parametric heavy-atom geometry model**. It requires no external molecular builder. A real PDB/mmCIF structure can be uploaded when experimentally determined coordinates are available.

> A literal side projection of a long DNA molecule is expected to be line-like. The molecular target used for cymatics is instead an **axial atomic-density projection** (or an explicitly labeled helical phase-folded projection). This is analogous to the axial molecular views used in structural DNA work.
        """
    )

    with gr.Tab("Gene / Sequence Retrieval"):
        with gr.Row():
            with gr.Column(scale=2):
                gene_query = gr.Textbox(value=DEFAULT_GENE_QUERY, label="Gene identifier / symbol / ZFIN URL")
            with gr.Column(scale=1):
                upstream_bp = gr.Slider(0, 10_000, value=DEFAULT_UPSTREAM_BP, step=100, label="Upstream sequence length (bp)")
                fetch_button = gr.Button("Fetch current gene sequences", variant="primary")
        retrieval_status = gr.Markdown()
        retrieval_table = gr.Dataframe(label="Retrieved sequence inventory", wrap=True, interactive=False)
        with gr.Row():
            sequence_choice = gr.Dropdown(label="Sequence to load into DNA workflow", choices=[], interactive=True)
            sequence_download = gr.File(label="Download FASTA package")
        retrieval_catalog = gr.State({})

    with gr.Tab("Structure → 2D Target → Modes → Audio"):
        with gr.Row():
            with gr.Column(scale=1):
                sequence = gr.Textbox(value=SAMPLE_DNA, lines=8, label="DNA sequence (5′→3′)", info="Canonical hoxb1a-201 = 1,507 nt")
                dna_model = gr.Radio(["sequence-dependent", "canonical"], value="sequence-dependent", label="Coarse DNA model")
                structure_source = gr.Radio(
                    ["Internal parametric heavy-atom model", "Uploaded PDB/mmCIF"],
                    value="Internal parametric heavy-atom model", label="3D structure source"
                )
                dna_form = gr.Dropdown(["A-DNA", "B-DNA", "C-DNA", "Z-DNA (idealized left-handed)", "A-RNA"], value="B-DNA", label="Nucleic-acid reference conformation")
                pdb_file = gr.File(file_types=[".pdb", ".cif", ".mmcif"], type="filepath", label="Optional PDB/mmCIF structure")
                dna_status = gr.Markdown("Sequence structure source: internal parametric model; no external molecular builder required.")
            with gr.Column(scale=1):
                projection_mode = gr.Radio(
                    ["Axial atomic density", "Single-turn axial density", "Helical phase-folded density"],
                    value="Single-turn axial density", label="2D molecular projection / target source",
                    info="Literal axial = whole molecule. Single-turn = one helical pitch. Phase-folded = all turns co-registered; derived, not a literal camera view."
                )
                atom_weighting = gr.Radio(["Uniform", "Atomic mass", "Electron count proxy"], value="Uniform", label="Atomic density weighting")
                projection_size = gr.Slider(128, 1024, value=512, step=64, label="2D resolution")
                atomic_blur_A = gr.Slider(0.0, 2.0, value=0.35, step=0.05, label="Atomic density blur (Å)")
                helical_pitch_A = gr.Slider(28, 60, value=34.0, step=0.1, label="Helical pitch for phase fold (Å)")
                auto_pitch = gr.Checkbox(value=True, label="Estimate phase-fold pitch from sequence-dependent step geometry")
                target_transform = gr.Radio(["Density", "Edge / nodal geometry"], value="Edge / nodal geometry", label="Cymatics target transform")
                membrane_radius_mm = gr.Slider(25, 500, value=150, step=5, label="Circular resonator radius (mm)")
                membrane_speed = gr.Slider(10, 2000, value=120, step=5, label="Membrane wave speed (m/s)")
                max_angular_mode = gr.Slider(0, 32, value=18, step=1, label="Maximum angular mode m")
                max_radial_mode = gr.Slider(1, 20, value=10, step=1, label="Maximum radial mode n")
                top_modes = gr.Slider(1, 20, value=8, step=1, label="Top resonant modes")
                target_type = gr.Radio(["Nodal / sand", "Displacement magnitude"], value="Nodal / sand", label="Target interpretation")
                musical_quantization = gr.Radio(["none", "chromatic"], value="none", label="Musical pitch quantization")
                physical_duration_per_mode_s = gr.Slider(0.5, 10, value=2.5, step=0.5, label="Physical drive seconds per mode")
                musical_duration_s = gr.Slider(8, 120, value=24, step=1, label="Musical WAV duration (s)")
                tempo_bpm = gr.Slider(40, 180, value=96, step=1, label="Musical tempo (BPM)")
                run = gr.Button("Build molecular target + resonant modes", variant="primary")

        gr.Markdown("## Results")
        with gr.Row():
            out_3d = gr.Plot(label="3D structure")
            out_projection_literal = gr.Plot(label="Literal axial atomic projection")
        with gr.Row():
            out_projection_target = gr.Plot(label="Cymatics target")
            out_recon = gr.Plot(label="Circular membrane mode-family reconstruction")
        out_spectrum = gr.Plot(label="Spatial Fourier spectrum")
        out_table = gr.Dataframe(label="Candidate circular membrane resonances", wrap=True)
        out_summary = gr.Markdown()
        with gr.Row():
            physical_audio = gr.Audio(label="Physical drive WAV — all candidate modes, exact frequencies", type="filepath")
            best_mode_audio = gr.Audio(label="Physical drive WAV — BEST SINGLE CANDIDATE MODE", type="filepath")
            musical_audio = gr.Audio(label="Musical sonification WAV (creative; not a physical cymatics drive)", type="filepath")
        out_bundle = gr.File(label="Full analysis bundle")
        reference_state = gr.State(None)

        run.click(
            fn=run_pipeline,
            inputs=[sequence, dna_model, structure_source, dna_form, pdb_file,
                    projection_mode, atom_weighting, projection_size, atomic_blur_A, helical_pitch_A, auto_pitch,
                    target_transform, membrane_radius_mm, membrane_speed, max_angular_mode, max_radial_mode, top_modes,
                    target_type, musical_quantization, physical_duration_per_mode_s, musical_duration_s, tempo_bpm],
            outputs=[out_3d, out_projection_literal, out_projection_target, out_spectrum, out_recon, out_table, out_summary,
                     physical_audio, best_mode_audio, musical_audio, out_bundle, reference_state],
        )

    with gr.Tab("Experimental Cymatics Verification"):
        gr.Markdown(
            """
### Quantitatively test a real resonator

1. Generate a molecular target and choose one candidate resonance.
2. Drive the physical resonator at the **exact, unquantized physical frequency**.
3. Record the plate geometry, mounting, actuator position, frequency and amplitude.
4. Photograph the resulting powder/sand/liquid pattern.
5. Upload the image and compare it with the molecular target.

The comparison includes translation registration, RMSE, Pearson correlation, Dice, IoU, a 95th-percentile boundary-distance metric, and angular-harmonic correlation.

A high image similarity score is evidence of pattern similarity; it is not, by itself, proof that the DNA caused that physical frequency or that the physical system is uniquely identified.
            """
        )
        measured_image = gr.Image(type="numpy", image_mode="L", label="Measured Chladni / cymatics image")
        threshold = gr.Slider(0.1, 0.9, value=0.55, step=0.01, label="Binary overlap threshold")
        verify = gr.Button("Compare measured pattern with molecular target", variant="primary")
        out_diff = gr.Plot(label="Difference map")
        out_metrics = gr.Markdown()
        verify.click(fn=verify_cymatics, inputs=[reference_state, measured_image, threshold], outputs=[out_diff, out_metrics])

    fetch_button.click(
        fn=fetch_gene_action,
        inputs=[gene_query, upstream_bp],
        outputs=[retrieval_status, retrieval_table, sequence_choice, retrieval_catalog, sequence, sequence_download],
    )
    sequence_choice.change(fn=load_retrieved_sequence, inputs=[sequence_choice, retrieval_catalog], outputs=[sequence])

    gr.Markdown(
        """
### Model boundaries

The primary sequence-only 3D structure path is an **internal pure-Python parametric heavy-atom geometry model**. It uses explicit base, deoxyribose/ribose and phosphate site templates, a straight global helical axis, and sequence-dependent B-DNA local twist/rise. It is a geometric reference for visualization and projection, not a force-field or quantum-chemistry calculation.

The **axial atomic-density image is a true projection of the loaded atom coordinates**. The **helical phase-folded image is a derived coordinate transform**, not a literal camera projection. Uploaded PDB/mmCIF coordinates remain the preferred source for quantitative structural analysis. The circular membrane solver is an ideal tension-dominated membrane; real Chladni plates require measured/calibrated resonator models, boundary conditions and actuator coupling.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, theme=gr.themes.Soft())
