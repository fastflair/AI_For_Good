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
from scipy.ndimage import zoom

from dna_atomic import (
    AtomicStructure,
    AtomicStructureError,
    INTERNAL_PARAMETRIC_SOURCE,
    UPLOADED_STRUCTURE_SOURCE,
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
    create_simultaneous_physical_drive_audio,
    apply_resonator_calibration,
    create_musical_audio_from_modes,
    prepare_resonator_target,
    observable_to_sand_artwork,
    PLATE_MATERIALS,
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

def _fit_plot(target, recon):
    target_arr = np.asarray(target, dtype=float)
    recon_arr = np.asarray(recon, dtype=float)
    if target_arr.shape != recon_arr.shape:
        recon_arr = zoom(recon_arr, (target_arr.shape[0] / recon_arr.shape[0], target_arr.shape[1] / recon_arr.shape[1]), order=1)
    residual = np.abs(target_arr - recon_arr)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=110)
    for ax, arr, title in zip(axes, [target_arr, recon_arr, residual], ["DNA cymatics target", "Predicted modal mixture", "Absolute fit residual"]):
        ax.imshow(arr, origin="lower", cmap="magma", aspect="equal")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


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
    if source == UPLOADED_STRUCTURE_SOURCE:
        if not pdb_file:
            raise AtomicStructureError("Upload a PDB or mmCIF structure first.")
        return load_structure(pdb_file), coarse, warning
    if source in {INTERNAL_PARAMETRIC_SOURCE, "Internal sequence-derived atom-site model"}:
        atomic = build_parametric_atomic_dna(coarse, dna_form=dna_form)
        warning = (
            "Using the built-in parametric heavy-atom geometry model. It uses standard nucleic-acid atom names "
            "and sequence-dependent B-DNA twist/rise, but it is not force-field minimized or experimentally refined."
        )
        return atomic, coarse, warning
    raise AtomicStructureError(f"Unsupported structure source: {source}")


def run_pipeline(
    sequence, dna_model, structure_source, dna_form, pdb_file,
    projection_mode, atom_weighting, atom_kernel, projection_size, atomic_blur_A, helical_pitch_A, auto_pitch, target_turn_start_bp, target_turn_bp,
    target_transform, target_smoothing_px, resonator_model, membrane_radius_mm, membrane_speed, plate_thickness_mm, plate_material,
    max_angular_mode, max_radial_mode, top_modes,
    candidate_pool, node_width, actuator_r_fraction, actuator_theta_deg, inverse_regularization,
    calibration_csv, calibration_tolerance_hz, use_calibrated_frequency,
    target_type, physical_duration_per_mode_s, physical_mix_duration_s, musical_quantization, musical_duration_s, tempo_bpm,
    musical_arrangement, musical_beats_per_note, musical_repeat_to_target, musical_interval_compression,
    musical_combination_count, musical_min_combination_size, musical_max_combination_size, musical_combination_seed,
    musical_include_all_tones, musical_combination_render, musical_combination_beats,
):
    try:
        seq = clean_sequence(sequence)
        coarse = build_dna_structure(seq, model=dna_model)
        atomic, coarse_for_plot, structure_warning = _make_atomic_structure(seq, structure_source, dna_form, pdb_file)
        effective_pitch_A = estimate_helical_pitch_A(coarse.step_df, default_A=float(helical_pitch_A)) if bool(auto_pitch) else float(helical_pitch_A)
        literal_proj = atomic_density_projection(
            atomic, mode="Axial atomic density", size=int(projection_size), blur_A=float(atomic_blur_A),
            point_weighting=atom_weighting, helical_pitch_A=effective_pitch_A, atom_kernel=atom_kernel,
        )
        if projection_mode == "Axial atomic density":
            selected_proj = literal_proj
        else:
            selected_proj = atomic_density_projection(
                atomic, mode=projection_mode, size=int(projection_size), blur_A=float(atomic_blur_A),
                point_weighting=atom_weighting, helical_pitch_A=effective_pitch_A, atom_kernel=atom_kernel,
                window_start_bp=int(target_turn_start_bp), turn_bp=int(target_turn_bp),
            )
        molecular_target = selected_proj.image.copy()
        target = molecular_target
        if target_transform == "Edge / nodal geometry":
            target = atomic_edge_target(target, blur_sigma_px=1.0, sharpen_power=2.5)
        resonator_target = prepare_resonator_target(target, smoothing_px=float(target_smoothing_px), radial_aperture=0.98)
        if resonator_model == "Clamped circular thin plate":
            young_pa, density_kg_m3, poisson = PLATE_MATERIALS[str(plate_material)]
        else:
            young_pa, density_kg_m3, poisson = 200e9, 7850.0, 0.30
        modes, mode_recon = rank_circular_membrane_modes(
            resonator_target,
            radius_m=float(membrane_radius_mm) / 1000.0,
            wave_speed_m_s=float(membrane_speed),
            max_angular_mode=int(max_angular_mode),
            max_radial_mode=int(max_radial_mode),
            top_modes=int(top_modes),
            target_type=target_type,
            candidate_pool=int(candidate_pool),
            node_width=float(node_width),
            actuator_r_fraction=float(actuator_r_fraction),
            actuator_theta_deg=float(actuator_theta_deg),
            fit_size=96,
            orientation_steps=24,
            resonator_model=str(resonator_model),
            plate_thickness_m=float(plate_thickness_mm) / 1000.0,
            plate_young_pa=float(young_pa),
            plate_density_kg_m3=float(density_kg_m3),
            plate_poisson=float(poisson),
            inverse_regularization=float(inverse_regularization),
        )

        calibration_applied = False
        if calibration_csv and bool(use_calibrated_frequency):
            modes = apply_resonator_calibration(modes, calibration_csv, max_delta_hz=float(calibration_tolerance_hz))
            calibration_applied = True
        drive_frequency_column = "drive_frequency_Hz" if calibration_applied and "drive_frequency_Hz" in modes.columns else "frequency_Hz"
        physical_amplitude_column = "drive_amplitude_physical_calibrated" if calibration_applied and "drive_amplitude_physical_calibrated" in modes.columns else "drive_amplitude_physical"

        # Primary physical signal: all fitted resonances simultaneously. Because the
        # inverse fit is in observable power, drive amplitude uses sqrt(weight) adjusted
        # for geometric actuator coupling. Exact physical frequencies are preserved.
        physical_mix_audio, physical_mix_events = create_simultaneous_physical_drive_audio(
            modes, duration_s=float(physical_mix_duration_s), phase_lock=True, frequency_column=drive_frequency_column,
            amplitude_column=physical_amplitude_column, weight_column=physical_amplitude_column,
        )
        physical_audio, physical_events = create_physical_drive_audio(
            modes, duration_per_mode_s=float(physical_duration_per_mode_s), use_exact_frequencies=True, frequency_column=drive_frequency_column
        )
        best_single_row = modes.sort_values("single_mode_correlation", ascending=False).head(1)
        best_mode_audio, best_mode_events = create_physical_drive_audio(
            best_single_row, duration_per_mode_s=max(float(physical_duration_per_mode_s), 3.0), use_exact_frequencies=True, frequency_column=drive_frequency_column, amplitude_column=physical_amplitude_column
        )
        musical_audio, musical_events = create_musical_audio_from_modes(
            modes, duration_s=float(musical_duration_s), tempo_bpm=float(tempo_bpm),
            quantization=musical_quantization, arrangement=musical_arrangement,
            repeat_to_target=bool(musical_repeat_to_target), beats_per_note=float(musical_beats_per_note),
            interval_compression=float(musical_interval_compression),
            combination_count=int(musical_combination_count),
            min_combination_size=int(musical_min_combination_size),
            max_combination_size=int(musical_max_combination_size),
            combination_seed=int(musical_combination_seed),
            include_all_tones=bool(musical_include_all_tones),
            combination_render=str(musical_combination_render),
            combination_beats=float(musical_combination_beats),
        )

        sand_prediction = observable_to_sand_artwork(mode_recon, mode=target_type)
        spectrum_components, spectrum = fft_components(
            resonator_target, canvas_width_m=2.0 * float(membrane_radius_mm) / 1000.0, top_n=min(24, int(top_modes)),
        )
        harmonics = polar_harmonic_spectrum(resonator_target, max_m=32)
        summary = dna_summary(coarse)

        run_dir = OUTPUT_ROOT / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        literal_png = run_dir / "atomic_axial_projection.png"
        target_png = run_dir / "cymatics_target_molecular.png"
        resonator_target_png = run_dir / "cymatics_target_resonator_conditioned.png"
        spectrum_png = run_dir / "spatial_spectrum.png"
        modes_png = run_dir / "circular_mode_reconstruction.png"
        fit_png = run_dir / "target_vs_modal_fit.png"
        sand_png = run_dir / "predicted_cymatics_sand_artwork.png"
        atomic_pdb = run_dir / "atomic_structure.pdb"
        physical_mix_wav = run_dir / "physical_drive_mode_mixture.wav"
        physical_wav = run_dir / "physical_drive_all_candidates_sequential.wav"
        best_mode_wav = run_dir / "physical_drive_best_mode.wav"
        musical_wav = run_dir / "dna_musical_sonification.wav"
        musical_melody_csv = run_dir / "musical_combination_melody.csv"
        mode_csv = run_dir / "circular_modes.csv"
        harmonic_csv = run_dir / "polar_harmonics.csv"
        spectrum_csv = run_dir / "spatial_frequency_components.csv"
        json_path = run_dir / "analysis.json"
        bundle_path = run_dir / "dna_cymatics_analysis.zip"

        sf.write(physical_mix_wav, physical_mix_audio, 44_100, subtype="PCM_16")
        sf.write(physical_wav, physical_audio, 44_100, subtype="PCM_16")
        sf.write(best_mode_wav, best_mode_audio, 44_100, subtype="PCM_16")
        sf.write(musical_wav, musical_audio, 44_100, subtype="PCM_16")
        pd.DataFrame(musical_events).to_csv(musical_melody_csv, index=False)
        modes.to_csv(mode_csv, index=False)
        harmonics.drop(columns=["complex_amplitude"], errors="ignore").to_csv(harmonic_csv, index=False)
        pd.DataFrame(spectrum_components).to_csv(spectrum_csv, index=False)
        write_pdb(atomic, atomic_pdb)

        for path, arr, title, cmap in [
            (literal_png, literal_proj.image, "Literal axial atomic-density projection", "magma"),
            (target_png, molecular_target, f"Molecular artwork target — {projection_mode} + {target_transform}", "magma"),
            (resonator_target_png, resonator_target, f"Resonator-conditioned target — smoothing {float(target_smoothing_px):.1f} px", "magma"),
            (spectrum_png, spectrum, "Spatial Fourier spectrum of conditioned target", "viridis"),
            (modes_png, mode_recon, f"{resonator_model} predicted observable ({target_type})", "magma"),
            (sand_png, sand_prediction, "Predicted cymatics sand/nodal artwork proxy", "magma"),
        ]:
            fig = _image_plot(arr, title, cmap=cmap)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        fig = _fit_plot(resonator_target, mode_recon)
        fig.savefig(fit_png, bbox_inches="tight")
        plt.close(fig)

        payload = {
            "software": {"version": "0.13", "application": "DNA-Cymatics"},
            "sequence": {"length_bp": len(seq), "sha256": __import__('hashlib').sha256(seq.encode()).hexdigest(), "model": dna_model},
            "structure": {
                "source": atomic.source, "dna_form": atomic.dna_form, "n_atoms": atomic.n_atoms,
                "metadata": atomic.metadata, "warning": structure_warning,
            },
            "projection": {
                "literal_mode": "Axial atomic density",
                "selected_mode": projection_mode,
                "target_transform": target_transform,
                "target_smoothing_px": float(target_smoothing_px),
                "inverse_regularization": float(inverse_regularization),
                "point_weighting": atom_weighting,
                "atom_kernel": atom_kernel,
                "helical_pitch_A": effective_pitch_A,
                "target_turn_start_bp": int(target_turn_start_bp),
                "target_turn_bp": int(target_turn_bp),
                "width_A": selected_proj.width_A,
                "height_A": selected_proj.height_A,
                "element_counts": selected_proj.element_counts,
            },
            "resonator": {
                "radius_m": float(membrane_radius_mm) / 1000.0,
                "wave_speed_m_s": float(membrane_speed),
                "model": resonator_model,
                "plate_thickness_m": float(plate_thickness_mm) / 1000.0,
                "plate_material": str(plate_material),
                "plate_young_pa": float(young_pa),
                "plate_density_kg_m3": float(density_kg_m3),
                "plate_poisson": float(poisson),
                "actuator_r_fraction": float(actuator_r_fraction),
                "actuator_theta_deg": float(actuator_theta_deg),
                "fit_note": "The inverse fit uses a linear, time-averaged modal observable model. The primary physical WAV preserves exact theoretical/measured resonance frequencies; the predicted sand artwork is the complement of displacement observable when appropriate. Physical output must be calibrated on the actual resonator.",
                "calibration_applied": calibration_applied,
                "drive_frequency_column": drive_frequency_column,
            },
            "modes": modes.to_dict(orient="records"),
            "fit": {
                "molecular_vs_conditioned_rmse": float(np.sqrt(np.mean((molecular_target - resonator_target) ** 2))),
                "rmse": float(modes["fit_rmse"].iloc[0]) if not modes.empty else None,
                "correlation": float(modes["fit_correlation"].iloc[0]) if not modes.empty else None,
                "relative_error": float(modes["relative_fit_error"].iloc[0]) if not modes.empty else None,
            },
            "physical_mix_events": physical_mix_events,
            "physical_events": physical_events,
            "best_mode_events": best_mode_events,
            "musical_events": musical_events,
            "musical_generator": {
                "arrangement": str(musical_arrangement),
                "combination_count": int(musical_combination_count),
                "min_combination_size": int(musical_min_combination_size),
                "max_combination_size": int(musical_max_combination_size),
                "seed": int(musical_combination_seed),
                "include_all_tones": bool(musical_include_all_tones),
                "combination_render": str(musical_combination_render),
                "combination_beats": float(musical_combination_beats),
                "melody_rule": "Every source tone once -> random subsets -> optional all-tones synthesis event.",
            },
            "summary": summary,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in [literal_png, target_png, resonator_target_png, spectrum_png, modes_png, sand_png, fit_png, atomic_pdb, physical_mix_wav, physical_wav, best_mode_wav, musical_wav, musical_melody_csv, mode_csv, harmonic_csv, spectrum_csv, json_path]:
                z.write(p, p.name)

        note_lines = [
            f"**{len(seq):,} bp** · GC **{summary['GC_percent']:.1f}%** · estimated turns **{summary['estimated_turns']:.2f}",
            f"3D structure: **{atomic.source}** · **{atomic.n_atoms:,} modeled heavy-atom sites**",
            f"2D molecular target: **{projection_mode}** · transform: **{target_transform}**",
            f"Atomic weighting: **{atom_weighting}** · kernel: **{atom_kernel}** · pitch for folding: **{effective_pitch_A:.3f} Å** · target window: **bp {int(target_turn_start_bp)}–{int(target_turn_start_bp)+int(target_turn_bp)-1}**",
            f"Resonator model: **{resonator_model}** · radius **{float(membrane_radius_mm):.1f} mm** · plate **{float(plate_thickness_mm):.3f} mm {plate_material}** · membrane speed **{float(membrane_speed):.1f} m/s** (membrane only)",
            f"Resonator conditioning: Gaussian smoothing **{float(target_smoothing_px):.1f} px** · molecular→conditioned RMSE **{float(np.sqrt(np.mean((molecular_target - resonator_target) ** 2))):.4f}**",
            f"Inverse modal fit: **RMSE {float(modes['fit_rmse'].iloc[0]):.4f}** · **correlation {float(modes['fit_correlation'].iloc[0]):.3f}** · **relative error {float(modes['relative_fit_error'].iloc[0]):.3f}**",
        ]
        if structure_warning:
            note_lines.append(f"⚠️ {structure_warning}")
        note_lines += [
            "",
            "**Physical-drive interpretation:** the **MODE MIXTURE WAV** is the primary inverse-model drive. It is a time-averaged multimode approximation, not a guarantee of one static Chladni figure. "
            "The predicted sand artwork is derived from the fitted observable. The sequential WAV is for resonance isolation/calibration, the BEST SINGLE MODE WAV isolates the strongest single-mode match, and the musical WAV is a separate creative sonification. The DNA Combination Melody arrangement includes each source tone once, then random tone subsets, then an optional all-tone synthesis event. When motif repetition is OFF, the musical WAV is automatically truncated to its actual generated event length (no silent tail); when ON, the complete combinatorial grammar repeats to the requested duration."
        ]

        return (
            _plotly_atomic_3d(atomic),
            _image_plot(literal_proj.image, "Literal axial atomic-density projection", cmap="magma", xlabel="Å", ylabel="Å"),
            _image_plot(molecular_target, f"Molecular artwork target — {projection_mode} + {target_transform}", cmap="magma", xlabel="normalized X", ylabel="normalized Y"),
            _image_plot(resonator_target, f"Resonator-conditioned target — smoothing {float(target_smoothing_px):.1f} px", cmap="magma", xlabel="normalized X", ylabel="normalized Y"),
            _image_plot(spectrum, "Spatial Fourier spectrum", cmap="viridis"),
            _image_plot(mode_recon, f"{resonator_model} predicted observable ({target_type})", cmap="magma"),
            _image_plot(sand_prediction, "Predicted cymatics sand/nodal artwork proxy", cmap="magma"),
            _fit_plot(target, mode_recon),
            modes,
            pd.DataFrame(musical_events),
            str(musical_melody_csv),
            "\n\n".join(note_lines),
            str(physical_mix_wav),
            str(physical_wav),
            str(best_mode_wav),
            str(musical_wav),
            str(bundle_path),
            {"molecular_target": molecular_target, "resonator_target": resonator_target, "mode_reconstruction": mode_recon, "sand_prediction": sand_prediction},
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


# --------------------------- experimental verification ---------------------------

def verify_cymatics(dna_reference, reference_kind, measured_image, threshold):
    if dna_reference is None:
        raise gr.Error("Generate a DNA target first.")
    if measured_image is None:
        raise gr.Error("Upload a measured cymatics image first.")
    try:
        measured = np.asarray(measured_image)
        if measured.ndim == 3:
            measured = np.mean(measured[..., :3], axis=2)
        if isinstance(dna_reference, dict):
            key = "resonator_target" if str(reference_kind).startswith("Resonator") else "molecular_target"
            reference = dna_reference.get(key)
        else:
            reference = np.asarray(dna_reference, dtype=float)
        if reference is None:
            raise gr.Error("Selected reference target is not available.")
        result = image_registration_metrics(np.asarray(reference, dtype=float), measured, threshold=float(threshold))
        aligned = result.pop("aligned_image")
        md = "### Experimental verification\n\n" + "\n".join(f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in result.items())
        md += (
            "\n\nThe registration is an image-processing step; it does not prove frequency causality. "
            "For a defensible physical validation, record the resonator geometry, boundary condition, actuator location, "
            "drive frequency, drive amplitude, and the camera image at the same time."
        )
        return _difference_plot(np.asarray(reference), aligned), md
    except Exception as exc:
        raise gr.Error(str(exc)) from exc



with gr.Blocks(title="DNA → Cymatics → Music") as demo:
    gr.Markdown(
        """
# DNA → 3D Atomic Geometry → 2D Molecular Pattern → Inverse Cymatics → Tones

The primary objective is to derive a 2-D molecular pattern from a DNA sequence and then synthesize the resonant tones whose *mode mixture* best reproduces that target on an idealized circular membrane. Music is a separate creative rendering of the same frequency signature.

This research application separates three different things that should not be conflated:

**(1) molecular structure**, **(2) a 2D target pattern derived from molecular coordinates**, and **(3) a physical resonator response**.

The canonical sequence is **zebrafish hoxb1a-201**, 1,507 nt. The default sequence-only structure path is a **pure-Python parametric heavy-atom geometry model**. It requires no external molecular builder. A real PDB/mmCIF structure can be uploaded when experimentally determined coordinates are available.

> A literal side projection of a long DNA molecule is expected to be line-like. For the snowflake/flower-like artwork target, the app uses an **axial cross-sectional projection of one helical turn** by default. A selected turn is an actual local subset of the sequence; the rolling-turn ensemble is a sequence-wide derived signature.
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
                    [INTERNAL_PARAMETRIC_SOURCE, UPLOADED_STRUCTURE_SOURCE],
                    value=INTERNAL_PARAMETRIC_SOURCE, label="3D structure source"
                )
                dna_form = gr.Dropdown(["A-DNA", "B-DNA", "C-DNA", "Z-DNA (idealized left-handed)", "A-RNA"], value="B-DNA", label="Nucleic-acid reference conformation")
                pdb_file = gr.File(file_types=[".pdb", ".cif", ".mmcif"], type="filepath", label="Optional PDB/mmCIF structure")
                dna_status = gr.Markdown("Sequence structure source: internal parametric model; no external molecular builder required.")
            with gr.Column(scale=1):
                projection_mode = gr.Radio(
                    ["Axial atomic density", "Single-turn axial density", "Helical phase-folded density", "Rolling-turn ensemble axial density"],
                    value="Rolling-turn ensemble axial density", label="2D molecular projection / target source",
                    info="Literal axial = whole molecule. Single-turn = one pitch window. Phase-folded = all turns aligned by helical phase. Rolling-turn ensemble = sequence-wide sliding one-turn projections averaged into a 2-D molecular signature."
                )
                atom_weighting = gr.Radio(["Uniform", "Atomic mass", "Electron count proxy", "VDW volume"], value="Electron count proxy", label="Atomic density weighting")
                atom_kernel = gr.Radio(["Element VDW Gaussian", "Point Gaussian"], value="Element VDW Gaussian", label="Atomic footprint model")
                projection_size = gr.Slider(128, 1024, value=384, step=64, label="2D resolution")
                atomic_blur_A = gr.Slider(0.0, 2.0, value=0.35, step=0.05, label="Atomic density blur (Å)")
                helical_pitch_A = gr.Slider(28, 60, value=34.0, step=0.1, label="Helical pitch for phase fold (Å)")
                auto_pitch = gr.Checkbox(value=True, label="Estimate helical pitch from sequence-dependent step geometry")
                target_turn_start_bp = gr.Number(value=1, minimum=1, maximum=200000, step=1, label="One-turn target start bp", info="Used for Single-turn axial density; clamped to sequence length.")
                target_turn_bp = gr.Slider(6, 20, value=11, step=1, label="One-turn target length (bp)", info="~10–11 bp for B-DNA; choose 10–12 to explore local sequence shape.")
                target_transform = gr.Radio(["Density", "Edge / nodal geometry"], value="Density", label="Molecular → cymatics target transform", info="The molecular density is the artwork target. The inverse solver fits a physical displacement observable; use Nodal / sand only as an exploratory phenomenological particle model.")
                target_smoothing_px = gr.Slider(0, 12, value=4, step=0.5, label="Resonator target smoothing (pixels)", info="Suppresses atomic-scale detail the macroscopic resonator cannot reproduce.")
                resonator_model = gr.Radio(["Clamped circular thin plate", "Circular membrane"], value="Clamped circular thin plate", label="Physical resonator model")
                membrane_radius_mm = gr.Slider(25, 500, value=150, step=5, label="Circular resonator radius (mm)")
                membrane_speed = gr.Slider(10, 2000, value=120, step=5, label="Membrane wave speed (m/s)", info="Used only by Circular membrane model.")
                plate_thickness_mm = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Plate thickness (mm)", info="Used by clamped circular thin-plate model.")
                plate_material = gr.Dropdown(list(PLATE_MATERIALS.keys()), value="Steel", label="Plate material", info="E, density and Poisson ratio are taken from the built-in reference table.")
                max_angular_mode = gr.Slider(0, 32, value=18, step=1, label="Maximum angular mode m")
                max_radial_mode = gr.Slider(1, 20, value=10, step=1, label="Maximum radial mode n")
                top_modes = gr.Slider(1, 20, value=8, step=1, label="Top resonant modes")
                target_type = gr.Radio(["Nodal / sand", "Displacement magnitude", "Displacement power"], value="Displacement power", label="Resonator observable model")
                candidate_pool = gr.Slider(8, 72, value=24, step=4, label="Inverse-fit candidate pool")
                inverse_regularization = gr.Slider(0.0, 0.05, value=0.002, step=0.0005, label="Inverse-fit L2 regularization", info="Small values favor stable/sparser mode solutions without changing the physical mode frequencies.")
                node_width = gr.Slider(0.03, 0.3, value=0.08, step=0.01, label="Nodal accumulation width")
                actuator_r_fraction = gr.Slider(0.0, 0.9, value=0.35, step=0.05, label="Actuator radial position r/R")
                actuator_theta_deg = gr.Slider(0, 360, value=0, step=5, label="Actuator angle (deg)")
                calibration_csv = gr.File(file_types=[".csv"], type="filepath", label="Optional resonator calibration CSV")
                gr.Markdown("CSV columns: `frequency_Hz` and optional `response`. Nearest measured resonance can replace the theoretical drive frequency.")
                calibration_tolerance_hz = gr.Slider(10, 1000, value=250, step=10, label="Calibration frequency matching tolerance (Hz)")
                use_calibrated_frequency = gr.Checkbox(value=True, label="Use measured resonance frequencies when calibration CSV is supplied")
                musical_quantization = gr.Radio(["none", "chromatic"], value="chromatic", label="Musical pitch quantization")
                musical_arrangement = gr.Radio(["DNA Combination Melody", "Salience contour", "Frequency ascending", "Angular symmetry"], value="DNA Combination Melody", label="Musical note arrangement", info="DNA Combination Melody guarantees every source tone, explores random combinations, and ends with an optional all-tone synthesis event.")
                musical_combination_count = gr.Slider(0, 48, value=12, step=1, label="Random combination events")
                musical_min_combination_size = gr.Slider(2, 8, value=2, step=1, label="Minimum combination size")
                musical_max_combination_size = gr.Slider(2, 20, value=8, step=1, label="Maximum combination size", info="Clamped automatically to one less than the number of available source tones.")
                musical_combination_seed = gr.Number(value=0, precision=0, label="Combination seed", info="Same seed + same tone count/order = same combinatorial melody. Change it to create another melody.")
                musical_include_all_tones = gr.Checkbox(value=True, label="Finish with combination of ALL tones")
                musical_combination_render = gr.Radio(["Arpeggio + chord", "Arpeggio", "Chord"], value="Arpeggio + chord", label="Combination rendering", info="Arpeggio + chord exposes each member melodically, then confirms the combination harmonically.")
                musical_combination_beats = gr.Slider(0.5, 4.0, value=1.5, step=0.25, label="Beats per combination")
                musical_beats_per_note = gr.Slider(0.25, 2.0, value=0.75, step=0.25, label="Beats per individual tone")
                musical_interval_compression = gr.Slider(0.45, 1.0, value=0.70, step=0.05, label="Musical interval compression", info="1.0 preserves physical frequency ratios; lower values compress extreme jumps into a more practical melodic range.")
                musical_repeat_to_target = gr.Checkbox(value=False, label="Repeat motif to target duration", info="OFF: WAV is automatically truncated to actual generated audio. ON: repeat the motif until the requested duration, then trim exactly.")
                physical_duration_per_mode_s = gr.Slider(0.5, 10, value=2.0, step=0.5, label="Calibration seconds per mode")
                physical_mix_duration_s = gr.Slider(3, 60, value=12, step=1, label="PRIMARY physical mode-mixture duration (s)")
                musical_duration_s = gr.Slider(4, 120, value=24, step=1, label="Musical WAV target duration (s)")
                tempo_bpm = gr.Slider(40, 180, value=96, step=1, label="Musical tempo (BPM)")
                run = gr.Button("Build DNA target + inverse resonant tone set", variant="primary")

        gr.Markdown("## Results")
        gr.Markdown("**Interpretation:** the molecular artwork target is derived from the DNA atom coordinates. The inverse solver searches the selected circular resonator basis for a sparse, physically driveable mode mixture. By default it fits **time-averaged displacement power** because that is a linear positive observable for distinct-frequency modes. The displayed sand/nodal artwork is a visualization proxy, not a particle-dynamics simulation. The PRIMARY physical-drive WAV contains the exact theoretical frequencies (or calibrated measured frequencies when supplied).")
        with gr.Row():
            out_3d = gr.Plot(label="3D structure")
            out_projection_literal = gr.Plot(label="Literal axial atomic projection")
        with gr.Row():
            out_projection_target = gr.Plot(label="Molecular artwork target")
            out_projection_resonator_target = gr.Plot(label="Resonator-conditioned target")
        with gr.Row():
            out_recon = gr.Plot(label="Predicted resonator observable")
            out_sand = gr.Plot(label="Predicted cymatics sand/nodal artwork proxy")
        out_fit = gr.Plot(label="DNA target vs modal-mixture fit")
        out_spectrum = gr.Plot(label="Spatial Fourier spectrum")
        out_table = gr.Dataframe(label="Candidate resonator modes", wrap=True)
        out_melody = gr.Dataframe(label="Generated combinatorial melody plan", wrap=True, interactive=False)
        out_melody_file = gr.File(label="Download combinatorial melody CSV")
        out_summary = gr.Markdown()
        with gr.Row():
            physical_mix_audio = gr.Audio(label="PRIMARY physical drive — simultaneous fitted mode mixture (exact Hz)", type="filepath")
            physical_audio = gr.Audio(label="Calibration drive — sequential fitted modes (exact Hz)", type="filepath")
            best_mode_audio = gr.Audio(label="Single-mode isolation drive (exact Hz)", type="filepath")
            musical_audio = gr.Audio(label="Musical sonification WAV (creative; not a physical drive)", type="filepath")
        out_bundle = gr.File(label="Full analysis bundle")
        reference_state = gr.State(None)

        run.click(
            fn=run_pipeline,
            inputs=[sequence, dna_model, structure_source, dna_form, pdb_file,
                    projection_mode, atom_weighting, atom_kernel, projection_size, atomic_blur_A, helical_pitch_A, auto_pitch, target_turn_start_bp, target_turn_bp,
                    target_transform, target_smoothing_px, resonator_model, membrane_radius_mm, membrane_speed, plate_thickness_mm, plate_material,
                    max_angular_mode, max_radial_mode, top_modes,
                    candidate_pool, node_width, actuator_r_fraction, actuator_theta_deg, inverse_regularization, calibration_csv, calibration_tolerance_hz, use_calibrated_frequency,
                    target_type, physical_duration_per_mode_s, physical_mix_duration_s, musical_quantization, musical_duration_s, tempo_bpm, musical_arrangement, musical_beats_per_note, musical_repeat_to_target, musical_interval_compression,
                    musical_combination_count, musical_min_combination_size, musical_max_combination_size, musical_combination_seed, musical_include_all_tones, musical_combination_render, musical_combination_beats],
            outputs=[out_3d, out_projection_literal, out_projection_target, out_projection_resonator_target, out_spectrum, out_recon, out_sand, out_fit, out_table, out_melody, out_melody_file, out_summary,
                     physical_mix_audio, physical_audio, best_mode_audio, musical_audio, out_bundle, reference_state],
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

The comparison includes translation + in-plane rotation registration, RMSE, Pearson correlation, Dice, IoU, a 95th-percentile boundary-distance metric, and angular-harmonic correlation.

A high image similarity score is evidence of pattern similarity; it is not, by itself, proof that the DNA caused that physical frequency or that the physical system is uniquely identified.
            """
        )
        reference_kind = gr.Radio(["Molecular target", "Resonator-conditioned target"], value="Resonator-conditioned target", label="Reference target for verification")
        measured_image = gr.Image(type="numpy", image_mode="L", label="Measured Chladni / cymatics image")
        threshold = gr.Slider(0.1, 0.9, value=0.55, step=0.01, label="Binary overlap threshold")
        verify = gr.Button("Compare measured pattern with selected target", variant="primary")
        out_diff = gr.Plot(label="Difference map")
        out_metrics = gr.Markdown()
        verify.click(fn=verify_cymatics, inputs=[reference_state, reference_kind, measured_image, threshold], outputs=[out_diff, out_metrics])

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

The **axial atomic-density image is a true projection of the loaded atom coordinates**. The **helical phase-folded and rolling-turn ensemble images are derived coordinate transforms/signatures**, not literal camera projections of the full gene. Uploaded PDB/mmCIF coordinates remain the preferred source for quantitative structural analysis. The circular membrane solver is an ideal tension-dominated membrane; real Chladni plates require measured/calibrated resonator models, boundary conditions, damping, and actuator coupling.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, theme=gr.themes.Soft())
