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

from dna_cymatics import (
    PLATE_MATERIALS,
    build_dna_structure,
    clean_sequence,
    create_mapped_audio,
    density_projection,
    dna_summary,
    fft_components,
    image_metrics,
    map_components,
    rank_square_plate_modes,
    reconstruct_from_components,
    hz_to_note,
    quantize_frequency,
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


def _plotly_3d(dna):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=dna.strand1[:, 0], y=dna.strand1[:, 1], z=dna.strand1[:, 2],
        mode="lines+markers", name="Strand 1", marker=dict(size=3), line=dict(width=6)
    ))
    fig.add_trace(go.Scatter3d(
        x=dna.strand2[:, 0], y=dna.strand2[:, 1], z=dna.strand2[:, 2],
        mode="lines+markers", name="Strand 2", marker=dict(size=3), line=dict(width=6)
    ))
    for i, edge in enumerate(dna.basepair_edges):
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2], mode="lines",
            showlegend=False, line=dict(width=2), hovertext=[f"bp {i+1}", f"bp {i+1}"],
        ))
    fig.update_layout(
        title="Sequence-dependent coarse-grained dsDNA",
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
    b2 = b
    if b.shape != a.shape:
        from scipy.ndimage import zoom
        b2 = zoom(b, (a.shape[0] / b.shape[0], a.shape[1] / b.shape[1]), order=1)
    diff = np.abs(a - b2)
    return _image_plot(diff, "Absolute target / measured difference", cmap="inferno")


def _sequence_choices_from_bundle(bundle):
    choices = list(bundle.sequence_catalog().keys())
    # Prefer the canonical hoxb1a-201 mature mRNA when multiple sequence classes are present.
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


def fetch_gene_action(query, upstream_bp):
    try:
        bundle = retrieve_gene_sequences(query, int(upstream_bp))
        choices = _sequence_choices_from_bundle(bundle)
        default_choice = next((c for c in choices if c.startswith("hoxb1a-201") and "mRNA/cDNA" in c), choices[0])
        catalog = bundle.sequence_catalog()
        status_lines = [
            f"### Retrieved **{bundle.symbol}**",
            f"Assembly: **{bundle.assembly}** ({bundle.assembly_accession}) · chromosome **{bundle.chromosome}** · "
            f"coordinates **{bundle.start:,}–{bundle.end:,}** · strand **{"plus" if bundle.strand == 1 else "minus"}**",
            f"Genomic sequence: **{len(bundle.genomic_sequence):,} nt** · promoter/upstream: **{len(bundle.promoter_sequence):,} nt**",
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


def run_pipeline(
    sequence, dna_model, projection, projection_size, blur_px, top_components,
    canvas_width_mm, mapping_mode, pattern_target, plate_material, thickness_mm,
    membrane_speed, musical_base, musical_octaves, frequency_multiplier,
    quantization, audio_style, duration_s, tempo_bpm, beats_per_note,
):
    try:
        seq = clean_sequence(sequence)
        dna = build_dna_structure(seq, model=dna_model)
        proj = density_projection(dna, projection=projection, size=int(projection_size), blur_px=float(blur_px))

        canvas_width_m = float(canvas_width_mm) / 1000.0
        comps, spectrum = fft_components(proj.image, canvas_width_m, top_n=int(top_components))
        E, rho, nu = PLATE_MATERIALS[plate_material]

        mode_note = ""
        if mapping_mode == "Square plate mode search":
            mode_df, recon = rank_square_plate_modes(
                proj.image,
                width_m=canvas_width_m,
                height_m=canvas_width_m,
                thickness_m=float(thickness_mm) / 1000.0,
                young_pa=E,
                density_kg_m3=rho,
                poisson=nu,
                max_mode=12,
                top_modes=int(top_components),
                target_type=pattern_target,
            )
            freq_df = pd.DataFrame({
                "component": mode_df["mode_rank"],
                "fx_cycles/m": np.nan,
                "fy_cycles/m": np.nan,
                "spatial_cycles/m": np.nan,
                "spectral_weight": mode_df["mixture_weight"],
                "phase_deg": np.nan,
                "plate_Hz": mode_df["frequency_Hz"],
                "membrane_Hz": np.nan,
                "musical_Hz": mode_df["frequency_Hz"],
                "mapped_Hz": mode_df["frequency_Hz"] * float(frequency_multiplier),
                "note_Hz": mode_df["frequency_Hz"] * float(frequency_multiplier),
                "note": [hz_to_note(float(f) * float(frequency_multiplier)) for f in mode_df["frequency_Hz"]],
                "pattern_correlation": mode_df["pattern_correlation"],
                "m": mode_df["m"], "n": mode_df["n"],
            })
            if quantization == "chromatic":
                q = [quantize_frequency(float(f), "chromatic")[0] for f in freq_df["note_Hz"]]
                freq_df["note_Hz"] = q
                freq_df["note"] = [hz_to_note(float(f)) for f in q]
            mode_note = (
                "The mode table ranks individual simply-supported analytical modes by spatial similarity. "
                "Different non-degenerate frequencies do not produce one static Chladni figure simultaneously."
            )
            from scipy.ndimage import zoom
            recon = zoom(recon, (proj.image.shape[0] / recon.shape[0], proj.image.shape[1] / recon.shape[1]), order=1)
        else:
            freq_df = map_components(
                comps,
                canvas_width_m=canvas_width_m,
                mapping_mode=mapping_mode,
                thickness_m=float(thickness_mm) / 1000.0,
                young_pa=E,
                density_kg_m3=rho,
                poisson=nu,
                membrane_speed_m_s=float(membrane_speed),
                musical_base_hz=float(musical_base),
                musical_octaves=float(musical_octaves),
                frequency_multiplier=float(frequency_multiplier),
                quantization=quantization,
            )
            recon = reconstruct_from_components(proj.image, comps)
            mode_note = (
                "The FFT mapping is a reproducible sonification of spatial structure. It is not a unique physical "
                "inverse of a real plate."
            )

        audio, note_sequence = create_mapped_audio(
            freq_df, style=audio_style, duration_s=float(duration_s), tempo_bpm=float(tempo_bpm),
            beats_per_note=float(beats_per_note),
        )

        run_dir = OUTPUT_ROOT / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        wav_path = run_dir / "dna_cymatics.wav"
        csv_path = run_dir / "frequency_components.csv"
        p_png = run_dir / "dna_projection.png"
        s_png = run_dir / "spatial_spectrum.png"
        r_png = run_dir / "reconstruction.png"
        json_path = run_dir / "analysis.json"
        bundle_path = run_dir / "dna_cymatics_bundle.zip"

        sf.write(wav_path, audio, 44_100, subtype="PCM_16")
        freq_df.to_csv(csv_path, index=False)

        figures = [
            (p_png, proj.image, f"DNA 2D projection — {projection}", "magma"),
            (s_png, spectrum, "Log spatial spectrum", "viridis"),
            (r_png, recon, "Model reconstruction / modal mixture", "magma"),
        ]
        for path, arr, title, cmap in figures:
            fig = _image_plot(arr, title, cmap=cmap)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        summary = dna_summary(dna)
        payload = {
            "dna_summary": summary,
            "settings": {
                "dna_model": dna_model, "projection": projection,
                "projection_size": int(projection_size), "blur_px": float(blur_px),
                "top_components": int(top_components), "canvas_width_mm": float(canvas_width_mm),
                "mapping_mode": mapping_mode, "pattern_target": pattern_target,
                "plate_material": plate_material, "thickness_mm": float(thickness_mm),
                "membrane_speed_m_s": float(membrane_speed), "frequency_multiplier": float(frequency_multiplier),
                "quantization": quantization, "audio_style": audio_style,
            },
            "note_sequence": [{"frequency_Hz": float(f), "note": n} for f, n in note_sequence],
            "components": freq_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in [wav_path, csv_path, p_png, s_png, r_png, json_path]:
                z.write(p, p.name)

        text = (
            f"**{len(seq):,} bp** · GC **{summary['GC_percent']:.1f}%** · "
            f"estimated turns **{summary['estimated_turns']:.2f}**\n\n{mode_note}\n\n"
            f"Generated **{len(freq_df)}** frequency/mode entries and a **{float(duration_s):.1f} s** WAV."
        )
        return (
            _plotly_3d(dna),
            _image_plot(proj.image, f"DNA 2D projection — {projection}", cmap="magma", xlabel=proj.x_label, ylabel=proj.y_label),
            _image_plot(spectrum, "Log spatial spectrum", cmap="viridis"),
            _image_plot(recon, "Model reconstruction / modal mixture", cmap="magma"),
            freq_df, text, str(wav_path), str(bundle_path),
            proj.image,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def verify_cymatics(dna_reference, measured_image, threshold):
    if dna_reference is None:
        raise gr.Error("Generate a DNA projection first.")
    if measured_image is None:
        raise gr.Error("Upload or capture a measured cymatics image first.")
    try:
        metrics = image_metrics(np.asarray(dna_reference, dtype=float), np.asarray(measured_image), threshold=float(threshold))
        measured_gray = np.asarray(measured_image)
        if measured_gray.ndim == 3:
            measured_gray = np.mean(measured_gray[..., :3], axis=2)
        ref = np.asarray(dna_reference, dtype=float)
        from scipy.ndimage import zoom
        measured_gray = zoom(measured_gray, (ref.shape[0] / measured_gray.shape[0], ref.shape[1] / measured_gray.shape[1]), order=1)
        diff_fig = _difference_plot(ref, measured_gray)
        md = "### Experimental verification\n\n" + "\n".join(f"**{k}:** {v:.4f}" for k, v in metrics.items())
        md += (
            "\n\nThese are image-similarity metrics only. They do not establish that the plate is being driven "
            "in the intended eigenmode unless the drive frequency, plate geometry, boundary condition and "
            "actuator coupling have also been recorded."
        )
        return diff_fig, md
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="DNA → Cymatics → Music") as demo:
    gr.Markdown(
        """
# DNA → 3D Shape → 2D Pattern → Frequencies → Music

This app performs a reproducible **DNA sequence → coarse-grained 3D geometry → 2D projection → spatial-frequency → audio** workflow and includes a database-backed sequence retrieval layer.

The initial canonical sequence is **hoxb1a-201**, the 1,507-nt zebrafish mature mRNA/cDNA sequence referenced by ZFIN. The retrieval tab can load current genomic DNA from **GRCz12tu**, all Ensembl transcripts and CDS sequences, NCBI RefSeq transcript cross-checks, and a configurable promoter/upstream region.

> **Physical interpretation:** DNA reconstruction is coarse-grained, the FFT route is sonification, and the plate-mode search is an analytical simply-supported plate model. Real cymatics depends on the actual resonator, boundary conditions, damping, actuator coupling, and measurement system.
        """
    )

    with gr.Tab("Gene / Sequence Retrieval"):
        with gr.Row():
            with gr.Column(scale=2):
                gene_query = gr.Textbox(
                    value=DEFAULT_GENE_QUERY,
                    label="Gene identifier / symbol / ZFIN URL",
                    info="Example: ZDB-GENE-990415-101, hoxb1a, or an NCBI GeneID.",
                )
            with gr.Column(scale=1):
                upstream_bp = gr.Slider(0, 10_000, value=DEFAULT_UPSTREAM_BP, step=100, label="Promoter / upstream length (bp)")
                fetch_button = gr.Button("Fetch current gene sequences", variant="primary")
        retrieval_status = gr.Markdown()
        retrieval_table = gr.Dataframe(label="Retrieved sequence inventory", wrap=True, interactive=False)
        with gr.Row():
            sequence_choice = gr.Dropdown(label="Sequence to load into DNA workflow", choices=[], interactive=True)
            sequence_download = gr.File(label="Download retrieved FASTA package")
        retrieval_catalog = gr.State({})

    with gr.Tab("DNA → Pattern → Music"):
        with gr.Row():
            with gr.Column(scale=1):
                sequence = gr.Textbox(
                    value=SAMPLE_DNA,
                    lines=8,
                    label="DNA sequence (5′→3′)",
                    info="Default: hoxb1a-201 mature mRNA/cDNA (1,507 nt). Genomic DNA and CDS can be loaded from the retrieval tab.",
                )
                dna_model = gr.Radio(["sequence-dependent", "canonical"], value="sequence-dependent", label="DNA geometry model")
                projection = gr.Dropdown(["XZ (side)", "XY (top)", "YZ (side)", "PCA"], value="XZ (side)", label="2D projection")
                projection_size = gr.Slider(128, 1024, value=512, step=64, label="Projection resolution")
                blur_px = gr.Slider(0.0, 6.0, value=1.4, step=0.1, label="2D rendering blur (pixels)")
                top_components = gr.Slider(4, 48, value=16, step=1, label="Frequency / mode count")
            with gr.Column(scale=1):
                canvas_width_mm = gr.Slider(50, 1000, value=300, step=10, label="Physical plate width (mm)")
                mapping_mode = gr.Radio(
                    ["Thin plate", "Membrane", "Musical radial", "Square plate mode search"],
                    value="Square plate mode search", label="Frequency model"
                )
                pattern_target = gr.Radio(["DNA density", "Nodal / sand"], value="Nodal / sand", label="Target interpretation")
                plate_material = gr.Dropdown(list(PLATE_MATERIALS.keys()), value="Steel", label="Plate material")
                thickness_mm = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Plate thickness (mm)")
                membrane_speed = gr.Slider(20, 1000, value=120, step=5, label="Membrane wave speed (m/s)")
                musical_base = gr.Slider(55, 440, value=110, step=5, label="Musical base frequency (Hz)")
                musical_octaves = gr.Slider(1, 8, value=4, step=0.25, label="Musical octave span")
                frequency_multiplier = gr.Slider(0.01, 100, value=1.0, step=0.01, label="Frequency multiplier")
                quantization = gr.Radio(["chromatic", "none"], value="chromatic", label="Musical quantization")
                audio_style = gr.Radio(["Arpeggio", "Chord", "Arpeggio + Drone"], value="Arpeggio", label="Audio style")
                duration_s = gr.Slider(4, 60, value=16, step=1, label="Audio duration (s)")
                tempo_bpm = gr.Slider(40, 180, value=100, step=1, label="Tempo (BPM)")
                beats_per_note = gr.Slider(0.25, 4, value=1, step=0.25, label="Beats per note")
                run = gr.Button("Generate", variant="primary")

        gr.Markdown("## Results")
        with gr.Row():
            out_3d = gr.Plot(label="3D reconstruction")
            out_projection = gr.Plot(label="2D projection")
        with gr.Row():
            out_spectrum = gr.Plot(label="Spatial spectrum")
            out_recon = gr.Plot(label="Pattern reconstruction")
        out_table = gr.Dataframe(label="Derived frequencies / plate modes", wrap=True)
        out_summary = gr.Markdown()
        with gr.Row():
            out_audio = gr.Audio(label="Generated WAV", type="filepath")
            out_bundle = gr.File(label="Analysis bundle")
        reference_state = gr.State(None)

        run.click(
            fn=run_pipeline,
            inputs=[sequence, dna_model, projection, projection_size, blur_px, top_components,
                    canvas_width_mm, mapping_mode, pattern_target, plate_material, thickness_mm,
                    membrane_speed, musical_base, musical_octaves, frequency_multiplier,
                    quantization, audio_style, duration_s, tempo_bpm, beats_per_note],
            outputs=[out_3d, out_projection, out_spectrum, out_recon, out_table, out_summary, out_audio, out_bundle, reference_state],
        )

    with gr.Tab("Experimental Cymatics Verification"):
        gr.Markdown(
            """
### Close the loop with a real plate

1. Generate a DNA target in the first tab.
2. Build a plate/resonator matching the dimensions and material settings.
3. Drive the plate with a calibrated sine wave at one candidate resonance at a time.
4. Photograph the resulting sand/flour/nodal pattern.
5. Upload the image here and compare it with the generated target.

For a **single static Chladni figure**, test one resonance/mode at a time. Playing several unrelated notes together usually creates a time-varying superposition rather than one static target pattern.
            """
        )
        measured_image = gr.Image(type="numpy", image_mode="L", label="Measured Chladni / cymatics image")
        threshold = gr.Slider(0.1, 0.9, value=0.55, step=0.01, label="Binary overlap threshold")
        verify = gr.Button("Compare measured pattern with DNA target", variant="primary")
        out_diff = gr.Plot(label="Difference map")
        out_metrics = gr.Markdown()
        verify.click(fn=verify_cymatics, inputs=[reference_state, measured_image, threshold], outputs=[out_diff, out_metrics])

    # Wire cross-tab events after every referenced component has been constructed.
    fetch_button.click(
        fn=fetch_gene_action,
        inputs=[gene_query, upstream_bp],
        outputs=[retrieval_status, retrieval_table, sequence_choice, retrieval_catalog, sequence, sequence_download],
    )
    sequence_choice.change(fn=load_retrieved_sequence, inputs=[sequence_choice, retrieval_catalog], outputs=[sequence])

    gr.Markdown(
        """
### Model boundaries

The DNA geometry is a sequence-dependent coarse-grained reconstruction using six standard base-pair-step variables: Tilt, Roll, Twist, Shift, Slide and Rise. The mode search is an analytical **simply-supported** plate model. It is not a finite-element simulation of an arbitrary real plate. For laboratory-grade prediction, calibrate the real resonator and actuator and then use measured images/frequencies in the verification loop.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, theme=gr.themes.Soft())
