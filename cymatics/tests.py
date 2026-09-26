from __future__ import annotations

import tempfile
from pathlib import Path
import zipfile

import numpy as np

from dna_atomic import (
    atomic_density_projection,
    atomic_edge_target,
    align_atomic_structure,
    load_structure,
    pca_align_axis,
    build_parametric_atomic_dna,
    DNA_FORM_PRESETS,
    INTERNAL_PARAMETRIC_SOURCE,
)
from dna_cymatics import (
    STEP_PARAMS,
    build_dna_structure,
    clean_sequence,
    circular_membrane_mode_field,
    circular_membrane_mode_frequency,
    circular_clamped_plate_eigenvalue,
    circular_clamped_plate_mode_frequency,
    circular_clamped_plate_mode_field,
    prepare_resonator_target,
    observable_to_sand_artwork,
    create_musical_audio_from_modes,
    generate_combination_melody_plan,
    create_physical_drive_audio,
    create_simultaneous_physical_drive_audio,
    apply_resonator_calibration,
    image_registration_metrics,
    polar_harmonic_spectrum,
    rank_circular_membrane_modes,
)
from dna_sources import (
    GeneSequenceBundle,
    HOXB1A_201_SEQUENCE,
    TranscriptRecord,
    write_sequence_bundle,
)

SEQ = "ACGTACGTACGT"


def test_sequence():
    assert clean_sequence(SEQ) == SEQ
    assert len(clean_sequence(HOXB1A_201_SEQUENCE)) == 1507


def test_hoxb1a_canonical():
    assert len(HOXB1A_201_SEQUENCE) == 1507
    assert HOXB1A_201_SEQUENCE.startswith("TCTCGATTTCTCAGGTTGTCCCT")
    assert HOXB1A_201_SEQUENCE.endswith("ACGATTCCTTCAGTGTACACGGTGGCG")


def test_sequence_source_helpers():
    bundle = GeneSequenceBundle(
        query="hoxb1a", symbol="hoxb1a", ncbi_gene_id="30337", ensembl_gene_id="ENSDARG00160005162",
        zfin_gene_id="ZDB-GENE-990415-101", assembly="GRCz12tu", assembly_accession="GCF_049306965.2",
        chromosome="3", start=22508764, end=22510570, strand=1,
        genomic_accession="NC_133178.1", genomic_sequence="ACGT" * 20,
        promoter_sequence="TGCA" * 20, promoter_start=22506764, promoter_end=22508763,
        upstream_bp=2000,
        transcripts=[TranscriptRecord(name="hoxb1a-201", ensembl_id="ENSDART00000110682", cDNA=HOXB1A_201_SEQUENCE, cds="ATG" * 10)],
        ncbi_refseq_accessions=["NM_131115.2"],
        zfin_transcript_links={"hoxb1a-201": "https://zfin.org/ZDB-TSCRIPT-090929-7277"},
        zfin_current_assembly="GRCz12tu", source_status=[],
        sources={"zfin_gene": "https://zfin.org/ZDB-GENE-990415-101", "ncbi_gene": "https://www.ncbi.nlm.nih.gov/gene/30337", "ensembl_gene": "https://www.ensembl.org/"},
    )
    catalog = bundle.sequence_catalog()
    assert any(k.startswith("hoxb1a-201") for k in catalog)
    assert any("CDS" in k for k in catalog)


def test_sequence_bundle_write():
    bundle = GeneSequenceBundle(
        query="hoxb1a", symbol="hoxb1a", ncbi_gene_id="30337", ensembl_gene_id="ENSDART00000110682",
        zfin_gene_id="ZDB-GENE-990415-101", assembly="GRCz12tu", assembly_accession="GCF_049306965.2",
        chromosome="3", start=22508764, end=22510570, strand=1,
        genomic_accession="NC_133178.1", genomic_sequence="ACGT" * 50,
        promoter_sequence="TGCA" * 25, promoter_start=22506764, promoter_end=22508763,
        upstream_bp=2000,
        transcripts=[TranscriptRecord(name="hoxb1a-201", ensembl_id="ENSDART00000110682", cDNA=HOXB1A_201_SEQUENCE, cds="ATG" * 10)],
        ncbi_refseq_accessions=["NM_131115.2"], zfin_transcript_links={}, zfin_current_assembly="GRCz12tu",
        source_status=[], sources={},
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle.zip"
        write_sequence_bundle(bundle, out)
        with zipfile.ZipFile(out) as z:
            names = set(z.namelist())
            assert "genomic_GRCz12tu.fna" in names
            assert "promoter_upstream.fna" in names
            assert "metadata.json" in names


def test_geometry_and_atomic_projection():
    dna = build_dna_structure(SEQ)
    atomic = build_parametric_atomic_dna(dna, dna_form="B-DNA")
    assert atomic.n_atoms > 100
    aligned, _, _ = pca_align_axis(atomic.atoms)
    assert aligned.shape[1] == 3
    aligned2, _, _ = align_atomic_structure(atomic)
    assert aligned2.shape == atomic.atoms.shape
    proj = atomic_density_projection(atomic, mode="Axial atomic density", size=192, blur_A=0.2)
    assert proj.image.shape == (192, 192)
    assert 0.0 <= proj.image.min() <= proj.image.max() <= 1.0
    assert proj.width_A > 10.0
    one_turn = atomic_density_projection(atomic, mode="Single-turn axial density", size=160, blur_A=0.2, helical_pitch_A=34.0)
    assert one_turn.image.shape == (160, 160)
    folded = atomic_density_projection(atomic, mode="Helical phase-folded density", size=128, blur_A=0.2)
    assert folded.image.shape == (128, 128)
    rolling = atomic_density_projection(atomic, mode="Rolling-turn ensemble axial density", size=128, blur_A=0.2, helical_pitch_A=34.0)
    assert rolling.image.shape == (128, 128)
    assert np.isfinite(rolling.image).all()
    edges = atomic_edge_target(one_turn.image)
    assert edges.shape == one_turn.image.shape
    assert np.isfinite(edges).all()




def test_structure_source_contract():
    # This exact label is emitted by the UI and must be accepted by the pipeline.
    dna = build_dna_structure(SEQ, model="sequence-dependent")
    atomic = build_parametric_atomic_dna(dna, dna_form="B-DNA")
    assert atomic.source == INTERNAL_PARAMETRIC_SOURCE

def test_internal_parametric_builder():
    dna = build_dna_structure(SEQ, model="sequence-dependent")
    for form in DNA_FORM_PRESETS:
        atomic = build_parametric_atomic_dna(dna, dna_form=form)
        assert atomic.n_atoms > 100
        assert atomic.atoms.shape[1] == 3
        assert len(atomic.elements) == atomic.n_atoms
        assert len(atomic.names) == atomic.n_atoms
        assert "not force-field minimized" in atomic.metadata["model_status"]

    b = build_parametric_atomic_dna(dna, dna_form="B-DNA")
    assert {"P", "O", "N", "C"}.issubset(set(b.elements))
    assert any(name == "N9" for name in b.names)
    assert any(name == "C1'" for name in b.names)
    assert any(name == "C7" for name in b.names)
    assert np.all(np.isfinite(b.atoms))

def test_circular_modes_and_audio():
    field = circular_membrane_mode_field(6, 1, size=96, nodal=True)
    assert field.shape == (96, 96)
    f = circular_membrane_mode_frequency(6, 1, 0.15, 120.0)
    assert f > 0
    target = field
    modes, recon = rank_circular_membrane_modes(target, 0.15, 120.0, max_angular_mode=8, max_radial_mode=4, top_modes=5, fit_size=96)
    assert len(modes) == 5
    assert recon.shape == (96, 96)
    physical_mix, mix_events = create_simultaneous_physical_drive_audio(modes, duration_s=1.0)
    assert physical_mix.ndim == 1 and mix_events
    physical, events = create_physical_drive_audio(modes, duration_per_mode_s=0.2)
    musical, mevents = create_musical_audio_from_modes(modes, duration_s=30.0, quantization="chromatic")
    assert physical.ndim == musical.ndim == 1
    assert events and mevents
    # Natural-length mode must not pad the file to the requested maximum duration.
    assert len(musical) / 44_100.0 < 30.0
    # Repeat mode must fill the requested duration exactly (within one sample).
    repeated, repeated_events = create_musical_audio_from_modes(
        modes, duration_s=3.0, quantization="chromatic", repeat_to_target=True
    )
    assert repeated_events
    assert abs(len(repeated) / 44_100.0 - 3.0) < 1 / 44_100
    # Physical frequency ratios are used as the musical pre-quantization relationship.
    none_audio, none_events = create_musical_audio_from_modes(
        modes, duration_s=2.0, quantization="none", arrangement="Frequency ascending", interval_compression=0.60
    )
    assert none_audio.ndim == 1 and none_events




def test_combinatorial_melody_plan():
    plan = generate_combination_melody_plan(5, tone_order=[0, 1, 2, 3, 4], combination_count=10, min_combination_size=2, max_combination_size=4, seed=123, include_all_tones=True)
    singles = [e["tone_indices"][0] for e in plan if e["event_type"] == "single"]
    assert singles == [0, 1, 2, 3, 4]
    combos = [e["tone_indices"] for e in plan if e["event_type"] == "combination"]
    assert len(combos) == 10
    assert all(2 <= len(c) <= 4 for c in combos)
    assert plan[-1]["event_type"] == "all"
    assert set(plan[-1]["tone_indices"]) == set(range(5))
    # Same seed reproduces the same combinatorial melody; a changed seed changes it.
    plan2 = generate_combination_melody_plan(5, tone_order=[0, 1, 2, 3, 4], combination_count=10, min_combination_size=2, max_combination_size=4, seed=123, include_all_tones=True)
    plan3 = generate_combination_melody_plan(5, tone_order=[0, 1, 2, 3, 4], combination_count=10, min_combination_size=2, max_combination_size=4, seed=124, include_all_tones=True)
    assert [p["tone_indices"] for p in plan] == [p["tone_indices"] for p in plan2]
    assert [p["tone_indices"] for p in plan] != [p["tone_indices"] for p in plan3]
    # Two-tone edge case remains valid.
    two = generate_combination_melody_plan(2, seed=1, combination_count=10)
    assert [p["event_type"] for p in two] == ["single", "single", "all"]

def test_combinatorial_melody_audio():
    field = circular_membrane_mode_field(5, 1, size=64, nodal=True)
    modes, _ = rank_circular_membrane_modes(field, 0.15, 120.0, max_angular_mode=7, max_radial_mode=3, top_modes=5, fit_size=64)
    audio, events = create_musical_audio_from_modes(
        modes, duration_s=20.0, quantization="chromatic", arrangement="DNA Combination Melody",
        combination_count=8, min_combination_size=2, max_combination_size=4, combination_seed=99,
        include_all_tones=True, combination_render="Arpeggio + chord", combination_beats=1.25,
    )
    assert audio.ndim == 1 and np.isfinite(audio).all() and np.max(np.abs(audio)) <= 0.93
    assert any(e["event_type"] == "single" for e in events)
    assert any(e["event_type"] == "combination" for e in events)
    assert any(e["event_type"] == "all" for e in events)
    singleton_indices = {int(e["tone_indices"].strip("[]").split(",")[0]) for e in events if e["event_type"] == "single"}
    assert singleton_indices == set(range(len(modes)))


def test_resonator_calibration():
    dna = build_dna_structure(SEQ)
    target = circular_membrane_mode_field(4, 1, size=64, nodal=True)
    modes, _ = rank_circular_membrane_modes(target, 0.15, 120.0, max_angular_mode=6, max_radial_mode=3, top_modes=3, fit_size=64)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cal.csv"
        path.write_text("frequency_Hz,response\n100,0.2\n500,1.0\n1000,0.5\n", encoding="utf-8")
        cal = apply_resonator_calibration(modes, str(path), max_delta_hz=10000)
        assert "drive_frequency_Hz" in cal.columns
        assert "calibration_response" in cal.columns
        assert cal["calibrated"].all()


def test_clamped_plate_physics_and_inverse():
    # Fundamental clamped circular plate frequency for a 300-mm diameter, 1-mm steel
    # plate should be close to the analytical benchmark (~110.5 Hz for nu≈0.30).
    lam = circular_clamped_plate_eigenvalue(0, 1)
    assert 3.0 < lam < 3.5
    f = circular_clamped_plate_mode_frequency(0, 1, 0.15, 0.001, 200e9, 7850.0, 0.30)
    assert 109.0 < f < 112.5
    field = circular_clamped_plate_mode_field(4, 2, size=96, nodal=False)
    assert field.shape == (96, 96) and np.isfinite(field).all()
    target = prepare_resonator_target(field, smoothing_px=2.0)
    modes, recon = rank_circular_membrane_modes(
        target, 0.15, 120.0, max_angular_mode=8, max_radial_mode=4, top_modes=4,
        fit_size=64, resonator_model="Clamped circular thin plate",
        plate_thickness_m=0.001, plate_young_pa=200e9, plate_density_kg_m3=7850.0, plate_poisson=0.30,
        target_type="Displacement power", candidate_pool=12, inverse_regularization=0.001,
    )
    assert len(modes) == 4
    assert (modes["frequency_Hz"] > 0).all()
    assert "radial_eigenvalue" in modes.columns
    sand = observable_to_sand_artwork(recon, "Displacement power")
    assert sand.shape == recon.shape and np.isfinite(sand).all()

def test_harmonics_and_registration():
    dna = build_dna_structure(SEQ)
    target = atomic_density_projection(build_parametric_atomic_dna(dna, dna_form="B-DNA"), size=128).image
    harmonics = polar_harmonic_spectrum(target, max_m=16)
    assert len(harmonics) == 17
    from scipy.ndimage import rotate
    rotated = rotate(target, 25, reshape=False, order=1, mode="constant", prefilter=False)
    result = image_registration_metrics(target, np.roll(rotated, 2, axis=1), threshold=0.55, rotation_step_deg=5.0)
    assert "RMSE" in result
    assert "angular-harmonic correlation" in result
    assert result["Pearson spatial correlation"] > 0.85
    assert "registration_rotation_deg" in result


if __name__ == "__main__":
    test_sequence(); test_hoxb1a_canonical(); test_sequence_source_helpers(); test_sequence_bundle_write()
    test_geometry_and_atomic_projection(); test_structure_source_contract(); test_internal_parametric_builder(); test_circular_modes_and_audio(); test_clamped_plate_physics_and_inverse(); test_resonator_calibration(); test_harmonics_and_registration()
    print("All tests passed.")
