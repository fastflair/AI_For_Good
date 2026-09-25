from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dna_cymatics import (  # noqa: E402
    build_dna_structure,
    clean_sequence,
    density_projection,
    fft_components,
    rank_square_plate_modes,
    map_components,
    create_mapped_audio,
    image_metrics,
    reverse_complement,
)
from dna_sources import (  # noqa: E402
    CANONICAL_METADATA,
    HOXB1A_201_SEQUENCE,
    TranscriptRecord,
    GeneSequenceBundle,
    _parse_fasta_records,
    _parse_ncbi_assembly_location,
    canonical_hoxb1a_bundle,
    write_sequence_bundle,
)


SEQ = "ATACAAAGGTGCGAGGTTTCTATGCTCCCACGATCGATCGTAGCTAGGCTACGATCG"


def test_sequence():
    assert clean_sequence("acgt\nac") == "ACGTAC"
    assert reverse_complement("ACGT") == "ACGT"
    try:
        clean_sequence("ACGX")
        assert False
    except ValueError:
        pass


def test_hoxb1a_canonical():
    assert len(HOXB1A_201_SEQUENCE) == 1507
    assert CANONICAL_METADATA["name"] == "hoxb1a-201"
    assert canonical_hoxb1a_bundle()["metadata"]["length_nt"] == 1507


def test_sequence_source_helpers():
    fasta = ">one test\nACGTNN\n>two\nTGCA\n"
    records = _parse_fasta_records(fasta)
    assert records == [("one test", "ACGTNN"), ("two", "TGCA")]

    xml = """
    <Entrezgene-Set>
      <Entrezgene>
        <Entrezgene_comments>
          <Gene-commentary>
            <Gene-commentary_heading>Gene Location History</Gene-commentary_heading>
            <Gene-commentary_comment>
              <Gene-commentary>
                <Gene-commentary_heading>GRCz12tu</Gene-commentary_heading>
                <Gene-commentary_accession>NC_133178.1</Gene-commentary_accession>
                <Gene-commentary_seqs>
                  <Seq-loc><Seq-loc_int><Seq-interval>
                    <Seq-interval_from>22508763</Seq-interval_from>
                    <Seq-interval_to>22510569</Seq-interval_to>
                    <Seq-interval_strand><Na-strand value="plus"/></Seq-interval_strand>
                  </Seq-interval></Seq-loc_int></Seq-loc>
                </Gene-commentary_seqs>
              </Gene-commentary>
            </Gene-commentary_comment>
          </Gene-commentary>
        </Entrezgene_comments>
      </Entrezgene>
    </Entrezgene-Set>
    """
    location = _parse_ncbi_assembly_location(xml, "GRCz12tu")
    assert location is not None
    assert location["accession"] == "NC_133178.1"
    assert location["start"] == 22508764
    assert location["end"] == 22510570
    assert location["strand"] == 1


def test_sequence_bundle_write():
    bundle = GeneSequenceBundle(
        query="ZDB-GENE-990415-101",
        symbol="hoxb1a",
        ncbi_gene_id="30337",
        ensembl_gene_id="ENSDARG00160005162",
        zfin_gene_id="ZDB-GENE-990415-101",
        assembly="GRCz12tu",
        assembly_accession="GCF_049306965.2",
        chromosome="3",
        start=22508764,
        end=22510570,
        strand=1,
        genomic_accession="NC_133178.1",
        genomic_sequence="ACGT" * 50,
        promoter_sequence="TGCA" * 25,
        promoter_start=22506764,
        promoter_end=22508763,
        upstream_bp=2000,
        transcripts=[
            TranscriptRecord(name="hoxb1a-201", ensembl_id="ENSDART00000110682", cDNA=HOXB1A_201_SEQUENCE, cds="ATG" * 10),
        ],
        ncbi_refseq_accessions=["NM_131115.2"],
        zfin_transcript_links={"hoxb1a-201": "https://zfin.org/ZDB-TSCRIPT-090929-7277"},
        zfin_current_assembly="GRCz12tu",
        source_status=[],
        sources={"zfin_gene": "https://zfin.org/ZDB-GENE-990415-101", "ncbi_gene": "https://www.ncbi.nlm.nih.gov/gene/30337", "ensembl_gene": "https://www.ensembl.org/"},
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle.zip"
        write_sequence_bundle(bundle, out)
        assert out.exists()
        with zipfile.ZipFile(out) as z:
            names = set(z.namelist())
            assert "genomic_GRCz12tu.fna" in names
            assert "promoter_upstream.fna" in names
            assert any("hoxb1a-201" in n and "cDNA" in n for n in names)
            assert any("hoxb1a-201" in n and "CDS" in n for n in names)
            assert "metadata.json" in names


def test_geometry_and_projection():
    dna = build_dna_structure(SEQ)
    assert dna.centers.shape == (len(SEQ), 3)
    assert dna.strand1.shape == dna.strand2.shape == dna.centers.shape
    proj = density_projection(dna, size=256)
    assert proj.image.shape == (256, 256)
    assert 0.0 <= proj.image.min() <= proj.image.max() <= 1.0


def test_fft_and_mapping():
    dna = build_dna_structure(SEQ)
    proj = density_projection(dna, size=256)
    comps, spectrum = fft_components(proj.image, 0.3, top_n=12)
    assert len(comps) == 12
    assert spectrum.shape == proj.image.shape
    df = map_components(
        comps, 0.3, "Thin plate", 0.001, 200e9, 7850, 0.30,
        120, 110, 4, 1, "chromatic"
    )
    assert not df.empty
    assert np.all(np.isfinite(df["note_Hz"]))


def test_modes_audio_and_metrics():
    dna = build_dna_structure(SEQ)
    proj = density_projection(dna, size=256)
    modes, recon = rank_square_plate_modes(
        proj.image, 0.3, 0.3, 0.001, 200e9, 7850, 0.30,
        max_mode=8, top_modes=8, target_type="Nodal / sand"
    )
    assert len(modes) == 8
    assert recon.shape == (160, 160)
    modes2 = modes.rename(columns={"mixture_weight": "spectral_weight", "frequency_Hz": "note_Hz"})
    modes2["note"] = ["A4"] * len(modes2)
    audio, used = create_mapped_audio(modes2, duration_s=2)
    assert audio.ndim == 1 and len(audio) == 2 * 44100
    assert used
    m = image_metrics(proj.image, proj.image)
    assert abs(m["Pearson spatial correlation"] - 1.0) < 1e-9


if __name__ == "__main__":
    test_sequence()
    test_hoxb1a_canonical()
    test_sequence_source_helpers()
    test_sequence_bundle_write()
    test_geometry_and_projection()
    test_fft_and_mapping()
    test_modes_audio_and_metrics()
    print("All tests passed.")
