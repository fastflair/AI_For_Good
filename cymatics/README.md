# DNA → 3D Shape → 2D Pattern → Frequencies → Music

Research-oriented application for exploring:

**DNA sequence → coarse-grained 3D geometry → 2D projection → spatial spectrum / plate modes → acoustic frequencies → music**

It also includes database-backed sequence retrieval and an experimental verification loop:

**database annotation → genomic / transcript / CDS / upstream sequence → DNA geometry → physical cymatics image → quantitative comparison**

## Canonical test sequence

The default DNA input is **zebrafish hoxb1a-201**, the ZFIN transcript record identified as `ZDB-TSCRIPT-090929-7277` / Ensembl transcript `ENSDART00000110682`, with **1,507 nt**. The bundled sequence keeps the app usable when live database services are unavailable.

## Sequence retrieval

The **Gene / Sequence Retrieval** tab accepts a ZFIN gene ID/URL, an NCBI GeneID, or a gene symbol. It resolves and cross-checks the gene using:

- **ZFIN** for gene/transcript provenance and current assembly context.
- **NCBI E-utilities** for the current `GRCz12tu` genomic mapping, genomic FASTA, promoter/upstream region, and independent RefSeq transcript cross-checks.
- **Ensembl REST** for the current gene model plus **all returned transcripts**, with cDNA and CDS sequence retrieval per transcript.

The retrieval tab supports the three requested DNA sequence classes:

1. **Genomic DNA** — the genomic locus, including introns and the sequence between exons within the locus.
2. **Mature mRNA/cDNA** — the spliced transcript sequence including UTRs.
3. **Protein-coding CDS** — the spliced coding sequence without UTRs.

The promoter/upstream output is explicitly labeled **upstream genomic sequence**. It should not be interpreted as an experimentally validated promoter unless external regulatory annotation confirms it.

The downloader writes a ZIP containing FASTA files and `metadata.json` with source IDs, assembly, accession, coordinates, strand, transcript identifiers, and source URLs.

## Current hoxb1a GRCz12tu reference

NCBI currently reports hoxb1a GeneID `30337` on chromosome 3 at `NC_133178.1:22508764..22510570` for **GRCz12tu** (`GCF_049306965.2`). The ZFIN gene record also identifies **GRCz12tu** as its current genomic assembly.

The code first attempts to obtain assembly-specific coordinates dynamically from NCBI Gene XML. For hoxb1a, a verified GRCz12tu mapping is retained as a safety fallback because public Gene XML layouts can vary over time.

## Scientific model

The DNA builder is a rigid-base-pair-step reconstruction using the six standard step variables: Tilt, Roll, Twist, Shift, Slide and Rise. This is not atomistic molecular dynamics and does not simulate solvent, ions, proteins, thermal fluctuations, or a unique physical conformation.

The FFT mode is a mathematically reproducible sonification. It is not a unique physical frequency inverse.

The square-plate search uses analytical simply-supported plate eigenmodes and ranks them by spatial similarity to the target. It does not model an arbitrary laboratory plate.

## Experimental verification

The verification tab compares a measured Chladni/cymatics image with the DNA-derived target using RMSE, Pearson spatial correlation, Dice overlap, and IoU. This is image-space verification; it does not prove that a particular frequency is a physical eigenfrequency without recording the plate geometry, boundary conditions, actuator coupling and measured frequency response.

For a static Chladni figure, test one resonance/mode at a time. A musical chord made from unrelated resonances generally produces a time-varying superposition rather than a single static nodal figure.

## Run

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown in the terminal.

## Network behavior

Live sequence retrieval requires outbound HTTPS access to ZFIN, NCBI, and Ensembl. The modeling and canonical hoxb1a-201 workflow remain usable offline. NCBI/Ensembl request timeouts and API errors are surfaced in the retrieval status panel rather than silently substituting a current genomic sequence.

## Next research step

For laboratory-grade prediction, replace the analytical plate basis with a finite-element eigenproblem using exact plate shape, thickness map, mounting and actuator location. Then use a measured frequency sweep and camera-based modal identification to estimate the real resonator transfer function and solve the inverse waveform problem.

## Important projection / cymatics distinction

A long DNA molecule viewed literally from the side is expected to look line-like. The previous implementation also accumulated local roll/tilt/slide/shift into the global molecular frame, which exaggerated that effect and produced an artificial bend. The current implementation keeps the B-DNA axis approximately straight along Z while applying those values locally.

The application now shows two separate 2D products:

1. **Geometric 2D projection** — a literal projection of the reconstructed 3D coarse-grained molecule. A long B-DNA sequence viewed from the side is line-like; viewed from the helical axis it is approximately an annulus.
2. **DNA-derived cymatic target** — a radial standing-wave field created from sequence information plus local helical-step information. This is intentionally snowflake/Chladni-like and is the image passed to the inverse frequency solver.

The second image is the mathematically explicit bridge to cymatics. It should not be described as an intrinsic 2D photograph of DNA.

The default inverse model is now an **ideal circular membrane eigenmode search** using Bessel-function modes. A real metal plate requires calibration/FEM because its modes depend on boundary conditions, thickness, material, damping, and actuator coupling.
