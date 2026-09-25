# DNA → 3D Atomic Geometry → 2D Molecular Target → Cymatics → Music

Version 0.5 research build

This application is an experimental scientific-computing workflow for turning a DNA sequence into a structural visualization, a two-dimensional molecular pattern, candidate resonator modes, and audio signals. It is deliberately designed to distinguish **molecular geometry**, **mathematical sonification**, and **physical cymatics**.

## Scientific objective

The intended pipeline is:

```text
DNA sequence
   ↓
sequence provenance / assembly / transcript selection
   ↓
3D molecular coordinates
   ↓
atomic-coordinate alignment to the molecular axis
   ↓
2D atomic projection
   ↓
optional one-turn or helical-phase transform
   ↓
optional edge/nodal target extraction
   ↓
spatial Fourier + polar harmonic analysis
   ↓
resonator mode search
   ↓
exact physical-drive WAV
   ↓
musical sonification WAV
   ↓
(optional) physical cymatics experiment + image verification
```

The application does **not** assume that a DNA sequence has a universal intrinsic audible frequency. A frequency reported by the resonator model is a property of the selected mathematical/physical resonator model and its parameters.

## Canonical test case

The default sequence is **zebrafish hoxb1a-201**, the mature mRNA/cDNA sequence used as the canonical software test sequence. The app also provides network retrieval of the current gene package from NCBI/Ensembl/ZFIN when the runtime has Internet access.

The canonical sequence is kept in `dna_sources.py` so that the application remains runnable offline.

## Sequence classes

The retrieval interface supports:

1. **Genomic DNA** — assembly-specific genomic locus, including introns and intergenic sequence inside the requested locus.
2. **Mature mRNA/cDNA** — transcript sequence including UTRs where provided.
3. **Protein-coding CDS** — translated coding sequence only.
4. **Promoter/upstream** — a configurable upstream genomic window. This is an upstream sequence window, not proof that every base is a promoter.

For gene retrieval, provenance is retained in `metadata.json`, including identifiers, assembly, chromosome/accession, coordinates, strand, transcript IDs and source URLs.

## 3D structure sources

The app supports three structural sources:

### 1. 3DNA sequence-dependent atomic rebuild

Preferred for sequence-dependent B-DNA when 3DNA `rebuild` and the standard B-DNA templates are available.

The app writes local base-pair-step parameters in 3DNA's `Shift Slide Rise Tilt Roll Twist` order and asks `rebuild -atomic` for atomic coordinates.

### 2. 3DNA atomistic fiber

Useful for idealized A-DNA, B-DNA, C-DNA, Z-DNA and RNA conformational reference structures.

### 3. Uploaded PDB/mmCIF

Use this when an experimentally determined structure or a trusted model is available. The atomic coordinates in the uploaded structure are used directly for visualization and projection.

### 4. Built-in parametric fallback

When 3DNA is unavailable, the app creates a deterministic atom-site surrogate from the sequence. It is useful for visualization and software testing only. It is **not** an atomistically validated chemical structure and must not be used for energetic or quantitative chemistry conclusions.

## Why the app shows two 2D images

A long DNA molecule has a large axial extent compared with its ~2 nm diameter. A side projection therefore looks like a line; that is expected.

The app therefore shows:

### Literal axial atomic-density projection

All loaded atom coordinates are aligned to the best-fit molecular axis and projected orthographically onto the transverse plane. This is the closest representation to an actual camera looking down the DNA axis.

### Selected cymatics target

Three target-source choices are available:

- **Axial atomic density** — the full-molecule literal projection.
- **Single-turn axial density** — a literal projection of one helical-pitch axial slab near the center of the structure. This is useful for comparison with finite axial molecular diagrams that show a few turns rather than an entire gene-length molecule.
- **Helical phase-folded density** — every atom is rotated by its axial helical phase before projection so successive turns are registered. This is a derived transform, not a literal camera view.

The selected target can then be converted to **edge / nodal geometry** before resonator fitting. Edge extraction is useful because Chladni sand accumulates near low-displacement nodal lines, while molecular density is a positive density field rather than a displacement field.

## Atomic projection weighting

Two weighting choices are exposed:

- **Uniform** — every atom contributes equal occupancy.
- **Atomic mass** — each atom is weighted by its elemental mass.

These are deliberately labeled alternatives. They represent different observables and should not be interpreted as interchangeable physical densities.

## Resonator model

The current analytical resonator is an ideal tension-dominated circular membrane. The mode shapes are based on Bessel functions:

```text
phi_mn(r,theta) = J_m(alpha_mn r/R) cos(m theta - phi)
```

where `alpha_mn` is the nth zero of the mth Bessel function.

The ideal membrane frequency model is:

```text
f_mn = c * alpha_mn / (2*pi*R)
```

where `c` is the supplied membrane wave speed and `R` is the resonator radius.

This is a mathematical membrane model, not a generic formula for a steel Chladni plate. A real plate requires measured or numerically identified boundary conditions, stiffness, thickness, density, damping and actuator coupling.

The mode solver searches the orientation of degenerate cosine/sine angular partners so the spatial mode is compared with the target in the best orientation.

## Audio outputs

### Physical drive WAV

The physical-drive WAV contains **exact unquantized resonator frequencies**, one candidate mode at a time. This is the file intended for a physical experiment.

Do not substitute the musical WAV for the physical-drive WAV when testing a resonator.

### Musical sonification WAV

The musical WAV is a creative sonification of the candidate physical frequencies. It may use musical pitch quantization and harmonics. It should **not** be expected to produce the same static Chladni figure as the physical-drive WAV.

A static resonant pattern is normally associated with a resonance/mode family. Driving several unrelated frequencies simultaneously generally produces a time-varying superposition rather than one stationary Chladni geometry.

## Experimental verification

The verification tab compares a measured cymatics image with the generated target. The software performs small translational registration and reports:

- RMSE
- Pearson spatial correlation
- Dice overlap
- IoU
- 95th-percentile nodal boundary distance
- angular-harmonic correlation

These image metrics establish image similarity only. They do not prove that a DNA sequence caused a frequency, that the model is unique, or that a frequency is a property of the DNA independent of the resonator.

## Recommended physical experiment

1. Use a single candidate resonance and the exact frequency from the physical-drive WAV.
2. Characterize the resonator with a frequency sweep before testing the DNA target.
3. Record the actual plate/membrane dimensions, thickness, material, mount and actuator location.
4. Measure the physical resonance frequency independently with an accelerometer, laser vibrometer, impedance measurement, microphone or equivalent instrumentation.
5. Photograph the resulting nodal pattern with fixed camera geometry.
6. Register the image to the target without changing the target definition after seeing the result.
7. Compare target and measured images.
8. Repeat using sequence controls such as a GC-matched random sequence, shuffled sequence, reverse complement, and periodic synthetic controls.

## Installation

### Basic installation

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown by the terminal.

### Optional 3DNA installation

Read `INSTALL_3DNA.txt`.

3DNA is optional. The fallback keeps the app functional, but the fallback structure is explicitly non-quantitative.

## Test suite

Run:

```bash
python tests.py
```

The tests cover:

- DNA validation
- canonical hoxb1a-201 length and sequence fingerprint
- sequence retrieval package structure
- PDB/atomic coordinate handling
- backbone-informed axis alignment
- literal and phase-folded atomic projections
- single-turn projection behavior
- nodal/edge target generation
- 3DNA parameter file generation
- circular membrane mode generation and ranking
- physical and musical WAV generation
- polar harmonics
- image registration metrics

## Reproducibility

Every analysis bundle contains an `analysis.json` file with the sequence hash, structure source, structural metadata, projection definition, resonator parameters, candidate modes and event timing.

For an actual experimental campaign, archive the entire generated ZIP together with the raw camera and sensor data.

## Accuracy hierarchy

From strongest to weakest sequence-to-coordinate provenance:

```text
Experimental / trusted PDB or mmCIF structure
        ↓
3DNA atomic sequence-dependent reconstruction
        ↓
3DNA idealized conformational fiber
        ↓
Built-in parametric atom-site surrogate
```

And separately, for the wave problem:

```text
Measured resonator + calibrated modal basis
        ↓
FEM / numerical eigenmode model matched to experiment
        ↓
Ideal circular membrane analytical model
        ↓
Simple frequency-to-note sonification
```

The application is most scientifically defensible when both sides of that chain are near the top.
