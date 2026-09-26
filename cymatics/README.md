# DNA → 3D Geometry → 2D Atomic Pattern → Resonator Modes → Audio

Version 0.6

This application explores a reproducible computational pipeline:

```text
DNA sequence
    ↓
sequence retrieval / provenance
    ↓
3D duplex geometry
    ↓
explicit heavy-atom coordinate model
    ↓
2D axial atomic-density projection
    ↓
helical folding / one-turn projection
    ↓
spatial + polar harmonic analysis
    ↓
resonator mode search
    ↓
exact physical-drive WAV
    ↓
music sonification WAV
    ↓
optional physical cymatics image verification
```

The application is deliberately self-contained for sequence-to-geometry generation. **It does not require 3DNA, AmberTools, NAB, a molecular-dynamics package, or a proprietary molecular builder.**

The highest-fidelity structural path remains an experimentally determined or externally validated **PDB/mmCIF** coordinate set.

## Canonical test sequence

The default sequence is zebrafish **hoxb1a-201**, 1,507 nt, retained as a reproducible canonical test case.

The sequence-retrieval tab can also fetch current genomic DNA, transcripts, CDS sequences and upstream sequence from public gene resources when Internet access is available.

## Structure sources

The application supports three paths.

### 1. Internal parametric heavy-atom model — default

A pure-Python geometry builder creates an explicit set of standard nucleic-acid heavy-atom sites for both Watson-Crick strands. The model contains:

- base-ring heavy atoms with standard atom names,
- deoxyribose or ribose sugar heavy atoms,
- phosphate and phosphate-oxygen sites,
- explicit element identities,
- a sequence-dependent B-DNA twist/rise component,
- idealized A-, B-, C- and left-handed Z-form geometry presets,
- local base inclination and handedness,
- a globally straight helix axis to prevent artificial long-range curvature.

The model is intended for **geometry, image formation, signal analysis and experimental target generation**. It is not a force-field minimized structure, molecular-dynamics trajectory, or quantum-chemical equilibrium structure.

Hydrogens are intentionally omitted from the generated sequence-only model. Their placement depends on protonation, tautomeric state and local geometry and is not necessary for the primary transverse heavy-atom density analysis.

### 2. Uploaded PDB/mmCIF

An experimental or externally validated structure can be loaded directly. The application preserves the supplied atomic coordinates and parses atom names and elements. This is the preferred route when a specific structural conformation is available.

### 3. Coarse sequence-dependent model

A lightweight base-pair-step representation remains available for rapid 3D visualization. It uses the standard local descriptors shift, slide, rise, tilt, roll and twist but is not atomistic.

## Why the sequence-only model is custom

The project originally experimented with external sequence-to-structure builders. They added installation and platform dependencies without solving the central scientific problem: the output still had to be projected and converted into a target spatial field suitable for a calibrated resonator.

The current design therefore keeps the sequence-to-coordinate layer deterministic and inspectable in Python. Every modeled atom has an element and atom name, and the generated PDB can be exported for inspection.

This is intentionally a **geometry model**, not a claim of chemical simulation.

## DNA conformational presets

The app exposes:

```text
A-DNA
B-DNA
C-DNA
Z-DNA (idealized left-handed)
A-RNA
```

A-DNA, B-DNA and Z-DNA use commonly reported idealized helical dimensions. C-DNA is included as an exploratory geometric preset rather than as a claim that the sequence uniquely determines a C-DNA conformation. Z-DNA is likewise an idealized left-handed geometry; a specific experimentally observed Z-DNA structure should be supplied as PDB/mmCIF when quantitative comparison is required.

For B-DNA, the built-in dinucleotide table modifies the local twist and rise by sequence. The global helix axis is not allowed to become a random walk from accumulated roll/tilt.

## 2D molecular projection

A long gene-length DNA molecule is hundreds of nanometers long while its transverse diameter is only a few nanometers, so a conventional side projection is supposed to look like a line.

The application therefore produces multiple views.

### Literal axial atomic density

All modeled atoms are aligned to the molecular axis and projected orthographically onto the transverse XY plane. This is the true camera-style projection of the loaded coordinates.

### Single-turn axial density

A one-pitch axial slab is selected and projected. This is useful for comparing the output with top-down helical illustrations.

### Helical phase-folded density

The axial coordinate is reduced modulo the helical pitch and the transverse coordinates are phase rotated so successive turns co-register. This is a derived signal-processing transform, not a literal camera view.

### Density weighting

The app can weight atoms by:

```text
Uniform
Atomic mass
Electron count proxy
```

These correspond to different mathematical observables. They are not interchangeable physical measurements.

## Atomic target vs. cymatics target

The molecular density field is not itself a mechanical displacement field.

For a cymatics experiment the application can derive an edge/nodal target from the atomic density using the spatial gradient magnitude.

Conceptually:

```text
atomic coordinates
       ↓
2D density ρ(x,y)
       ↓
|∇ρ(x,y)|
       ↓
geometric target
```

This prevents the software from claiming that molecular occupancy is literally the displacement function of a vibrating plate.

## Spatial harmonic analysis

The application calculates both a 2D FFT and polar angular harmonics.

For an image expressed in polar coordinates:

```text
ρ(r, θ)
```

it estimates angular components:

```text
C_m(r) = integral ρ(r,θ) exp(-i m θ) dθ
```

The angular order `m` is particularly relevant for circular resonators because it corresponds to the number of nodal-diameter families.

## Resonator model

The built-in physical model is an ideal tension-dominated circular membrane. Its spatial modes use Bessel functions:

```text
φ_mn(r,θ) = J_m(α_mn r/R) cos(mθ - φ)
```

where `α_mn` is a zero of the Bessel function of order `m` and `R` is the resonator radius.

The mode frequency is modeled as:

```text
f_mn = c α_mn / (2πR)
```

where `c` is the assumed transverse wave speed.

The solver searches angular order, radial order and orientation and ranks candidate modes according to spatial similarity to the target.

This is an ideal resonator model. It is not a universal DNA frequency model.

## Physical frequency vs. musical frequency

These are now intentionally separated.

### Physical-drive WAV

Contains the **exact unquantized model resonances**. Each candidate mode is driven sequentially or the best single candidate can be isolated.

This is the signal intended for a real cymatics experiment.

### Musical sonification WAV

The same spatially derived frequency information can be mapped into a musical scale and expanded with rhythm/harmonics for creative use.

This file is **not** expected to maintain a static physical Chladni pattern because changing notes changes the spatial resonance conditions.

## Experimental verification

A physical verification run should be treated as a closed-loop measurement rather than a visual analogy:

```text
DNA target
   ↓
mode solver
   ↓
exact frequency
   ↓
physical resonator
   ↓
sand / powder / liquid
   ↓
camera
   ↓
image registration
   ↓
quantitative comparison
```

The application reports:

- RMSE,
- Pearson spatial correlation,
- Dice overlap,
- IoU,
- 95th percentile boundary distance,
- angular-harmonic correlation.

Similarity is evidence of pattern similarity only. It does not demonstrate causality or uniqueness.

## Recommended controls

A serious experiment should include:

1. the target sequence,
2. a shuffled sequence with matched length and base composition,
3. a reverse-complement control where applicable,
4. synthetic periodic sequences,
5. repeat measurements at the same resonance,
6. frequency-neighbor controls above and below the candidate.

Record plate dimensions, thickness, material, mounting, actuator location, drive amplitude, drive frequency, camera geometry and environmental conditions.

## Installation

Python 3.11+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\\Scripts\\activate
```

Linux/macOS/WSL:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

Windows:

```text
launch_windows.bat
```

Linux/macOS/WSL:

```bash
./launch_linux_mac.sh
```

## Dependencies

Required Python packages are intentionally limited to open-source scientific and UI libraries:

- Gradio,
- NumPy,
- SciPy,
- pandas,
- Plotly,
- Matplotlib,
- SoundFile,
- BeautifulSoup4,
- Biopython.

Biopython is used only for parsing uploaded PDB/mmCIF structures. The sequence-only coordinate builder is implemented inside this repository.

## Testing

Run:

```bash
python tests.py
```

The tests cover:

- sequence validation,
- hoxb1a-201 canonical sequence length and sentinel bases,
- gene-sequence package creation,
- 3D geometry construction,
- artificial-bend prevention,
- parametric heavy-atom generation,
- PDB writing,
- atomic 2D projection modes,
- density weighting,
- circular resonator modes,
- physical-drive audio,
- musical sonification,
- polar harmonic analysis,
- cymatics registration metrics.

No external molecular builder is needed for the test suite.

## Accuracy hierarchy

For a specific physical conformation:

```text
experimentally determined / validated PDB or mmCIF
        ↓
internal parametric heavy-atom geometry
        ↓
coarse sequence-dependent geometry
```

The internal model should be interpreted as a reproducible **coordinate hypothesis** for the image-generation experiment, not as evidence that those exact atom coordinates exist in vivo.

## Output bundle

Each analysis creates a ZIP containing:

- atomic PDB,
- literal axial projection PNG,
- selected cymatics target PNG,
- FFT spectrum PNG,
- resonator reconstruction PNG,
- physical-drive WAV,
- best-mode WAV,
- musical WAV,
- candidate-mode CSV,
- polar-harmonic CSV,
- spatial-frequency CSV,
- `analysis.json` with all parameters and hashes.

The objective is complete reproducibility: the same sequence, projection parameters and resonator parameters should recreate the same computational target.
