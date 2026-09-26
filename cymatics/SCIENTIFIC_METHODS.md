# Scientific Methods

## Objective

The experimental objective is to construct a reproducible chain:

**DNA sequence → 3-D molecular geometry → 2-D molecular target → resonator mode decomposition → exact frequencies → physical cymatics image.**

The workflow is intentionally divided into a molecular-geometry layer and a resonator-physics layer. No claim is made that DNA possesses a unique intrinsic audible frequency.

## 1. Sequence provenance

The bundled canonical input is the zebrafish `hoxb1a-201` mature mRNA/cDNA sequence (1,507 nt). The application also retrieves genomic DNA, all available transcripts/cDNA, CDS, and an upstream sequence from public sequence resources when the machine running the app has network access.

A mature cDNA is not genomic double-stranded DNA. When a cDNA is used in the internal double-stranded geometry builder, the generated duplex is a **modeling construct**, not a claim about the in-vivo structure of that transcript.

## 2. 3-D structure representation

When a PDB/mmCIF is supplied, those coordinates are used directly after axis alignment.

When no structure is supplied, the app builds an explicit heavy-atom model from sequence and structural templates. The model includes base, sugar, and phosphate heavy-atom sites and applies a DNA-form-specific helical scaffold. Local sequence-dependent base-pair-step parameters alter the geometry without allowing the global axis to drift arbitrarily.

The internal builder is deterministic and self-contained. It is not a force-field calculation and does not represent an experimentally minimized structure.

## 3. Axial atomic projection

The structure is aligned so the helical axis is approximately the image-normal direction. Each heavy atom contributes an element-dependent footprint. The footprint can be a point Gaussian or a Gaussian whose width is derived from a van-der-Waals radius.

Optional density weights are:

- uniform site count,
- atomic mass,
- electron-count proxy (atomic number),
- van-der-Waals volume proxy.

These are different visualization measures; none should be interpreted as an exact X-ray or electron-scattering density calculation.

## 4. Sequence-wide molecular artwork

A whole-gene side projection is strongly elongated. For compact molecular artwork, the application provides:

### Single-turn axial density

A selected local ~one-turn window is projected directly.

### Helical phase-folded density

All atoms are rotated into a common helical-phase frame.

### Rolling-turn ensemble axial density

Every sliding one-turn window is phase-aligned and accumulated into the same transverse coordinate frame. This is the default because it uses the **entire sequence** rather than an arbitrary local window while retaining sequence-dependent differences in local atom occupancy.

This projection is a sequence-derived molecular signature. It is not a literal photograph of the entire gene folded into one physical turn.

## 5. Resonator conditioning

A macroscopic plate cannot reproduce arbitrary atom-scale detail. Before inverse fitting, the target is normalized, Gaussian-smoothed, and tapered to the circular resonator aperture.

This step is a spatial low-pass model: it asks what portion of the molecular artwork exists at spatial scales that the selected resonator basis can represent.

## 6. Resonator modes

Two idealized resonators are supported:

### Circular membrane

`phi_mn(r,theta) = J_m(alpha_mn r) cos(m theta - theta0)`

where `alpha_mn` is a zero of the Bessel function `J_m`.

### Clamped circular thin plate

The plate displacement field is represented by the axisymmetric plate basis

`phi_mn(r,theta) = [J_m(lambda_mn r) + B_mn I_m(lambda_mn r)] cos(m theta - theta0)`

with `B_mn = -J_m(lambda_mn)/I_m(lambda_mn)` and the clamped-edge characteristic equation

`J'_m(lambda) I_m(lambda) - J_m(lambda) I'_m(lambda) = 0`.

The plate frequency is computed from

`f = lambda^2/(2*pi*R^2) * sqrt(D/(rho*h))`

and

`D = E h^3/[12(1-nu^2)]`.

For a real apparatus, the boundary condition must match the mechanical construction. A plate that is simply supported, free-edged, bolted at discrete points, bonded, or driven at a finite-area actuator will not exactly follow the ideal clamped model.

## 7. Inverse mode fitting

Each mode family is orientation-matched. Mode fields are scaled by a point-actuator coupling proxy. Non-negative least squares then finds a positive mixture of modal observables.

A small optional L2 regularization is used to discourage unstable solutions. After selecting the strongest `N` modes, the selected basis is **re-fit** rather than merely truncating and renormalizing the original coefficients.

The reported `mixture_weight` is a modeled observable-power contribution. The physical drive amplitude is approximately proportional to its square root, with an additional actuator-coupling compensation proxy.

## 8. Default physical observable

The default inverse target is **displacement power**. This is convenient because, for distinct frequencies in a linear system, cross terms can average out over a sufficiently long observation interval, allowing a positive sum of modal power-like contributions to serve as a first-order inverse design model.

This does not constitute a complete mechanical model of powder/sand transport.

The application also provides a `Nodal / sand` proxy. That mode concentrates image intensity near predicted zero-displacement lines, with gradient weighting to suppress broad low-amplitude regions. It is explicitly phenomenological.

## 9. Frequency generation

The physical WAV preserves the exact modeled resonance frequencies. The musical WAV is generated separately and may quantize frequencies into musical notes.

Therefore:

`physical frequency != musical note`

unless an experiment deliberately chooses a pitch-quantized drive.

## 10. Experimental calibration

A calibration CSV may contain:

```csv
frequency_Hz,response
110.7,0.20
267.1,0.91
453.2,0.54
```

The current calibration layer maps theoretical candidate frequencies to nearby measured resonances and optionally weights them by measured response. It does not reconstruct a complete complex transfer function or guarantee the measured mode shape is identical to the ideal mode.

The experimental verification module registers a measured image using in-plane rotation and translation and then reports spatial metrics. Registration improves comparability but does not prove frequency causality.

## 11. What would constitute strong physical validation

A strong experiment would record:

1. plate dimensions and material,
2. mounting/boundary condition,
3. actuator location and contact geometry,
4. input waveform and voltage/current,
5. actual resonant frequencies,
6. camera image at each resonance,
7. orientation and scale registration,
8. repeat measurements.

The measured mode library can then replace the ideal analytical mode library. That is the preferred path for future versions.


## Combinatorial musical sonification

The musical layer is an interpretive mapping, not part of the physical resonator inverse problem. The input to this layer is a finite set of calculated source resonances. Let the tone identities be

\[
\mathcal{T}=\{T_1,T_2,\ldots,T_N\}.
\]

The melody generator constructs an ordered event sequence

\[
M = [T_1, T_2, \ldots, T_N, S_1, S_2, \ldots, S_K, \mathcal{T}],
\]

where each `S_k` is a randomly selected subset of `\mathcal{T}` with cardinality between user-defined bounds. The random process is seeded, so the sequence is reproducible. The final set event explicitly contains all tones.

For audio, a combination can be represented as an arpeggio, chord, or an arpeggio followed by a chord. The latter is the default because it exposes both individual tone identities and the combined harmonic object. Musical pitch quantization occurs only in this creative layer; the physical-drive WAV continues to use the calculated/calibrated resonant frequencies.
