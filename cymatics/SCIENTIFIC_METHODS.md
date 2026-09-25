# Scientific Methods

## 1. Sequence provenance

The software distinguishes genomic DNA, mature mRNA/cDNA and CDS. Sequence hashes are recorded so an analysis can be reproduced exactly.

Gene retrieval is intentionally split across NCBI, Ensembl and ZFIN. The retrieved package records the database identifiers and URLs rather than treating a copied sequence as anonymous text.

## 2. Coarse sequence-dependent DNA model

The built-in coarse model uses local dinucleotide base-pair-step descriptors:

- Shift
- Slide
- Rise
- Tilt
- Roll
- Twist

The implementation is a sequence-dependent geometric model. It is not molecular dynamics, a force-field minimization, or an experimental structure determination.

The coarse model is used for software fallback, sanity checks and an estimated helix pitch.

## 3. Atomistic reconstruction

The preferred sequence-only route uses 3DNA where available. For B-DNA, a local base-pair-step parameter file is supplied to `rebuild -atomic`.

The standard 3DNA fiber mode is an idealized conformational reference. It is useful for comparing A/B/C/Z forms, but it is not a claim that the full genomic locus adopts one homogeneous conformation in vivo.

Uploaded PDB/mmCIF coordinates take precedence over sequence-only models when the user has a specific experimentally determined or otherwise trusted structure.

## 4. Molecular-axis alignment

The projection first estimates a molecular axis by PCA. Phosphorus atoms are preferred as the reference coordinates because they lie along the nucleic-acid phosphate backbones. If too few P atoms exist, all loaded coordinates are used.

The resulting orthonormal basis is deterministic up to the canonical sign rule and is applied to every atom.

## 5. Literal axial atomic projection

For each atom coordinate `(x,y,z)` after alignment, the `x,y` coordinates are rasterized onto a square image.

The image is therefore an actual orthographic coordinate projection under the selected atom weighting. Gaussian smoothing is a visualization kernel, not part of the molecular coordinate model.

## 6. Single-turn projection

A one-pitch axial slab centered within the molecule is selected:

```text
|z-z_center| <= pitch/2
```

and the selected atoms are projected literally. This is useful when comparing the application with finite helical cross-sectional drawings.

It should not be described as the projection of the entire gene.

## 7. Helical phase folding

For a pitch `P`, each atom is rotated by:

```text
phi = -2*pi*(z mod P)/P
```

before dropping the axial coordinate.

This co-registers successive turns. It is a derived transform intended to study repeat geometry and angular harmonics. It is not equivalent to a real camera image.

When `auto_pitch` is enabled, a geometric pitch estimate is computed from sequence-dependent mean rise and twist:

```text
P ≈ mean(rise) * 360 / mean(twist)
```

This estimate is only used for visualization.

## 8. Edge / nodal target

Molecular density is not a resonator displacement field. To compare geometry with Chladni nodal lines, the software can convert density to gradient magnitude:

```text
E(x,y) = sqrt((dI/dx)^2 + (dI/dy)^2)
```

This produces a reproducible geometric target emphasizing molecular boundaries and high spatial gradients.

The edge target is a derived analysis representation, not a claim that the plate experiences the molecular density field as a mechanical force.

## 9. Spatial Fourier analysis

A Hann-windowed 2-D FFT is used to identify dominant spatial frequencies and phase components. Opposite Fourier bins are paired to produce non-redundant real-image components.

The Fourier output is a spatial-spectrum diagnostic. Mapping a spatial cycle to an audible pitch is sonification, not a physical wave-equation solution.

## 10. Polar angular harmonics

For an approximately circular target, angular orders are calculated from:

```text
C_m = sum I(r,theta) exp(-i*m*theta)
```

or equivalent discretized sums over the image.

Angular order `m` is useful because circular resonators have modes with the same angular index. A high `m` component in a molecular target is therefore a meaningful candidate feature to compare with a circular-resonator mode.

## 11. Circular membrane modal model

For an ideal circular tension-dominated membrane:

```text
phi_mn(r,theta) = J_m(alpha_mn*r/R) cos(m*theta-phi)
```

where `alpha_mn` is a Bessel-function zero.

The frequency relation is:

```text
f_mn = c*alpha_mn/(2*pi*R)
```

where `c` is the membrane wave speed supplied by the user.

The model assumes a specific idealized resonator. It is not a generic thin-plate bending solution.

## 12. Mode fitting

The application evaluates a grid of angular and radial mode indices and searches degenerate angular orientation for the best target correlation.

The returned ranking answers:

> Which ideal circular membrane mode has the greatest image-space similarity to this target under this model?

It does not answer:

> Which frequency is intrinsically encoded in the DNA?

## 13. Audio synthesis

### Physical drive

Candidate model frequencies are emitted exactly, without musical quantization, one mode at a time.

### Musical sonification

Candidate frequencies may be rounded to chromatic notes and harmonically enriched. That output is intended for listening and downstream generative-music systems.

Musical quantization is intentionally separated from physical-drive frequencies.

## 14. Experimental verification

The software can compare a camera image of a real cymatics experiment with the precomputed target. Registration is limited to modest translation so the target cannot be arbitrarily warped into agreement.

Verification should be performed on an image collected after the frequency was selected. The target definition should not be modified after observing the experimental result.

Strong evidence requires independent controls and repeated trials. Image similarity alone is insufficient to establish causation or uniqueness.

## 15. Recommended extension: calibrated inverse resonator

The most rigorous future version is an experimentally identified modal basis:

```text
frequency sweep
      ↓
measured resonance peaks
      ↓
measured mode images
      ↓
image basis / transfer function
      ↓
non-negative or sparse inverse fit
      ↓
exact drive frequencies, amplitudes and phases
```

For complicated plates, the preferred model is FEM or experimentally measured eigenmodes. Modern Chladni experiments show that boundary conditions and actuator position can change the observed nodal patterns and that mode mixing can occur at higher frequencies.
