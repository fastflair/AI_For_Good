# Build Algorithm

## End-to-end pipeline

```text
DNA sequence
    ↓
sequence validation + provenance
    ↓
3-D coarse sequence geometry
    ↓
explicit heavy-atom parametric model OR uploaded PDB/mmCIF
    ↓
axis alignment
    ↓
atomic-density rasterization
    ↓
2-D molecular target
    ↓
sequence-wide rolling-turn / phase folding
    ↓
resonator conditioning
    ↓
spatial spectrum + polar harmonics
    ↓
ideal circular resonator mode library
    ↓
orientation matching + actuator coupling
    ↓
regularized NNLS inverse fit
    ↓
sparse top-N re-fit
    ↓
exact physical frequencies + amplitudes
    ↓
physical WAV + musical WAV
    ↓
optional physical camera verification
```

## Canonical projection

The default projection is `Rolling-turn ensemble axial density`.

Let each atom have aligned coordinates `(x,y,z)` and a sequence-associated helical phase `theta_bp`. For each sliding window start `s`, rotate the window by `-theta_s` so all windows share a common phase origin. Rasterize the transformed atoms and average/accumulate them.

The resulting target is sequence-wide:

`T(x,y) = aggregate_s density_s(x,y)`

Unlike a single-turn crop, every eligible base-pair window contributes.

## Atom rasterization

Each atom contributes a normalized weight `w_i`:

`w_i ∈ {1, mass_i, Z_i, vdw_i^3}`

The default atom footprint is a Gaussian with standard deviation based on the atom's van-der-Waals radius plus user-selected blur.

## Resonator conditioning

```text
T → min/max normalization
  → Gaussian low-pass
  → circular aperture/taper
```

This prevents the inverse solver from wasting modes on atom-scale detail that a millimeter-scale plate cannot represent.

## Circular thin-plate modes

For mode `(m,n)`, solve

`J'_m(lambda) I_m(lambda) - J_m(lambda) I'_m(lambda) = 0`

for the appropriate positive root. Build

`phi = [J_m(lambda r) + B I_m(lambda r)] cos(m theta - theta0)`

with

`B = -J_m(lambda)/I_m(lambda)`.

Frequency:

`f_mn = lambda_mn^2/(2*pi*R^2) * sqrt(D/(rho*h))`

where

`D = E h^3/[12(1-nu^2)]`.

## Mode orientation

For `m > 0`, cosine/sine partners share the same ideal frequency. The app samples orientation `theta0` over the requested angular grid and keeps the orientation with the highest target correlation.

For a real plate, imperfections and mounting can split or rotate nominally degenerate mode families. Calibration is therefore more authoritative than the ideal analytical orientation.

## Actuator coupling

A point-actuator proxy evaluates the modal displacement at `(r0,theta0_actuator)`. Low-coupling modes are downweighted during inverse fitting.

This is not a full finite-element transfer function.

## Sparse inverse fitting

For selected mode basis columns `A` and target vector `b`, solve

`min ||A x - b||_2^2 + lambda ||x||_2^2`

subject to

`x >= 0`.

The L2 term is implemented through NNLS augmentation. The strongest contributors are retained and then re-fit in the sparse basis.

## Audio

The physical drive preserves the exact modeled/measured frequencies. Amplitude is derived from modal power and actuator coupling.

The musical renderer separately maps the logarithmic ratios into a useful musical register and may quantize to chromatic pitch.

## Verification

A measured camera image is normalized, resized, rotated, translated, and compared with the selected reference target. Metrics include:

- RMSE
- Pearson spatial correlation
- Dice
- IoU
- 95th percentile boundary distance
- angular-harmonic correlation

Registration is intended to remove camera alignment nuisance variables. It is not evidence of causal equivalence by itself.
