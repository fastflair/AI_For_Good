# Changelog

## v0.10 — sequence-wide inverse cymatics workflow

### Major changes

- Made `Rolling-turn ensemble axial density` the default molecular artwork projection so the entire DNA sequence contributes to the derived 2-D target.
- Kept literal axial and single-turn projections available for comparison.
- Added a more explicit separation between molecular artwork target, resonator observable, and predicted sand/nodal artwork.
- Changed the default inverse observable to `Displacement power`, which is better suited to a positive linear modal inverse than the phenomenological sand proxy.
- Added `observable_to_sand_artwork()` to visualize the low-displacement complement as a cymatics sand/nodal proxy.
- Added small NNLS L2 regularization.
- Re-fit the retained sparse mode basis after top-N selection instead of only truncating and renormalizing the full solution.
- Added actuator-coupling-compensated physical drive amplitudes.
- Changed the "best single mode" WAV to use the candidate with highest single-mode spatial correlation rather than the largest mixture weight.
- Added predicted sand/nodal artwork output and archive.
- Updated mode-table labeling from circular membrane to generic resonator modes.
- Updated the scientific and experimental documentation to distinguish model outputs from experimentally measured cymatics.

### Validation

- Python compilation passes.
- Full automated test suite passes.
- Application imports successfully.
- hoxb1a-201 sequence remains the 1,507-nt canonical test input.

### Remaining known limitations

The analytical mode library does not model the actual finite-element resonator, mounting hardware, actuator transfer function, damping, or granular particle dynamics. Those are intended to be calibrated experimentally in the next stage.
