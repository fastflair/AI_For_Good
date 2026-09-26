## v0.13 — melody artifacts as first-class outputs

- Exposes the generated combinatorial melody plan directly in the main results area.
- Adds a downloadable `musical_combination_melody.csv` output alongside the musical WAV.
- The main output now contains both the rendered melody and its machine-readable event plan.
- Keeps the physical-drive WAV independent of the musical arrangement.

## v0.12 — DNA combinatorial melody generator

- Added `DNA Combination Melody` as the default musical arrangement.
- Guarantees one singleton event for every fitted source tone.
- Adds seeded pseudo-random tone combinations with configurable count and subset-size range.
- Adds an explicit final all-tones synthesis event.
- Added `Arpeggio + chord`, `Arpeggio`, and `Chord` combination rendering modes.
- Added deterministic combination seed for reproducible melody generation; changing the seed produces a different valid melody without changing the source tone set.
- Added generated melody event table in the UI and `musical_combination_melody.csv` to every analysis bundle.
- Handles the 1-tone and 2-tone edge cases without invalid combination-size errors.
- Added regression tests for singleton coverage, combination sizes, all-tone synthesis, reproducibility, and seed variation.

## v0.11 — Musical sonification and WAV-length fix

- Fixed trailing-silence/over-allocation behavior in the musical WAV.
- Added optional motif repetition to intentionally fill a requested target duration.
- Added selectable musical note arrangements and global interval compression.
