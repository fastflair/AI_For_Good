# Changelog

## v0.5 — Atomic target and resonator-inverse cleanup

- Added backbone-informed atomic-axis alignment using phosphorus atoms when available.
- Added literal whole-structure axial atomic-density projection.
- Added single-helical-turn axial projection.
- Added explicit helical phase folding and labeling as a derived transform.
- Added molecular-density edge/nodal target extraction.
- Changed the default cymatics target source to a single helical-turn projection so finite helical cross-section geometry is visible rather than collapsing the entire gene into an averaged ring.
- Vectorized circular membrane orientation fitting for substantially faster mode search.
- Kept exact physical resonances separate from musical pitch quantization.
- Corrected missing `generate_3dna_sequence_dependent_atomic` import in the UI layer.
- Added stronger metadata describing target source, target transform and phase-fold pitch.
- Added additional generated images to analysis bundles.
- Expanded tests for atomic projection and target generation.
- Rewrote scientific methods, limitations, experimental protocol and installation documentation.

## v0.4

- Added optional 3DNA integration for `fiber` and sequence-dependent `rebuild -atomic` workflows.
- Added PDB/mmCIF upload and atomic-coordinate parsing.
- Added deterministic sequence-only atom-site surrogate.
- Added atomic-density projection and polar harmonic analysis.
- Added circular membrane eigenmode search.
- Added physical-drive WAV with exact candidate frequencies.
- Added separate musical sonification WAV.
- Added image registration and cymatics comparison metrics.

## v0.3

- Added hoxb1a-201 canonical sequence retrieval and GRCz12tu genomic/transcript/CDS download.
- Added sequence provenance and FASTA package export.
- Added initial experimental verification tab.
