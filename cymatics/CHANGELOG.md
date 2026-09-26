# Changelog

## 0.6 — self-contained sequence-to-geometry build

- Removed all AmberTools/NAB runtime code and configuration.
- Removed all NAB-specific UI controls and installation documentation.
- Removed NAB-specific tests.
- Added a pure-Python parametric heavy-atom geometry builder.
- Added explicit base-specific standard heavy-atom names for A, G, C and T.
- Added sugar and phosphate heavy-atom site templates.
- Added A-DNA, B-DNA, C-DNA, idealized Z-DNA and A-RNA geometry presets.
- Kept B-DNA sequence dependence through local dinucleotide twist/rise parameters.
- Prevented local roll/tilt from accumulating into artificial global molecular-axis curvature.
- Added electron-count proxy density weighting.
- Updated the 2D projection metadata and documentation to distinguish literal from derived projections.
- Updated tests to validate all sequence-only structure families without external molecular software.
- Updated scientific methods, limitations, references and experimental protocol.
- The application now requires no external sequence-to-structure builder.
