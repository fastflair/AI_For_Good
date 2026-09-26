# Internal Sequence-to-3D Heavy-Atom Algorithm

## Design goal

Create a deterministic 3D coordinate field from a DNA sequence without depending on
an external molecular builder.

## Coordinate frames

Each base pair receives a local helical frame:

```text
x = radial direction
 y = tangential direction
 z = helix axis
```

The global axis rotates by the helical twist and advances by the helical rise.
Local inclination changes the base-plane normal but is not fed back into the global
axis. This prevents long sequences from developing an artificial macroscopic bend.

## Base placement

Each base is represented by its standard heavy-atom names. Base-specific 2D templates
approximate the ring layout. A small local scale factor preserves the distinction
between purines and pyrimidines.

## Sugar placement

Each nucleotide receives a seven-site sugar template. Small axial offsets represent a
puckered sugar rather than a perfectly planar ring.

## Phosphate placement

Each nucleotide receives P, O1P, O2P and O5P sites on the outer helical backbone.
The coordinates are arranged so the phosphate cloud follows the helical envelope.

## Sequence dependence

For B-DNA, local dinucleotide parameters provide sequence-dependent twist and rise.
Shift and slide are applied as a restrained transverse modulation. The cumulative
transverse displacement is centered so that it does not create an arbitrary drift.

## Conformational families

The idealized presets change rise, twist, diameter/radius, handedness and base
inclination for A-, B-, C- and Z-form explorations.

## Output

The resulting coordinate set is an `AtomicStructure` with:

- `atoms`: Nx3 coordinates in Å,
- `elements`: element symbols,
- `names`: standard atom names,
- `residues`: residue/base labels,
- `source`: provenance string,
- `dna_form`: selected conformational family,
- `metadata`: model assumptions and warnings.

## Scientific status

This is a geometry hypothesis generator. It is not intended to replace experimental
coordinates, crystallographic rebuilding, molecular dynamics, or force-field
minimization when quantitative chemistry is required.
