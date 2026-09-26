from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from Bio.PDB import MMCIFParser, PDBParser
except ImportError:  # Optional: only required for uploaded PDB/mmCIF files.
    MMCIFParser = PDBParser = None

BASES = set("ACGT")
ELEMENT_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06}
ELEMENT_Z = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "P": 15.0, "S": 16.0}


class AtomicStructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class AtomicStructure:
    atoms: np.ndarray
    elements: tuple[str, ...]
    names: tuple[str, ...]
    residues: tuple[str, ...]
    source: str
    dna_form: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.shape[0])


@dataclass(frozen=True)
class AtomicProjection:
    image: np.ndarray
    points_2d: np.ndarray
    points_3d_aligned: np.ndarray
    element_counts: dict[str, int]
    x_label: str
    y_label: str
    width_A: float
    height_A: float
    mode: str


# Reference geometric presets. These describe idealized conformational families;
# the internal sequence-dependent model modifies B-DNA local twist/rise only.
# A-DNA / B-DNA / Z-DNA values are consistent with commonly reported ranges;
# C-DNA is retained as a geometric exploratory preset because it is less common.
DNA_FORM_PRESETS = {
    "B-DNA": {
        "rise_A": 3.38,
        "twist_deg": 36.0,
        "radius_A": 10.0,
        "handedness": 1.0,
        "base_inclination_deg": -6.0,
    },
    "A-DNA": {
        "rise_A": 2.81,
        "twist_deg": 32.7,
        "radius_A": 11.5,
        "handedness": 1.0,
        "base_inclination_deg": 13.0,
    },
    "C-DNA": {
        "rise_A": 3.31,
        "twist_deg": 39.5,
        "radius_A": 9.5,
        "handedness": 1.0,
        "base_inclination_deg": 0.0,
    },
    "Z-DNA": {
        "rise_A": 3.70,
        "twist_deg": -30.0,
        "radius_A": 9.0,
        "handedness": -1.0,
        "base_inclination_deg": -7.0,
    },
    "A-RNA": {
        "rise_A": 2.81,
        "twist_deg": 32.7,
        "radius_A": 11.5,
        "handedness": 1.0,
        "base_inclination_deg": 13.0,
    },
}


# The atom templates below are intentionally geometric rather than a force-field
# topology. They use standard nucleic-acid atom names and elements, with local
# coordinates arranged to reproduce the major molecular features needed by the
# projection: planar bases, ribose/deoxyribose rings, and phosphate backbones.
# Hydrogens are omitted because proton positions depend strongly on protonation,
# tautomer and local geometry and do not materially improve the axial heavy-atom
# pattern at this resolution.
BASE_TEMPLATES = {
    "A": [
        ("N9", "N", -1.20, 0.00), ("C8", "C", 0.00, 1.25), ("N7", "N", 1.15, 0.70),
        ("C5", "C", 0.55, -0.45), ("C6", "C", -0.55, -0.85), ("N6", "N", -1.55, -1.35),
        ("N1", "N", -1.05, -1.95), ("C2", "C", 0.00, -2.25), ("N3", "N", 1.05, -1.70),
        ("C4", "C", 1.35, -0.55),
    ],
    "G": [
        ("N9", "N", -1.20, 0.00), ("C8", "C", 0.00, 1.25), ("N7", "N", 1.15, 0.70),
        ("C5", "C", 0.55, -0.45), ("C6", "C", -0.55, -0.85), ("O6", "O", -1.55, -1.35),
        ("N1", "N", -1.05, -1.95), ("C2", "C", 0.00, -2.25), ("N2", "N", 1.10, -1.95),
        ("N3", "N", 1.05, -0.80), ("C4", "C", 1.35, -0.25),
    ],
    "C": [
        ("N1", "N", -1.00, -1.45), ("C2", "C", 0.00, -2.15), ("O2", "O", 1.15, -1.75),
        ("N3", "N", 1.15, -0.55), ("C4", "C", 0.35, 0.35), ("C5", "C", -0.85, 0.10),
        ("C6", "C", -1.35, -0.75),
    ],
    "T": [
        ("N1", "N", -1.00, -1.45), ("C2", "C", 0.00, -2.15), ("O2", "O", 1.15, -1.75),
        ("N3", "N", 1.15, -0.55), ("C4", "C", 0.35, 0.35), ("O4", "O", 1.35, 0.95),
        ("C5", "C", -0.85, 0.10), ("C7", "C", -1.65, 1.05), ("C6", "C", -1.35, -0.75),
    ],
    "U": [
        ("N1", "N", -1.00, -1.45), ("C2", "C", 0.00, -2.15), ("O2", "O", 1.15, -1.75),
        ("N3", "N", 1.15, -0.55), ("C4", "C", 0.35, 0.35), ("O4", "O", 1.35, 0.95),
        ("C5", "C", -0.85, 0.10), ("C6", "C", -1.35, -0.75),
    ],
}

# Approximate deoxyribose heavy-atom template. Coordinates are in a local plane
# whose x direction points radially outward from the helix axis and y is tangential.
# The small z offsets mimic sugar puckering instead of forcing everything into one plane.
SUGAR_TEMPLATE = [
    ("C1'", "C", -1.70, 0.00, 0.25),
    ("O4'", "O", -0.65, 1.05, -0.15),
    ("C4'", "C", 0.65, 0.75, 0.20),
    ("C3'", "C", 1.15, -0.55, -0.20),
    ("O3'", "O", 1.85, -1.35, -0.05),
    ("C2'", "C", 0.00, -1.55, 0.30),
    ("C5'", "C", -1.05, 1.75, 0.40),
]

RNA_SUGAR_TEMPLATE = [
    ("C1'", "C", -1.70, 0.00, 0.25),
    ("O4'", "O", -0.65, 1.05, -0.15),
    ("C4'", "C", 0.65, 0.75, 0.20),
    ("C3'", "C", 1.15, -0.55, -0.20),
    ("O3'", "O", 1.85, -1.35, -0.05),
    ("C2'", "C", 0.00, -1.55, 0.30),
    ("O2'", "O", 0.80, -2.35, 0.10),
    ("C5'", "C", -1.05, 1.75, 0.40),
]

PHOSPHATE_TEMPLATE = [
    ("P", "P", 0.00, 0.00, 0.00),
    ("O1P", "O", 1.15, 0.85, 0.35),
    ("O2P", "O", -1.05, 0.95, -0.35),
    ("O5P", "O", -0.10, -1.25, 0.20),
]


# --------------------------- structure loading ---------------------------

def _parse_pdb_text(text: str) -> AtomicStructure:
    coords: list[list[float]] = []
    elements: list[str] = []
    names: list[str] = []
    residues: list[str] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        try:
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
        except ValueError:
            continue
        atom_name = line[12:16].strip() or "?"
        res_name = line[17:20].strip() or "?"
        element = line[76:78].strip().upper()
        if not element:
            m = re.match(r"([A-Za-z]+)", atom_name)
            element = m.group(1)[0].upper() if m else "C"
        if element == "D":
            element = "H"
        coords.append([x, y, z])
        elements.append(element)
        names.append(atom_name)
        residues.append(res_name)
    if not coords:
        raise AtomicStructureError("No ATOM/HETATM coordinates were found in the PDB file.")
    return AtomicStructure(np.asarray(coords, dtype=float), tuple(elements), tuple(names), tuple(residues), "PDB/mmCIF upload", "Unknown", {})


def load_structure(path: str | Path) -> AtomicStructure:
    p = Path(path)
    if not p.exists():
        raise AtomicStructureError(f"Structure file does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix in {".cif", ".mmcif"} and MMCIFParser is not None:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("uploaded", str(p))
        return _biopython_to_atomic(structure)
    if suffix == ".pdb" and PDBParser is not None:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("uploaded", str(p))
        return _biopython_to_atomic(structure)
    return _parse_pdb_text(p.read_text(encoding="utf-8", errors="replace"))


def _biopython_to_atomic(structure) -> AtomicStructure:
    coords=[]; elements=[]; names=[]; residues=[]
    for atom in structure.get_atoms():
        c = np.asarray(atom.coord, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            continue
        element = (atom.element or "C").upper()
        if element == "D":
            element = "H"
        residue = atom.get_parent()
        resname = str(residue.get_resname()).strip() or "?"
        coords.append(c.tolist())
        elements.append(element)
        names.append(str(atom.get_name()).strip())
        residues.append(resname)
    if not coords:
        raise AtomicStructureError("No atomic coordinates were found in the uploaded structure.")
    return AtomicStructure(np.asarray(coords, dtype=float), tuple(elements), tuple(names), tuple(residues), "PDB/mmCIF upload", "Unknown", {})


def load_pdb(path: str | Path) -> AtomicStructure:
    return load_structure(path)


# --------------------------- sequence -> geometry ---------------------------

def _orthonormal_frame(theta_rad: float, inclination_deg: float = 0.0, handedness: float = 1.0) -> np.ndarray:
    """Build a local base-plane frame without accumulating global curvature."""
    ct, st = math.cos(theta_rad), math.sin(theta_rad)
    radial = np.array([ct, st, 0.0], dtype=float)
    tangential = np.array([-handedness * st, handedness * ct, 0.0], dtype=float)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    a = math.radians(float(inclination_deg))
    # Rotate the base-plane normal away from z by a small inclination around tangent.
    normal = math.cos(a) * z + math.sin(a) * radial
    xaxis = radial
    yaxis = np.cross(normal, xaxis)
    yaxis /= max(np.linalg.norm(yaxis), 1e-12)
    normal /= max(np.linalg.norm(normal), 1e-12)
    return np.column_stack([xaxis, yaxis, normal])


def _canonical_parameters(sequence: str, dna_form: str, step_df=None) -> tuple[np.ndarray, np.ndarray]:
    preset = DNA_FORM_PRESETS[dna_form]
    n = len(sequence)
    twists = np.full(max(0, n - 1), float(preset["twist_deg"]), dtype=float)
    rises = np.full(max(0, n - 1), float(preset["rise_A"]), dtype=float)
    # Sequence-dependent local parameters are meaningful for B-DNA in this model.
    if dna_form == "B-DNA" and step_df is not None and len(step_df) == n - 1:
        twists = np.asarray(step_df["twist"], dtype=float)
        rises = np.asarray(step_df["rise"], dtype=float)
    return twists, rises


def build_parametric_atomic_dna(dna_structure, dna_form: str = "B-DNA") -> AtomicStructure:
    """Generate an explicit sequence-derived heavy-atom coordinate model in pure Python.

    This is a deterministic geometry builder, not a force-field or quantum model.
    It creates standard nucleotide heavy-atom names for both Watson-Crick strands,
    using local base, sugar and phosphate templates placed along an idealized helix.
    It is designed for spatial projection, visualization and inverse-wave experiments.
    Experimental PDB/mmCIF coordinates remain the preferred route for quantitative
    structural chemistry.
    """
    if dna_form in {"Z-DNA-like (left-handed B reference)", "Z-DNA (idealized left-handed)"}:
        dna_form = "Z-DNA"
    if dna_form not in DNA_FORM_PRESETS:
        raise AtomicStructureError(f"Unsupported internal conformation: {dna_form}")

    seq = str(dna_structure.sequence).upper()
    if not seq or any(b not in BASES for b in seq):
        raise AtomicStructureError("Sequence must contain A/C/G/T only.")

    preset = DNA_FORM_PRESETS[dna_form]
    twists, rises = _canonical_parameters(seq, dna_form, getattr(dna_structure, "step_df", None))
    n = len(seq)
    theta_deg = np.zeros(n, dtype=float)
    z_A = np.zeros(n, dtype=float)
    for i in range(1, n):
        theta_deg[i] = theta_deg[i - 1] + float(twists[i - 1])
        z_A[i] = z_A[i - 1] + float(rises[i - 1])

    # A small transverse displacement field retains the local sequence-dependent
    # geometry without permitting roll/tilt to produce an artificial macroscopic bend.
    # Keep sequence-dependent shift/slide local. Accumulating these small values
    # over hundreds of base pairs would create an unphysical transverse walk.
    offset_x = np.zeros(n, dtype=float)
    offset_y = np.zeros(n, dtype=float)
    if dna_form == "B-DNA" and getattr(dna_structure, "step_df", None) is not None:
        for i, row in dna_structure.step_df.iterrows():
            offset_x[i + 1] = 0.35 * float(row.get("shift", 0.0))
            offset_y[i + 1] = 0.35 * float(row.get("slide", 0.0))

    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    points: list[np.ndarray] = []
    elements: list[str] = []
    names: list[str] = []
    residues: list[str] = []

    def add(atom_xyz, element: str, atom_name: str, residue: str) -> None:
        points.append(np.asarray(atom_xyz, dtype=float))
        elements.append(element)
        names.append(atom_name)
        residues.append(residue)

    # Convert local (radial, tangential, axial) coordinates into global XYZ.
    def place(frame: np.ndarray, origin: np.ndarray, q: Iterable[float]) -> np.ndarray:
        return origin + frame @ np.asarray(q, dtype=float)

    for i, base in enumerate(seq):
        theta = math.radians(float(theta_deg[i]))
        frame = _orthonormal_frame(theta, float(preset["base_inclination_deg"]), float(preset["handedness"]))
        axis_center = np.array([float(offset_x[i]), float(offset_y[i]), float(z_A[i])], dtype=float)

        # Z-DNA has alternating local glycosidic orientation. We mimic the principal
        # syn/anti alternation geometrically without claiming a full Z-DNA force-field model.
        for strand_index, b in enumerate((base, comp[base])):
            strand_sign = 1.0 if strand_index == 0 else -1.0
            display_base = ("U" if dna_form == "A-RNA" and b == "T" else b)
            radial_center = strand_sign * float(preset["radius_A"] * 0.40)
            base_origin = axis_center + frame[:, 0] * radial_center

            base_flip = 1.0
            if dna_form == "Z-DNA" and b in {"G", "C"}:
                base_flip = -1.0 if (i % 2 == 0 and b == "G") or (i % 2 == 1 and b == "C") else 1.0
            for atom_name, element, x, y in BASE_TEMPLATES[display_base]:
                qx = x * base_flip
                qy = y
                add(place(frame, base_origin, (qx, qy, 0.0)), element, atom_name, f"{display_base}{i+1:04d}")

            # Sugar sits between the base and phosphate backbone. RNA receives O2'.
            sugar_origin = axis_center + frame[:, 0] * (strand_sign * 5.8) + frame[:, 2] * 0.0
            sugar_template = RNA_SUGAR_TEMPLATE if dna_form == "A-RNA" else SUGAR_TEMPLATE
            for atom_name, element, x, y, zz in sugar_template:
                add(place(frame, sugar_origin, (x * 0.72, y * 0.72, zz)), element, atom_name, f"{display_base}{i+1:04d}")

            # Phosphate is placed on the outer backbone, slightly advanced along z.
            phosphate_origin = axis_center + frame[:, 0] * (strand_sign * 9.6) + frame[:, 2] * (0.9 if strand_sign > 0 else -0.9)
            for atom_name, element, x, y, zz in PHOSPHATE_TEMPLATE:
                add(place(frame, phosphate_origin, (x * 0.85, y * 0.85, zz)), element, atom_name, f"{b}{i+1:04d}")

    atoms = np.vstack(points) if points else np.empty((0, 3), dtype=float)
    return AtomicStructure(
        atoms,
        tuple(elements),
        tuple(names),
        tuple(residues),
        "Internal sequence-derived parametric heavy-atom model",
        dna_form,
        {
            "generator": "pure-python-parametric-nucleic-acid-geometry",
            "sequence_length_bp": str(n),
            "atom_scope": "heavy atoms only; hydrogens omitted",
            "model_status": "geometry reference for visualization/projection; not force-field minimized",
            "sequence_dependence": "B-DNA uses local twist/rise and small shift/slide modulation from the built-in dinucleotide model",
            "macroscopic_axis": "kept straight; local inclination/handedness do not accumulate into global curvature",
        },
    )



# --------------------------- alignment and projection ---------------------------

def pca_align_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise AtomicStructureError("At least three Nx3 coordinates are required for alignment.")
    center = p.mean(axis=0)
    centered = p - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    k = int(np.argmax(np.abs(axis)))
    if axis[k] < 0:
        axis = -axis
    u = vt[1]
    v = np.cross(axis, u)
    v /= max(np.linalg.norm(v), 1e-15)
    u = np.cross(v, axis)
    u /= max(np.linalg.norm(u), 1e-15)
    basis = np.vstack([u, v, axis])
    aligned = centered @ basis.T
    return aligned, center, basis


def _axis_reference_points(structure: AtomicStructure) -> np.ndarray:
    p = np.asarray(structure.atoms, dtype=float)
    elem = np.asarray(structure.elements, dtype=object)
    p_atoms = p[elem == "P"]
    return p_atoms if len(p_atoms) >= 4 else p


def align_atomic_structure(structure: AtomicStructure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = _axis_reference_points(structure)
    if len(ref) < 3:
        raise AtomicStructureError("At least three atomic coordinates are required for alignment.")
    _, center, basis = pca_align_axis(ref)
    aligned = (np.asarray(structure.atoms, dtype=float) - center) @ basis.T
    return aligned, center, basis


def estimate_helical_pitch_A(step_df=None, default_A: float = 33.8) -> float:
    if step_df is None or len(step_df) == 0:
        return float(default_A)
    rise = float(np.nanmean(np.asarray(step_df["rise"], dtype=float)))
    twist = float(np.nanmean(np.asarray(step_df["twist"], dtype=float)))
    if not np.isfinite(rise) or not np.isfinite(twist) or abs(twist) < 1e-9:
        return float(default_A)
    return float(abs(rise * 360.0 / twist))


def _phase_fold(points_aligned: np.ndarray, pitch_A: float) -> np.ndarray:
    if pitch_A <= 0:
        raise ValueError("pitch_A must be positive")
    p = np.asarray(points_aligned, dtype=float).copy()
    phase = -2.0 * np.pi * np.mod(p[:, 2], pitch_A) / pitch_A
    c = np.cos(phase); s = np.sin(phase)
    x = p[:, 0] * c - p[:, 1] * s
    y = p[:, 0] * s + p[:, 1] * c
    p[:, 0] = x
    p[:, 1] = y
    p[:, 2] = np.mod(p[:, 2], pitch_A)
    return p


def atomic_density_projection(
    structure: AtomicStructure,
    mode: str = "Axial atomic density",
    size: int = 512,
    blur_A: float = 0.35,
    point_weighting: str = "Uniform",
    helical_pitch_A: float = 33.8,
) -> AtomicProjection:
    """Render every modeled atom into a square 2-D density field."""
    if not 64 <= int(size) <= 2048:
        raise ValueError("Atomic projection size must be between 64 and 2048 pixels.")
    if blur_A < 0:
        raise ValueError("blur_A must be non-negative.")
    if point_weighting not in {"Uniform", "Atomic mass", "Electron count proxy"}:
        raise ValueError("Unsupported atom weighting")
    valid_modes = {"Axial atomic density", "Helical phase-folded density", "Single-turn axial density", "Best-fit axial PCA"}
    if mode not in valid_modes:
        raise ValueError("Unsupported atomic projection mode")

    aligned, _, _ = align_atomic_structure(structure)
    if mode == "Helical phase-folded density":
        p = _phase_fold(aligned, helical_pitch_A)
        elems = np.asarray(structure.elements, dtype=object)
    elif mode == "Single-turn axial density":
        z = aligned[:, 2]
        z_mid = 0.5 * (float(z.min()) + float(z.max()))
        mask = np.abs(z - z_mid) <= float(helical_pitch_A) / 2.0
        if int(mask.sum()) < 4:
            raise AtomicStructureError("The selected one-turn window contains too few atoms.")
        p = aligned[mask]
        elems = np.asarray(structure.elements, dtype=object)[mask]
    else:
        p = aligned
        elems = np.asarray(structure.elements, dtype=object)

    xy = p[:, :2]
    lo = xy.min(axis=0); hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    # Use a symmetric canvas so the axial molecular cross-section is not stretched.
    side = float(max(span) * 1.10)
    center = 0.5 * (lo + hi)
    lo2 = center - side / 2.0
    hi2 = center + side / 2.0

    if point_weighting == "Uniform":
        weights = np.ones(len(xy), dtype=float)
    elif point_weighting == "Atomic mass":
        weights = np.asarray([ELEMENT_MASS.get(e, 12.0) for e in elems], dtype=float)
    else:
        weights = np.asarray([ELEMENT_Z.get(e, 6.0) for e in elems], dtype=float)

    edges_x = np.linspace(lo2[0], hi2[0], int(size) + 1)
    edges_y = np.linspace(lo2[1], hi2[1], int(size) + 1)
    image, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(edges_y, edges_x), weights=weights)
    if blur_A > 0:
        px_A = side / int(size)
        sigma_px = max(0.20, float(blur_A) / max(px_A, 1e-12))
        image = gaussian_filter(image, sigma=sigma_px, mode="constant")
    image -= image.min()
    peak = float(image.max())
    if peak > 0:
        image /= peak

    counts = {e: int(np.sum(elems == e)) for e in sorted(set(elems.tolist()))}
    return AtomicProjection(
        image=image,
        points_2d=xy,
        points_3d_aligned=p,
        element_counts=counts,
        x_label="Transverse X (Å)",
        y_label="Transverse Y (Å)",
        width_A=side,
        height_A=side,
        mode=mode,
    )


def atomic_edge_target(image: np.ndarray, blur_sigma_px: float = 1.0) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    work = img.copy()
    work -= work.min()
    if work.max() > 0:
        work /= work.max()
    if blur_sigma_px > 0:
        work = gaussian_filter(work, sigma=float(blur_sigma_px))
    gy, gx = np.gradient(work)
    edges = np.hypot(gx, gy)
    edges -= edges.min()
    if edges.max() > 0:
        edges /= edges.max()
    return edges


def element_color_sequence(elements: Iterable[str]) -> list[str]:
    colors = {"H": "lightgray", "C": "gray", "N": "royalblue", "O": "red", "P": "orange", "S": "yellow"}
    return [colors.get(e, "white") for e in elements]


def write_pdb(structure: AtomicStructure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for i, (xyz, element, name, residue) in enumerate(zip(structure.atoms, structure.elements, structure.names, structure.residues), start=1):
            x, y, z = [float(v) for v in xyz]
            atom_name = (name[:4] if name else "X").rjust(4)
            res = (residue[:3] if residue else "DNA").rjust(3)
            fh.write(f"ATOM  {i:5d} {atom_name} {res} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")
        fh.write("END\n")
    return p
