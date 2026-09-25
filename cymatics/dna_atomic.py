from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter
try:
    from Bio.PDB import MMCIFParser, PDBParser
except ImportError:  # optional until an uploaded PDB/mmCIF is used
    MMCIFParser = PDBParser = None

BASES = set("ACGT")
ELEMENT_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06}


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


def find_3dna_fiber(explicit_path: str | None = None) -> str | None:
    candidates: list[str] = []
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.is_dir():
            candidates += [str(p / "bin" / "fiber"), str(p / "fiber"), str(p / "bin" / "fiber.exe"), str(p / "fiber.exe")]
        else:
            candidates.append(str(p))
    x3dna = os.environ.get("X3DNA")
    if x3dna:
        root = Path(x3dna).expanduser()
        candidates += [str(root / "bin" / "fiber"), str(root / "bin" / "fiber.exe"), str(root / "fiber"), str(root / "fiber.exe")]
    path = shutil.which("fiber") or shutil.which("fiber.exe")
    if path:
        candidates.append(path)
    for c in candidates:
        p = Path(c)
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


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
    arr = np.asarray(coords, dtype=float)
    return AtomicStructure(arr, tuple(elements), tuple(names), tuple(residues), "PDB/mmCIF", "Unknown", {})


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
    text = p.read_text(encoding="utf-8", errors="replace")
    return _parse_pdb_text(text)


def _biopython_to_atomic(structure) -> AtomicStructure:
    coords=[]; elements=[]; names=[]; residues=[]
    for atom in structure.get_atoms():
        c = np.asarray(atom.coord, dtype=float)
        if c.shape != (3,) or not np.all(np.isfinite(c)):
            continue
        element = (atom.element or "C").upper()
        if element == "D": element = "H"
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
    """Backward-compatible alias."""
    return load_structure(path)


def generate_3dna_fiber(
    sequence: str,
    dna_form: str = "B-DNA",
    executable: str | None = None,
    timeout_s: int = 180,
) -> AtomicStructure:
    seq = re.sub(r"\s+", "", sequence).upper()
    if not seq or any(c not in BASES for c in seq):
        raise AtomicStructureError("3DNA input sequence must contain A/C/G/T only.")
    if len(seq) > 5000:
        raise AtomicStructureError("3DNA atomistic mode is limited to 5,000 bp for interactive use.")

    fiber = find_3dna_fiber(executable)
    if fiber is None:
        raise AtomicStructureError(
            "3DNA 'fiber' executable was not found. Install 3DNA and set X3DNA or provide the executable path."
        )

    switches = {"A-DNA": ["-a"], "B-DNA": ["-b"], "C-DNA": ["-c"], "Z-DNA": ["-z"], "A-RNA": ["-rna"]}
    if dna_form not in switches:
        raise AtomicStructureError(f"Unsupported 3DNA fiber form: {dna_form}")

    with tempfile.TemporaryDirectory(prefix="dna_cymatics_3dna_") as td:
        out = Path(td) / "fiber.pdb"
        cmd = [fiber, *switches[dna_form], f"-seq={seq}", str(out)]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        except OSError as exc:
            raise AtomicStructureError(f"Could not execute 3DNA fiber: {exc}") from exc
        if completed.returncode != 0:
            msg = (completed.stderr or completed.stdout or "3DNA fiber failed").strip()
            raise AtomicStructureError(f"3DNA fiber failed (exit {completed.returncode}): {msg[-1200:]}")
        if not out.exists():
            # Some 3DNA builds may normalize the extension/name. Find the first PDB in the temp directory.
            pdbs = list(Path(td).glob("*.pdb"))
            if len(pdbs) == 1:
                out = pdbs[0]
        if not out.exists():
            raise AtomicStructureError("3DNA completed without producing a PDB output file.")
        parsed = load_structure(out)
    return AtomicStructure(
        parsed.atoms,
        parsed.elements,
        parsed.names,
        parsed.residues,
        "3DNA fiber",
        dna_form,
        {"3dna_executable": fiber, "sequence_length_bp": str(len(seq)), "command": " ".join(cmd)},
    )


def pca_align_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align coordinates to a deterministic principal axis.

    The returned coordinates use an orthonormal basis [u, v, axis]. The sign
    of the principal axis is canonicalized for reproducibility.
    """
    p = np.asarray(points, dtype=float)
    center = p.mean(axis=0)
    centered = p - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    # Canonicalize sign to make results reproducible.
    k = int(np.argmax(np.abs(axis)))
    if axis[k] < 0:
        axis = -axis
    u = vt[1]
    v = vt[2]
    # Make a right-handed orthonormal basis.
    v = np.cross(axis, u)
    v /= max(np.linalg.norm(v), 1e-15)
    u = np.cross(v, axis)
    u /= max(np.linalg.norm(u), 1e-15)
    basis = np.vstack([u, v, axis])
    aligned = centered @ basis.T
    return aligned, center, basis


def _axis_reference_points(structure: AtomicStructure) -> np.ndarray:
    """Return the most structure-specific points available for helix-axis fitting.

    Phosphorus atoms are preferred for nucleic-acid structures because they are
    distributed along the two phosphate backbones. For generic uploaded structures
    without P atoms, all atoms are used.
    """
    p = np.asarray(structure.atoms, dtype=float)
    elem = np.asarray(structure.elements, dtype=object)
    p_atoms = p[elem == "P"]
    if len(p_atoms) >= 4:
        return p_atoms
    return p


def align_atomic_structure(structure: AtomicStructure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align all atoms using a backbone-informed principal molecular axis."""
    ref = _axis_reference_points(structure)
    if len(ref) < 3:
        raise AtomicStructureError("At least three atomic coordinates are required for alignment.")
    _, center, basis = pca_align_axis(ref)
    aligned = (np.asarray(structure.atoms, dtype=float) - center) @ basis.T
    return aligned, center, basis


def estimate_helical_pitch_A(step_df=None, default_A: float = 34.0) -> float:
    """Estimate helical pitch from mean local rise and twist.

    pitch = rise * 360 / twist. This is a geometric estimate and is only used
    for the explicit phase-folded visualization; it is not a measured material
    property.
    """
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


def select_axial_window(
    structure: AtomicStructure,
    window_A: float,
    center_fraction: float = 0.5,
) -> AtomicStructure:
    """Return atoms inside an axial slab of a molecular structure.

    The function is useful for viewing a single helical turn or a short structural
    repeat. Unlike phase folding, this is an actual subset of the coordinates.
    ``center_fraction`` is the fractional position along the molecular axis.
    """
    if window_A <= 0:
        raise ValueError("window_A must be positive")
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError("center_fraction must be in [0, 1]")
    aligned, center, basis = align_atomic_structure(structure)
    z = aligned[:, 2]
    z0 = float(z.min() + center_fraction * (z.max() - z.min()))
    mask = np.abs(z - z0) <= float(window_A) / 2.0
    if int(mask.sum()) < 4:
        raise AtomicStructureError("Axial window contains too few atoms; increase the window size.")
    atoms_local = aligned[mask]
    # Reconstruct original-coordinate representation so downstream code can align it again.
    atoms_original = (atoms_local @ basis) + center
    elems = tuple(np.asarray(structure.elements, dtype=object)[mask].tolist())
    names = tuple(np.asarray(structure.names, dtype=object)[mask].tolist())
    residues = tuple(np.asarray(structure.residues, dtype=object)[mask].tolist())
    return AtomicStructure(
        atoms_original, elems, names, residues,
        f"{structure.source} — axial window", structure.dna_form,
        {**structure.metadata, "axial_window_A": f"{window_A:.6f}", "center_fraction": f"{center_fraction:.6f}"},
    )


def atomic_density_projection(
    structure: AtomicStructure,
    mode: str = "Axial atomic density",
    size: int = 512,
    blur_A: float = 0.35,
    point_weighting: str = "Uniform",
    helical_pitch_A: float = 34.0,
) -> AtomicProjection:
    """Project every atom into a calibrated 2-D molecular density image.

    Modes:
      * Axial atomic density: literal orthographic view looking down the best-fit
        molecular axis.
      * Helical phase-folded density: an explicitly derived visualization that
        removes axial periodicity and co-registers successive turns. It is not a
        literal camera projection.
      * Single-turn axial density: literal projection of a one-pitch axial slab
        centered within the structure. This is the closest sequence-derived view
        to the finite helical cross-sections shown in structural diagrams.
      * Best-fit axial PCA: backward-compatible alias for the literal axial view.

    Atom weighting is intentionally explicit because uniform occupancy, atomic
    mass and other contrast models represent different physical observables.
    """
    if not 64 <= int(size) <= 2048:
        raise ValueError("Atomic projection size must be between 64 and 2048 pixels.")
    if blur_A < 0:
        raise ValueError("blur_A must be non-negative.")
    if point_weighting not in {"Uniform", "Atomic mass"}:
        raise ValueError("Unsupported atom weighting")
    if mode not in {"Axial atomic density", "Helical phase-folded density", "Single-turn axial density", "Best-fit axial PCA"}:
        raise ValueError("Unsupported atomic projection mode")

    aligned, _, _ = align_atomic_structure(structure)
    if mode == "Helical phase-folded density":
        p = _phase_fold(aligned, helical_pitch_A)
    elif mode == "Single-turn axial density":
        z = aligned[:, 2]
        z_mid = 0.5 * (float(z.min()) + float(z.max()))
        mask = np.abs(z - z_mid) <= float(helical_pitch_A) / 2.0
        if int(mask.sum()) < 4:
            raise AtomicStructureError("The selected one-turn window contains too few atoms.")
        p = aligned[mask]
    else:
        p = aligned

    xy = p[:, :2]
    lo = xy.min(axis=0); hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    side = float(max(span) * 1.10)
    center = 0.5 * (lo + hi)
    lo2 = center - side / 2.0
    hi2 = center + side / 2.0

    if point_weighting == "Uniform":
        weights = np.ones(len(xy), dtype=float)
    else:
        weights = np.asarray([ELEMENT_MASS.get(e, 12.0) for e in structure.elements], dtype=float)

    edges_x = np.linspace(lo2[0], hi2[0], int(size) + 1)
    edges_y = np.linspace(lo2[1], hi2[1], int(size) + 1)
    image, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=(edges_y, edges_x), weights=weights
    )
    if blur_A > 0:
        px_A = side / int(size)
        sigma_px = max(0.20, float(blur_A) / max(px_A, 1e-12))
        image = gaussian_filter(image, sigma=sigma_px, mode="constant")
    image -= image.min()
    peak = float(image.max())
    if peak > 0:
        image /= peak

    counts = {e: structure.elements.count(e) for e in sorted(set(structure.elements))}
    label = "X/Y transverse coordinates (Å)"
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
    """Convert molecular density into a nodal-line target using gradient magnitude.

    This is the preferred target for comparing against Chladni nodal lines: the
    molecular density itself is not a displacement field, while its strong spatial
    gradients provide a reproducible geometric feature set.
    """
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
    # Plotly accepts CSS colors. Keeping this deterministic makes the 3D view reproducible.
    colors = {"H": "lightgray", "C": "dimgray", "N": "royalblue", "O": "red", "P": "orange", "S": "yellow"}
    return [colors.get(e, "white") for e in elements]

# Lightweight sequence-only atom-site surrogate. It is intentionally not used for
# quantitative chemistry; it supplies base-specific heavy-atom-like sites when
# 3DNA is unavailable so the axial-density visualization remains meaningful.
BASE_ATOM_COUNTS = {"A": 9, "G": 10, "C": 8, "T": 9}
BASE_RING = {
    "C": np.array([[-2.9, -1.1], [-1.4, -2.0], [0.1, -1.1], [0.1, 0.9], [-1.4, 1.8], [-2.9, 0.7]], dtype=float),
    "T": np.array([[-2.9, -1.1], [-1.4, -2.0], [0.1, -1.1], [0.1, 0.9], [-1.4, 1.8], [-2.9, 0.7]], dtype=float),
    "A": np.array([[-2.8, -1.0], [-1.4, -2.0], [0.1, -1.0], [0.7, 0.5], [-0.4, 1.8], [-1.9, 1.5]], dtype=float),
    "G": np.array([[-2.8, -1.0], [-1.4, -2.0], [0.1, -1.0], [0.7, 0.5], [-0.4, 1.8], [-1.9, 1.5]], dtype=float),
}


def coarse_to_atomic_surrogate(dna_structure) -> AtomicStructure:
    """Create a deterministic straight-axis heavy-atom-like site cloud.

    This fallback intentionally keeps the molecular axis straight and applies
    the sequence's local twist/rise values. It is a visualization surrogate,
    not a chemically validated atomistic reconstruction.
    """
    seq = dna_structure.sequence
    # Build a straight helical axis so a long sequence remains a compact axial ring
    # rather than inheriting curvature artifacts from the coarse visualization model.
    n = len(seq)
    cumulative_twist = np.zeros(n, dtype=float)
    cumulative_rise = np.zeros(n, dtype=float)
    if getattr(dna_structure, "step_df", None) is not None and not dna_structure.step_df.empty:
        for i, row in dna_structure.step_df.iterrows():
            cumulative_twist[int(i)] = cumulative_twist[int(i) - 1] + float(row["twist"])
            cumulative_rise[int(i)] = cumulative_rise[int(i) - 1] + float(row["rise"])
    else:
        cumulative_twist[:] = np.arange(n) * 34.3
        cumulative_rise[:] = np.arange(n) * 3.4

    pts: list[np.ndarray] = []
    elems: list[str] = []
    names: list[str] = []
    residues: list[str] = []
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    for i, base in enumerate(seq):
        theta = math.radians(float(cumulative_twist[i]))
        frame = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        c = np.array([0.0, 0.0, float(cumulative_rise[i])])
        for strand_sign, b in ((1.0, base), (-1.0, complement[base])):
            center = c + frame @ np.array([strand_sign * 3.0, 0.0, 0.0])
            ring = BASE_RING[b].copy()
            if b in {"A", "G"}:
                ring = np.vstack([ring, [1.3, 1.2]])
            ring *= 0.95 if b in {"C", "T"} else 1.05
            for j, (x, y) in enumerate(ring):
                local = np.array([0.55 * x, 0.42 * y, 0.08 * math.sin(j)])
                pts.append(center + frame @ local)
                elems.append("N" if (j + (0 if strand_sign > 0 else 1)) % 3 == 0 else "C")
                names.append(f"{b}_BASE_{strand_sign:+.0f}_{j}")
                residues.append(b)
            for k, local in enumerate(((0.0, -4.5, 0.8), (0.8, -5.4, -0.6), (-0.8, -5.4, -0.6))):
                pts.append(center + frame @ np.asarray(local))
                elems.append("O" if k else "C"); names.append(f"SUGAR_{strand_sign:+.0f}_{k}"); residues.append(b)
            for k, local in enumerate(((strand_sign * 0.0, 9.3, 1.0), (strand_sign * 0.0, 10.1, 1.6), (strand_sign * 0.0, 10.2, -1.2))):
                pts.append(c + frame @ np.asarray(local))
                elems.append("P" if k == 0 else "O"); names.append(f"PHOS_{strand_sign:+.0f}_{k}"); residues.append(b)
    atoms = np.vstack(pts) if pts else np.empty((0, 3))
    return AtomicStructure(
        atoms, tuple(elems), tuple(names), tuple(residues),
        "Parametric straight-axis atom-site surrogate", "B-DNA",
        {"warning": "Visualization surrogate; not atomistically validated", "axis": "straight Z"},
    )


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


def find_3dna_tool(name: str, explicit_path: str | None = None) -> str | None:
    root = Path(explicit_path).expanduser() if explicit_path else None
    candidates: list[str] = []
    roots: list[Path] = []
    if root:
        roots.append(root if root.is_dir() else root.parent)
    x3dna = os.environ.get("X3DNA")
    if x3dna:
        roots.append(Path(x3dna).expanduser())
    for r in roots:
        candidates.extend([str(r / "bin" / name), str(r / "bin" / f"{name}.exe"), str(r / name), str(r / f"{name}.exe")])
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        candidates.append(found)
    for c in candidates:
        p = Path(c)
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


def write_3dna_bp_step_file(sequence: str, step_params: dict[str, dict[str, float]], path: str | Path) -> Path:
    """Write the local-step parameter format accepted by 3DNA rebuild.

    The first base is a zero-parameter anchor; each following row carries
    Shift, Slide, Rise, Tilt, Roll, Twist for the corresponding dinucleotide.
    """
    seq = re.sub(r"\s+", "", sequence).upper()
    p = Path(path)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(seq)} # bases\n")
        fh.write("0 # *** local step parameters ***\n")
        fh.write("# Shift Slide Rise Tilt Roll Twist\n")
        fh.write(f"{seq[0]} 0.00 0.00 0.00 0.00 0.00 0.00\n")
        for base_index in range(1, len(seq)):
            step = seq[base_index - 1 : base_index + 1]
            if step not in step_params:
                raise AtomicStructureError(f"No step parameters available for {step}")
            q = step_params[step]
            fh.write(
                f"{seq[base_index]} {q['shift']:.3f} {q['slide']:.3f} {q['rise']:.3f} "
                f"{q['tilt']:.3f} {q['roll']:.3f} {q['twist']:.3f}\n"
            )
    return p


def generate_3dna_sequence_dependent_atomic(
    sequence: str,
    step_params: dict[str, dict[str, float]],
    dna_form: str = "B-DNA",
    executable: str | None = None,
    timeout_s: int = 180,
) -> AtomicStructure:
    """Build an all-heavy-atom sequence-dependent model through 3DNA rebuild.

    For B-DNA, the standard B-DNA sugar-phosphate geometry can be installed in
    the temporary build directory with cp_std before rebuild. The base-pair
    geometry is generated from the supplied local step parameters.
    """
    seq = re.sub(r"\s+", "", sequence).upper()
    if not seq or any(c not in BASES for c in seq):
        raise AtomicStructureError("Sequence must contain A/C/G/T only")
    if len(seq) > 5000:
        raise AtomicStructureError("3DNA atomistic rebuild is limited to 5,000 bp for interactive use.")
    if dna_form != "B-DNA":
        raise AtomicStructureError("Sequence-dependent atomic rebuild is currently standardized to B-DNA; use fiber for A/C/Z.")
    rebuild = find_3dna_tool("rebuild", executable)
    if rebuild is None:
        raise AtomicStructureError("3DNA 'rebuild' executable was not found.")
    cp_std = find_3dna_tool("cp_std", executable)
    with tempfile.TemporaryDirectory(prefix="dna_cymatics_rebuild_") as td:
        root = Path(td)
        par = write_3dna_bp_step_file(seq, step_params, root / "bp_step.par")
        if cp_std:
            try:
                subprocess.run([cp_std, "BDNA"], cwd=root, capture_output=True, text=True, timeout=60, check=True)
            except Exception:
                # Rebuild can still create the exact base coordinates even when
                # standard backbone setup is unavailable; metadata records that.
                cp_std = None
        out = root / "rebuild.pdb"
        completed = subprocess.run([rebuild, "-atomic", str(par), str(out)], cwd=root, capture_output=True, text=True, timeout=timeout_s, check=False)
        if completed.returncode != 0 or not out.exists():
            msg = (completed.stderr or completed.stdout or "3DNA rebuild failed").strip()
            raise AtomicStructureError(f"3DNA rebuild failed (exit {completed.returncode}): {msg[-1500:]}")
        parsed = load_structure(out)
    return AtomicStructure(
        parsed.atoms, parsed.elements, parsed.names, parsed.residues,
        "3DNA sequence-dependent rebuild", "B-DNA",
        {
            "3dna_rebuild": rebuild,
            "cp_std_BDNA": cp_std or "unavailable",
            "sequence_length_bp": str(len(seq)),
            "local_step_parameter_source": "application STEP_PARAMS table",
        },
    )
