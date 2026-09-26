from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple, Optional, Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom, rotate
from scipy.optimize import nnls


# The 10 unique DNA dinucleotide steps are sufficient because of reverse-complement
# symmetry. Values are representative averages from a large high-resolution
# protein-DNA structural dataset. Units: Angstrom / degree.
# Young et al., Biophysical Reviews (2024), Table 1.
UNIQUE_STEPS: Dict[str, Dict[str, float]] = {
    "CG": dict(tilt=0.0, roll=6.4, twist=34.3, shift=0.00, slide=0.36, rise=3.34),
    "CA": dict(tilt=0.0, roll=5.6, twist=35.1, shift=-0.06, slide=0.18, rise=3.33),
    "TA": dict(tilt=0.0, roll=2.3, twist=37.5, shift=0.00, slide=0.20, rise=3.35),
    "AG": dict(tilt=-0.4, roll=3.6, twist=32.2, shift=-0.03, slide=-0.34, rise=3.30),
    "GG": dict(tilt=0.0, roll=4.9, twist=33.2, shift=-0.01, slide=-0.33, rise=3.36),
    "AA": dict(tilt=0.0, roll=-0.2, twist=35.1, shift=0.02, slide=-0.27, rise=3.24),
    "GA": dict(tilt=-0.1, roll=1.9, twist=36.3, shift=-0.02, slide=-0.11, rise=3.28),
    "AT": dict(tilt=0.0, roll=0.1, twist=30.7, shift=0.00, slide=-0.69, rise=3.22),
    "AC": dict(tilt=0.2, roll=1.7, twist=31.9, shift=0.00, slide=-0.64, rise=3.26),
    "GC": dict(tilt=0.0, roll=2.7, twist=33.3, shift=0.00, slide=-0.45, rise=3.29),
}

BASES = "ACGT"
PAIR_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

PLATE_MATERIALS: Dict[str, Tuple[float, float, float]] = {
    # Young's modulus [Pa], density [kg/m^3], Poisson ratio.
    "Steel": (200e9, 7850.0, 0.30),
    "Aluminum": (69e9, 2700.0, 0.33),
    "Glass": (70e9, 2500.0, 0.22),
    "Brass": (100e9, 8500.0, 0.34),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True)
class DNA3D:
    sequence: str
    centers: np.ndarray
    strand1: np.ndarray
    strand2: np.ndarray
    basepair_edges: np.ndarray
    frames: np.ndarray
    step_df: pd.DataFrame


@dataclass(frozen=True)
class Projection2D:
    image: np.ndarray
    xy: np.ndarray
    x_label: str
    y_label: str
    width_A: float
    height_A: float


def clean_sequence(sequence: str) -> str:
    """Normalize and strictly validate a DNA sequence."""
    if sequence is None:
        raise ValueError("DNA sequence is required.")
    seq = "".join(str(sequence).upper().split())
    if not seq:
        raise ValueError("DNA sequence is empty.")
    bad = sorted(set(seq) - set(BASES))
    if bad:
        raise ValueError(f"Invalid DNA characters: {', '.join(bad)}. Use A/C/G/T only.")
    if len(seq) < 2:
        raise ValueError("Sequence must contain at least 2 bases.")
    if len(seq) > 200_000:
        raise ValueError("Sequence is too long for interactive mode; use <= 200,000 bp.")
    return seq


def reverse_complement(sequence: str) -> str:
    seq = clean_sequence(sequence)
    return "".join(PAIR_COMPLEMENT[b] for b in reversed(seq))


def _build_step_table() -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for key, vals in UNIQUE_STEPS.items():
        table[key] = vals.copy()
        rc = reverse_complement(key)
        if rc not in table:
            mirrored = vals.copy()
            # Under reverse-complement reversal, Tilt and Shift change sign;
            # Roll, Twist, Slide and Rise retain the scalar convention used here.
            mirrored["shift"] *= -1.0
            mirrored["tilt"] *= -1.0
            table[rc] = mirrored
    missing = [a + b for a in BASES for b in BASES if a + b not in table]
    if missing:
        raise RuntimeError(f"Missing dinucleotide parameters: {missing}")
    return table


STEP_PARAMS = _build_step_table()


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_dna_structure(
    sequence: str,
    model: str = "sequence-dependent",
    backbone_radius_A: float = 10.0,
) -> DNA3D:
    """Build a sequence-dependent coarse-grained duplex without artificial global bending.

    The helical axis is propagated using twist/rise plus small step translations.
    Local tilt/roll are recorded in the step table but are not recursively fed back
    into the global helix-axis orientation. This keeps a long sequence from becoming
    an unphysical random walk while still preserving the sequence-dependent local
    geometry that is useful for visualization and the parametric atom-site model.
    """
    seq = clean_sequence(sequence)
    if model not in {"sequence-dependent", "canonical"}:
        raise ValueError("model must be 'sequence-dependent' or 'canonical'.")
    if backbone_radius_A <= 0:
        raise ValueError("backbone_radius_A must be positive.")

    n = len(seq)
    centers = np.zeros((n, 3), dtype=float)
    frames = np.zeros((n, 3, 3), dtype=float)
    frames[0] = np.eye(3)

    step_rows: List[dict] = []
    for i in range(n - 1):
        step = seq[i : i + 2]
        if model == "canonical":
            p = dict(tilt=0.0, roll=0.0, twist=36.0, shift=0.0, slide=0.0, rise=3.38)
        else:
            p = STEP_PARAMS[step]

        theta = math.radians(float(p["twist"]))
        # Global axis frame rotates only about z. Local roll/tilt remain local
        # descriptors rather than cumulatively tilting the entire molecule.
        local_R = _rz(theta)
        frames[i + 1] = frames[i] @ local_R
        local_t = np.array([p["shift"], p["slide"], p["rise"]], dtype=float)
        centers[i + 1] = centers[i] + frames[i] @ local_t
        step_rows.append({"index": i + 1, "step": step, **p})

    strand1 = np.zeros_like(centers)
    strand2 = np.zeros_like(centers)
    basepair_edges = np.zeros((n, 2, 3), dtype=float)
    phase = math.radians(12.0)
    a = backbone_radius_A * np.array([math.cos(phase), math.sin(phase), 0.0])
    b = backbone_radius_A * np.array([math.cos(phase + math.pi), math.sin(phase + math.pi), 0.0])
    for i in range(n):
        strand1[i] = centers[i] + frames[i] @ a
        strand2[i] = centers[i] + frames[i] @ b
        basepair_edges[i, 0] = centers[i] + frames[i] @ (0.92 * a)
        basepair_edges[i, 1] = centers[i] + frames[i] @ (0.92 * b)

    step_df = pd.DataFrame(step_rows)
    return DNA3D(seq, centers, strand1, strand2, basepair_edges, frames, step_df)

def project_points(points: np.ndarray, projection: str) -> Tuple[np.ndarray, str, str]:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if projection == "XY (top)":
        return p[:, [0, 1]], "X (Å)", "Y (Å)"
    if projection == "XZ (side)":
        return p[:, [0, 2]], "X (Å)", "Z (Å)"
    if projection == "YZ (side)":
        return p[:, [1, 2]], "Y (Å)", "Z (Å)"
    if projection == "PCA":
        centered = p - p.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, "PC1 (Å)", "PC2 (Å)"
    raise ValueError(f"Unknown projection: {projection}")


def density_projection(
    dna: DNA3D,
    projection: str = "XZ (side)",
    size: int = 512,
    blur_px: float = 1.4,
) -> Projection2D:
    """Render the coarse DNA geometry to a calibrated square 2D density field."""
    size = int(size)
    if not 64 <= size <= 2048:
        raise ValueError("Projection size must be between 64 and 2048 pixels.")
    if blur_px < 0:
        raise ValueError("blur_px must be non-negative.")

    pts = np.vstack([dna.strand1, dna.strand2, dna.centers])
    xy, xlabel, ylabel = project_points(pts, projection)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    side = float(max(span) * 1.08)
    center = 0.5 * (lo + hi)
    lo2 = center - side / 2.0
    hi2 = center + side / 2.0

    counts, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=size, range=[[lo2[1], hi2[1]], [lo2[0], hi2[0]]]
    )
    image = counts.astype(float)
    if blur_px > 0:
        image = gaussian_filter(image, sigma=max(0.1, float(blur_px)), mode="constant")
    image -= image.min()
    peak = image.max()
    if peak > 0:
        image /= peak
    return Projection2D(image=image, xy=xy, x_label=xlabel, y_label=ylabel, width_A=side, height_A=side)


def fft_components(
    image: np.ndarray,
    canvas_width_m: float,
    top_n: int = 16,
    min_radius_bins: float = 2.0,
) -> Tuple[List[dict], np.ndarray]:
    """Extract strongest non-redundant spatial Fourier components."""
    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError("FFT image must be square.")
    if canvas_width_m <= 0:
        raise ValueError("canvas_width_m must be positive.")
    top_n = int(top_n)
    if not 1 <= top_n <= img.size // 2:
        raise ValueError("Invalid top_n.")

    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    work = (img - img.mean()) * window
    F = np.fft.fftshift(np.fft.fft2(work))
    mag = np.abs(F)

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.hypot(yy - cy, xx - cx)
    mag[rr < float(min_radius_bins)] = 0.0

    fx = np.fft.fftshift(np.fft.fftfreq(w, d=canvas_width_m / w))
    fy = np.fft.fftshift(np.fft.fftfreq(h, d=canvas_width_m / h))

    flat = np.argsort(mag.ravel())[::-1]
    used = set()
    comps: List[dict] = []
    for k in flat:
        iy, ix = np.unravel_index(k, mag.shape)
        key = (int(iy), int(ix))
        partner = ((-iy + h) % h, (-ix + w) % w)
        if key in used or partner in used:
            continue
        used.add(key)
        used.add(partner)
        comps.append(
            {
                "iy": int(iy),
                "ix": int(ix),
                "iy2": int(partner[0]),
                "ix2": int(partner[1]),
                "fx": float(fx[ix]),
                "fy": float(fy[iy]),
                "spatial_cycles_m": float(math.hypot(fx[ix], fy[iy])),
                "magnitude": float(mag[iy, ix]),
                "phase": float(np.angle(F[iy, ix])),
            }
        )
        if len(comps) >= top_n:
            break

    scale = max((c["magnitude"] for c in comps), default=0.0)
    for c in comps:
        c["amplitude_norm"] = c["magnitude"] / scale if scale > 0 else 0.0
    return comps, np.log1p(mag)


def reconstruct_from_components(image: np.ndarray, comps: List[dict]) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * window))
    out = np.zeros_like(F, dtype=complex)
    for c in comps:
        i, j = c["iy"], c["ix"]
        i2, j2 = c["iy2"], c["ix2"]
        out[i, j] = F[i, j]
        out[i2, j2] = np.conj(F[i, j])
    rec = np.real(np.fft.ifft2(np.fft.ifftshift(out)))
    rec -= rec.min()
    peak = rec.max()
    if peak > 0:
        rec /= peak
    return rec


def plate_frequency(
    spatial_cycles_m: float,
    width_m: float,
    thickness_m: float,
    young_pa: float,
    density_kg_m3: float,
    poisson: float,
) -> float:
    """Thin-plate flexural-wave dispersion approximation.

    For spatial cycles q [cycles/m], beta=2*pi*q and
    omega=sqrt(D/(rho*h))*beta^2, hence f=omega/(2*pi).
    """
    q = float(spatial_cycles_m)
    if q <= 0 or width_m <= 0 or thickness_m <= 0 or density_kg_m3 <= 0:
        return 0.0
    D = young_pa * thickness_m**3 / (12.0 * (1.0 - poisson**2))
    return float(2.0 * math.pi * math.sqrt(D / (density_kg_m3 * thickness_m)) * q**2)


def membrane_frequency(spatial_cycles_m: float, wave_speed_m_s: float) -> float:
    return float(max(0.0, spatial_cycles_m) * max(0.0, wave_speed_m_s))


def hz_to_note(freq: float) -> str:
    if not np.isfinite(freq) or freq <= 0:
        return "—"
    midi = int(round(69 + 12 * math.log2(freq / 440.0)))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def quantize_frequency(freq: float, mode: str = "chromatic") -> Tuple[float, str]:
    if not np.isfinite(freq) or freq <= 0:
        return 0.0, "—"
    if mode == "none":
        return float(freq), hz_to_note(freq)
    if mode != "chromatic":
        raise ValueError("quantization must be 'chromatic' or 'none'.")
    midi = int(round(69 + 12 * math.log2(freq / 440.0)))
    f = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    return float(f), hz_to_note(f)


def map_components(
    comps: List[dict],
    canvas_width_m: float,
    mapping_mode: str,
    thickness_m: float,
    young_pa: float,
    density_kg_m3: float,
    poisson: float,
    membrane_speed_m_s: float,
    musical_base_hz: float,
    musical_octaves: float,
    frequency_multiplier: float,
    quantization: str = "chromatic",
) -> pd.DataFrame:
    if not comps:
        return pd.DataFrame()
    spatial_max = max(c["spatial_cycles_m"] for c in comps)
    mags = np.asarray([c["magnitude"] for c in comps], dtype=float)
    mag_max = max(float(mags.max()), 1e-12)
    rows = []
    for idx, (c, mag) in enumerate(zip(comps, mags), start=1):
        pf = plate_frequency(c["spatial_cycles_m"], canvas_width_m, thickness_m, young_pa, density_kg_m3, poisson)
        mf = membrane_frequency(c["spatial_cycles_m"], membrane_speed_m_s)
        radial = np.clip(c["spatial_cycles_m"] / max(spatial_max, 1e-12), 0.0, 1.0)
        mus = musical_base_hz * 2.0 ** (musical_octaves * radial)
        if mapping_mode == "Thin plate":
            raw = pf
        elif mapping_mode == "Membrane":
            raw = mf
        elif mapping_mode == "Musical radial":
            raw = mus
        else:
            raise ValueError(f"Unsupported mapping_mode: {mapping_mode}")
        raw *= max(0.0, frequency_multiplier)
        qf, note = quantize_frequency(raw, quantization)
        rows.append(
            {
                "component": idx,
                "fx_cycles/m": c["fx"],
                "fy_cycles/m": c["fy"],
                "spatial_cycles/m": c["spatial_cycles_m"],
                "spectral_weight": float(mag / mag_max),
                "phase_deg": math.degrees(c["phase"]),
                "plate_Hz": pf,
                "membrane_Hz": mf,
                "musical_Hz": mus,
                "mapped_Hz": raw,
                "note_Hz": qf,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def square_plate_mode_frequency(
    m: int,
    n: int,
    width_m: float,
    height_m: float,
    thickness_m: float,
    young_pa: float,
    density_kg_m3: float,
    poisson: float,
) -> float:
    if min(m, n, width_m, height_m, thickness_m, density_kg_m3) <= 0:
        return 0.0
    D = young_pa * thickness_m**3 / (12.0 * (1.0 - poisson**2))
    q2 = (m / width_m) ** 2 + (n / height_m) ** 2
    return float(math.pi * math.sqrt(D / (density_kg_m3 * thickness_m)) * q2 / 2.0)


def _resize_square(image: np.ndarray, size: int = 160) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("Expected a 2D image.")
    scale_y = size / img.shape[0]
    scale_x = size / img.shape[1]
    out = zoom(img, (scale_y, scale_x), order=1)
    out -= out.min()
    peak = out.max()
    if peak > 0:
        out /= peak
    return out


def _nodal_density(phi: np.ndarray, sigma: float = 0.16) -> np.ndarray:
    """Soft indicator for sand accumulating near displacement nodes."""
    return np.exp(-((np.abs(phi) / max(sigma, 1e-4)) ** 2))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.ravel() - float(a.mean())
    bb = b.ravel() - float(b.mean())
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _fit_nonnegative_basis(A: np.ndarray, b: np.ndarray, regularization: float = 0.0) -> np.ndarray:
    """Solve NNLS, optionally with a small L2 penalty implemented by augmentation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    lam = max(float(regularization), 0.0)
    if lam > 0.0 and A.shape[1] > 0:
        A_aug = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
        b_aug = np.concatenate([b, np.zeros(A.shape[1], dtype=float)])
        coeff, _ = nnls(A_aug, b_aug)
    else:
        coeff, _ = nnls(A, b)
    return np.maximum(coeff, 0.0)


def rank_square_plate_modes(
    target_image: np.ndarray,
    width_m: float,
    height_m: float,
    thickness_m: float,
    young_pa: float,
    density_kg_m3: float,
    poisson: float,
    max_mode: int = 12,
    top_modes: int = 12,
    target_type: str = "DNA density",
    fit_size: int = 160,
    nodal_sigma: float = 0.16,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Rank analytical simply-supported square/rectangular plate modes by image similarity.

    This is a mode-search tool, not a free-edge finite-element solution. For a
    true Chladni prediction, the experimental boundary conditions and actuator
    coupling must be calibrated.
    """
    if target_type not in {"DNA density", "Nodal / sand"}:
        raise ValueError("Unknown target_type")
    max_mode = int(max_mode)
    top_modes = int(top_modes)
    if max_mode < 1 or top_modes < 1:
        raise ValueError("max_mode and top_modes must be positive")

    target = np.asarray(target_image, dtype=float)
    target = _resize_square(target, fit_size)
    if target_type == "DNA density":
        target = target
    else:
        # User is saying the bright target represents where sand/nodes should be.
        target = target

    y = np.linspace(0.0, height_m, fit_size)
    x = np.linspace(0.0, width_m, fit_size)
    X, Y = np.meshgrid(x, y)

    records = []
    recon = np.zeros_like(target)
    candidates = []

    for m in range(1, max_mode + 1):
        for n in range(1, max_mode + 1):
            phi = np.sin(m * math.pi * X / width_m) * np.sin(n * math.pi * Y / height_m)
            disp = (phi - phi.min()) / max(phi.max() - phi.min(), 1e-12)
            nodal = _nodal_density(phi, sigma=nodal_sigma)
            field = disp if target_type == "DNA density" else nodal
            score = _corr(target, field)
            freq = square_plate_mode_frequency(
                m, n, width_m, height_m, thickness_m, young_pa, density_kg_m3, poisson
            )
            candidates.append((score, m, n, freq, field))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:top_modes]

    # A visual mixture is useful for exploration only. It should not be interpreted
    # as one static Chladni figure unless the participating modes are degenerate or
    # otherwise phase/coherently driven at the same frequency.
    positive_scores = np.asarray([max(0.0, c[0]) for c in selected], dtype=float)
    if positive_scores.sum() > 0:
        weights = positive_scores / positive_scores.sum()
    else:
        weights = np.ones(len(selected), dtype=float) / max(len(selected), 1)

    rows = []
    for rank, (weight, candidate) in enumerate(zip(weights, selected), start=1):
        score, m, n, freq, field = candidate
        recon += weight * field
        rows.append(
            {
                "mode_rank": rank,
                "m": m,
                "n": n,
                "frequency_Hz": float(freq),
                "note": hz_to_note(freq),
                "pattern_correlation": float(score),
                "mixture_weight": float(weight),
            }
        )

    recon -= recon.min()
    peak = recon.max()
    if peak > 0:
        recon /= peak
    return pd.DataFrame(rows), recon


def create_mapped_audio(
    frequency_df: pd.DataFrame,
    style: str = "Arpeggio",
    duration_s: float = 16.0,
    sample_rate: int = 44_100,
    tempo_bpm: float = 100.0,
    beats_per_note: float = 1.0,
    tone_harmonics: int = 4,
) -> Tuple[np.ndarray, List[Tuple[float, str]]]:
    """Generate a conservative WAV sonification from the derived frequencies."""
    if frequency_df.empty:
        raise ValueError("No frequency components available.")
    style = str(style)
    if style not in {"Arpeggio", "Chord", "Arpeggio + Drone"}:
        raise ValueError("Unknown audio style")
    duration_s = float(duration_s)
    sample_rate = int(sample_rate)
    if duration_s <= 0 or sample_rate < 8_000:
        raise ValueError("Invalid duration or sample rate")

    freqs = np.asarray(frequency_df["note_Hz"], dtype=float)
    weights = np.asarray(frequency_df["spectral_weight"], dtype=float)
    notes = frequency_df["note"].astype(str).tolist()
    keep = np.isfinite(freqs) & (freqs > 0) & (freqs < sample_rate / 2.5)
    freqs, weights = freqs[keep], weights[keep]
    notes = [n for n, ok in zip(notes, keep) if ok]
    if len(freqs) == 0:
        raise ValueError("No mapped frequencies fall inside the audible synthesis range.")

    total = int(round(duration_s * sample_rate))
    audio = np.zeros(total, dtype=np.float64)
    beat_seconds = 60.0 / max(float(tempo_bpm), 1.0)
    note_len = max(0.08, beat_seconds * max(float(beats_per_note), 0.25))

    def add_tone(start_s: float, dur_s: float, f: float, amp: float) -> None:
        start = max(0, int(round(start_s * sample_rate)))
        if start >= total:
            return
        n = min(int(round(dur_s * sample_rate)), total - start)
        if n <= 0:
            return
        t = np.arange(n, dtype=float) / sample_rate
        attack = np.minimum(1.0, t / 0.02)
        release = np.minimum(1.0, np.maximum(0.0, (dur_s - t) / 0.08))
        env = attack * release
        signal = np.zeros_like(t)
        for h in range(1, max(1, int(tone_harmonics)) + 1):
            signal += (1.0 / h) * np.sin(2.0 * math.pi * f * h * t)
        signal /= sum(1.0 / h for h in range(1, max(1, int(tone_harmonics)) + 1))
        audio[start : start + n] += amp * env * signal

    max_w = max(float(weights.max()), 1e-9)
    weights = weights / max_w
    used: List[Tuple[float, str]] = []
    cycle = list(zip(freqs, weights, notes))

    if style == "Chord":
        for f, w, note in cycle:
            add_tone(0.0, duration_s * 0.97, float(f), 0.08 + 0.14 * float(w))
            used.append((float(f), note))
    else:
        t0 = 0.0
        idx = 0
        while t0 < duration_s:
            f, w, note = cycle[idx % len(cycle)]
            add_tone(t0, min(note_len * 0.93, duration_s - t0), float(f), 0.12 + 0.18 * float(w))
            used.append((float(f), note))
            t0 += note_len
            idx += 1
        if style == "Arpeggio + Drone":
            f0 = float(cycle[0][0]) / 2.0
            if f0 > 0:
                add_tone(0.0, duration_s * 0.98, f0, 0.05)

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = 0.92 * audio / peak
    return audio.astype(np.float32), used


def dna_summary(dna: DNA3D) -> dict:
    seq = dna.sequence
    gc = sum(1 for b in seq if b in "GC") / len(seq)
    span = np.ptp(dna.centers, axis=0)
    total_twist = float(dna.step_df["twist"].sum()) if not dna.step_df.empty else 0.0
    return {
        "length_bp": len(seq),
        "GC_fraction": float(gc),
        "GC_percent": float(100.0 * gc),
        "centerline_span_A": span.tolist(),
        "mean_rise_A": float(dna.step_df["rise"].mean()) if not dna.step_df.empty else 0.0,
        "mean_twist_deg": float(dna.step_df["twist"].mean()) if not dna.step_df.empty else 0.0,
        "estimated_turns": total_twist / 360.0,
        "reverse_complement": reverse_complement(seq),
    }


def image_metrics(reference: np.ndarray, measured: np.ndarray, threshold: float = 0.55) -> dict:
    """Image-space verification metrics after resizing measured to reference."""
    ref = np.asarray(reference, dtype=float)
    meas = np.asarray(measured, dtype=float)
    if ref.ndim != 2 or meas.ndim != 2:
        raise ValueError("Both reference and measured images must be 2D grayscale arrays.")
    meas = _resize_square(meas, ref.shape[0])
    ref = ref - ref.min()
    meas = meas - meas.min()
    if ref.max() > 0:
        ref = ref / ref.max()
    if meas.max() > 0:
        meas = meas / meas.max()

    mse = float(np.mean((ref - meas) ** 2))
    rmse = math.sqrt(mse)
    correlation = _corr(ref, meas)
    ref_mask = ref >= threshold
    meas_mask = meas >= threshold
    intersection = np.logical_and(ref_mask, meas_mask).sum()
    union = np.logical_or(ref_mask, meas_mask).sum()
    dice = float(2 * intersection / max(ref_mask.sum() + meas_mask.sum(), 1))
    iou = float(intersection / max(union, 1))
    return {
        "RMSE": rmse,
        "Pearson spatial correlation": correlation,
        "Dice overlap": dice,
        "IoU": iou,
    }

# ---------- Atomic/circular-mode research pipeline ----------
from scipy.special import jn_zeros, jv, iv, ive, jvp, ivp
from scipy.optimize import brentq
from scipy.ndimage import binary_erosion, distance_transform_edt


def circular_membrane_mode_frequency(m: int, n: int, radius_m: float, wave_speed_m_s: float) -> float:
    """Ideal circular, tension-dominated membrane eigenfrequency.

    f_mn = c * alpha_mn / (2*pi*R), where alpha_mn is the n-th zero of J_m.
    This is a membrane model, not a flexural plate model.
    """
    if int(m) < 0 or int(n) < 1 or radius_m <= 0 or wave_speed_m_s <= 0:
        return 0.0
    alpha = float(jn_zeros(int(m), int(n))[-1])
    return float(wave_speed_m_s * alpha / (2.0 * math.pi * radius_m))



@lru_cache(maxsize=256)
def circular_clamped_plate_eigenvalue(m: int, n: int) -> float:
    """Return the nth positive clamped-circular-plate eigenvalue lambda*R.

    For a uniform, isotropic Kirchhoff-Love circular plate with a clamped edge,
    the characteristic equation is

        J'_m(lambda) I_m(lambda) - J_m(lambda) I'_m(lambda) = 0.

    The positive roots are found by bracketing each root around the corresponding
    Bessel J_m zero. This avoids a large global root scan while retaining a fully
    deterministic numerical solution.
    """
    m = int(m); n = int(n)
    if m < 0 or n < 1 or m > 64 or n > 32:
        raise ValueError("Unsupported plate mode indices")

    def characteristic(lam: float) -> float:
        return float(jvp(m, lam, 1) * iv(m, lam) - jv(m, lam) * ivp(m, lam, 1))

    # Clamped-plate roots lie just above the corresponding J_m zeros. The bracket
    # is deliberately generous and is expanded if required for extreme modes.
    bessel_zero = float(jn_zeros(m, n)[-1])
    a = max(float(m) + 1e-4, bessel_zero + 0.20)
    b = bessel_zero + 1.40
    fa = characteristic(a); fb = characteristic(b)
    for _ in range(8):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0:
            return float(brentq(characteristic, a, b, xtol=1e-11, rtol=1e-11, maxiter=100))
        b += 0.50
        fb = characteristic(b)
    # Fall back to a local scan only if numerical behavior at very high modes
    # defeats the normal bracket. This keeps the normal path fast.
    grid = np.linspace(a, b, 1500)
    vals = np.asarray([characteristic(float(x)) for x in grid])
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0.0:
            return float(brentq(characteristic, float(grid[i]), float(grid[i + 1]), xtol=1e-11, rtol=1e-11))
    raise RuntimeError(f"Could not determine clamped circular plate root m={m}, n={n}")


def circular_clamped_plate_mode_frequency(
    m: int,
    n: int,
    radius_m: float,
    thickness_m: float,
    young_pa: float,
    density_kg_m3: float,
    poisson: float,
) -> float:
    """Thin-plate bending resonance of an ideal clamped circular plate.

    f = lambda^2/(2*pi*R^2) * sqrt(D/(rho*h)),
    D = E*h^3/(12*(1-nu^2)).
    """
    m = int(m); n = int(n)
    radius_m = float(radius_m); thickness_m = float(thickness_m)
    young_pa = float(young_pa); density_kg_m3 = float(density_kg_m3); poisson = float(poisson)
    if radius_m <= 0 or thickness_m <= 0 or young_pa <= 0 or density_kg_m3 <= 0 or abs(poisson) >= 0.5:
        raise ValueError("Invalid circular plate properties")
    lam = circular_clamped_plate_eigenvalue(m, n)
    D = young_pa * thickness_m**3 / (12.0 * (1.0 - poisson**2))
    return float((lam**2 / (2.0 * math.pi * radius_m**2)) * math.sqrt(D / (density_kg_m3 * thickness_m)))


def circular_clamped_plate_mode_field(m: int, n: int, size: int = 192, nodal: bool = True) -> np.ndarray:
    """Normalized clamped circular plate mode field for image synthesis."""
    if not (0 <= int(m) <= 64 and 1 <= int(n) <= 32):
        raise ValueError("Mode indices outside supported range")
    size = int(size)
    if size < 32:
        raise ValueError("size must be >= 32")
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    lam = circular_clamped_plate_eigenvalue(int(m), int(n))
    # B = -J_m(lambda)/I_m(lambda), evaluated directly for numerical transparency.
    B = -float(jv(int(m), lam) / iv(int(m), lam))
    radial = jv(int(m), lam * r) + B * iv(int(m), lam * r)
    phi = radial * np.cos(int(m) * theta)
    inside = r <= 1.0
    field = _mode_target_field(phi, inside, "Nodal / sand" if nodal else "Displacement magnitude", node_width=0.18)
    return field


def circular_resonator_mode_frequency(
    m: int,
    n: int,
    radius_m: float,
    model: str = "Circular membrane",
    wave_speed_m_s: float = 120.0,
    thickness_m: float = 0.001,
    young_pa: float = 200e9,
    density_kg_m3: float = 7850.0,
    poisson: float = 0.30,
) -> float:
    if model == "Circular membrane":
        return circular_membrane_mode_frequency(m, n, radius_m, wave_speed_m_s)
    if model == "Clamped circular thin plate":
        return circular_clamped_plate_mode_frequency(m, n, radius_m, thickness_m, young_pa, density_kg_m3, poisson)
    raise ValueError(f"Unsupported resonator model: {model}")


def circular_resonator_mode_field(
    m: int,
    n: int,
    size: int = 192,
    model: str = "Circular membrane",
    nodal: bool = True,
) -> np.ndarray:
    if model == "Circular membrane":
        return circular_membrane_mode_field(m, n, size=size, nodal=nodal)
    if model == "Clamped circular thin plate":
        return circular_clamped_plate_mode_field(m, n, size=size, nodal=nodal)
    raise ValueError(f"Unsupported resonator model: {model}")


def prepare_resonator_target(
    image: np.ndarray,
    smoothing_px: float = 3.0,
    radial_aperture: float = 0.98,
) -> np.ndarray:
    """Condition a molecular target to the spatial scale representable by a resonator.

    Molecular projections can contain atom-scale detail far above the spatial bandwidth
    of a macroscopic resonator. Gaussian prefiltering is therefore applied before inverse
    mode fitting. The full-resolution molecular target remains available for artwork and
    later experimental comparison.
    """
    target = np.asarray(image, dtype=float)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError("Target image must be a square 2-D array")
    target = target - float(target.min())
    if target.max() > 0:
        target = target / float(target.max())
    sigma = max(0.0, float(smoothing_px))
    if sigma > 0:
        target = gaussian_filter(target, sigma=sigma, mode="nearest")
    size = target.shape[0]
    yy, xx = np.indices(target.shape, dtype=float)
    xx = 2.0 * xx / max(size - 1, 1) - 1.0
    yy = 2.0 * yy / max(size - 1, 1) - 1.0
    r = np.hypot(xx, yy)
    # Smoothly taper the square canvas to the circular resonator aperture.
    ap = min(max(float(radial_aperture), 0.80), 1.0)
    taper = np.clip((ap + 0.08 - r) / 0.08, 0.0, 1.0)
    target *= taper
    target[r > 1.0] = 0.0
    peak = float(target.max())
    if peak > 0:
        target /= peak
    return target

def circular_membrane_mode_field(
    m: int,
    n: int,
    size: int = 192,
    nodal: bool = True,
) -> np.ndarray:
    """Return a normalized circular membrane displacement/nodal-density field."""
    if not (0 <= int(m) <= 64 and 1 <= int(n) <= 32):
        raise ValueError("Mode indices outside supported range")
    size = int(size)
    if size < 32:
        raise ValueError("size must be >= 32")
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    alpha = float(jn_zeros(int(m), int(n))[-1])
    phi = jv(int(m), alpha * r) * np.cos(int(m) * theta)
    inside = r <= 1.0
    if nodal:
        # Sand tends to accumulate near displacement nodes; this is a soft node indicator.
        scale = max(float(np.percentile(np.abs(phi[inside]), 75)), 1e-6)
        field = np.exp(-((np.abs(phi) / (0.22 * scale)) ** 2))
    else:
        field = np.abs(phi)
    field[~inside] = 0.0
    field -= field.min()
    peak = float(field.max())
    if peak > 0:
        field /= peak
    return field


def _resample_square(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D image")
    zoom_y = size / arr.shape[0]
    zoom_x = size / arr.shape[1]
    out = zoom(arr, (zoom_y, zoom_x), order=1)
    out -= out.min()
    peak = float(out.max())
    if peak > 0:
        out /= peak
    return out


def _normalized_projection_target(image: np.ndarray, nodal_target: bool = True, size: int = 192) -> np.ndarray:
    target = _resample_square(image, size)
    if nodal_target:
        # Bright image means target nodal/sand accumulation; keep as density target.
        pass
    return target


def _mode_target_field(phi: np.ndarray, inside: np.ndarray, target_type: str, node_width: float = 0.18) -> np.ndarray:
    """Convert displacement into an observable-like field.

    For a sand/nodal image, a raw low-displacement indicator incorrectly treats the
    center of high-angular-order modes as a broad node because the radial Bessel term
    is small there. Instead, the nodal proxy is concentrated at zero crossings and
    weighted by local displacement gradient, which is much closer to a thin nodal-line
    image. It remains a phenomenological particle-accumulation proxy rather than a
    granular-dynamics simulation.
    """
    if target_type == "Nodal / sand":
        vals = np.abs(phi)
        scale = max(float(np.percentile(vals[inside], 75)), 1e-9)
        node_sigma = max(float(node_width) * scale, 1e-9)
        line_likelihood = np.exp(-((vals / node_sigma) ** 2))
        gy, gx = np.gradient(phi)
        grad = np.hypot(gx, gy)
        gscale = max(float(np.percentile(grad[inside], 90)), 1e-9)
        gradient_weight = np.clip(grad / gscale, 0.0, 1.0)
        field = line_likelihood * (0.10 + gradient_weight) ** 1.5
    elif target_type == "Displacement magnitude":
        field = np.abs(phi)
    elif target_type == "Displacement power":
        field = phi ** 2
    else:
        raise ValueError(f"Unsupported target_type: {target_type}")
    field = np.asarray(field, dtype=float)
    field[~inside] = 0.0
    mn = float(field[inside].min())
    mx = float(field[inside].max())
    if mx > mn:
        field = (field - mn) / (mx - mn)
    return field


def rank_circular_membrane_modes(
    target_image: np.ndarray,
    radius_m: float,
    wave_speed_m_s: float,
    max_angular_mode: int = 20,
    max_radial_mode: int = 12,
    top_modes: int = 8,
    target_type: str = "Displacement power",
    fit_size: int = 128,
    candidate_pool: int | None = None,
    orientation_steps: int = 36,
    node_width: float = 0.18,
    actuator_r_fraction: float = 0.35,
    actuator_theta_deg: float = 0.0,
    actuator_coupling_floor: float = 0.03,
    resonator_model: str = "Circular membrane",
    plate_thickness_m: float = 0.001,
    plate_young_pa: float = 200e9,
    plate_density_kg_m3: float = 7850.0,
    plate_poisson: float = 0.30,
    inverse_regularization: float = 0.002,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inverse-fit circular-membrane mode intensities to a DNA-derived 2-D target.

    Each (m,n) family is first orientation-matched because cosine/sine partners are
    frequency-degenerate. A non-negative least-squares fit then finds the simultaneous
    mode mixture whose *time-averaged modal observable* best matches the target.

    For distinct drive frequencies and a linear system, the cross terms between modes
    average out over an observation interval that is long compared with the beat periods,
    so a weighted sum of modal power/particle-density proxies is a useful first-order
    inverse model for image synthesis. It is not a rigorous nonlinear sand-transport model.
    The returned mixture_weight represents observable power. A small non-negative
    regularization can favor a compact/stable solution, and the final sparse mode set is
    re-fit after truncation so the reported waveform corresponds to the reported modes.
    The physical displacement/drive amplitude scales approximately with sqrt(weight), with
    an additional actuator-coupling compensation proxy in the physical-drive output.
    """
    if target_type not in {"Nodal / sand", "Displacement magnitude", "Displacement power"}:
        raise ValueError("Unsupported target_type")
    radius_m = float(radius_m); wave_speed_m_s = float(wave_speed_m_s)
    if radius_m <= 0 or wave_speed_m_s <= 0:
        raise ValueError("radius_m and wave_speed_m_s must be positive")
    max_angular_mode = int(max_angular_mode); max_radial_mode = int(max_radial_mode)
    top_modes = int(top_modes); fit_size = int(fit_size)
    orientation_steps = max(1, int(orientation_steps))
    actuator_r_fraction = float(actuator_r_fraction)
    if not (0.0 <= actuator_r_fraction <= 0.99):
        raise ValueError("actuator_r_fraction must be in [0, 0.99]")
    if not (0 <= max_angular_mode <= 64 and 1 <= max_radial_mode <= 32):
        raise ValueError("Unsupported mode search limits")
    if not (1 <= top_modes <= (max_angular_mode + 1) * max_radial_mode):
        raise ValueError("Invalid top_modes")
    if fit_size < 32:
        raise ValueError("fit_size must be >= 32")

    target = _resize_square(target_image, fit_size)
    y, x = np.indices((fit_size, fit_size), dtype=float)
    x = 2.0 * x / (fit_size - 1) - 1.0
    y = 2.0 * y / (fit_size - 1) - 1.0
    r = np.hypot(x, y); theta = np.arctan2(y, x)
    inside = r <= 1.0
    target_inside = np.asarray(target[inside], dtype=float)
    target_inside -= target_inside.mean()
    target_norm = float(np.linalg.norm(target_inside))

    rows: list[dict] = []
    fields: dict[tuple[int, int], np.ndarray] = {}
    # For NNLS we need a consistent set of oriented fields. Store best orientation per family.
    for m in range(max_angular_mode + 1):
        zeros = jn_zeros(m, max_radial_mode)
        angles = np.array([0.0]) if m == 0 else np.linspace(0.0, math.pi / m, orientation_steps, endpoint=False)
        for n, alpha0 in enumerate(zeros, start=1):
            # IMPORTANT: for a thin plate the spatial eigenvalue lambda_mn is NOT the
            # membrane Bessel zero alpha_mn. The plate field must be evaluated with the
            # same lambda used by the plate frequency/eigenvalue equation.
            alpha = float(alpha0) if resonator_model == "Circular membrane" else float(circular_clamped_plate_eigenvalue(m, n))
            radial = np.asarray(jv(m, alpha * r), dtype=float)
            disp = radial[:, :, None] * np.cos(m * theta[:, :, None] - angles[None, None, :])
            field_stack = np.empty_like(disp, dtype=float)
            for j in range(disp.shape[-1]):
                field_stack[:, :, j] = _mode_target_field(disp[:, :, j], inside, target_type, node_width=node_width)
            flat = field_stack[inside, :].T
            flat -= flat.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(flat, axis=1)
            scores = (flat @ target_inside) / np.maximum(norms * target_norm, 1e-12) if target_norm > 1e-12 else np.zeros(len(angles))
            best_idx = int(np.argmax(scores))
            best_field = field_stack[:, :, best_idx].copy()
            # Point-actuator geometric participation factor. It is a simple controllability
            # indicator, not a full actuator/plate transfer function.
            rr0 = actuator_r_fraction
            tt0 = math.radians(float(actuator_theta_deg))
            if resonator_model == "Circular membrane":
                radial_at_actuator = float(jv(m, alpha * rr0))
            else:
                B = -float(jv(m, alpha) / iv(m, alpha))
                radial_at_actuator = float(jv(m, alpha * rr0) + B * iv(m, alpha * rr0))
            coupling = abs(float(radial_at_actuator * math.cos(m * tt0 - float(angles[best_idx])))) if m > 0 else abs(float(radial_at_actuator))
            coupling = max(coupling, float(actuator_coupling_floor))
            fields[(m, n)] = best_field
            freq = circular_resonator_mode_frequency(
                m, n, radius_m, model=resonator_model, wave_speed_m_s=wave_speed_m_s,
                thickness_m=plate_thickness_m, young_pa=plate_young_pa,
                density_kg_m3=plate_density_kg_m3, poisson=plate_poisson,
            )
            rows.append({
                "m": int(m), "n": int(n), "frequency_Hz": float(freq),
                "angular_order": int(m), "radial_order": int(n),
                "radial_eigenvalue": float(alpha), "membrane_bessel_zero": float(alpha0),
                "orientation_deg": float(math.degrees(float(angles[best_idx]))),
                "single_mode_correlation": float(scores[best_idx]),
                "actuator_coupling": coupling,
                "mode_frequency_family": f"(m={m}, n={n})",
                "resonator_model": resonator_model,
            })

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("No resonant modes generated")
    raw = raw.sort_values("single_mode_correlation", ascending=False).reset_index(drop=True)
    pool_n = int(candidate_pool or max(top_modes * 4, top_modes))
    pool_n = max(top_modes, min(pool_n, len(raw)))

    # Preserve the dominant angular orders of the DNA target in the candidate library.
    # A pure single-mode correlation can otherwise over-select visually busy high-m modes
    # while missing the target's primary rotational symmetry (for example m=10 for a
    # roughly decagonal B-DNA cross-sectional signature).
    harmonic = polar_harmonic_spectrum(target, max_m=min(max_angular_mode, 32))
    harmonic_orders = [int(v) for v in harmonic.sort_values("energy_fraction", ascending=False)["angular_order_m"].tolist()]
    keep_orders = set(harmonic_orders[:min(6, len(harmonic_orders))])
    symmetry_pool = raw[raw["m"].isin(keep_orders)]
    remainder = raw[~raw.index.isin(symmetry_pool.index)]
    take_sym = min(len(symmetry_pool), max(top_modes, pool_n // 2))
    take_rem = max(0, pool_n - take_sym)
    pool = pd.concat([symmetry_pool.head(take_sym), remainder.head(take_rem)], ignore_index=True)
    if len(pool) < top_modes:
        pool = raw.head(max(top_modes, len(pool))).copy()

    # NNLS operates on normalized observable fields. Low coupling is penalized by
    # scaling the basis downward: weakly coupled modes require more drive for the same
    # physical effect and therefore are less attractive unless necessary for shape fit.
    columns = []
    for _, row in pool.iterrows():
        field = fields[(int(row["m"]), int(row["n"]))]
        coupling = float(row["actuator_coupling"])
        vec = field[inside].astype(float) * coupling
        columns.append(vec)
    A = np.column_stack(columns)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms < 1e-12] = 1.0
    A_n = A / col_norms[None, :]
    b = np.maximum(target_inside - target_inside.min(), 0.0)
    b_norm = np.linalg.norm(b)
    b_n = b / b_norm if b_norm > 1e-12 else b
    coeff = _fit_nonnegative_basis(A_n, b_n, regularization=float(inverse_regularization))
    if np.all(coeff <= 0):
        coeff = np.zeros_like(coeff); coeff[:top_modes] = 1.0
    # Convert coefficient space back to observable power weights.
    power = np.maximum(coeff / col_norms, 0.0)
    if np.sum(power) <= 0:
        power[:top_modes] = 1.0
    power = power / np.sum(power)
    # Keep only the strongest contributors, then re-fit NNLS on that sparse basis.
    # Re-fitting is important: truncating a full NNLS solution and merely renormalizing
    # its coefficients can unnecessarily degrade the reconstructed spatial pattern.
    keep = np.argsort(power)[::-1][:top_modes]
    selected = pool.iloc[keep].copy().reset_index(drop=True)
    selected_fields = [fields[(int(row["m"]), int(row["n"]))] for _, row in selected.iterrows()]
    A_sel = np.column_stack([f[inside].astype(float) * float(row["actuator_coupling"]) for f, (_, row) in zip(selected_fields, selected.iterrows())])
    sel_norms = np.linalg.norm(A_sel, axis=0)
    sel_norms[sel_norms < 1e-12] = 1.0
    A_sel_n = A_sel / sel_norms[None, :]
    sel_coeff = _fit_nonnegative_basis(A_sel_n, b_n, regularization=float(inverse_regularization))
    sparse_power = np.maximum(sel_coeff / sel_norms, 0.0)
    if float(sparse_power.sum()) <= 0.0:
        sparse_power = np.maximum(selected_power[:len(selected)], 0.0)
    if float(sparse_power.sum()) <= 0.0:
        sparse_power = np.ones(len(selected), dtype=float)
    sparse_power /= float(sparse_power.sum())

    recon = np.zeros((fit_size, fit_size), dtype=float)
    for w, (_, row) in zip(sparse_power, selected.iterrows()):
        recon += float(w) * fields[(int(row["m"]), int(row["n"]))]
    recon[~inside] = 0.0
    peak = float(recon.max())
    if peak > 0:
        recon /= peak

    # Fit quality: compare target and reconstructed observable directly.
    fit_a = target[inside].astype(float); fit_r = recon[inside].astype(float)
    fit_rmse = float(np.sqrt(np.mean((fit_a - fit_r) ** 2)))
    fit_corr = float(_corr(fit_a, fit_r))
    relative_error = fit_rmse / max(float(np.sqrt(np.mean((fit_a - float(fit_a.mean())) ** 2))), 1e-9)

    selected.insert(0, "mode_rank", np.arange(1, len(selected) + 1))
    selected["mixture_weight"] = sparse_power
    selected["drive_amplitude_proxy"] = np.sqrt(sparse_power)
    selected["actuator_compensated_amplitude_proxy"] = np.sqrt(sparse_power) / np.maximum(selected["actuator_coupling"].to_numpy(float), 1e-9)
    phys_amp = selected["actuator_compensated_amplitude_proxy"].to_numpy(float)
    if phys_amp.max() > 0:
        phys_amp = phys_amp / phys_amp.max()
    selected["drive_amplitude_physical"] = phys_amp
    selected["note"] = [hz_to_note(float(f)) for f in selected["frequency_Hz"]]
    selected["fit_rmse"] = fit_rmse
    selected["fit_correlation"] = fit_corr
    selected["relative_fit_error"] = relative_error
    selected["inverse_regularization"] = float(inverse_regularization)
    return selected, recon


def observable_to_sand_artwork(observable_image: np.ndarray, mode: str = "Displacement power") -> np.ndarray:
    """Convert a predicted displacement observable into a sand/nodal artwork proxy.

    For displacement power, particles are expected to accumulate preferentially near low
    displacement. The returned image is therefore the normalized complement of power.
    This is a visualization proxy, not a granular-dynamics simulation.
    """
    img = _resample_square(observable_image, np.asarray(observable_image).shape[0])
    img = np.clip(img, 0.0, 1.0)
    if mode == "Displacement power":
        out = 1.0 - img
    elif mode == "Displacement magnitude":
        out = 1.0 - img
    elif mode == "Nodal / sand":
        out = img
    else:
        raise ValueError(f"Unsupported observable mode: {mode}")
    out -= out.min()
    peak = float(out.max())
    if peak > 0:
        out /= peak
    return out


def polar_harmonic_spectrum(image: np.ndarray, max_m: int = 32, radial_bins: int = 128) -> pd.DataFrame:
    """Compute rotation-sensitive angular harmonic energy from a square image."""
    img = _resample_square(image, int(radial_bins) * 2)
    size = img.shape[0]
    y, x = np.indices(img.shape)
    cx = (size - 1) / 2.0; cy = (size - 1) / 2.0
    xx = x - cx; yy = y - cy
    r = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    # Only analyze the inscribed circular field; including the square-canvas corners
    # would inject artificial angular harmonics unrelated to the circular resonator.
    radius = (size - 1) / 2.0
    mask = r <= radius
    records = []
    for m in range(0, int(max_m) + 1):
        coeff = np.sum(img[mask] * np.exp(-1j * m * theta[mask]))
        records.append({"angular_order_m": m, "complex_amplitude": coeff, "energy": float(np.abs(coeff) ** 2)})
    df = pd.DataFrame(records)
    if not df.empty and float(df["energy"].sum()) > 0:
        df["energy_fraction"] = df["energy"] / float(df["energy"].sum())
    else:
        df["energy_fraction"] = 0.0
    return df


def _translate_image(meas: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(meas)
    y0 = max(0, dy); y1 = min(meas.shape[0], meas.shape[0] + dy)
    x0 = max(0, dx); x1 = min(meas.shape[1], meas.shape[1] + dx)
    sy0 = max(0, -dy); sy1 = sy0 + (y1 - y0)
    sx0 = max(0, -dx); sx1 = sx0 + (x1 - x0)
    if y1 > y0 and x1 > x0:
        shifted[y0:y1, x0:x1] = meas[sy0:sy1, sx0:sx1]
    return shifted


def image_registration_metrics(reference: np.ndarray, measured: np.ndarray, threshold: float = 0.55, rotation_step_deg: float = 5.0) -> dict:
    """Registration-aware image metrics with translation + in-plane rotation search.

    Rotation is especially important for circular cymatics images because the camera can
    be rotated arbitrarily relative to the DNA-derived target. The search is an image
    registration operation; it is not a scientific acceptance score.
    """
    ref = np.asarray(reference, dtype=float).copy()
    meas = np.asarray(measured, dtype=float)
    if ref.ndim != 2 or meas.ndim != 2:
        raise ValueError("Images must be grayscale 2-D arrays")
    meas = _resample_square(meas, ref.shape[0])
    ref -= ref.min(); meas -= meas.min()
    if ref.max() > 0: ref /= ref.max()
    if meas.max() > 0: meas /= meas.max()
    if rotation_step_deg <= 0:
        rotation_step_deg = 360.0
    angles = np.arange(0.0, 360.0, float(rotation_step_deg))
    lim = max(2, ref.shape[0] // 20)
    step_px = max(1, lim // 10)
    best = None
    for angle in angles:
        rotated = rotate(meas, float(angle), reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
        # A coarse translation search is sufficient after the camera image is square-cropped.
        for dy in range(-lim, lim + 1, step_px):
            for dx in range(-lim, lim + 1, step_px):
                shifted = _translate_image(rotated, dx, dy)
                corr = _corr(ref, shifted)
                if best is None or corr > best[0]:
                    best = (corr, dx, dy, float(angle), shifted)
    _, dx, dy, angle, aligned = best
    mse = float(np.mean((ref - aligned) ** 2))
    rmse = math.sqrt(mse)
    corr = _corr(ref, aligned)
    ref_mask = ref >= float(threshold)
    meas_mask = aligned >= float(threshold)
    intersection = np.logical_and(ref_mask, meas_mask).sum()
    union = np.logical_or(ref_mask, meas_mask).sum()
    dice = float(2 * intersection / max(ref_mask.sum() + meas_mask.sum(), 1))
    iou = float(intersection / max(union, 1))
    ref_edge = ref_mask ^ binary_erosion(ref_mask)
    meas_edge = meas_mask ^ binary_erosion(meas_mask)
    if ref_edge.any() and meas_edge.any():
        d_ref = distance_transform_edt(~ref_edge)
        d_meas = distance_transform_edt(~meas_edge)
        hd95 = float(max(np.percentile(d_meas[ref_edge], 95), np.percentile(d_ref[meas_edge], 95)))
    else:
        hd95 = float("nan")
    ref_h = polar_harmonic_spectrum(ref, max_m=24)
    meas_h = polar_harmonic_spectrum(aligned, max_m=24)
    harmonic_corr = _corr(ref_h["energy_fraction"].to_numpy(), meas_h["energy_fraction"].to_numpy())
    return {
        "RMSE": rmse,
        "Pearson spatial correlation": corr,
        "Dice overlap": dice,
        "IoU": iou,
        "95th-percentile nodal boundary distance (pixels)": hd95,
        "registration_dx_pixels": int(dx),
        "registration_dy_pixels": int(dy),
        "registration_rotation_deg": float(angle),
        "angular-harmonic correlation": float(harmonic_corr),
        "aligned_image": aligned,
    }


def apply_resonator_calibration(mode_table: pd.DataFrame, csv_path: str, max_delta_hz: float = 250.0) -> pd.DataFrame:
    """Attach measured resonance frequencies/response from a user calibration CSV.

    Expected CSV columns: `frequency_Hz` and optionally `response`. The nearest measured
    resonance is assigned to each theoretical mode when within max_delta_hz. The shape
    fit remains based on the mathematical mode family; only the physical drive frequency
    is replaced by the measured resonance when available.
    """
    if mode_table.empty:
        return mode_table.copy()
    if not csv_path:
        return mode_table.copy()
    cal = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in cal.columns}
    fcol = cols.get("frequency_hz") or cols.get("frequency")
    if not fcol:
        raise ValueError("Calibration CSV must contain a frequency_Hz or frequency column.")
    rf = pd.to_numeric(cal[fcol], errors="coerce").to_numpy(float)
    valid = np.isfinite(rf) & (rf > 0)
    if not np.any(valid):
        raise ValueError("Calibration CSV contains no positive frequencies.")
    rf = rf[valid]
    response = None
    if "response" in cols:
        response = pd.to_numeric(cal.loc[valid, cols["response"]], errors="coerce").to_numpy(float)
        response = np.where(np.isfinite(response), np.maximum(response, 0.0), 0.0)
    out = mode_table.copy()
    assigned = []; resp = []; delta = []
    tol = float(max_delta_hz)
    for f in out["frequency_Hz"].to_numpy(float):
        j = int(np.argmin(np.abs(rf - f)))
        d = float(abs(rf[j] - f))
        if d <= tol:
            assigned.append(float(rf[j]))
            resp.append(float(response[j]) if response is not None else 1.0)
            delta.append(d)
        else:
            assigned.append(float(f)); resp.append(0.0); delta.append(d)
    out["drive_frequency_Hz"] = assigned
    out["calibration_response"] = resp
    out["calibration_delta_Hz"] = delta
    out["calibrated"] = [d <= tol for d in delta]
    # If measured response is supplied, use it as a soft coupling factor. It is not a
    # complete FRF/Q model, but it prevents driving frequencies with essentially no measured response.
    if response is not None and np.max(out["calibration_response"].to_numpy(float)) > 0:
        r = out["calibration_response"].to_numpy(float)
        out["mixture_weight_calibrated"] = out["mixture_weight"].to_numpy(float) * (r / max(float(r.max()), 1e-12))
    else:
        out["mixture_weight_calibrated"] = out["mixture_weight"].to_numpy(float)
    calibrated_power = np.maximum(out["mixture_weight_calibrated"].to_numpy(float), 0.0)
    if calibrated_power.max() > 0:
        calibrated_power = calibrated_power / float(calibrated_power.sum())
    out["drive_amplitude_calibrated"] = np.sqrt(calibrated_power)
    out["drive_amplitude_calibrated"] = out["drive_amplitude_calibrated"] / max(float(out["drive_amplitude_calibrated"].max()), 1e-12)
    # Preserve actuator controllability compensation after calibration. Measured response
    # changes the required power weighting, while coupling still affects how efficiently
    # a specific point actuator excites the selected mode.
    phys = np.sqrt(calibrated_power) / np.maximum(out["actuator_coupling"].to_numpy(float), 1e-9)
    out["drive_amplitude_physical_calibrated"] = phys / max(float(phys.max()), 1e-12)
    return out


def _fade_envelope(n: int, sr: int, attack_s: float = 0.05, release_s: float = 0.12) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)
    env = np.ones(n, dtype=float)
    a = min(n, max(1, int(round(float(attack_s) * sr))))
    r = min(n, max(1, int(round(float(release_s) * sr))))
    env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    env[-r:] *= np.linspace(1.0, 0.0, r, endpoint=True)
    return env


def create_physical_drive_audio(
    mode_table: pd.DataFrame,
    duration_per_mode_s: float = 3.0,
    sample_rate: int = 44_100,
    amplitude_column: str = "drive_amplitude_proxy",
    use_exact_frequencies: bool = True,
    frequency_column: str = "frequency_Hz",
) -> tuple[np.ndarray, list[dict]]:
    """Generate a sequential mode sweep using exact physical frequencies."""
    if mode_table.empty:
        raise ValueError("No modes available")
    sr = int(sample_rate); d = float(duration_per_mode_s)
    if d <= 0:
        raise ValueError("duration_per_mode_s must be positive")
    total = int(round(len(mode_table) * d * sr))
    out = np.zeros(total, dtype=np.float64)
    events: list[dict] = []
    for idx, (_, row) in enumerate(mode_table.iterrows()):
        freq_key = frequency_column if frequency_column in row.index else "frequency_Hz"
        f = float(row[freq_key] if use_exact_frequencies else row.get("note_Hz", row[freq_key]))
        amp = float(row.get(amplitude_column, 1.0))
        start = int(round(idx * d * sr)); n = min(int(round(d * sr)), total - start)
        if n <= 0 or f <= 0 or f >= sr / 2:
            continue
        t = np.arange(n, dtype=float) / sr
        env = _fade_envelope(n, sr)
        out[start:start+n] += amp * env * np.sin(2.0 * math.pi * f * t)
        events.append({"index": int(idx + 1), "start_s": float(idx * d), "duration_s": d, "frequency_Hz": f, "amplitude_proxy": amp})
    peak = float(np.max(np.abs(out)))
    if peak > 0: out = 0.95 * out / peak
    return out.astype(np.float32), events


def create_simultaneous_physical_drive_audio(
    mode_table: pd.DataFrame,
    duration_s: float = 12.0,
    sample_rate: int = 44_100,
    amplitude_column: str = "drive_amplitude_physical",
    phase_lock: bool = True,
    frequency_column: str = "frequency_Hz",
    weight_column: str = "drive_amplitude_physical",
) -> tuple[np.ndarray, list[dict]]:
    """Generate the simultaneous multi-tone drive intended to reproduce the fitted target.

    The inverse model fits observable power, so drive amplitude is approximately the
    square root of modal mixture weight, adjusted by the geometric actuator-coupling
    proxy. Exact mode frequencies are retained; no musical quantization is applied.
    """
    if mode_table.empty:
        raise ValueError("No modes available")
    sr = int(sample_rate); duration_s = float(duration_s)
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    total = int(round(duration_s * sr))
    t = np.arange(total, dtype=float) / sr
    audio = np.zeros(total, dtype=float)
    selected_weight_column = weight_column if weight_column in mode_table.columns else amplitude_column
    if selected_weight_column in mode_table.columns:
        raw_amps = np.asarray(mode_table[selected_weight_column], dtype=float)
    else:
        raw_amps = np.sqrt(np.asarray(mode_table.get("mixture_weight", np.ones(len(mode_table))), dtype=float))
    raw_amps = np.maximum(raw_amps, 0.0)
    if raw_amps.max() > 0:
        raw_amps = raw_amps / raw_amps.max()
    events: list[dict] = []
    for i, (_, row) in enumerate(mode_table.iterrows()):
        freq_key = frequency_column if frequency_column in row.index else "frequency_Hz"
        f = float(row[freq_key])
        if f <= 0 or f >= sr / 2:
            continue
        amp = float(raw_amps[i])
        phase = 0.0 if phase_lock else 2.0 * math.pi * ((i * 0.173) % 1.0)
        audio += amp * np.sin(2.0 * math.pi * f * t + phase)
        events.append({
            "mode_index": int(i + 1), "frequency_Hz": f, "relative_amplitude": amp,
            "phase_deg": float(math.degrees(phase)), "duration_s": duration_s,
        })
    env = _fade_envelope(total, sr, attack_s=0.20, release_s=0.25)
    audio *= env
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = 0.92 * audio / peak
    return audio.astype(np.float32), events



def generate_combination_melody_plan(
    n_tones: int,
    tone_order: Optional[Sequence[int]] = None,
    combination_count: Optional[int] = None,
    min_combination_size: int = 2,
    max_combination_size: Optional[int] = None,
    seed: int = 0,
    include_all_tones: bool = True,
) -> list[dict]:
    """Build a reproducible combinatorial melody plan from *tone indices* only.

    The algorithm deliberately does not require the actual frequencies.  It operates
    on the known number of tones and a canonical order, then leaves pitch assignment
    to :func:`create_musical_audio_from_modes`.

    Structure:
      1. Every distinct tone is presented exactly once.
      2. A deterministic pseudo-random development section adds subset combinations
         with sizes spanning ``min_combination_size..max_combination_size``.
      3. An optional final ALL-TONES event synthesizes the complete set.

    Each combination is stored as an ordered tuple of source tone indices.  The random
    generator is seeded so identical inputs reproduce the same melody.  Changing the
    seed produces a different valid combinatorial melody without changing the tone set.
    """
    n = int(n_tones)
    if n < 1:
        raise ValueError("n_tones must be >= 1")

    if tone_order is None:
        order = list(range(n))
    else:
        order = [int(i) for i in tone_order]
        if sorted(order) != list(range(n)):
            raise ValueError("tone_order must be a permutation of 0..n_tones-1")

    if n == 1:
        plan = [{"section": "declaration", "event_type": "single", "tone_indices": [order[0]], "event_number": 1}]
        if include_all_tones:
            plan.append({"section": "synthesis", "event_type": "all", "tone_indices": [order[0]], "event_number": 2})
        return plan
    if n == 2:
        # There is no proper intermediate subset size: every 2-tone subset is already
        # the ALL-tones set. Keep the grammar well-defined rather than raising.
        plan = [
            {"section": "declaration", "event_type": "single", "tone_indices": [int(order[0])], "event_number": 1},
            {"section": "declaration", "event_type": "single", "tone_indices": [int(order[1])], "event_number": 2},
        ]
        if include_all_tones:
            plan.append({"section": "synthesis", "event_type": "all", "tone_indices": [int(i) for i in order], "event_number": 3})
        return plan

    kmin = max(2, int(min_combination_size))
    kmax = n - 1 if max_combination_size is None else min(int(max_combination_size), n - 1)
    if kmin > kmax:
        raise ValueError("Combination size range is empty; choose min size < n_tones")

    count = (2 * n if combination_count is None else int(combination_count))
    if count < 0:
        raise ValueError("combination_count must be >= 0")

    rng = random.Random(int(seed))
    position = {idx: pos for pos, idx in enumerate(order)}

    plan: list[dict] = []
    event_no = 1
    # Section 1: every tone exactly once. This guarantees coverage independent of
    # subsequent random combinations and preserves the requested global order.
    for idx in order:
        plan.append({
            "section": "declaration",
            "event_type": "single",
            "tone_indices": [int(idx)],
            "event_number": event_no,
        })
        event_no += 1

    # Section 2: randomly selected combinations. We deliberately cycle combination
    # sizes so the melody contains pairs, triads, and larger groupings rather than
    # clustering at one subset size.
    seen: set[tuple[int, ...]] = set()
    for j in range(count):
        size = kmin + (j % (kmax - kmin + 1))
        chosen = tuple(sorted(rng.sample(range(n), size), key=lambda i: position[i]))
        if not chosen:
            continue
        # Avoid an identical subset unless all unique subsets of this size are exhausted.
        attempts = 0
        while chosen in seen and attempts < 12:
            chosen = tuple(sorted(rng.sample(range(n), size), key=lambda i: position[i]))
            attempts += 1
        seen.add(chosen)
        # Randomly vary the internal contour while retaining the global tone identity.
        members = list(chosen)
        if len(members) > 2:
            roll = rng.randrange(len(members))
            members = members[roll:] + members[:roll]
        if rng.random() < 0.35:
            members.reverse()
        plan.append({
            "section": "combinatorial",
            "event_type": "combination",
            "tone_indices": [int(i) for i in members],
            "event_number": event_no,
        })
        event_no += 1

    # Section 3: one explicit synthesis event containing every tone.
    if include_all_tones:
        plan.append({
            "section": "synthesis",
            "event_type": "all",
            "tone_indices": [int(i) for i in order],
            "event_number": event_no,
        })

    return plan


def _musical_frequency_mapping(physical: np.ndarray, interval_compression: float, center_hz: float = 220.0) -> np.ndarray:
    """Map physical resonances to a musical register using one global transform."""
    gmean = float(np.exp(np.mean(np.log(np.maximum(physical, 1e-9)))))
    log_ratio = np.log2(np.maximum(physical, 1e-9) / max(gmean, 1e-9))
    musical = center_hz * np.power(2.0, interval_compression * log_ratio)
    while float(np.median(musical)) < 110.0:
        musical *= 2.0
    while float(np.median(musical)) > 880.0:
        musical /= 2.0
    return musical


def create_musical_audio_from_modes(
    mode_table: pd.DataFrame,
    duration_s: float = 24.0,
    sample_rate: int = 44_100,
    tempo_bpm: float = 96.0,
    quantization: str = "chromatic",
    harmonic_order: int = 4,
    arrangement: str = "DNA Combination Melody",
    repeat_to_target: bool = False,
    beats_per_note: float = 0.75,
    phrase_gap_beats: float = 0.25,
    interval_compression: float = 0.70,
    combination_count: int = 12,
    min_combination_size: int = 2,
    max_combination_size: Optional[int] = None,
    combination_seed: int = 0,
    include_all_tones: bool = True,
    combination_render: str = "Arpeggio + chord",
    combination_beats: float = 1.5,
) -> tuple[np.ndarray, list[dict]]:
    """Create a listening/Suno-oriented DNA combinatorial melody.

    The new ``DNA Combination Melody`` arrangement is intentionally defined on the
    *identity and count of source tones*, not on their numeric values. It therefore
    works even when the frequencies are only known after the resonator inverse solve.

    Melody grammar:

        [T1 T2 ... TN]
        -> [random subsets of T1..TN]
        -> [T1 T2 ... TN] (all-tone synthesis)

    Every tone appears once as a singleton. Random combination events then explore
    different subset sizes. The final all-tone event explicitly combines the complete
    source set. Combination events are rendered as an arpeggio followed by a short
    simultaneous chord by default, so a music model can perceive both the individual
    identities and their harmonic combination.

    This is a creative sonification, NOT the physical cymatics drive waveform.
    """
    if mode_table.empty:
        raise ValueError("No modes available")
    sr = int(sample_rate)
    target_duration = float(duration_s)
    if target_duration <= 0 or sr < 8_000:
        raise ValueError("duration_s must be positive and sample_rate must be >= 8000")
    beat = 60.0 / max(float(tempo_bpm), 1.0)
    single_beats = max(float(beats_per_note), 0.25)
    combo_beats = max(float(combination_beats), single_beats)
    interval_compression = float(interval_compression)
    if not (0.25 <= interval_compression <= 1.0):
        raise ValueError("interval_compression must be between 0.25 and 1.0")
    gap_beats = max(float(phrase_gap_beats), 0.0)
    if combination_render not in {"Arpeggio + chord", "Chord", "Arpeggio"}:
        raise ValueError("Unknown combination_render")

    df = mode_table.copy().reset_index(drop=True)
    physical = pd.to_numeric(df["frequency_Hz"], errors="coerce").to_numpy(float)
    valid = np.isfinite(physical) & (physical > 0.0) & (physical < sr / 2.0)
    df = df.loc[valid].reset_index(drop=True)
    physical = physical[valid]
    if len(df) == 0:
        raise ValueError("No valid mode frequencies available")

    weights = pd.to_numeric(df.get("mixture_weight", pd.Series(np.ones(len(df)))), errors="coerce").fillna(0.0).to_numpy(float)
    corr = pd.to_numeric(df.get("single_mode_correlation", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0.0).to_numpy(float)
    w = np.maximum(weights, 0.0)
    if w.max() > 0:
        w = w / w.max()
    c = np.clip((corr + 1.0) * 0.5, 0.0, 1.0)
    salience = 0.65 * w + 0.35 * c
    df["_salience"] = salience

    m = pd.to_numeric(df.get("m", pd.Series(np.zeros(len(df)))), errors="coerce").fillna(0).to_numpy(int)
    n = pd.to_numeric(df.get("n", pd.Series(np.ones(len(df)))), errors="coerce").fillna(1).to_numpy(int)
    if arrangement in {"Frequency ascending", "Frequency contour"}:
        tone_order = list(np.argsort(physical, kind="stable"))
        arrangement_label = arrangement
    elif arrangement == "Angular symmetry":
        tone_order = list(np.lexsort((n, physical, m)))
        arrangement_label = arrangement
    else:
        # Default and DNA Combination Melody use spatial importance as the canonical
        # source-tone order. The random subset generator itself uses only indices/count.
        tone_order = list(np.argsort(-salience, kind="stable"))
        arrangement_label = "DNA Combination Melody"

    if arrangement_label == "DNA Combination Melody":
        plan = generate_combination_melody_plan(
            len(df), tone_order=tone_order, combination_count=int(combination_count),
            min_combination_size=int(min_combination_size), max_combination_size=max_combination_size,
            seed=int(combination_seed), include_all_tones=bool(include_all_tones),
        )
    else:
        # Preserve the previous simple arrangements for comparison.
        plan = [
            {"section": "linear", "event_type": "single", "tone_indices": [int(i)], "event_number": k + 1}
            for k, i in enumerate(tone_order)
        ]

    musical_raw = _musical_frequency_mapping(physical, interval_compression)
    H = max(1, int(harmonic_order))
    harmonic_den = sum(1.0 / (h ** 1.15) for h in range(1, H + 1))

    # Estimate natural motif duration so the no-repeat renderer never manufactures a
    # silent tail. Combination events consume more musical time because they contain a
    # richer amount of information.
    natural_beats = 0.0
    for p in plan:
        natural_beats += combo_beats if p["event_type"] in {"combination", "all"} else single_beats
        natural_beats += gap_beats
    natural_duration = max(beat, natural_beats * beat)
    if repeat_to_target:
        render_duration = target_duration
    else:
        render_duration = min(target_duration, natural_duration)

    total = max(1, int(round(render_duration * sr)))
    audio = np.zeros(total, dtype=float)
    events: list[dict] = []
    cursor_s = 0.0

    def render_tone(start_s: float, duration: float, freq: float, amp: float, phase: float = 0.0):
        start = int(round(start_s * sr))
        if start >= total:
            return
        count = min(int(round(duration * sr)), total - start)
        if count <= 0:
            return
        t = np.arange(count, dtype=float) / sr
        env = _fade_envelope(count, sr, attack_s=min(0.025, duration * 0.12), release_s=min(0.12, duration * 0.25))
        signal = np.zeros(count, dtype=float)
        for h in range(1, H + 1):
            partial = freq * h * (1.0 + 0.0006 * (h - 1) ** 2)
            if partial >= sr / 2.0:
                break
            signal += (1.0 / (h ** 1.15)) * np.sin(2.0 * math.pi * partial * t + phase * h)
        signal /= max(harmonic_den, 1e-9)
        audio[start:start + count] += amp * env * signal

    def render_chord(start_s: float, duration: float, tone_indices: list[int], base_amp: float):
        k = max(len(tone_indices), 1)
        for j, idx in enumerate(tone_indices):
            # Small deterministic phase offsets prevent completely identical wave starts
            # in a large chord while remaining reproducible.
            phase = 2.0 * math.pi * ((j * 0.173) % 1.0)
            render_tone(start_s, duration, float(musical_raw[idx]), base_amp / math.sqrt(k), phase)

    def event_note_name(idx: int) -> str:
        f = float(musical_raw[idx])
        if quantization == "chromatic":
            return quantize_frequency(f, "chromatic")[1]
        return hz_to_note(f)

    # Loop over one motif. If repetition is enabled, the complete combinatorial grammar
    # repeats from its beginning; if not, it ends naturally at the final event.
    repetitions = 1
    if repeat_to_target and natural_duration > 0:
        repetitions = max(1, int(math.ceil(target_duration / natural_duration)))
    full_plan = plan * repetitions

    for rep, event in enumerate(full_plan):
        indices = [int(i) for i in event["tone_indices"]]
        if not indices:
            continue
        is_combo = event["event_type"] in {"combination", "all"}
        event_beats = combo_beats if is_combo else single_beats
        event_duration = event_beats * beat
        if cursor_s >= render_duration:
            break

        if not is_combo:
            idx = indices[0]
            f_phys = float(physical[idx])
            f_mus = float(musical_raw[idx])
            if quantization == "chromatic":
                f_render, note = quantize_frequency(f_mus, "chromatic")
            elif quantization == "none":
                f_render, note = f_mus, hz_to_note(f_mus)
            else:
                raise ValueError("quantization must be 'none' or 'chromatic'")
            if not (0.0 < f_render < sr / 2.0):
                cursor_s += event_duration + gap_beats * beat
                continue
            dur = min(event_duration * 0.90, max(0.0, render_duration - cursor_s))
            render_tone(cursor_s, dur, f_render, 0.24 + 0.26 * float(salience[idx]))
            events.append({
                "event_number": int(event["event_number"]),
                "repeat_number": rep + 1,
                "section": event["section"],
                "event_type": "single",
                "tone_indices": str(indices),
                "tone_count": 1,
                "physical_frequencies_Hz": str([round(f_phys, 6)]),
                "musical_frequencies_Hz": str([round(float(f_render), 6)]),
                "notes": note,
                "start_s": float(cursor_s),
                "duration_s": float(dur),
                "combination_render": "Single",
                "combination_seed": int(combination_seed),
            })
        else:
            # Preserve the chord identity while also producing an audible melodic line:
            # arpeggiate through the subset, then add a short simultaneous chord tail.
            converted = []
            for idx in indices:
                f_mus = float(musical_raw[idx])
                if quantization == "chromatic":
                    f_render, note = quantize_frequency(f_mus, "chromatic")
                elif quantization == "none":
                    f_render, note = f_mus, hz_to_note(f_mus)
                else:
                    raise ValueError("quantization must be 'none' or 'chromatic'")
                if 0.0 < f_render < sr / 2.0:
                    converted.append((idx, float(f_render), note))
            if not converted:
                cursor_s += event_duration + gap_beats * beat
                continue

            k = len(converted)
            if combination_render == "Chord":
                render_chord(cursor_s, min(event_duration * 0.90, render_duration - cursor_s), [x[0] for x in converted], 0.33)
            else:
                arp_fraction = 0.65 if combination_render == "Arpeggio + chord" else 0.95
                arp_total = max(0.03, event_duration * arp_fraction)
                per_note = arp_total / k
                for pos, (idx, f_render, note) in enumerate(converted):
                    render_tone(cursor_s + pos * per_note, min(per_note * 0.88, render_duration - (cursor_s + pos * per_note)), f_render, (0.28 + 0.18 * float(salience[idx])) / math.sqrt(k))
                if combination_render == "Arpeggio + chord":
                    chord_start = cursor_s + arp_total * 0.78
                    chord_dur = min(event_duration - (chord_start - cursor_s), render_duration - chord_start)
                    if chord_dur > 0:
                        render_chord(chord_start, chord_dur * 0.92, [x[0] for x in converted], 0.30)

            events.append({
                "event_number": int(event["event_number"]),
                "repeat_number": rep + 1,
                "section": event["section"],
                "event_type": event["event_type"],
                "tone_indices": str(indices),
                "tone_count": len(indices),
                "physical_frequencies_Hz": str([round(float(physical[i]), 6) for i in indices]),
                "musical_frequencies_Hz": str([round(float(f), 6) for _, f, _ in converted]),
                "notes": " ".join(note for _, _, note in converted),
                "start_s": float(cursor_s),
                "duration_s": float(min(event_duration, render_duration - cursor_s)),
                "combination_render": combination_render,
                "combination_seed": int(combination_seed),
            })

        cursor_s += event_duration + gap_beats * beat

    if not repeat_to_target and events:
        end_s = max(e["start_s"] + e["duration_s"] for e in events)
        total = max(1, int(round(end_s * sr)))
        audio = audio[:total]
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = 0.92 * audio / peak
    return audio.astype(np.float32), events

