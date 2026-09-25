from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom
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
    """Build a sequence-dependent coarse-grained dsDNA reconstruction.

    This is a rigid-base-pair-step reconstruction, not an atomistic molecular
    dynamics simulation. It captures sequence-dependent local twist, roll,
    tilt, shift, slide and rise and then places two coarse backbone loci around
    each local base-pair frame.
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
            p = dict(tilt=0.0, roll=0.0, twist=34.3, shift=0.0, slide=0.0, rise=3.4)
        else:
            p = STEP_PARAMS[step]

        local_R = (
            _rz(math.radians(p["twist"]))
            @ _ry(math.radians(p["roll"]))
            @ _rx(math.radians(p["tilt"]))
        )
        local_t = np.array([p["shift"], p["slide"], p["rise"]], dtype=float)

        # Matrix convention: frame[i] maps local coordinates into global.
        frames[i + 1] = frames[i] @ local_R
        centers[i + 1] = centers[i] + frames[i] @ local_t
        step_rows.append({"index": i + 1, "step": step, **p})

    strand1 = np.zeros_like(centers)
    strand2 = np.zeros_like(centers)
    basepair_edges = np.zeros((n, 2, 3), dtype=float)

    # The simple backbone radius is intentionally a visualization scale rather
    # than an atomically reconstructed phosphate/sugar backbone.
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
from scipy.special import jn_zeros, jv
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


def rank_circular_membrane_modes(
    target_image: np.ndarray,
    radius_m: float,
    wave_speed_m_s: float,
    max_angular_mode: int = 20,
    max_radial_mode: int = 12,
    top_modes: int = 16,
    target_type: str = "Nodal / sand",
    fit_size: int = 128,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rank ideal circular membrane modes against a 2-D target.

    The circular membrane approximation uses Bessel-function eigenfunctions.
    For m>0, cosine/sine partners are frequency-degenerate, so the implementation
    searches the orientation analytically by evaluating a vectorized orientation
    grid. This preserves the frequency family while fitting its spatial phase.

    This is a *candidate mode identification* model. It is not a calibrated model
    of a metal plate; boundary conditions, stiffness, damping and actuator coupling
    must be measured for physical prediction.
    """
    if target_type not in {"Nodal / sand", "Displacement magnitude"}:
        raise ValueError("Unsupported target_type")
    radius_m = float(radius_m); wave_speed_m_s = float(wave_speed_m_s)
    if radius_m <= 0 or wave_speed_m_s <= 0:
        raise ValueError("radius_m and wave_speed_m_s must be positive")
    max_angular_mode = int(max_angular_mode)
    max_radial_mode = int(max_radial_mode)
    top_modes = int(top_modes)
    fit_size = int(fit_size)
    if not (0 <= max_angular_mode <= 64 and 1 <= max_radial_mode <= 32):
        raise ValueError("Unsupported mode search limits")
    if not (1 <= top_modes <= (max_angular_mode + 1) * max_radial_mode):
        raise ValueError("Invalid top_modes")
    if fit_size < 32:
        raise ValueError("fit_size must be >= 32")

    target = _resize_square(target_image, fit_size)
    x = np.linspace(-1.0, 1.0, fit_size)
    y = np.linspace(-1.0, 1.0, fit_size)
    X, Y = np.meshgrid(x, y)
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    inside = r <= 1.0
    target_inside = target[inside].astype(float)
    target_inside -= target_inside.mean()
    target_norm = float(np.linalg.norm(target_inside))

    rows: list[dict] = []
    fields: dict[tuple[int, int], np.ndarray] = {}

    for m in range(max_angular_mode + 1):
        zeros = jn_zeros(m, max_radial_mode)
        radial_base = [jv(m, float(alpha) * r) for alpha in zeros]
        for n in range(1, max_radial_mode + 1):
            alpha = float(zeros[n - 1])
            radial = np.asarray(radial_base[n - 1], dtype=float)
            angles = np.array([0.0], dtype=float) if m == 0 else np.linspace(0.0, math.pi / m, 25)

            # Vectorized orientation evaluation: (y, x, orientation).
            disp = radial[:, :, None] * np.cos(m * theta[:, :, None] - angles[None, None, :])
            if target_type == "Nodal / sand":
                vals = np.abs(disp[:, :, :])
                scale = max(float(np.percentile(vals[inside, :], 75)), 1e-6)
                field_stack = np.exp(-((vals / (0.22 * scale)) ** 2))
            else:
                field_stack = np.abs(disp)
            field_stack[~inside, :] = 0.0

            flat = field_stack[inside, :].T
            flat -= flat.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(flat, axis=1)
            if target_norm > 1e-12:
                scores = (flat @ target_inside) / np.maximum(norms * target_norm, 1e-12)
            else:
                scores = np.zeros(len(angles), dtype=float)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_field = field_stack[:, :, best_idx].copy()
            peak = float(best_field.max())
            if peak > 0:
                best_field /= peak
            fields[(m, n)] = best_field

            freq = circular_membrane_mode_frequency(m, n, radius_m, wave_speed_m_s)
            rows.append({
                "m": int(m),
                "n": int(n),
                "frequency_Hz": float(freq),
                "angular_order": int(m),
                "radial_order": int(n),
                "bessel_zero": alpha,
                "orientation_deg": float(math.degrees(float(angles[best_idx]))),
                "pattern_correlation": best_score,
                "mode_frequency_family": f"(m={m}, n={n})",
            })

    rows_df = pd.DataFrame(rows).sort_values("pattern_correlation", ascending=False).reset_index(drop=True)
    selected = rows_df.head(top_modes).copy()
    positive = np.maximum(selected["pattern_correlation"].to_numpy(float), 0.0)
    weights = positive / positive.sum() if positive.sum() > 0 else np.ones(len(selected)) / len(selected)
    recon = np.zeros((fit_size, fit_size), dtype=float)
    for weight, (_, row) in zip(weights, selected.iterrows()):
        recon += float(weight) * fields[(int(row["m"]), int(row["n"]))]
    recon[~inside] = 0.0
    if recon.max() > 0:
        recon /= recon.max()

    selected.insert(0, "mode_rank", np.arange(1, len(selected) + 1))
    selected["mixture_weight"] = weights
    selected["note"] = [hz_to_note(float(f)) for f in selected["frequency_Hz"]]
    return selected, recon


def polar_harmonic_spectrum(image: np.ndarray, max_m: int = 32, radial_bins: int = 128) -> pd.DataFrame:
    """Compute rotation-sensitive angular harmonic energy from a square image."""
    img = _resample_square(image, int(radial_bins) * 2)
    size = img.shape[0]
    y, x = np.indices(img.shape)
    cx = (size - 1) / 2.0; cy = (size - 1) / 2.0
    xx = x - cx; yy = y - cy
    r = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    mask = r <= r.max()
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


def image_registration_metrics(reference: np.ndarray, measured: np.ndarray, threshold: float = 0.55) -> dict:
    """Registration-aware image metrics plus boundary-distance error."""
    ref = np.asarray(reference, dtype=float)
    meas = np.asarray(measured, dtype=float)
    if ref.ndim != 2 or meas.ndim != 2:
        raise ValueError("Images must be grayscale 2-D arrays")
    meas = _resample_square(meas, ref.shape[0])
    ref -= ref.min(); meas -= meas.min()
    if ref.max() > 0: ref /= ref.max()
    if meas.max() > 0: meas /= meas.max()
    # Search integer pixel translations around the center to compensate for camera framing.
    best = None
    lim = max(2, ref.shape[0] // 20)
    for dy in range(-lim, lim + 1, max(1, lim // 10)):
        for dx in range(-lim, lim + 1, max(1, lim // 10)):
            shifted = np.zeros_like(meas)
            y0 = max(0, dy); y1 = min(ref.shape[0], ref.shape[0] + dy)
            x0 = max(0, dx); x1 = min(ref.shape[1], ref.shape[1] + dx)
            sy0 = max(0, -dy); sy1 = sy0 + (y1 - y0)
            sx0 = max(0, -dx); sx1 = sx0 + (x1 - x0)
            shifted[y0:y1, x0:x1] = meas[sy0:sy1, sx0:sx1]
            corr = _corr(ref, shifted)
            if best is None or corr > best[0]:
                best = (corr, dx, dy, shifted)
    _, dx, dy, aligned = best
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
        "angular-harmonic correlation": float(harmonic_corr),
        "aligned_image": aligned,
    }


def create_physical_drive_audio(
    mode_table: pd.DataFrame,
    duration_per_mode_s: float = 3.0,
    sample_rate: int = 44_100,
    amplitude_column: str = "mixture_weight",
    use_exact_frequencies: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Generate a one-mode-at-a-time physical-drive WAV.

    Frequencies are not quantized. This output is intended for a calibrated
    resonator experiment, not for conventional musical listening.
    """
    if mode_table.empty:
        raise ValueError("No modes available")
    sr = int(sample_rate)
    d = float(duration_per_mode_s)
    total = int(round(len(mode_table) * d * sr))
    out = np.zeros(total, dtype=np.float64)
    events: list[dict] = []
    for i, row in mode_table.iterrows():
        f = float(row["frequency_Hz"] if use_exact_frequencies else row.get("note_Hz", row["frequency_Hz"]))
        amp = float(row.get(amplitude_column, 1.0))
        start = int(round(i * d * sr)); n = min(int(round(d * sr)), total - start)
        if n <= 0 or f <= 0 or f >= sr / 2:
            continue
        t = np.arange(n, dtype=float) / sr
        attack = min(0.1, d / 5.0)
        release = min(0.15, d / 5.0)
        env = np.minimum(1.0, t / max(attack, 1e-6)) * np.minimum(1.0, (d - t) / max(release, 1e-6))
        out[start:start+n] += amp * env * np.sin(2.0 * math.pi * f * t)
        events.append({"index": int(i + 1), "start_s": float(i * d), "duration_s": d, "frequency_Hz": f, "amplitude": amp})
    peak = float(np.max(np.abs(out)))
    if peak > 0: out = 0.95 * out / peak
    return out.astype(np.float32), events


def create_musical_audio_from_modes(
    mode_table: pd.DataFrame,
    duration_s: float = 24.0,
    sample_rate: int = 44_100,
    tempo_bpm: float = 96.0,
    quantization: str = "chromatic",
    harmonic_order: int = 3,
) -> tuple[np.ndarray, list[dict]]:
    """Map physical mode frequencies to musical pitches while retaining order/weights."""
    if mode_table.empty:
        raise ValueError("No modes available")
    sr = int(sample_rate); total = int(round(float(duration_s) * sr))
    beat = 60.0 / max(float(tempo_bpm), 1.0)
    note_len = beat
    audio = np.zeros(total, dtype=float)
    events: list[dict] = []
    weights = np.asarray(mode_table.get("mixture_weight", np.ones(len(mode_table))), dtype=float)
    weights = weights / max(float(weights.max()), 1e-9)
    for idx, (_, row) in enumerate(mode_table.iterrows()):
        physical = float(row["frequency_Hz"])
        musical = physical
        if quantization == "chromatic":
            musical, note = quantize_frequency(physical, "chromatic")
        else:
            note = hz_to_note(musical)
        start = int(round((idx * note_len) * sr)) % total
        n = min(int(round(note_len * 0.9 * sr)), total - start)
        if n <= 0 or musical <= 0 or musical >= sr / 2: continue
        t = np.arange(n, dtype=float) / sr
        env = np.minimum(1.0, t / 0.025) * np.minimum(1.0, (note_len * 0.9 - t) / 0.08)
        signal = np.sin(2 * math.pi * musical * t)
        for h in range(2, max(1, int(harmonic_order)) + 1):
            signal += (1.0 / h) * np.sin(2 * math.pi * musical * h * t)
        signal /= sum(1.0 / h for h in range(1, max(1, int(harmonic_order)) + 1))
        audio[start:start+n] += (0.15 + 0.32 * weights[idx]) * env * signal
        events.append({"mode_index": idx + 1, "physical_frequency_Hz": physical, "musical_frequency_Hz": float(musical), "note": note})
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = 0.92 * audio / peak
    return audio.astype(np.float32), events
