from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

LOGGER = logging.getLogger("extract_nifti")
NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class Window:
    """CT display window defined by level (centre) and width in Hounsfield units."""

    name: str
    level: float
    width: float


DEFAULT_STRIPPED_WINDOWS: tuple[Window, ...] = (
    Window("brain_stripped", level=40.0, width=80.0),
    Window("subdural_stripped", level=80.0, width=200.0),
)
DEFAULT_NORMAL_WINDOWS: tuple[Window, ...] = (
    Window("brain_normal", level=40.0, width=80.0),
)


@dataclass(frozen=True)
class PatientJob:
    """The trio of input volumes (any may be missing) for a single patient."""

    patient_id: str
    stripped: Path | None
    normal: Path | None
    mask: Path | None


def configure_logging(verbose: bool) -> None:
    """Initialise the root logger with a compact, timestamped format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def discover_volumes(root: Path | None) -> dict[str, Path]:
    """Return a ``{stem: path}`` map for every NIfTI directly under ``root``."""
    if root is None or not root.is_dir():
        return {}
    found: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name.lower()
        if name.endswith(".nii.gz"):
            stem = entry.name[:-7]
        elif name.endswith(".nii"):
            stem = entry.name[:-4]
        else:
            continue
        found[stem] = entry
    return found


def normalize_patient_id(stem: str) -> str:
    """Zero-pad numeric stems to 3 digits so they match the dataset CSV format."""
    try:
        return f"{int(stem):03d}"
    except ValueError:
        return stem


def load_volume(path: Path, canonical: bool) -> np.ndarray:
    """Load a NIfTI volume as ``float32``, optionally reoriented to RAS+."""
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    return np.asanyarray(img.dataobj, dtype=np.float32)


def apply_window(slice_hu: np.ndarray, window: Window) -> np.ndarray:
    """Clip Hounsfield units to ``[level - width/2, level + width/2]`` and rescale to uint8."""
    lo = window.level - window.width / 2.0
    hi = window.level + window.width / 2.0
    clipped = np.clip(slice_hu, lo, hi)
    scaled = (clipped - lo) * (255.0 / max(window.width, 1e-6))
    return scaled.astype(np.uint8)


def to_mask_uint8(slice_arr: np.ndarray) -> np.ndarray:
    """Threshold any positive voxel to 255 and zero elsewhere for a clean binary PNG."""
    binary = slice_arr > 0.5
    return binary.astype(np.uint8) * 255


def write_png(path: Path, image: np.ndarray) -> None:
    """Persist a single grayscale PNG, creating parent dirs lazily."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def _iter_slices(volume: np.ndarray, axis: int, rot90: int):
    """Yield ``(index, 2D slice)`` along ``axis`` with optional 90° CCW rotations."""
    n = volume.shape[axis]
    for i in range(n):
        sl = np.take(volume, i, axis=axis)
        if rot90:
            sl = np.rot90(sl, k=rot90)
        yield i, sl


def _emit_image_volume(
    path: Path,
    patient_dir: Path,
    windows: tuple[Window, ...],
    slice_axis: int,
    canonical: bool,
    rot90: int,
) -> int:
    """Slice an HU volume and write one PNG per (slice, window) pair; return slice count."""
    volume = load_volume(path, canonical)
    count = 0
    for idx, sl in _iter_slices(volume, slice_axis, rot90):
        slice_name = f"{idx:04d}"
        for window in windows:
            write_png(
                patient_dir / f"{slice_name}_{window.name}.png",
                apply_window(sl, window),
            )
        count += 1
    return count


def _emit_mask_volume(
    path: Path,
    patient_dir: Path,
    slice_axis: int,
    canonical: bool,
    rot90: int,
) -> int:
    """Slice a binary mask volume and write one PNG per slice; return slice count."""
    volume = load_volume(path, canonical)
    count = 0
    for idx, sl in _iter_slices(volume, slice_axis, rot90):
        write_png(patient_dir / f"{idx:04d}.png", to_mask_uint8(sl))
        count += 1
    return count


def process_patient(
    job: PatientJob,
    images_root: Path,
    masks_root: Path,
    stripped_windows: tuple[Window, ...],
    normal_windows: tuple[Window, ...],
    slice_axis: int,
    canonical: bool,
    rot90: int,
) -> tuple[str, int, str | None]:
    """Slice every available volume for one patient; returns ``(id, slices, error)``."""
    try:
        slices = 0
        img_dir = images_root / job.patient_id
        mask_dir = masks_root / job.patient_id

        if job.stripped is not None:
            slices = max(
                slices,
                _emit_image_volume(
                    job.stripped,
                    img_dir,
                    stripped_windows,
                    slice_axis,
                    canonical,
                    rot90,
                ),
            )
        if job.normal is not None:
            slices = max(
                slices,
                _emit_image_volume(
                    job.normal, img_dir, normal_windows, slice_axis, canonical, rot90
                ),
            )
        if job.mask is not None:
            slices = max(
                slices,
                _emit_mask_volume(job.mask, mask_dir, slice_axis, canonical, rot90),
            )
        return job.patient_id, slices, None
    except Exception as exc:
        return job.patient_id, 0, str(exc)


def build_jobs(
    stripped_dir: Path | None,
    normal_dir: Path | None,
    masks_dir: Path | None,
) -> list[PatientJob]:
    """Pair NIfTI files across the three input folders by filename stem."""
    s = discover_volumes(stripped_dir)
    n = discover_volumes(normal_dir)
    m = discover_volumes(masks_dir)
    stems = sorted(set(s) | set(n) | set(m))
    return [
        PatientJob(
            patient_id=normalize_patient_id(stem),
            stripped=s.get(stem),
            normal=n.get(stem),
            mask=m.get(stem),
        )
        for stem in stems
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-slice PNGs from NIfTI volumes for the ICH dataset. "
            "Pairs files across --stripped-dir, --normal-dir, --masks-dir by stem."
        ),
    )
    parser.add_argument(
        "--stripped-dir",
        type=Path,
        default=None,
        help="Folder of skull-stripped NIfTI scans (yields brain_stripped + subdural_stripped).",
    )
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=None,
        help="Folder of un-stripped NIfTI scans (yields brain_normal).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Folder of binary mask NIfTI volumes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root; PNGs land under {output}/images and {output}/masks.",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=2,
        help="Volume axis to slice along (default 2 = axial after canonical reorientation).",
    )
    parser.add_argument(
        "--no-canonical",
        action="store_true",
        help="Skip nib.as_closest_canonical reorientation.",
    )
    parser.add_argument(
        "--rot90",
        type=int,
        default=1,
        help="Number of 90° CCW rotations applied to each emitted slice (default 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Patients processed concurrently via a process pool (default 1).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero when any patient fails."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not any([args.stripped_dir, args.normal_dir, args.masks_dir]):
        LOGGER.error(
            "at least one of --stripped-dir / --normal-dir / --masks-dir is required"
        )
        return 2

    output_root: Path = args.output_dir.resolve()
    images_root = output_root / "images"
    masks_root = output_root / "masks"
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.stripped_dir, args.normal_dir, args.masks_dir)
    if not jobs:
        LOGGER.warning("no NIfTI volumes discovered in the provided folders")
        return 0

    workers = max(1, args.workers)
    LOGGER.info("processing %d patients with %d worker(s)", len(jobs), workers)

    kw = {
        "images_root": images_root,
        "masks_root": masks_root,
        "stripped_windows": DEFAULT_STRIPPED_WINDOWS,
        "normal_windows": DEFAULT_NORMAL_WINDOWS,
        "slice_axis": args.slice_axis,
        "canonical": not args.no_canonical,
        "rot90": args.rot90,
    }

    if workers == 1:
        results = (process_patient(j, **kw) for j in jobs)
    else:
        pool = cf.ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(process_patient, j, **kw) for j in jobs]
        results = (f.result() for f in cf.as_completed(futures))

    failures = 0
    for idx, (pid, n_slices, err) in enumerate(results, start=1):
        if err is None:
            LOGGER.info(
                "[%d/%d] OK   patient=%s slices=%d", idx, len(jobs), pid, n_slices
            )
        else:
            failures += 1
            LOGGER.error("[%d/%d] FAIL patient=%s :: %s", idx, len(jobs), pid, err)

    LOGGER.info("done: %d patients, %d failures", len(jobs), failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
