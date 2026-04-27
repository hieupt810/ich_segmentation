"""Batch skull-stripping of NIfTI volumes via the SynthStrip Docker image.

Runs ``freesurfer/synthstrip`` against every ``.nii`` / ``.nii.gz`` file in an
input directory and writes stripped volumes (and optional binary masks) to an
output directory. Intended for brain CT/MRI. CT inputs should be pre-clipped
to a sane Hounsfield range (e.g. [0, 100] HU) for optimal mask quality, since
SynthStrip was primarily trained on MRI-like intensity distributions and is
only approximately contrast-agnostic at CT extremes.

Usage
-----
    python synthstrip_batch.py /data/ct_raw /data/ct_stripped \\
        --pull --workers 4 --border 1

References
----------
SynthStrip: Hoopes et al., NeuroImage 2022. Docker image tag
``freesurfer/synthstrip`` exposes ``synthstrip`` as its entrypoint, so all
flags after the image name are forwarded to the binary directly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("synthstrip_batch")

DEFAULT_IMAGE = "freesurfer/synthstrip:latest"
NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class StripJob:
    """A single-file skull-stripping task."""

    input_path: Path
    output_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class RunConfig:
    """Immutable runtime configuration shared across jobs."""

    image: str
    use_gpu: bool
    border: int
    no_csf: bool
    extra_args: tuple[str, ...]
    timeout_s: int


def configure_logging(verbose: bool) -> None:
    """Initialise root logger with a compact, timestamped format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_docker_available() -> None:
    """Fail fast if the Docker CLI is missing or the daemon is unreachable."""
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found on PATH")
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"docker daemon unreachable: {stderr}") from exc


def pull_image_if_missing(image: str) -> None:
    """Pull the SynthStrip image when it is not yet present locally."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode == 0:
        LOGGER.debug("image %s already present", image)
        return
    LOGGER.info("pulling image %s", image)
    subprocess.run(["docker", "pull", image], check=True)


def discover_inputs(root: Path) -> list[Path]:
    """Return all NIfTI files beneath ``root``, sorted for reproducibility."""
    found: list[Path] = []
    for entry in sorted(root.rglob("*")):
        if entry.is_file() and entry.name.lower().endswith(NIFTI_SUFFIXES):
            found.append(entry)
    return found


def _append_stem(filename: str, suffix: str) -> str:
    """Append ``suffix`` before the NIfTI extension of ``filename``."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return f"{filename[:-7]}{suffix}.nii.gz"
    if lower.endswith(".nii"):
        return f"{filename[:-4]}{suffix}.nii"
    raise ValueError(f"not a NIfTI filename: {filename}")


def derive_outputs(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    write_mask: bool,
) -> tuple[Path, Path | None]:
    """Map one input path to its stripped (and optional mask) output paths."""
    rel = input_path.relative_to(input_root)
    stripped = output_root / rel.with_name(rel.name)
    mask = None
    if write_mask:
        mask = output_root / rel.with_name(_append_stem(rel.name, "_mask"))
    return stripped, mask


def build_command(
    job: StripJob,
    input_root: Path,
    output_root: Path,
    cfg: RunConfig,
) -> list[str]:
    """Assemble the ``docker run`` invocation for a single job."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    gid = os.getgid() if hasattr(os, "getgid") else 0

    in_rel = job.input_path.relative_to(input_root).as_posix()
    out_rel = job.output_path.relative_to(output_root).as_posix()

    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{input_root.resolve()}:/in:ro",
        "-v",
        f"{output_root.resolve()}:/out",
    ]
    if cfg.use_gpu:
        cmd += ["--gpus", "all"]
    cmd.append(cfg.image)
    cmd += [
        "-i",
        f"/in/{in_rel}",
        "-o",
        f"/out/{out_rel}",
        "-b",
        str(cfg.border),
    ]
    if job.mask_path is not None:
        mask_rel = job.mask_path.relative_to(output_root).as_posix()
        cmd += ["-m", f"/out/{mask_rel}"]
    if cfg.no_csf:
        cmd.append("--no-csf")
    if cfg.use_gpu:
        cmd.append("-g")
    cmd += list(cfg.extra_args)
    return cmd


def run_job(
    job: StripJob,
    input_root: Path,
    output_root: Path,
    cfg: RunConfig,
) -> tuple[StripJob, int, str]:
    """Execute one job, returning (job, returncode, stderr)."""
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if job.mask_path is not None:
        job.mask_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(job, input_root, output_root, cfg)
    LOGGER.debug("cmd: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=cfg.timeout_s,
        )
    except subprocess.TimeoutExpired:
        return job, 124, f"timeout after {cfg.timeout_s}s"
    stderr = proc.stderr.decode(errors="replace")
    return job, proc.returncode, stderr


def process(
    jobs: Iterable[StripJob],
    input_root: Path,
    output_root: Path,
    cfg: RunConfig,
    workers: int,
) -> int:
    """Dispatch jobs sequentially or via a thread pool; return failure count."""
    jobs_list = list(jobs)
    total = len(jobs_list)
    failures = 0

    if workers <= 1:
        results = (run_job(j, input_root, output_root, cfg) for j in jobs_list)
    else:
        pool = cf.ThreadPoolExecutor(max_workers=workers)
        futures = [
            pool.submit(run_job, j, input_root, output_root, cfg) for j in jobs_list
        ]
        results = (f.result() for f in cf.as_completed(futures))

    for idx, (job, rc, err) in enumerate(results, start=1):
        if rc == 0:
            LOGGER.info(
                "[%d/%d] OK   %s -> %s",
                idx,
                total,
                job.input_path.name,
                job.output_path.name,
            )
        else:
            failures += 1
            LOGGER.error(
                "[%d/%d] FAIL rc=%d %s :: %s",
                idx,
                total,
                rc,
                job.input_path.name,
                err.strip()[-500:],
            )
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Batch SynthStrip skull-stripping via Docker.",
    )
    parser.add_argument(
        "input_dir", type=Path, help="folder containing .nii/.nii.gz volumes"
    )
    parser.add_argument(
        "output_dir", type=Path, help="destination folder (created if missing)"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help=f"Docker image (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="enable CUDA via --gpus all and synthstrip -g",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=1,
        help="mask border in mm (synthstrip -b, default 1)",
    )
    parser.add_argument(
        "--no-csf",
        action="store_true",
        help="exclude CSF from mask (synthstrip --no-csf)",
    )
    parser.add_argument(
        "--no-mask", action="store_true", help="skip writing the binary mask file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel containers (forced to 1 when --gpu)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-file timeout in seconds (default 1800)",
    )
    parser.add_argument(
        "--pull", action="store_true", help="pull the image before running"
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args forwarded to synthstrip (after --)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX-style exit code."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    input_root: Path = args.input_dir.resolve()
    output_root: Path = args.output_dir.resolve()
    if not input_root.is_dir():
        LOGGER.error("input_dir is not a directory: %s", input_root)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        ensure_docker_available()
        if args.pull:
            pull_image_if_missing(args.image)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        LOGGER.error("docker preflight failed: %s", exc)
        return 3

    inputs = discover_inputs(input_root)
    if not inputs:
        LOGGER.warning("no NIfTI files found under %s", input_root)
        return 0

    jobs: list[StripJob] = []
    for path in inputs:
        stripped, mask = derive_outputs(
            path,
            input_root,
            output_root,
            write_mask=not args.no_mask,
        )
        jobs.append(StripJob(path, stripped, mask))

    cfg = RunConfig(
        image=args.image,
        use_gpu=args.gpu,
        border=args.border,
        no_csf=args.no_csf,
        extra_args=tuple(args.extra),
        timeout_s=args.timeout,
    )

    workers = 1 if args.gpu else max(1, args.workers)
    LOGGER.info(
        "processing %d files with %d worker(s), image=%s, gpu=%s",
        len(jobs),
        workers,
        cfg.image,
        cfg.use_gpu,
    )
    failures = process(jobs, input_root, output_root, cfg, workers)
    LOGGER.info("done: %d files, %d failures", len(jobs), failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
