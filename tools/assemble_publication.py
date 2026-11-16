"""Rebuild this public benchmark archive from the local script2video checkout.

The selection keeps source, measured evidence, report media, and native Nsight
captures. It omits regenerable tensor/frame intermediates, exported SQLite
databases, installed dependencies, and expired signed publication links.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path


REPORT_DIRS = {
    "h3-pika-flex-comprehensive-rtx6000",
    "h3-pika-flex-fast-rtx6000",
    "h3-pika-flex-chunk-copy-configurations",
    "original_image_parallel_report",
    "vae_configuration_report",
    "h3-pika-flex-nsight-rtx6000",
    "h3-pika-flex-five-ten-rtx6000",
}
SKIP_PARTS = {"__pycache__", "node_modules", "site-packages", "fast_dependencies", "publication"}
SKIP_SUFFIXES = {".pt", ".rgb", ".sqlite", ".pyc", ".so", ".o", ".pack", ".idx", ".gz", ".safetensors", ".ckpt"}
SKIP_NAMES = {
    "reference_report.html", "source_report.html", "source_report.html.txt",
    "publish_verification.json", "comprehensive_publication_status.json",
    "followup_publication_status.json",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".html", ".log", ".yaml", ".yml", ".sh", ".patch"}
MEDIA_SUFFIXES = {".mp4", ".wav", ".jpg", ".jpeg", ".png", ".gif"}
PROFILE_SUFFIXES = {".nsys-rep", ".ncu-rep"}


def select(rel: Path) -> tuple[bool, str]:
    if rel.parts[0] == "h3-pika-flex-batch-two-rtx6000":
        return False, "duplicate fixed-workload report"
    if any(part in SKIP_PARTS for part in rel.parts):
        return False, "installed or duplicate dependency"
    if rel.name in SKIP_NAMES or ("upload" in rel.name and rel.suffix == ".log") or rel.name.startswith("latest_"):
        return False, "expired publication link or upload log"
    if rel.parts[0] in REPORT_DIRS and rel.suffix == ".gz":
        return True, "complete browser report"
    if rel.suffix in SKIP_SUFFIXES:
        return False, "regenerable intermediate or model weight"
    if rel.suffix in PROFILE_SUFFIXES:
        return True, "native profile"
    if rel.parts[0] in REPORT_DIRS:
        return True, "complete browser report"
    if rel.suffix in TEXT_SUFFIXES:
        return True, "source or measured evidence"
    if rel.suffix in MEDIA_SUFFIXES:
        if rel.parts[0] == "runs" and rel.name == "output.mp4" and rel.parts[1] in {
            "followup_nsys_v2_1t2v_control", "followup_nsys_v2_2t2v",
            "followup_nsys_v2_3t1v", "followup_nsys_v2_3t2v_staged",
        }:
            return True, "profiled video output"
        if rel.parts[0] == "inputs" or rel.parts[:2] == ("five_ten", "inputs") and rel.name == "audio_master.wav":
            return True, "benchmark input"
        if rel.parts[:2] == ("five_ten", "matched_ab_share"):
            return True, "matched comparison output"
        return False, "repeat media present in report"
    if rel.suffix == ".zip":
        return False, "duplicate reproduction archive"
    return False, "unneeded generated artifact"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="script2video checkout")
    parser.add_argument("--destination", type=Path, required=True, help="public repository root")
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    benchmark = source / "agent" / "rtx6000_benchmark"
    if not benchmark.is_dir():
        raise SystemExit(f"Missing source benchmark: {benchmark}")
    copied = Counter()
    omitted = Counter()
    for item in benchmark.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        rel = item.relative_to(benchmark)
        include, reason = select(rel)
        if not include:
            omitted[reason] += 1
            continue
        if item.suffix in TEXT_SUFFIXES and item.stat().st_size <= 8_000_000:
            data = item.read_text(errors="ignore")
            if "X-Amz-Signature=" in data:
                omitted["expired signed URL in text"] += 1
                continue
        target = destination / "agent" / "rtx6000_benchmark" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        if rel.as_posix() == "h3-pika-flex-comprehensive-rtx6000/reproducibility.zip":
            clean = target.with_suffix(".clean.zip")
            with zipfile.ZipFile(target) as original, zipfile.ZipFile(clean, "w") as sanitized:
                for member in original.infolist():
                    data = original.read(member.filename)
                    if b"X-Amz-Signature=" in data:
                        continue
                    sanitized.writestr(member, data)
            clean.replace(target)
        copied[reason] += 1
    core = source / "ml_modules" / "minimax_h3_turbo"
    for item in core.rglob("*"):
        if item.is_file() and not item.is_symlink() and item.suffix in {".py", ".md"}:
            target = destination / "ml_modules" / "minimax_h3_turbo" / item.relative_to(core)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied["pinned inference source"] += 1
    reference_note = destination / "agent/rtx6000_benchmark/h3-pika-flex-comprehensive-rtx6000/inputs/source_report.html.txt"
    reference_note.parent.mkdir(parents=True, exist_ok=True)
    reference_note.write_text("The original reference report HTML contained an expired signed S3 URL and is omitted from the public archive. The benchmark input manifest and experiment reports preserve its provenance and measured comparison data.\n")
    print("Copied:", dict(copied))
    print("Omitted:", dict(omitted))


if __name__ == "__main__":
    main()
