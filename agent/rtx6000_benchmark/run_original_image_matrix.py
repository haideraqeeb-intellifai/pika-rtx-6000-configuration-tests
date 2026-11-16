"""Replay the practical one-GPU matrix with the portrait at both ends of every chunk.

The existing run_case idle gate stops the matrix if the GPU is contested or its
utilization telemetry is unhealthy. This runner never bypasses that gate.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def case(name, chunks, plan="aligned", *extra):
    anchor = ["--anchor-mode", "hard"] if "--anchor-mode" not in extra else []
    return ["--name", f"original_image_{name}", "--reference", "original",
            *anchor, "--chunks", str(chunks), "--chunk-plan", plan, *extra]


CASES = [
    # First establish the visual and timing effect of this new generation policy.
    case("4_report_repeat", 4, "report"),
    case("7_aligned_overlap", 7, "aligned", "--encoding-overlap"),
    case("10_aligned_overlap", 10, "aligned", "--encoding-overlap"),
    case("10_aligned_overlap_repeat2", 10, "aligned", "--encoding-overlap"),
    case("14_aligned", 14),
    # Match the earlier chunk-count sweep; these are independent-chunk semantics.
    *[case(f"{count}_uniform", count, "uniform") for count in (5, 6, 7, 8, 9, 10, 12)],
    case("10_aligned_no_overlap", 10),
    case("4_aligned_overlap", 4, "aligned", "--encoding-overlap"),
    # Keep the portrait as every start image while varying the end-anchor policy.
    case("10_anchor_every2_overlap", 10, "aligned", "--anchor-mode", "every-2", "--encoding-overlap"),
    case("10_anchor_every2_overlap_repeat2", 10, "aligned", "--anchor-mode", "every-2", "--encoding-overlap"),
    case("10_anchor_every3_overlap", 10, "aligned", "--anchor-mode", "every-3", "--encoding-overlap"),
    case("10_anchor_first_only_overlap", 10, "aligned", "--anchor-mode", "first-only", "--encoding-overlap"),
    # Residency, model copies, VAE entry point, and storage on the same ten-chunk plan.
    case("10_two_transformers", 10, "aligned", "--transformers", "2"),
    case("10_two_vaes", 10, "aligned", "--vaes", "2"),
    case("10_four_vaes", 10, "aligned", "--vaes", "4"),
    case("10_highvram", 10, "aligned", "--residency", "highvram"),
    case("10_cpu_text", 10, "aligned", "--clip-device", "cpu"),
    case("10_tiled_vae", 10, "aligned", "--vae-decode", "tiled"),
    case("10_disk_frames", 10, "aligned", "--frame-storage", "disk"),
    case("4_repeat_all_loaders", 4, "report", "--loader-policy", "repeat_all"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute cases; otherwise print the plan")
    args = parser.parse_args()
    for number, options in enumerate(CASES, 1):
        name = options[options.index("--name") + 1]
        result_path = RUNS / name / "result.json"
        if result_path.exists():
            if json.loads(result_path.read_text()).get("status") == "success":
                print(f"{number}/{len(CASES)} {name}: existing valid result", flush=True)
                continue
            raise RuntimeError(f"{name} has a failed run directory; inspect it before retrying")
        command = [sys.executable, "-m", "agent.rtx6000_benchmark.run_case", *options]
        print(f"{number}/{len(CASES)} {' '.join(command)}", flush=True)
        if args.run:
            with (RUNS / f"{name}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
