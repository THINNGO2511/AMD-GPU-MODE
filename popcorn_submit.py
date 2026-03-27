#!/usr/bin/env python3
"""
popcorn_submit.py — Non-interactive submission wrapper for popcorn-cli v1.3.6+
Uses --no-tui flag for clean stdout output.

Usage:
    python3 popcorn_submit.py <leaderboard> <mode> <filepath> [--timeout 600]
"""

import subprocess
import sys
import os
import json
import time
import re
import argparse
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "auto_research_logs"
LOG_DIR.mkdir(exist_ok=True)

POPCORN_CLI = os.path.expanduser("~/.local/bin/popcorn-cli")
if not os.path.exists(POPCORN_CLI):
    POPCORN_CLI = "popcorn-cli"


def submit(leaderboard: str, mode: str, filepath: str, timeout: int = 600) -> dict:
    """Submit to popcorn-cli and return parsed results."""
    cmd = [
        POPCORN_CLI, "submit",
        "--gpu", "MI355X",
        "--leaderboard", leaderboard,
        "--mode", mode,
        filepath,
        "--no-tui",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SCRIPT_DIR),
        )
        duration = round(time.time() - start, 1)
        output = result.stdout + "\n" + result.stderr

        # Parse results
        parsed = {
            "timestamp": datetime.now().isoformat(),
            "leaderboard": leaderboard,
            "mode": mode,
            "filepath": filepath,
            "duration_s": duration,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
            "raw_output": output,
        }

        # Extract benchmark results if present
        parsed.update(_parse_output(output))

        return parsed

    except subprocess.TimeoutExpired:
        return {
            "timestamp": datetime.now().isoformat(),
            "leaderboard": leaderboard,
            "mode": mode,
            "filepath": filepath,
            "duration_s": timeout,
            "exit_code": -1,
            "success": False,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "leaderboard": leaderboard,
            "mode": mode,
            "filepath": filepath,
            "duration_s": round(time.time() - start, 1),
            "exit_code": -1,
            "success": False,
            "error": str(e),
        }


def _parse_output(output: str) -> dict:
    """Parse benchmark/test results from popcorn output."""
    info = {}

    # Check for test results
    test_pass = re.findall(r'Passed (\d+)/(\d+) tests', output)
    if test_pass:
        info["tests_passed"] = int(test_pass[0][0])
        info["tests_total"] = int(test_pass[0][1])

    # Check for benchmark results
    benchmarks = re.findall(r'benchmark\.(\d+)\.mean:\s*([\d.]+)', output)
    if benchmarks:
        means = [float(b[1]) for b in benchmarks]
        info["benchmark_means_ns"] = means
        # Compute geomean in microseconds
        from math import exp, log
        means_us = [m / 1000 for m in means]  # ns to μs
        if means_us:
            info["geomean_us"] = round(exp(sum(log(m) for m in means_us) / len(means_us)), 3)
            info["per_shape_us"] = [round(m, 3) for m in means_us]

    # Check for errors
    if "error" in output.lower() and "success" not in output.lower():
        error_lines = [l for l in output.split('\n') if 'error' in l.lower() or 'Error' in l]
        if error_lines:
            info["error"] = error_lines[0][:200]

    # Check for rate limit
    if "rate limit" in output.lower() or "429" in output:
        info["rate_limited"] = True

    return info


def log_result(result: dict):
    """Append result to JSONL log."""
    log_file = LOG_DIR / "submissions.jsonl"
    # Don't log raw output to save space
    log_entry = {k: v for k, v in result.items() if k != "raw_output"}
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Submit to popcorn-cli")
    parser.add_argument("leaderboard", help="Leaderboard name")
    parser.add_argument("mode", help="test, benchmark, or leaderboard")
    parser.add_argument("filepath", help="Path to submission file")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Submitting {args.filepath} → {args.leaderboard} ({args.mode})")

    result = submit(args.leaderboard, args.mode, args.filepath, args.timeout)
    log_result(result)

    if args.json:
        # Don't print raw_output in JSON mode
        out = {k: v for k, v in result.items() if k != "raw_output"}
        print(json.dumps(out, indent=2))
    else:
        print(f"\n--- Result ({result['duration_s']}s, exit={result['exit_code']}) ---")
        if "geomean_us" in result:
            print(f"Geomean: {result['geomean_us']}μs")
            print(f"Per shape: {result.get('per_shape_us', [])}")
        if "tests_passed" in result:
            print(f"Tests: {result['tests_passed']}/{result['tests_total']}")
        if "error" in result:
            print(f"Error: {result['error']}")
        if not result["success"]:
            # Print last part of raw output for debugging
            raw = result.get("raw_output", "")
            if raw:
                print(f"\n--- Last 1000 chars ---\n{raw[-1000:]}")


if __name__ == "__main__":
    main()
