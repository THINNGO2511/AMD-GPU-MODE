#!/usr/bin/env python3
"""GEMM Autonomous Research Runner for AMD GPU MODE Hackathon (MI355X / gfx950).

Iterates through experiment submissions, submits via popcorn, tracks results,
handles rate limits, and reports findings.

Usage:
    python gemm_autoresearch.py                    # Run all pending experiments
    python gemm_autoresearch.py --status           # Show results summary
    python gemm_autoresearch.py --dry-run          # Show what would be submitted
    python gemm_autoresearch.py --only 01_probe    # Run specific experiment(s)
    python gemm_autoresearch.py --interval 300     # Custom wait between submissions (sec)
    python gemm_autoresearch.py --skip-wait        # No wait between submissions (careful!)
    python gemm_autoresearch.py --rerun 03_stages3 # Force re-run even if cached
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

PROJECT_DIR = os.path.expanduser("~/Downloads/code/AMD-GPU-MODE/mxfp4-mm")
REPO_ROOT = os.path.expanduser("~/Downloads/code/AMD-GPU-MODE")
POPCORN = os.path.expanduser("~/.local/bin/popcorn-cli")
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "gemm_experiments")
RESULTS_FILE = os.path.join(PROJECT_DIR, "gemm_autoresearch_results.json")
LOG_DIR = os.path.join(PROJECT_DIR, "gemm_experiments", "logs")
DEFAULT_INTERVAL = 390  # seconds between submissions (~10/hour with margin)

# Current best reference
CURRENT_BEST = {
    "file": "submission_prewarm.py",
    "benchmark_geomean": 9.9,
    "ranked_geomean": 16.5,
}


def load_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def geomean(values):
    if not values:
        return float("inf")
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def parse_output(output):
    """Parse popcorn benchmark output for timing data and probe info."""
    result = {"raw_output_len": len(output)}

    # Check for rate limiting
    if "Rate limit" in output or "rate limit" in output.lower():
        wait_match = re.search(r"[Tt]ry again in (\d+)", output)
        wait_secs = int(wait_match.group(1)) + 15 if wait_match else 400
        result["status"] = "rate_limited"
        result["wait_seconds"] = wait_secs
        return result

    # Check for errors
    error_keywords = [
        "TypeError", "RuntimeError", "SyntaxError", "AttributeError",
        "ValueError", "NameError", "KeyError", "IndexError",
        "Benchmarking failed", "AssertionError", "ModuleNotFoundError",
        "CompilationError", "ImportError",
    ]
    if any(e in output for e in error_keywords):
        error_lines = [
            l.strip() for l in output.split("\n")
            if any(e in l for e in ["Error", "error", "Traceback", "FAIL"])
        ]
        result["status"] = "error"
        result["errors"] = error_lines[:5]
        return result

    # Extract probe data (lines containing PROBE:)
    probes = [l.strip() for l in output.split("\n") if "PROBE:" in l]
    if probes:
        result["probes"] = probes

    # Parse benchmark timings
    times = {}
    lines = output.split("\n")
    for i, line in enumerate(lines):
        km = re.match(r".*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)", line)
        if km and i + 1 < len(lines):
            tm = re.search(r"([\d.]+)\s*[^\d\s]", lines[i + 1])
            if tm:
                t_us = float(tm.group(1))
                if t_us > 0.1:
                    shape_key = "%sx%sx%s" % (km.group(2), km.group(3), km.group(1))
                    times[shape_key] = t_us

    if times:
        result["status"] = "ok"
        result["times"] = times
        result["geomean"] = geomean(list(times.values()))
        result["num_shapes"] = len(times)
    elif probes:
        result["status"] = "probe_only"
    elif "success" in output.lower() or "passed" in output.lower():
        result["status"] = "success_no_timings"
    else:
        result["status"] = "no_output"

    return result


def submit(filepath, mode="benchmark", max_retries=3):
    """Submit a file via popcorn and return parsed results."""
    rel_path = os.path.relpath(filepath, REPO_ROOT)

    cmd = [
        POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
        "--mode", mode, rel_path, "--no-tui",
    ]

    for attempt in range(max_retries):
        try:
            print("  Submitting (attempt %d/%d)..." % (attempt + 1, max_retries),
                  end=" ", flush=True)
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=700, cwd=REPO_ROOT
            )
            output = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            if attempt < max_retries - 1:
                continue
            return {"status": "timeout"}

        # Save raw output
        os.makedirs(LOG_DIR, exist_ok=True)
        log_name = os.path.splitext(os.path.basename(filepath))[0]
        log_path = os.path.join(LOG_DIR, "%s_attempt%d.txt" % (log_name, attempt + 1))
        with open(log_path, "w") as f:
            f.write(output)

        result = parse_output(output)

        if result["status"] == "rate_limited":
            wait = result.get("wait_seconds", 400)
            print("RATE LIMITED, waiting %ds..." % wait)
            time.sleep(wait)
            continue

        return result

    return {"status": "max_retries_exhausted"}


def show_status(results):
    """Print summary of all experiment results."""
    print("\n" + "=" * 70)
    print("GEMM AUTORESEARCH STATUS — %d experiments tracked" % len(results))
    print("Current best: %s (benchmark %.1fμs, ranked %.1fμs)" % (
        CURRENT_BEST["file"], CURRENT_BEST["benchmark_geomean"],
        CURRENT_BEST["ranked_geomean"]))
    print("=" * 70)

    if not results:
        print("  No results yet. Run: python gemm_autoresearch.py")
        return

    # Categorize
    ok_results = []
    probe_results = []
    errors = []
    other = []

    for name, r in sorted(results.items()):
        status = r.get("status", "unknown")
        if status == "ok":
            ok_results.append((name, r))
        elif status in ("probe_only", "probe"):
            probe_results.append((name, r))
        elif status == "error":
            errors.append((name, r))
        else:
            other.append((name, r))

    # Benchmark results sorted by geomean
    if ok_results:
        print("\n--- Benchmark Results (sorted by geomean) ---")
        ok_results.sort(key=lambda x: x[1].get("geomean", float("inf")))
        for name, r in ok_results:
            gm = r.get("geomean", 0)
            delta = gm - CURRENT_BEST["benchmark_geomean"]
            marker = " ** IMPROVEMENT **" if delta < -0.1 else ""
            print("  %6.2fμs (%+.2f) %s%s" % (gm, delta, name, marker))
            if "times" in r:
                for shape, t in sorted(r["times"].items()):
                    print("           %s: %.2fμs" % (shape, t))

    # Probe results
    if probe_results:
        print("\n--- Probe Results ---")
        for name, r in probe_results:
            print("  %s:" % name)
            for p in r.get("probes", [])[:10]:
                print("    %s" % p)

    # Errors
    if errors:
        print("\n--- Errors ---")
        for name, r in errors:
            errs = r.get("errors", ["unknown"])
            print("  %s: %s" % (name, errs[0][:100] if errs else "unknown"))

    # Other
    if other:
        print("\n--- Other ---")
        for name, r in other:
            print("  %s: %s" % (name, r.get("status", "unknown")))

    # Best overall
    if ok_results:
        best_name, best_r = ok_results[0]
        best_gm = best_r["geomean"]
        print("\n" + "-" * 70)
        if best_gm < CURRENT_BEST["benchmark_geomean"]:
            print("NEW BEST: %s at %.2fμs (was %.1fμs, %.1f%% improvement)" % (
                best_name, best_gm, CURRENT_BEST["benchmark_geomean"],
                (1 - best_gm / CURRENT_BEST["benchmark_geomean"]) * 100))
        else:
            print("No improvement over current best (%.1fμs)" %
                  CURRENT_BEST["benchmark_geomean"])


def run_experiments(args):
    """Main experiment loop."""
    results = load_results()
    os.makedirs(LOG_DIR, exist_ok=True)

    # Find experiment files
    if not os.path.isdir(EXPERIMENTS_DIR):
        print("ERROR: No experiments directory: %s" % EXPERIMENTS_DIR)
        sys.exit(1)

    files = sorted([f for f in os.listdir(EXPERIMENTS_DIR)
                    if f.endswith(".py") and not f.startswith("_")])

    if not files:
        print("ERROR: No experiment files in %s" % EXPERIMENTS_DIR)
        sys.exit(1)

    # Filter by --only
    if args.only:
        patterns = args.only
        files = [f for f in files if any(p in f for p in patterns)]
        if not files:
            print("ERROR: No experiments matching: %s" % patterns)
            sys.exit(1)

    # Force re-run
    if args.rerun:
        for pattern in args.rerun:
            to_remove = [k for k in results if pattern in k]
            for k in to_remove:
                del results[k]
                print("Cleared cached result: %s" % k)
        save_results(results)

    interval = 0 if args.skip_wait else args.interval
    submitted = 0

    print("\n" + "=" * 70)
    print("GEMM AUTORESEARCH — %d experiments queued" % len(files))
    print("Interval: %ds between submissions" % interval)
    print("=" * 70)

    for i, fname in enumerate(files):
        filepath = os.path.join(EXPERIMENTS_DIR, fname)

        # Skip completed experiments
        if fname in results and results[fname].get("status") in ("ok", "probe_only"):
            gm = results[fname].get("geomean", "N/A")
            print("\n[%d/%d] %s: CACHED (geomean=%s)" % (i + 1, len(files), fname, gm))
            continue

        print("\n" + "=" * 70)
        print("[%d/%d] %s" % (i + 1, len(files), fname))
        print("=" * 70)

        if args.dry_run:
            print("  [DRY RUN] Would submit: %s" % fname)
            with open(filepath) as f:
                lines = f.readlines()
            print("  Lines: %d" % len(lines))
            # Show first comment/docstring
            for line in lines[:10]:
                if line.strip().startswith("#") or line.strip().startswith('"""'):
                    print("  %s" % line.rstrip())
            continue

        # Wait between submissions (rate limit)
        if submitted > 0 and interval > 0:
            print("  Waiting %ds for rate limit..." % interval)
            for remaining in range(interval, 0, -30):
                print("    %ds remaining..." % remaining, flush=True)
                time.sleep(min(30, remaining))

        result = submit(filepath)
        result["timestamp"] = datetime.now().isoformat()
        result["experiment"] = fname
        results[fname] = result
        save_results(results)
        submitted += 1

        # Print result
        status = result.get("status", "unknown")
        if status == "ok":
            gm = result["geomean"]
            delta = gm - CURRENT_BEST["benchmark_geomean"]
            print("  RESULT: geomean=%.2fμs (%+.2f vs current best)" % (gm, delta))
            for shape, t in sorted(result.get("times", {}).items()):
                print("    %s: %.2fμs" % (shape, t))
            if delta < -0.1:
                print("  ** IMPROVEMENT FOUND! **")
        elif status == "probe_only":
            print("  PROBE DATA:")
            for p in result.get("probes", [])[:15]:
                print("    %s" % p)
        elif status == "error":
            print("  ERROR:")
            for e in result.get("errors", [])[:3]:
                print("    %s" % e[:120])
        else:
            print("  STATUS: %s" % status)

    # Final summary
    show_status(results)


def main():
    p = argparse.ArgumentParser(description="GEMM Autonomous Research Runner")
    p.add_argument("--status", action="store_true", help="Show results summary")
    p.add_argument("--dry-run", action="store_true", help="Don't submit, just show")
    p.add_argument("--only", nargs="+", help="Only run experiments matching these patterns")
    p.add_argument("--rerun", nargs="+", help="Force re-run experiments matching patterns")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                   help="Seconds between submissions (default %d)" % DEFAULT_INTERVAL)
    p.add_argument("--skip-wait", action="store_true",
                   help="No waiting between submissions (risk rate limit)")
    args = p.parse_args()

    if args.status:
        show_status(load_results())
    else:
        run_experiments(args)


if __name__ == "__main__":
    main()
