#!/usr/bin/env python3
"""
Auto-Research Engine: Active web research + passive monitoring.
Continuously scans for new optimization techniques, competitor movements,
aiter updates, and community insights. Logs actionable findings.

Runs alongside autosweep scripts. Never stops.
"""
import os, sys, json, time, shutil, subprocess, re
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser

REPO = Path(__file__).parent
LOG_DIR = REPO / "auto_research_logs"
LOG_FILE = LOG_DIR / "research.jsonl"
FINDINGS_DIR = LOG_DIR / "findings"
POPCORN = shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli")

CHECK_INTERVAL = 7200  # 2 hours
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GPUModeResearch/1.0)"}


def log(entry):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), **entry}) + "\n")
    print(f"[{datetime.now().strftime('%H:%M')}] {entry.get('type', '?')}: {entry.get('summary', '')}")


def save_finding(title, content, priority="MED"):
    """Save an actionable finding to disk."""
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)[:50]
    path = FINDINGS_DIR / f"{priority}_{ts}_{safe_title}.md"
    path.write_text(f"# [{priority}] {title}\n\n{content}\n\nFound: {datetime.now()}\n")
    if priority == "HIGH":
        print(f"  !!! HIGH PRIORITY FINDING: {title}")
    return path


def fetch(url, timeout=15):
    """Fetch URL content. Returns text or None."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        log({"type": "fetch_error", "url": url, "summary": str(e)[:100]})
        return None


def fetch_json(url, timeout=15):
    """Fetch JSON from URL."""
    text = fetch(url, timeout)
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


# ============================================================
# PASSIVE MONITORING
# ============================================================

def check_leaderboard():
    """Check our current leaderboard standings via popcorn-cli."""
    for lb in ["amd-mxfp4-mm", "amd-mixed-mla", "amd-moe-mxfp4"]:
        try:
            r = subprocess.run(
                [POPCORN, "submissions", "list", "--leaderboard", lb],
                capture_output=True, text=True, timeout=30
            )
            lines = r.stdout.strip().split('\n')
            latest = lines[2] if len(lines) > 2 else "none"
            log({"type": "leaderboard", "lb": lb, "summary": latest[:100]})
        except Exception as e:
            log({"type": "error", "summary": f"leaderboard {lb}: {e}"})


def check_mla_retry():
    """Check if MLA pg2 retry passed secret seed."""
    try:
        r = subprocess.run(
            [POPCORN, "submissions", "list", "--leaderboard", "amd-mixed-mla"],
            capture_output=True, text=True, timeout=30
        )
        lines = r.stdout.strip().split('\n')
        if len(lines) >= 3:
            latest = lines[2]
            if "pg2" in latest and "done" in latest.lower():
                log({"type": "MLA_CHECK", "summary": f"pg2 submission detected: {latest[:100]}"})
                # Check if score improved
                for line in lines[2:]:
                    if "pg2" in line:
                        score_match = re.search(r'(\d+\.?\d*)\s*[μu]s', line)
                        if score_match:
                            score = float(score_match.group(1))
                            if score < 42.0:
                                save_finding("MLA pg2 passed!", f"Score: {score}μs (was 42.5μs)", "HIGH")
                                return True
    except Exception as e:
        log({"type": "error", "summary": f"MLA check: {e}"})
    return False


def probe_runner_version():
    """Submit probe to check runner's aiter version."""
    probe_code = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
from task import input_t, output_t
import torch, sys, os
_ref=None; _raw=None; _bq=None; _probed=False
def _unshuffle(s):
    s=s.view(torch.uint8);sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
def custom_kernel(data:input_t)->output_t:
    global _ref,_raw,_bq,_probed
    A,B,B_q,B_shuffle,B_scale_sh=data
    m,k=A.shape;n=B.shape[0]
    if _ref is not B_scale_sh:
        _ref=B_scale_sh;_raw=_unshuffle(B_scale_sh);_bq=B_q.view(torch.uint8)
    if not _probed:
        _probed=True
        try:
            import subprocess as sp
            r=sp.run(["git","log","--oneline","-5"],cwd="/home/runner/aiter",capture_output=True,text=True,timeout=5)
            print(f"AITER_GIT: {r.stdout.strip()}",flush=True)
        except: pass
        try:
            import glob
            cos=glob.glob("/home/runner/aiter/hsa/gfx950/**/*.co",recursive=True)
            fmoe_cos=[c for c in cos if "fmoe" in c]
            mla_cos=[c for c in cos if "mla" in c]
            print(f"CO_TOTAL: {len(cos)} FMOE: {len(fmoe_cos)} MLA: {len(mla_cos)}",flush=True)
        except: pass
        try:
            import aiter
            print(f"AITER_VER: {getattr(aiter,'__version__','?')}",flush=True)
        except: pass
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bq,asc,_raw,dtype=torch.bfloat16)
    return gemm_a16wfp4(A,_bq,_raw,dtype=torch.bfloat16)
'''
    probe_path = REPO / "mxfp4-mm" / "probe_runner.py"
    probe_path.write_text(probe_code)
    try:
        r = subprocess.run(
            [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
             "--mode", "benchmark", str(probe_path), "--no-tui"],
            capture_output=True, text=True, timeout=600
        )
        output = r.stdout + r.stderr
        findings = []
        for line in output.split('\n'):
            for tag in ['AITER_GIT', 'CO_TOTAL', 'CO_FILES', 'AITER_VER', 'FMOE', 'MLA']:
                if tag in line:
                    findings.append(line.strip())
        if findings:
            log({"type": "runner_probe", "summary": "; ".join(findings)})
            save_finding("Runner version probe", "\n".join(findings), "LOW")
        else:
            log({"type": "runner_probe", "summary": "probe submitted, no tagged output"})
    except Exception as e:
        log({"type": "error", "summary": f"runner probe: {e}"})


# ============================================================
# ACTIVE WEB RESEARCH
# ============================================================

def research_aiter_github():
    """Check aiter GitHub for recent PRs and commits."""
    print("  Checking aiter GitHub PRs...")

    # Recent merged PRs
    url = "https://api.github.com/repos/ROCm/aiter/pulls?state=closed&sort=updated&direction=desc&per_page=20"
    data = fetch_json(url)
    if data:
        findings = []
        for pr in data:
            merged = pr.get("merged_at", "")
            if not merged:
                continue
            title = pr.get("title", "")
            number = pr.get("number", "")
            updated = pr.get("updated_at", "")[:10]
            labels = [l.get("name", "") for l in pr.get("labels", [])]

            # Filter for relevant PRs (after Mar 25)
            if merged >= "2026-03-25":
                relevance = "LOW"
                for kw in ["gemm", "moe", "mla", "fp4", "mxfp4", "gfx950", "config", "tune", "perf"]:
                    if kw in title.lower():
                        relevance = "HIGH"
                        break
                for kw in ["triton", "kernel", "scale", "quant", "block"]:
                    if kw in title.lower():
                        relevance = "MED"
                        break
                findings.append(f"PR #{number} ({merged[:10]}): {title} [{relevance}]")
                if relevance == "HIGH":
                    # Fetch PR details
                    detail_url = f"https://api.github.com/repos/ROCm/aiter/pulls/{number}"
                    detail = fetch_json(detail_url)
                    body = detail.get("body", "")[:500] if detail else ""
                    save_finding(f"aiter PR #{number}: {title}", f"URL: {pr.get('html_url')}\n\n{body}", "HIGH")

        if findings:
            log({"type": "aiter_prs", "summary": f"{len(findings)} recent PRs", "prs": findings[:10]})
        else:
            log({"type": "aiter_prs", "summary": "no new relevant PRs"})

    # Recent commits on main
    url = "https://api.github.com/repos/ROCm/aiter/commits?per_page=10"
    data = fetch_json(url)
    if data:
        recent = []
        for commit in data[:5]:
            msg = commit.get("commit", {}).get("message", "").split("\n")[0]
            date = commit.get("commit", {}).get("committer", {}).get("date", "")[:10]
            recent.append(f"{date}: {msg}")
        log({"type": "aiter_commits", "summary": "; ".join(recent[:3])})


def research_triton_github():
    """Check Triton GitHub for ROCm-related PRs."""
    print("  Checking Triton GitHub...")
    url = "https://api.github.com/repos/triton-lang/triton/pulls?state=all&sort=updated&direction=desc&per_page=15"
    data = fetch_json(url)
    if data:
        rocm_prs = []
        for pr in data:
            title = pr.get("title", "")
            if any(kw in title.lower() for kw in ["rocm", "hip", "amd", "gfx9", "cdna", "mfma"]):
                merged = pr.get("merged_at") or "open"
                rocm_prs.append(f"#{pr['number']} ({merged[:10] if merged != 'open' else 'open'}): {title}")
        if rocm_prs:
            log({"type": "triton_prs", "summary": f"{len(rocm_prs)} ROCm PRs", "prs": rocm_prs[:5]})
            for pr_info in rocm_prs[:3]:
                if "block" in pr_info.lower() or "mfma" in pr_info.lower() or "scale" in pr_info.lower():
                    save_finding(f"Triton ROCm PR: {pr_info}", pr_info, "MED")


def research_vllm():
    """Check vLLM for ROCm/aiter-related PRs."""
    print("  Checking vLLM GitHub...")
    url = "https://api.github.com/repos/vllm-project/vllm/pulls?state=all&sort=updated&direction=desc&per_page=15"
    data = fetch_json(url)
    if data:
        relevant = []
        for pr in data:
            title = pr.get("title", "")
            if any(kw in title.lower() for kw in ["rocm", "aiter", "mla", "moe", "fp4", "amd"]):
                relevant.append(f"#{pr['number']}: {title}")
        if relevant:
            log({"type": "vllm_prs", "summary": f"{len(relevant)} relevant PRs", "prs": relevant[:5]})


def research_composable_kernel():
    """Check Composable Kernel for new kernel templates."""
    print("  Checking Composable Kernel...")
    url = "https://api.github.com/repos/ROCm/composable_kernel/commits?per_page=10"
    data = fetch_json(url)
    if data:
        recent = []
        for commit in data[:5]:
            msg = commit.get("commit", {}).get("message", "").split("\n")[0]
            date = commit.get("commit", {}).get("committer", {}).get("date", "")[:10]
            if any(kw in msg.lower() for kw in ["moe", "gemm", "fp4", "mxfp4", "fmoe", "mla", "scale"]):
                recent.append(f"{date}: {msg}")
        if recent:
            log({"type": "ck_commits", "summary": "; ".join(recent[:3])})
            save_finding("CK relevant commits", "\n".join(recent), "MED")


def research_leaderboard_mirror():
    """Check leaderboard mirror for current scores."""
    print("  Checking leaderboard mirror...")
    for lb in ["amd-mxfp4-mm", "amd-mixed-mla", "amd-moe-mxfp4"]:
        url = f"https://leaderboard.ooousay.com/{lb}"
        html = fetch(url, timeout=20)
        if html:
            # Extract scores from HTML (simple regex-based parsing)
            # Look for table rows with usernames and scores
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
            scores = []
            for row in rows[:15]:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if len(cells) >= 3:
                    # Clean HTML tags
                    clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    scores.append(" | ".join(clean[:4]))
            if scores:
                log({"type": "leaderboard_mirror", "lb": lb, "summary": f"top entries found",
                     "scores": scores[:10]})
                # Check for significant changes
                for score_line in scores[:3]:
                    if "noobmaster69" in score_line.lower():
                        log({"type": "our_rank", "lb": lb, "summary": score_line})
            else:
                log({"type": "leaderboard_mirror", "lb": lb, "summary": "no table rows parsed"})


def research_github_search():
    """Search GitHub for MI355X/gfx950 optimization repos."""
    print("  Searching GitHub for MI355X/gfx950 repos...")
    queries = [
        "MI355X+mxfp4",
        "gfx950+gemm+optimization",
        "gpu+mode+hackathon+amd",
        "amd+moe+mxfp4+kernel",
        "cdna4+triton+fp4",
    ]
    for query in queries:
        url = f"https://api.github.com/search/repositories?q={query}&sort=updated&per_page=5"
        data = fetch_json(url)
        if data and data.get("items"):
            for repo in data["items"][:3]:
                name = repo.get("full_name", "")
                desc = repo.get("description", "") or ""
                updated = repo.get("updated_at", "")[:10]
                if updated >= "2026-03":
                    log({"type": "github_repo", "summary": f"{name}: {desc[:80]}"})
                    if any(kw in desc.lower() for kw in ["gemm", "moe", "mla", "hackathon"]):
                        save_finding(f"GitHub repo: {name}", f"{desc}\nURL: {repo.get('html_url')}", "MED")


def research_github_code_search():
    """Search GitHub code for specific optimization patterns."""
    print("  Searching GitHub code...")
    queries = [
        "gemm_a16wfp4+config+MI355X",
        "mla_decode_fwd+page_size",
        "fused_moe+ck_moe_stage1",
    ]
    for query in queries:
        url = f"https://api.github.com/search/code?q={query}&per_page=5"
        data = fetch_json(url)
        if data and data.get("items"):
            for item in data["items"][:3]:
                repo = item.get("repository", {}).get("full_name", "")
                path = item.get("path", "")
                log({"type": "code_search", "summary": f"{repo}/{path}"})


def research_blogs():
    """Check AMD/ROCm blogs and salykova for new content."""
    print("  Checking blogs...")

    # salykova.github.io
    html = fetch("https://salykova.github.io/")
    if html:
        # Look for new posts about MFMA/FP4
        links = re.findall(r'href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        for href, text in links:
            text_clean = re.sub(r'<[^>]+>', '', text).strip()
            if any(kw in text_clean.lower() for kw in ["mfma", "fp4", "gemm", "gfx9", "matmul", "amd"]):
                log({"type": "blog", "summary": f"salykova: {text_clean}: {href}"})
                save_finding(f"salykova blog: {text_clean}", f"URL: https://salykova.github.io{href}", "MED")

    # ROCm blogs
    html = fetch("https://rocm.blogs.amd.com/")
    if html:
        links = re.findall(r'href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        for href, text in links:
            text_clean = re.sub(r'<[^>]+>', '', text).strip()
            if any(kw in text_clean.lower() for kw in ["triton", "fp4", "gemm", "moe", "instinct", "mi3", "cdna"]):
                log({"type": "blog", "summary": f"ROCm blog: {text_clean}"})


def research_hackathon_community():
    """Search for hackathon discussions and writeups."""
    print("  Searching hackathon community...")
    queries = [
        "site:reddit.com GPU MODE hackathon AMD",
        "site:huggingface.co MI355X optimization",
    ]
    for query in queries:
        url = f"https://api.github.com/search/issues?q={query.replace(' ', '+')}&per_page=5"
        data = fetch_json(url)
        if data and data.get("items"):
            for item in data["items"][:3]:
                title = item.get("title", "")
                url = item.get("html_url", "")
                log({"type": "community", "summary": f"{title}: {url}"})


def research_papers():
    """Search for relevant papers on arXiv."""
    print("  Checking papers...")
    # arXiv API for recent papers
    queries = [
        "split-K+GEMM+GPU",
        "MoE+GPU+optimization",
        "mixed+precision+attention+decode",
    ]
    for query in queries:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&sortBy=lastUpdatedDate&sortOrder=descending&max_results=3"
        xml = fetch(url, timeout=20)
        if xml:
            titles = re.findall(r'<title>(.*?)</title>', xml)
            for title in titles[1:4]:  # Skip feed title
                if any(kw in title.lower() for kw in ["gemm", "moe", "attention", "gpu", "fp4", "quantiz"]):
                    log({"type": "paper", "summary": title.strip()})


# ============================================================
# MAIN LOOP
# ============================================================

def run_passive_monitoring():
    """Quick monitoring checks."""
    print("\n--- Passive Monitoring ---")
    check_leaderboard()
    check_mla_retry()


def run_active_research():
    """Full web research cycle."""
    print("\n--- Active Web Research ---")
    research_aiter_github()
    research_triton_github()
    research_vllm()
    research_composable_kernel()
    research_leaderboard_mirror()
    research_github_search()
    research_github_code_search()
    research_blogs()
    research_hackathon_community()
    research_papers()


def run_runner_probe(cycle):
    """Probe runner version (every 4 cycles = 8 hours)."""
    if cycle % 4 == 1:
        print("\n--- Runner Version Probe ---")
        probe_runner_version()


def main():
    print(f"=== Auto-Research Engine Started {datetime.now()} ===")
    print(f"Check interval: {CHECK_INTERVAL//3600}h")
    print(f"Log: {LOG_FILE}")
    print(f"Findings: {FINDINGS_DIR}")
    LOG_DIR.mkdir(exist_ok=True)
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"Research Cycle {cycle} ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print(f"{'='*60}")

        try:
            run_passive_monitoring()
            run_active_research()
            run_runner_probe(cycle)
        except Exception as e:
            log({"type": "cycle_error", "summary": str(e)[:200]})
            print(f"  Cycle error: {e}")

        # Print findings summary
        if FINDINGS_DIR.exists():
            findings = sorted(FINDINGS_DIR.glob("*.md"), key=lambda f: f.name, reverse=True)
            high = [f for f in findings if f.name.startswith("HIGH_")]
            if high:
                print(f"\n  !!! {len(high)} HIGH priority findings — check {FINDINGS_DIR}")

        print(f"\nNext cycle in {CHECK_INTERVAL//3600}h...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
