"""
pr450 self-test: one command that checks every pipeline stage and says
whether all is well.

    python -m pr450.selftest            # full run (~3 min, includes end-to-end)
    python -m pr450.selftest --quick    # skip the slow end-to-end check (~30 s)
    python -m pr450.selftest --stage eca --stage caf

What is checked, per stage (see tests/):
  * conformance against independent 3GPP implementations (py3gpp, srsRAN 4G)
  * analytic checks against physics / closed-form results / invariants
    (tests/analytic_checks.py): link budget, kT, ADC SQNR, processing gain,
    CFAR Pfa and Pd formulas, LS orthogonality, truth recovery, ...
  * the unit tests of each module
  * mutation tests: realistic bugs are injected on purpose and must be caught,
    which shows the checks themselves are sensitive

Writes results/selftest/report.md and report.json and exits non-zero if
anything failed. Passing checks cannot prove the absence of every logical
error, but each stage is verified against references that do not depend on
the code under test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
ORDER = ["conformance", "waveform", "receiver", "reference", "eca", "caf", "cfar", "clean", "end_to_end",
         "mutation", "other"]


class _Collector:
    def __init__(self, stages: list[str] | None = None):
        self.only = set(stages) if stages else None
        self.stage_of: dict[str, str] = {}
        self.results: dict[str, dict] = {}

    def pytest_collection_modifyitems(self, session, config, items):
        from conftest import stage_of  # noqa: PLC0415
        keep, drop = [], []
        for item in items:
            stage = stage_of(item)
            if self.only is None or stage in self.only:
                self.stage_of[item.nodeid] = stage
                keep.append(item)
            else:
                drop.append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep

    def pytest_runtest_logreport(self, report):
        r = self.results.setdefault(report.nodeid, {"outcome": "passed", "duration": 0.0, "message": "",
                                                    "values": []})
        r["duration"] += report.duration
        if report.when == "call":
            r["values"] = [f"{k}: {v}" for k, v in report.user_properties]
        if report.failed:
            r["outcome"] = "failed"
            text = str(report.longrepr)
            lines = [ln for ln in text.splitlines() if ln.startswith("E ")] or text.splitlines()[-3:]
            r["message"] = " | ".join(ln[1:].strip() if ln.startswith("E ") else ln.strip() for ln in lines[:4])
        elif report.skipped and r["outcome"] != "failed":
            r["outcome"] = "skipped"
            r["message"] = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)


def _fingerprint() -> str:
    h = hashlib.sha256()
    for path in sorted(list((ROOT / "pr450").rglob("*.py")) + list(TESTS.glob("*.py")) +
                       list((ROOT / "configs").glob("*.yaml"))):
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _environment() -> dict:
    import numpy
    import scipy
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
           "platform": platform.platform(), "code_fingerprint": _fingerprint()}
    try:
        import py3gpp  # noqa: F401
        from importlib.metadata import version
        env["py3gpp"] = version("py3gpp")
    except Exception:
        env["py3gpp"] = "not installed (NR conformance checks skipped)"
    src = TESTS / "data" / "srsran" / "SOURCE.txt"
    env["srsran_vectors"] = src.read_text().strip() if src.exists() else "missing (LTE conformance checks skipped)"
    return env


def main(argv=None) -> int:
    import pytest
    sys.path.insert(0, str(TESTS))
    from conftest import STAGES  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="pr450 self-test", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip slow checks (end-to-end)")
    ap.add_argument("--stage", action="append", choices=list(STAGES), help="only these stages (repeatable)")
    ap.add_argument("--report", default=str(ROOT / "results" / "selftest"))
    ap.add_argument("-v", "--verbose", action="store_true", help="list every check with its measured values")
    args = ap.parse_args(argv)

    collector = _Collector(args.stage)
    pytest_args = [str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider", "-o", "console_output_style=classic",
                   "--tb=short", "-rN"]
    if not args.quick:
        pytest_args.append("--runslow")
    t0 = time.time()
    print(f"pr450 self-test {'(quick) ' if args.quick else ''}- running checks ...", flush=True)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pytest.main(pytest_args, plugins=[collector])
    elapsed = time.time() - t0

    by_stage: dict[str, list] = {}
    for nodeid, r in collector.results.items():
        by_stage.setdefault(collector.stage_of.get(nodeid, "other"), []).append((nodeid, r))

    env = _environment()
    lines = [f"# pr450 self-test report", "",
             f"{dt.datetime.now().isoformat(timespec='seconds')} - {elapsed:.0f} s - code fingerprint "
             f"`{env['code_fingerprint']}`", "",
             "| Stage | Checks | Passed | Failed | Skipped | Verdict |", "|---|---|---|---|---|---|"]
    console = ["", f"{'Stage':<58s} {'checks':>6s} {'pass':>5s} {'fail':>5s} {'skip':>5s}  verdict", "-" * 92]
    total_fail = total_skip = 0
    for stage in ORDER:
        items = by_stage.get(stage, [])
        if not items:
            continue
        n_pass = sum(r["outcome"] == "passed" for _, r in items)
        n_fail = sum(r["outcome"] == "failed" for _, r in items)
        n_skip = sum(r["outcome"] == "skipped" for _, r in items)
        total_fail += n_fail
        total_skip += n_skip
        verdict = "FAIL" if n_fail else ("OK (some skipped)" if n_skip else "OK")
        title = STAGES.get(stage, stage)
        console.append(f"{title[:58]:<58s} {len(items):>6d} {n_pass:>5d} {n_fail:>5d} {n_skip:>5d}  {verdict}")
        lines.append(f"| {title} | {len(items)} | {n_pass} | {n_fail} | {n_skip} | {verdict} |")

    if total_fail:
        overall = f"PROBLEMS FOUND: {total_fail} check(s) failed"
    elif total_skip:
        overall = f"ALL RUN CHECKS PASSED ({total_skip} skipped - see below)"
    else:
        overall = "ALL CHECKS PASSED"
    console += ["-" * 92, f"OVERALL: {overall}", ""]
    lines += ["", f"**Overall: {overall}**", ""]

    def detail(title, outcome):
        rows = [(s, n, r) for s in ORDER for n, r in by_stage.get(s, []) if r["outcome"] == outcome]
        if not rows:
            return
        console.append(title)
        lines.extend([f"## {title}", ""])
        for stage, nodeid, r in rows:
            name = nodeid.split("::", 1)[-1]
            console.append(f"  [{stage}] {name}\n      {r['message']}")
            lines.append(f"- **[{stage}] {name}**: {r['message']}")
            for v in r["values"]:
                console.append(f"      {v}")
                lines.append(f"  - {v}")
        console.append("")
        lines.append("")

    detail("Failed checks", "failed")
    detail("Skipped checks", "skipped")
    if args.verbose:
        detail("Passed checks", "passed")
    lines += ["## Measured values of the analytic checks", ""]
    for stage in ORDER:
        for nodeid, r in by_stage.get(stage, []):
            if r["values"]:
                lines.append(f"- [{stage}] {nodeid.split('::', 1)[-1]}")
                lines.extend(f"  - {v}" for v in r["values"])
    lines += ["", "## Environment", ""] + [f"- {k}: {v}" for k, v in env.items()]

    print("\n".join(console))
    out = Path(args.report)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps({
        "overall": overall, "elapsed_s": elapsed, "environment": env,
        "checks": [{"stage": collector.stage_of.get(n, "other"), "id": n, **r} for n, r in collector.results.items()],
    }, indent=1, default=str), encoding="utf-8")
    print(f"Report: {out / 'report.md'}")
    if total_fail:
        print("\n--- pytest output (tail) ---\n" + "\n".join(buf.getvalue().splitlines()[-40:]))
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
