#!/usr/bin/env python3
"""Differential gate for the cuda-samples interception smoke test.

Compares a BASELINE run (native — no cedana GPU interception) against a
CANDIDATE run (under cedana's GPU interceptor) and fails ONLY on regressions:
samples that pass natively but do NOT pass under interception.

Why differential: a chunk of cuda-samples fail or waive on any given host for
reasons that have nothing to do with cedana (no display, compute capability,
GPU count, missing libs, etc.). Gating on the candidate's absolute pass/fail
would make the test red regardless of cedana. By measuring native vs intercepted
on the *same* runner, environmental failures appear in both and cancel out, so
the only thing that can fail this gate is a real interception regression.

Classification (key = "<exe_name> <run-description>"):
  - baseline Passed, candidate not Passed  -> REGRESSION (gates)
  - baseline Passed, candidate missing      -> REGRESSION (didn't run under cedana)
  - baseline not Passed                      -> environmental, ignored (logged)
  - baseline not Passed, candidate Passed    -> improvement (logged)

Exit code: 1 if any regressions, else 0.

Usage:
    compare-results.py --baseline native.json --candidate intercepted.json
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        data = json.load(f)
    return {r["key"]: r for r in data.get("results", [])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True,
                   help="native results.json (no interception)")
    p.add_argument("--candidate", required=True,
                   help="intercepted results.json (under cedana)")
    a = p.parse_args()

    base = load(a.baseline)
    cand = load(a.candidate)

    PASS = "Passed"

    regressions = []     # (key, candidate_status, candidate_return_code)
    missing = []         # key
    base_failures = []   # (key, baseline_status)
    improvements = []    # key

    for key, b in base.items():
        c = cand.get(key)
        if b["status"] != PASS:
            base_failures.append((key, b["status"]))
            if c is not None and c["status"] == PASS:
                improvements.append(key)
            continue
        # Baseline passed -> candidate must pass too.
        if c is None:
            missing.append(key)
        elif c["status"] != PASS:
            regressions.append((key, c["status"], c.get("return_code")))

    print(f"baseline (native):      {len(base)} runs")
    print(f"candidate (intercepted): {len(cand)} runs")

    if base_failures:
        print(f"\nIgnored — failed/waived natively too, not cedana's fault ({len(base_failures)}):")
        for k, s in sorted(base_failures):
            print(f"  {k}: {s}")
    if improvements:
        print(f"\nPassed under cedana but not natively ({len(improvements)}):")
        for k in sorted(improvements):
            print(f"  {k}")
    if missing:
        print(f"\nREGRESSION — passed natively, did not run under cedana ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}")
    if regressions:
        print(f"\nREGRESSION — passed natively, broke under cedana ({len(regressions)}):")
        for k, s, rc in sorted(regressions):
            print(f"  {k}: {s} (code {rc})")

    total = len(regressions) + len(missing)
    if total:
        print(f"\nFAIL: {total} cedana-induced regression(s).")
        return 1
    print("\nPASS: no cedana-induced regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
