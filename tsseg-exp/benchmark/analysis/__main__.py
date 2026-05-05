"""Run ``python -m analysis`` to (re)generate every paper figure.

Use ``python -m analysis --refresh`` to bypass the on-disk cache and
re-fetch raw metrics from MLflow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure benchmark/ is on sys.path when invoked as ``python -m analysis``
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (
    data,
    feature_sensitivity,
    performance,
    runtime,
    runtime_split,
    scatter,
    sms_breakdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch raw metrics from MLflow (slow).")
    parser.add_argument("--refresh-built", action="store_true",
                        help="Rebuild df_cpd/df_sd/... from cached raw data.")
    parser.add_argument("--only", nargs="+",
                        choices=["performance", "runtime", "runtime_split",
                                 "scatter", "sms", "feature_sensitivity"],
                        help="Generate only the listed figures.")
    args = parser.parse_args()

    d = data.load_data(refresh=args.refresh, refresh_built=args.refresh_built)
    targets = set(args.only or [
        "performance", "runtime", "runtime_split",
        "scatter", "sms", "feature_sensitivity",
    ])

    if "performance" in targets:
        performance.make_all_figures(d)
    if "runtime" in targets:
        runtime.make_figure(d)
    if "runtime_split" in targets:
        runtime_split.make_figure(d)
    if "scatter" in targets:
        scatter.make_figure(d)
    if "sms" in targets:
        sms_breakdown.make_figure()
    if "feature_sensitivity" in targets:
        feature_sensitivity.make_figure(d)


if __name__ == "__main__":
    main()
