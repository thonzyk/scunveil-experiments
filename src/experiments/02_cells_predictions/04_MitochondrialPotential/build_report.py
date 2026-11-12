#!/usr/bin/env python3
"""Refresh /workspace/results/analysis/MitochondrialPotential/report/ safely during collection."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/results/analysis/MitochondrialPotential"))
    parser.add_argument("--max-repetitions", type=int, default=100, help="First N completed repetitions; 0 means all")
    args = parser.parse_args()
    if args.max_repetitions < 0:
        parser.error("--max-repetitions must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    from report import build_report
    root = args.collection
    if root is None:
        manifests = list((args.output_root / "collections").glob("*/collection.json"))
        if not manifests:
            parser.error("No collection yet. Run collect_replicates.py first; prepared data alone is not an experiment.")
        root = max(manifests, key=lambda p: p.stat().st_mtime).parent
    # Scientific collection code stays unchanged; report dispatch supports both formats.
    import json
    if json.loads((root / "collection.json").read_text()).get("kind") == "full_data_pca_comparison":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pca_report import build_report
    if json.loads((root / "collection.json").read_text()).get("kind") == "unscaled_benchmark_v2":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmark_report import build_report
    try:
        path = build_report(root, max_repetitions=args.max_repetitions or None)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Report: {path / 'REPORT.md'}")


if __name__ == "__main__":
    main()
