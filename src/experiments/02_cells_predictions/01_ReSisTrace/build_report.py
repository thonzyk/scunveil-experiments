#!/usr/bin/env python3
"""Report a snapshot of completed repetitions; safe while collection continues."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, help="Exact collection directory printed by the runner")
    parser.add_argument("--max-repetitions", type=int, default=100,
                        help="Use the first N completed repetitions (default 100); 0 uses all")
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/results/analysis/ReSisTrace"),
                        help="Find latest collection here if --collection is omitted")
    args = parser.parse_args()
    if args.max_repetitions < 0:
        parser.error("--max-repetitions must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    from replication_report import build_report
    if args.collection:
        root = args.collection
    else:
        choices = sorted((args.output_root / "collections").glob("*/collection.json"),
                         key=lambda p: p.stat().st_mtime)
        if not choices:
            parser.error("No collections found; run collect_replicates.py first")
        root = choices[-1].parent
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
        output = build_report(root, max_repetitions=args.max_repetitions or None)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Report: {root.resolve().parent.parent / 'report' / 'REPORT.md'}", flush=True)
    print(f"Snapshot: {output}", flush=True)


if __name__ == "__main__":
    main()
