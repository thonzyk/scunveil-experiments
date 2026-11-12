#!/usr/bin/env python3
"""Continuously collect paired ReSisTrace split/dilution/training repetitions."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal

from replication_common import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Overrides of replication_config.json")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), help="Embedding device; sklearn probes always use CPU")
    parser.add_argument("--max-replicates", type=int, help="Optional TOTAL completed target; default runs until interrupted")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.device:
        settings["device"] = args.device
    if args.max_replicates is not None and args.max_replicates < 1:
        parser.error("--max-replicates must be positive")
    if settings["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(settings["threads"])
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from replication import collect
    try:
        collect(settings, args.max_replicates)
    except KeyboardInterrupt:
        # Also handle a stop during initial preparation/identity setup.
        print("Stopped. Completed repetitions, if any, remain intact.", flush=True)


if __name__ == "__main__":
    main()
