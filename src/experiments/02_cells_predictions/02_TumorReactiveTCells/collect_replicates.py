#!/usr/bin/env python3
"""Run tumor-reactivity paired repetitions until interrupted; resume with the same command."""
import argparse
import os
from pathlib import Path
import signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--max-replicates", type=int, help="Optional TOTAL completed target; default infinite")
    args = parser.parse_args()
    from tcr_common import load_settings
    settings = load_settings(args.config)
    if args.device:
        settings["device"] = args.device
    if settings["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[name] = str(settings["threads"])
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    from experiment import collect
    collect(settings, args.max_replicates)


if __name__ == "__main__":
    main()
