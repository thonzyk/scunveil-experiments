#!/usr/bin/env bash
# Fresh/resume and cache controls are parsed centrally, before any worker starts.
set -Eeuo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -u "$script_dir/benchmark_runner.py" "$@"
