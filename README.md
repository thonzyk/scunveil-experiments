# scUNVEIL Experiment Reproducibility Repository

## Dev container setup

This repo provides a `Dockerfile` and `.devcontainer/devcontainer.json` for a reproducible GPU training environment.

Before opening the repo in the dev container, define where your data and results live on the host machine:

```bash
export SCUNVEIL_DATA_DIR="/path/to/data"
export SCUNVEIL_RESULTS_DIR="/path/to/results"