#!/usr/bin/env bash

while true; do
    echo "Starting script at $(date)"
    python3 0100_Hao_PBMC_benchmark.py
    echo "Script exited with code $? at $(date)"
    sleep 1
done