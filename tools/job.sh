#!/bin/bash
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --partition=skylake
#SBATCH --cpus-per-task=40
#SBATCH --account=def-sgkang09

module load intel-one/2025.3
uv run main.py
