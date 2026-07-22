#!/bin/bash
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=64
#SBATCH --account=def-sgkang09
#SBATCH --output=./slurm/slurm-%j.out

module load intel-one/2025.3
uv run ./tools/main.py -a "SE" -n 64
