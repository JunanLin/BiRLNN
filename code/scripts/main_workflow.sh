#!/usr/bin/env bash
# Orchestrate: clean training set -> sample -> compute FCD -> constrained gen -> analysis -> RL
# Usage (from repo root or anywhere): bash code/scripts/main_workflow.sh
set -euo pipefail

# Compute repo root based on this script location
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Prefer conda's C++ runtime to avoid CXXABI mismatches with SciPy/sklearn
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Python executable (override with PYTHON env var)
PYTHON="${PYTHON:-python3}"

# Run training
"$PYTHON" code/scripts/main_trainer.py

# Run evaluation
"$PYTHON" evaluation/main_evaluator.py

# Run sampling (will raise error and stop if it fails)
"$PYTHON" code/analysis/collective_sample_30k.py

# On success, compute FCD (adjust args as needed)
# Produces data for Table 1
"$PYTHON" code/analysis/compute_fcd.py --gen-path evaluation/ --ref data/SMILES_training.csv

# Run constrained generation
"$PYTHON" code/scripts/main_cons_generator.py

# Analyze constrained generation results
# Produces Table 2, Figures 2-5, stored under evaluation/analysis/
"$PYTHON" code/analysis/main_analyzer.py

# Perform RL fine-tuning with multiple experiments and reward weights
# Produces Table 3, Figures 7-8, stored under evaluation/rl/reinforce/
# Can modify multiple_RL.sh to change experiments/reward weights
bash code/scripts/multiple_RL.sh

# Visualize QED vs. SA for RL-generated molecules
"$PYTHON" code/model/visualize_generated.py