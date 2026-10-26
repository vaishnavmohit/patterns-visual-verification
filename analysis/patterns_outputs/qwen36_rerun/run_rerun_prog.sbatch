#!/bin/bash
#SBATCH -J rq36_prog
#SBATCH -p gpu
#SBATCH --account=project_exai
#SBATCH --gres=gpu:A100:1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o slurm-prog-%j.out
#SBATCH -e slurm-prog-%j.err
# Rerun of the 81 Qwen3.6:35b symbolic items whose archived (7-14 May) output
# said no input was provided. Filtered item files per context; outputs go to a
# separate tree so ~/output_human_context is never touched. Settings match the
# archive: temperature 0, thinking off, config budgets (4096 out / 16384 ctx).
set -e
R=$HOME/rerun_qwen36_blind
cd $HOME/code/human-machine-reasoning
git rev-parse HEAD > $R/code_commit.txt
git diff > $R/code_uncommitted.diff
sha256sum $SMBHOME/ollama/models/manifests/registry.ollama.ai/library/qwen3.6/35b > $R/model_manifest_sha256.txt
for ctx in aps aps_concept; do
  DATA_DIR=$R/$ctx/data OUTPUT_DIR=$R/$ctx/output HMR_CONTEXTS=$ctx HMR_MODELS=qwen3.6:35b \
  HMR_THINK=false HMR_NUM_SAMPLES=50 bash scripts/hpc/run_logo_symbolic_home.sh
done
echo "ALL DONE"
