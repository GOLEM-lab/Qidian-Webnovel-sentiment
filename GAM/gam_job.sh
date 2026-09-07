#!/bin/bash
#SBATCH --job-name=qidian-gam
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=logs/gam_%j.out
#SBATCH --error=logs/gam_%j.err

# The whole GAM analysis in one job: fitting, sensitivity fits, diagnostics,
# the lme4 comparison and every figure (GAM_analysis_habrok.R).
#
# Longer and larger than the old 02:00:00 / 16GB job because it now also runs
# the sensitivity fits and the lme4 comparison that used to be done locally.
#
# Run from the repo root, which must also contain logs/:
#     mkdir -p logs && sbatch GAM/gam_job.sh
#
# logs/ has to exist BEFORE submission -- SLURM opens the --output/--error
# paths above before the script runs, so creating it here would be too late
# and the job would fail at launch with no log saying why. The repo ships a
# logs/.gitkeep for this reason.

# SLURM starts the job in the submission directory, and every path below is
# relative to the repository root. Fail loudly rather than with a confusing
# missing-file error from R.
if [ ! -d september-data ] || [ ! -d GAM ]; then
  echo "ERROR: run this from the repository root (september-data/ and GAM/ not found here)" >&2
  echo "       cwd is: $(pwd)" >&2
  exit 1
fi

module purge
module load R/4.5.1-gfbf-2025a

export GAM_DATA="september-data/long_combined_qidian_webnovel_paragraphs_comments_sentiment.csv"
export GAM_OUT="GAM"
export GAM_FIG="figures"

Rscript GAM/GAM_analysis_habrok.R
