#!/bin/bash
# one registry fit per array task (registry.py run): its task in registry-fits.csv,
# each arm's y drawn ten times on the fit's design and refitted by lme4.
# submit from the repository root once `uv sync --group registry` ran through Slurm, the log
# directory made, with the site's partition and QOS, mixedbench's cache and the survey's data
# directory, and n the tasks in registry-fits.csv:
#   mkdir -p experiments/scale-draw/registry-runs/logs
#   sbatch -p <partition> -q <qos> --array=0-$(( n - 1 )) experiments/scale-draw/registry.sbatch \
#       <cache> <survey>
# then, once the array has ended:
#   uv run --group registry python experiments/scale-draw/registry.py collect
# a CPU per task; fifty refits of at most half an hour each, most of them a second
#SBATCH --job-name=registry
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/scale-draw/registry-runs/logs/%A_%a.log

# no sync per task, which concurrent tasks would race on in the shared .venv
export UV_NO_SYNC=1
# R's session directory beside the run directories: a node whose /tmp is broken fails every task
export TMPDIR="$PWD/experiments/scale-draw/registry-runs/tmp"
mkdir -p "$TMPDIR"
uv run --group registry python experiments/scale-draw/registry.py run "$SLURM_ARRAY_TASK_ID" \
    --cache "$1" --survey "$2"
