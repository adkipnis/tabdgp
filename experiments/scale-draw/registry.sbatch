#!/bin/bash
# one registry fit per array task (registry.py run): its task in registry-fits.csv,
# each arm's y drawn at the draws first to stop - 1 on the fit's design and refitted by lme4,
# or one arm's, which past draw 38 draws only while short of 39 refits with estimates.
# submit from the repository root once `uv sync --group registry` ran through Slurm, the log
# directory made, with the site's partition and QOS, mixedbench's cache and the survey's data
# directory, first draws 0-38 of every arm in chunks, then once they have ended draws 39-77 per arm:
#   mkdir -p experiments/scale-draw/registry-runs/logs
#   sbatch -p <partition> -q <qos> --array=<tasks> experiments/scale-draw/registry.sbatch \
#       <cache> <survey> <first> <stop> [<arm>]
# then, once every array has ended:
#   uv run --group registry python experiments/scale-draw/registry.py collect
# a CPU per task; a chunk's refits of at most two hours each, most of them a second
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
    --cache "$1" --survey "$2" --draws "$3" "$4" ${5:+--arms "$5"}
