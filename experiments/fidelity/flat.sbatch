#!/bin/bash
# one step of flat.py per job: the large tables' measure, or an arm's draws of the tables
# outside the default sizes, which hold up to half a million rows.
# submit from the repository root once `uv sync --group registry --group tabicl` ran through
# Slurm, the log directory made, with the site's partition and QOS and mixedbench's cache.
# the measure rewrites flat-real.csv, which every run reads, so the runs wait for its end:
#   mkdir -p experiments/fidelity/flat-logs
#   sbatch -p <partition> -q <qos> experiments/fidelity/flat.sbatch measure --cache <cache> --large
#   sbatch -p <partition> -q <qos> experiments/fidelity/flat.sbatch run default outside
#   sbatch -p <partition> -q <qos> experiments/fidelity/flat.sbatch run independent outside
#   sbatch -p <partition> -q <qos> experiments/fidelity/flat.sbatch run tabicl outside
# then, once every job has ended:
#   uv run --group registry python experiments/fidelity/flat.py report
# a CPU per job; the largest table's design holds 480 million cells
#SBATCH --job-name=fidelity
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/fidelity/flat-logs/%j.log

# no sync per job, which concurrent jobs would race on in the shared .venv
export UV_NO_SYNC=1
# uv's lock file beside the logs: a node whose /tmp is broken fails every job
export TMPDIR="$PWD/experiments/fidelity/flat-logs/tmp"
mkdir -p "$TMPDIR"
uv run --group registry --group tabicl python experiments/fidelity/flat.py "$@"
