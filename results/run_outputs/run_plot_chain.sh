#!/bin/bash
# Submit plot_chain.jl as a 1-CPU Slurm job on the DAC cluster.
#
# Usage (from this directory):
#   sbatch run_plot_chain.sh <chain.fits>
#
# Paths to Octofitter_imbh.jl and plot_chain.jl are resolved relative to the
# location of this script, so no manual path editing is needed.

#SBATCH --account=def-vhenault
#SBATCH --job-name=plot_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/plot_chain_%j.out
#SBATCH --error=../logs/plot_chain_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vincent.henault@smu.ca

set -euo pipefail

# Under Slurm, BASH_SOURCE[0] points to a copy in /localscratch, so use
# SLURM_SUBMIT_DIR (the directory sbatch was invoked from) when available.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OCTOFITTER_DIR="$(cd "${SCRIPT_DIR}/../../../Octofitter_imbh.jl" && pwd)"
PLOT_SCRIPT="$(cd "${SCRIPT_DIR}/../../launch_scripts" && pwd)/plot_chain.jl"
LOG_DIR="$(cd "${SCRIPT_DIR}/../logs" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $(basename "$0") <chain.fits>" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

module load julia/1.10.10

exec julia --project="${OCTOFITTER_DIR}" "${PLOT_SCRIPT}" "$1"
