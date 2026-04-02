#!/usr/bin/env bash
# Run plot_chain.jl on a chain FITS file.
#
# Usage (from this directory):
#   ./run_plot_chain.sh <chain.fits>
#
# Paths to Octofitter_imbh.jl and plot_chain.jl are resolved relative to the
# location of this script, so no manual path editing is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCTOFITTER_DIR="$(cd "${SCRIPT_DIR}/../../../Octofitter_imbh.jl" && pwd)"
PLOT_SCRIPT="$(cd "${SCRIPT_DIR}/../../launch_scripts" && pwd)/plot_chain.jl"

if [[ $# -lt 1 ]]; then
    echo "Usage: $(basename "$0") <chain.fits>" >&2
    exit 1
fi

exec julia --project="${OCTOFITTER_DIR}" "${PLOT_SCRIPT}" "$1"
