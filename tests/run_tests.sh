#!/bin/bash
#SBATCH --account=def-vhenault
#SBATCH --job-name=octo_tests
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=logs/octo_tests_%j.out
#SBATCH --error=logs/octo_tests_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vincent.henault@smu.ca

mkdir -p logs

module load julia/1.10.10

OCTOFITTER_PROJECT=../../Octofitter_imbh.jl
TESTS_DIR="$(dirname "$0")"

run_test() {
    local script="$1"
    echo "========================================"
    echo "Running: $script"
    echo "========================================"
    julia --project="$OCTOFITTER_PROJECT" -t 4 "$TESTS_DIR/$script"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "FAILED: $script (exit code $status)"
        exit $status
    fi
    echo "PASSED: $script"
    echo ""
}

run_test test_likelihoods.jl
run_test test_ad_compatibility.jl
run_test test_small_fit.jl

echo "========================================"
echo "All tests passed."
echo "========================================"
