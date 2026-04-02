#!/bin/bash
#SBATCH --account=def-vhenault
#SBATCH --job-name=octo_imbh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-cpu=3G
#SBATCH --time=23:59:00
#SBATCH --output=/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/logs/octo_imbh_%j.out
#SBATCH --error=/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/logs/octo_imbh_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vincent.henault@smu.ca

mkdir -p /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/logs
mkdir -p /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/run_outputs

module load julia/1.10.10

export JULIA_CONDAPKG_BACKEND=Null

julia --project=/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../../Octofitter_imbh.jl -e 'using Pkg; Pkg.instantiate(); Pkg.add(["CairoMakie", "PairPlots", "Distributions", "Unitful"])' 2>&1 | tee /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/logs/instantiate_${SLURM_JOB_ID}.log

julia --project=/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../../Octofitter_imbh.jl -t 192 \
    /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/launch_scripts/octo_orbit_direct_likelihoods.jl \
    /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/test_run2.toml 2>&1 | tee /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/logs/output_${SLURM_JOB_ID}.log
