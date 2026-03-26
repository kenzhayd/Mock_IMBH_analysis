#!/bin/bash
#SBATCH --account=def-vhenault
#SBATCH --job-name=octo_imbh_192
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-cpu=3G
#SBATCH --time=72:00:00
#SBATCH --output=logs/octo_imbh_%j.out
#SBATCH --error=logs/octo_imbh_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vincent.henault@smu.ca

mkdir -p logs

module load julia/1.10.10

julia --project=../../Octofitter_imbh.jl -t 192 ./octo_orbit_direct_likelihoods.jl 2>&1 | tee logs/output_direct_likelihoods_192.log
