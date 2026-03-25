#!/bin/bash
#SBATCH --account=def-vhenault
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --mem-per-cpu=3G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vincent.henault@smu.ca
#SBATCH --time=72:00:00

module load julia/1.10.10

julia -t 192 ./octo_orbit_julia_192c_18r_all_MCMC_centre.jl  2>&1 | tee output_192_all_18r_MCMC_centre.log
