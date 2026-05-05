#!/usr/bin/env julia

"""
    mock_job.sh

To resume a previous run:
    1. Set restart_job_id = "SLURM_JOB_ID"
    2. Set n_rounds = TOTAL desired rounds (NOT extra rounds)

"""

using Dates
using JSON3

# Load module 
include(joinpath(@__DIR__, "mock_utils.jl"))
using .mock_utils

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────

# Give a good name for this run
IDENTIFY_THIS_RUN = "BEANS"

# Test masses, error
# Difference of masses
m_diff = 0.0
# Scaling factor for uncertainties
err_scale = 1.0


# === MOCK SYSTEM OBSERVATIONS BASED ON starsACDEF_192c_18r_cont_10836842

MOCK_MASS = 6.373e4 - m_diff # Solar masses
MOCK_NAME = "mock_10836842" # Parameter set with median values from starsACDEF_192c_18r_cont_10836842
Z_SIGMA   = 4558.0 # Los prior 
PLX       = 0.191 # Parallax
# Or PLX = 1 / DISTANCE_KPC

DISTANCE_KPC = 5.43      # Omega Cen distance

# Reference epoch (MJD)
TREF = 55197.0 # 2010  FROM RUN 10836842

# Uncertainties 

SIGMA_RA_OFF     = 0.5   # mas 
SIGMA_DEC_OFF    = 0.5   # mas

# RA uncertainties (Haberle): 0.038, 0.182, 0.127, 0.082, 0.025, 0.017, 0.098
# Ave = 0.081
SIGMA_PM_RA  = 0.081    # mas/yr
# Dec uncertainties (Haberle): 0.055, 0.081, 0.056, 0.061, 0.037, 0.016, 0.062
# Ave = 0.053
SIGMA_PM_DEC = 0.053    # mas/yr

# RA Acc uncertainties (Haberle): 0.0083, 0.0239, 0.0333, 0.0177, 0.0042, 0.0038, 0.0267
# Ave = 0.0168
SIGMA_ACC_RA = 0.0168* err_scale     # mas/yr²
# Dec uncertainties (Haberle): 0.0098, 0.0157, 0.0123, 0.0162, 0.0075, 0.0038, 0.0170
# Ave = 0.0118
SIGMA_ACC_DEC= 0.0118* err_scale     # mas/yr²

# Average of measured uncertainties 4000 m/s (F) and 2700 m/s (E)
SIGMA_RV     = 3350.0  # m/s


# Cluster center from Anderson & van der Marel (2010), ApJ 710, 1032:
# (α, δ) = (13:26:47.24, −47:28:46.45)  →  201.6968333°, −47.4795694°
# https://iopscience.iop.org/article/10.1088/0004-637X/710/2/1032
# Used as the origin of the relative astrometry frame; the IMBH position
# is a free parameter (offsetx, offsety) relative to this point.

# Omega Centauri Center RA and Dec in deg
OCEN_CENTER_RA  = 201.6968333 # [Deg]
OCEN_CENTER_DEC = -47.4795694 # [Deg]

# IMBH location
# Offsets from starsACDEF_192c_18r_cont_10836842
offsetx = -84.443  # [mas]
offsety = 730.613  # [mas]

ra_offset_deg  = offsetx / 3.6e6 # [deg]
dec_offset_deg = offsety / 3.6e6 # [deg]

IMBH_RA = OCEN_CENTER_RA + ra_offset_deg # [deg]
IMBH_DEC = OCEN_CENTER_DEC + dec_offset_deg # [deg]

# Cluster systemic radial velocity (Baumgardt catalogue)
RV_CLUSTER     = 232780.0   # [m/s]  (232.78 ± 0.21 km/s)
RV_CLUSTER_ERR = 210.0      # [m/s]


# Pigeons 
restart_job_id = ""

n_rounds = 16
n_chains = 192
n_chains_variational = 192
include_acceleration = true

# Slurm config
job_name      = IDENTIFY_THIS_RUN
account       = "def-vhenault"
nodes         = 1
cpus_per_task = 192
mem           = "192G"
mem_per_cpu   = "3G"
time          = "18:00:00"
julia_module  = "julia/1.11.3"
julia_threads = 192
mail_type     = "ALL"
mail_user     = "Mackenzie.hayduk@smu.ca"

# Paths
project_dir = "/home/kenzhayd/projects/def-vhenault/kenzhayd/octoIMBH_env"
base = "/home/kenzhayd/projects/def-vhenault/kenzhayd/mock_cluster/"

fitting_script = "/home/kenzhayd/projects/def-vhenault/kenzhayd/mock_cluster/mock_inference.jl"



# ─────────────────────────────────────────────────────────────
# Make run ID
function make_run_id(job_name, n_rounds, n_chains)
    return "$(job_name)_$(n_rounds)r_$(n_chains)c"
end

# Results and Logs
run_id = make_run_id(job_name, n_rounds, n_chains)

results_dir = joinpath(base, "results", run_id)
log_dir = joinpath(results_dir, "logs")
mkpath(results_dir)
mkpath(log_dir)

println("Run ID: $run_id")
println("Results directory: $results_dir")

# Save configuration with CLI arguments 
config_dict = Dict(
    "mock_name" => MOCK_NAME,
    "M_IMBH" => MOCK_MASS,
    "z_prior_sigma" => Z_SIGMA,
    "plx" => PLX,
    "n_rounds" => n_rounds,
    "n_chains" => n_chains,
    "n_chains_variational" => n_chains_variational,
    "distance_kpc" => DISTANCE_KPC,
    "tref" => TREF,
    "sigma_ra_off" => SIGMA_RA_OFF,
    "sigma_dec_off" => SIGMA_DEC_OFF,
    "sigma_pm_ra" => SIGMA_PM_RA,
    "sigma_pm_dec" => SIGMA_PM_DEC,
    "sigma_acc_ra" => SIGMA_ACC_RA,
    "sigma_acc_dec" => SIGMA_ACC_DEC,
    "sigma_rv" => SIGMA_RV,
    "imbh_ra" => IMBH_RA,
    "imbh_dec" => IMBH_DEC,
    "rv_cluster" => RV_CLUSTER,
    "rv_cluster_err" => RV_CLUSTER_ERR,
    "include_acc" => include_acceleration,
    "job_name" => job_name,
    "results_dir" => results_dir
)

config_json = JSON3.write(config_dict; indent=4)

# Save CLI args exactly 
args_string = """
mock_name=$MOCK_NAME
M_IMBH=$MOCK_MASS
z_prior_sigma=$Z_SIGMA
plx=$PLX
n_rounds=$n_rounds
n_chains=$n_chains
n_chains_variational=$n_chains_variational
distance_kpc=$DISTANCE_KPC
tref=$TREF
sigma_ra_off=$SIGMA_RA_OFF
sigma_dec_off=$SIGMA_DEC_OFF
sigma_pm_ra=$SIGMA_PM_RA
sigma_pm_dec=$SIGMA_PM_DEC
sigma_acc_ra=$SIGMA_ACC_RA
sigma_acc_dec=$SIGMA_ACC_DEC
sigma_rv=$SIGMA_RV
imbh_ra=$IMBH_RA
imbh_dec=$IMBH_DEC
rv_cluster=$RV_CLUSTER
rv_cluster_err=$RV_CLUSTER_ERR
include_acc=$include_acceleration
results_dir=$results_dir
job_name=$job_name
"""

# Resume sampling
resume_arg = ""
job_id_str = restart_job_id

if !isempty(job_id_str)
    candidates = filter(
        f -> endswith(f, "_pt_location.txt") &&
             occursin("_$(job_id_str)_", f),
        readdir(results_dir; join=true, recursive=true)
    )

    isempty(candidates) && error(
        "No pt_location file found for job_id=$(job_id_str) in $(results_dir). " *
        "Ensure the previous run completed with checkpoint=true."
    )

    length(candidates) > 1 && @warn(
        "Multiple pt_location files match job_id=$(job_id_str); using: $(candidates[1])"
    )

    pt_exec_folder = strip(read(candidates[1], String))

    isdir(pt_exec_folder) || error("PT exec folder does not exist: $pt_exec_folder")

    resume_arg = " --resume $(pt_exec_folder)"

    println("Resume job_id=$(job_id_str) → PT folder: $pt_exec_folder")
end

julia_cmd = """
julia --project=$project_dir -t $julia_threads \
    $fitting_script \
    mock_name=$MOCK_NAME \
    M_IMBH=$MOCK_MASS \
    z_prior_sigma=$Z_SIGMA \
    plx=$PLX \
    n_rounds=$n_rounds \
    n_chains=$n_chains \
    n_chains_variational=$n_chains_variational \
    distance_kpc=$DISTANCE_KPC \
    tref=$TREF \
    sigma_ra_off=$SIGMA_RA_OFF \
    sigma_dec_off=$SIGMA_DEC_OFF \
    sigma_pm_ra=$SIGMA_PM_RA \
    sigma_pm_dec=$SIGMA_PM_DEC \
    sigma_acc_ra=$SIGMA_ACC_RA \
    sigma_acc_dec=$SIGMA_ACC_DEC \
    sigma_rv=$SIGMA_RV \
    imbh_ra=$IMBH_RA \
    imbh_dec=$IMBH_DEC \
    rv_cluster=$RV_CLUSTER \
    rv_cluster_err=$RV_CLUSTER_ERR \
    include_acc=$include_acceleration \
    results_dir=$results_dir \
    job_name=$job_name $resume_arg
"""

script = """
#!/bin/bash
#SBATCH --account=$(account)
#SBATCH --job-name=$(job_name)
#SBATCH --nodes=$(nodes)
#SBATCH --cpus-per-task=$(cpus_per_task)
#SBATCH --mem-per-cpu=$(mem_per_cpu)
#SBATCH --time=$(time)
#SBATCH --output=$(log_dir)/$(job_name)_%j.out
#SBATCH --error=$(log_dir)/$(job_name)_%j.err
#SBATCH --mail-type=$(mail_type)
#SBATCH --mail-user=$(mail_user)

module purge
module load $julia_module

mkdir -p $(log_dir)
mkdir -p $(results_dir)

# Save CLI args 
cat << EOF > $(results_dir)/config.json
$config_json
EOF

cat << EOF > $(results_dir)/args.txt
$args_string
EOF

echo "Saved config → $(results_dir)/config.json"
echo "Saved args   → $(results_dir)/args.txt"

export JULIA_CONDAPKG_BACKEND=Null

$julia_cmd
"""


timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
script_path = joinpath(base, "job_$(job_name)_$timestamp.sh")

write(script_path, script)

println("Generated: $script_path")
