# Run Summary

- **Date:** 2026-04-21T10:10:02.879
- **Slurm Job ID:** 10741829
- **Stars:** A, C, D, E, F
- **Reference epoch:** 55197.0 MJD (2010.0 yr)
- **Config file:** /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/noAccel_16r_x2_zprior.toml
- **Run type:** fresh

## Sampling Parameters

| Parameter | Value |
|---|---|
| n_rounds | 16 |
| n_chains | 192 |
| n_chains_variational | 192 |
| checkpoint | true |

## System Priors

| Parameter | Prior |
|---|---|
| offsetx | Uniform(-3000, 3000) |
| M | Uniform(100, 100000) |
| offsety | Uniform(-3000, 3000) |
| plx | truncated(Normal(0.19, 0.004), lower=0) |
| z_prior | Normal(0, 9000.0) AU |

## Companion Priors (defaults)

| Parameter | Prior |
|---|---|
| theta | UniformCircular() |
| e | Uniform(0.0, 0.99) |
| P | Uniform(10, 2_000_000) |
| omega | UniformCircular() |
| Omega | UniformCircular() |
| i | Sine() |

## Full Configuration

```toml
# ============================================================
# Omega Centauri IMBH Orbit Fitting — Run Configuration
# ============================================================
#
# This file controls all tuneable parameters for a single run.
# Copy it, edit the copy, then launch:
#
#   julia submit_job.jl configs/my_run.toml        # HPC (generates + submits Slurm job)
#   julia octo_orbit_direct_likelihoods.jl configs/my_run.toml   # local
#
# Star observational data (positions, PMs, accelerations) lives in
# octo_utils.jl and is NOT duplicated here.
# ============================================================

# --- Run metadata (written into the run summary) ---
[meta]
system_name = "Omega_Cen"
description = "5-star direct likelihood fit"

# --- Star selection ---
# Available: A, B, C, D, E, F, G  (B and G have lower-quality data)
[stars]
selected = ["A", "C", "D", "E", "F"]

# --- Reference epoch ---
# The single epoch at which position, PM, and acceleration are evaluated.
# Specified in decimal years; converted internally to MJD via Octofitter.years2mjd().
[epoch]
year = 2010.0

# === Priors =============================================================
# Prior strings are parsed at runtime into Distributions.jl objects.
# Supported forms:
#   "Uniform(lo, hi)"
#   "Normal(mu, sigma)"
#   "truncated(Normal(mu, sigma), lower=L)"
#   "truncated(Normal(mu, sigma), lower=L, upper=U)"
#   "Sine()"
#   "UniformCircular()"
# Underscores in numbers (e.g. 2_000_000) are allowed inside strings.
# ========================================================================

# System-level priors (shared across all companions)
[priors.system]
plx     = "truncated(Normal(0.19, 0.004), lower=0)"   # Parallax [mas]
M       = "Uniform(100, 100000)"                        # IMBH mass [solar masses]
offsetx = "Uniform(-3000, 3000)"                        # IMBH RA offset from assumed center [mas]; ±3" covers Haberle+2024 MCMC centre (0.77" NE of AvdM10)
offsety = "Uniform(-3000, 3000)"                        # IMBH Dec offset from assumed center [mas]

# Default companion (per-star) priors — applied to every star unless overridden below
[priors.companion_defaults]
P     = "Uniform(10, 2_000_000)"    # Orbital period [yr]
e     = "Uniform(0.0, 0.99)"        # Eccentricity
i     = "Sine()"                     # Inclination [rad]
omega = "UniformCircular()"          # Argument of periastron [rad]
Omega = "UniformCircular()"          # Longitude of ascending node [rad]
theta = "UniformCircular()"          # Mean anomaly at reference epoch [rad]

# Per-star prior overrides (optional)
# Only list parameters that differ from companion_defaults.
# Uncomment and adjust as needed:
#
# [priors.overrides.B]
# P = "Uniform(1, 2_000_000)"
#
# [priors.overrides.G]
# P = "Uniform(1, 200_000)"

# --- Data selection ---
# Which observation types to include per star.  Defaults apply to all stars.
# Set to false to exclude a data type for a specific star.
# "radial_velocity" is only used when the star has RV data in octo_utils.jl.
[data.defaults]
position        = true
proper_motion   = true
acceleration    = false
radial_velocity = true

z_prior         = true      # LOS position prior (cluster membership constraint)

# Line-of-sight (z) prior — constrains each star's LOS offset from the IMBH.
# sigma_z_au is the width of a Normal(0, σ) prior in AU.
# 0.0221 pc = 4558 AU (one-dimensional positional standard deviation from Haberle)
# Omega Cen core radius ≈ 4.1 pc ≈ 845,000 AU; half-light radius ≈ 7.9 pc ≈ 1,629,000 AU.
[data.z_prior]
sigma_z_au = 9000           # 1σ = 0.0221 pc [AU]

# Per-star overrides (uncomment to disable specific data for a star):
# [data.overrides.A]
# acceleration = false

# --- Resume a previous run (optional) ---
# Set job_id to the Slurm job ID of the run to resume; leave empty for a fresh run.
# Set n_rounds to the desired TOTAL round count (e.g. 20 to add 5 to a 15-round run).
[restart]
job_id = ""

# --- Sampling ---
[sampling]
n_rounds             = 16
n_chains             = 192
n_chains_variational = 192
checkpoint           = true

# --- Slurm / HPC ---
[slurm]
account       = "def-vhenault"
job_name      = "octo_imbh"
nodes         = 1
cpus_per_task = 192
mem_per_cpu   = "3G"
time          = "23:59:00"
julia_module  = "julia/1.10.10"
julia_threads = 192
mail_type     = "ALL"
mail_user     = "vincent.henault@smu.ca"

# --- Paths ---
# Relative paths are resolved from the directory containing this config file.
[paths]
project    = "../../Octofitter_imbh.jl"     # --project= argument for Julia
output_dir = "../results/run_outputs"        # chain files, plots, summaries
log_dir    = "../results/logs"               # Slurm stdout/stderr and tee logs

```

## Sampling Result


## Pigeons PT Checkpoint

The PT exec folder contains intermediate checkpoint files.
It can be deleted once the chain FITS file has been verified.

| | Path |
|---|---|
| PT exec folder | `/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/launch_scripts/results/all/2026-04-21-10-11-31-XywzeOPO` |
| PT location file | `/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/run_outputs/starsACDEF_192c_16r_10741829_pt_location.txt` |
