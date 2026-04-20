# Run Summary

- **Date:** 2026-04-19T23:25:48.105
- **Slurm Job ID:** 10691237
- **Stars:** A, C, D, E, F
- **Reference epoch:** 55197.0 MJD (2010.0 yr)
- **Config file:** /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/noAccel_17r_cont10549504.toml
- **Run type:** continuation (target: 17 total rounds)
- **Resumed from PT checkpoint:** /home/vhenault/links/projects/def-vhenault/vhenault/OCen_IMBH/Ocen_IMBH_analysis/launch_scripts/results/all/2026-04-17-15-46-45-SSKpVEU1

## Sampling Parameters

| Parameter | Value |
|---|---|
| n_rounds | 17 (total) |
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
| z_prior | Normal(0, 20000.0) AU |

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
# Continuation of job 10549504 (noAccel_16r) for 1 additional round
# ============================================================

[meta]
system_name = "Omega_Cen"
description = "5-star direct likelihood fit (no accel) — continuation of job 10549504"

[stars]
selected = ["A", "C", "D", "E", "F"]

[epoch]
year = 2010.0

[priors.system]
plx     = "truncated(Normal(0.19, 0.004), lower=0)"
M       = "Uniform(100, 100000)"
offsetx = "Uniform(-3000, 3000)"
offsety = "Uniform(-3000, 3000)"

[priors.companion_defaults]
P     = "Uniform(10, 2_000_000)"
e     = "Uniform(0.0, 0.99)"
i     = "Sine()"
omega = "UniformCircular()"
Omega = "UniformCircular()"
theta = "UniformCircular()"

[data.defaults]
position        = true
proper_motion   = true
acceleration    = false
radial_velocity = true
z_prior         = true

[data.z_prior]
sigma_z_au = 20000

[restart]
job_id = "10549504"

[sampling]
n_rounds             = 17
n_chains             = 192
n_chains_variational = 192
checkpoint           = true

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

[paths]
project    = "../../Octofitter_imbh.jl"
output_dir = "../results/run_outputs"
log_dir    = "../results/logs"

```

## Sampling Result

- **Additional rounds run:** 1
- **Total rounds:** 17

## Pigeons PT Checkpoint

The PT exec folder contains intermediate checkpoint files.
It can be deleted once the chain FITS file has been verified.

| | Path |
|---|---|
| PT exec folder | `/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/launch_scripts/results/all/2026-04-19-23-25-49-p67JpLv3` |
| PT location file | `/lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/../results/run_outputs/starsACDEF_192c_17r_cont_10691237_pt_location.txt` |
