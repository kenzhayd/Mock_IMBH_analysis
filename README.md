# Ocen IMBH Analysis

Scripts for fitting orbits of high-velocity stars around an intermediate-mass black hole (IMBH) candidate in ω Centauri. Orbit fitting uses [`Octofitter_imbh.jl`](../Octofitter_imbh.jl), a development fork of [Octofitter](https://github.com/sefffal/Octofitter.jl), sampling with Pigeons (parallel tempering HMC/NUTS).

---

## Repository Structure

```
Ocen_IMBH_analysis/
├── configs/
│   └── default.toml                          # Reference run configuration
├── launch_scripts/
│   ├── octo_utils.jl                         # Star data and observation builder
│   ├── octo_orbit_direct_likelihoods.jl      # Main fitting script (v8 API)
│   ├── parse_config.jl                       # TOML loader and prior parser
│   ├── submit_job.jl                         # Slurm job generator/submitter
│   └── old_scripts/
│       └── octo_orbit_192c_18r.jl                    # Legacy script (v7 API)
└── tests/
    ├── run_tests.sh                          # Slurm batch script to run all tests
    ├── test_likelihoods.jl                   # Unit tests for PM/accel likelihoods
    ├── test_ad_compatibility.jl              # ForwardDiff gradient verification
    └── test_small_fit.jl                     # End-to-end 2-star sampling test
```

---

## Prerequisites

- Julia ≥ 1.10 with the `Octofitter_imbh.jl` package (the sibling directory `../Octofitter_imbh.jl`)
- On DAC HPC clusters: `module load julia/1.10.10`

All scripts are run with `--project=path/to/Octofitter_imbh.jl` so that the local fork is used rather than any registered Octofitter version.

---

## Configuration File

All run parameters live in a single TOML file. The reference configuration is [`configs/default.toml`](configs/default.toml). Copy and edit it to create a new run variant:

```bash
cp configs/default.toml configs/my_run.toml
# edit configs/my_run.toml
```

### Key sections

#### `[meta]`
Human-readable labels written into the run summary.

```toml
[meta]
system_name = "Omega_Cen"
description = "5-star direct likelihood fit"
```

#### `[stars]`
Which stars to include. Available labels: `A`, `B`, `C`, `D`, `E`, `F`, `G`. Stars B and G have lower-quality data.

```toml
[stars]
selected = ["A", "C", "D", "E", "F"]
```

#### `[epoch]`
Reference epoch at which position, proper motion, and acceleration are evaluated (decimal years; converted to MJD internally).

```toml
[epoch]
year = 2010.0
```

#### `[priors.system]`
System-level parameters shared across all stars.

| Parameter | Meaning | Units |
|-----------|---------|-------|
| `plx` | Parallax | mas |
| `M` | IMBH mass | M☉ |
| `offsetx` | IMBH RA offset from assumed centre | mas |
| `offsety` | IMBH Dec offset from assumed centre | mas |

```toml
[priors.system]
plx     = "truncated(Normal(0.19, 0.004), lower=0)"
M       = "Uniform(100, 120000)"
offsetx = "Normal(0, 10)"
offsety = "Normal(0, 10)"
```

#### `[priors.companion_defaults]`
Default orbital priors applied to every star.

| Parameter | Meaning |
|-----------|---------|
| `P` | Period [yr] |
| `e` | Eccentricity |
| `i` | Inclination (use `"Sine()"`) |
| `omega` | Argument of periastron (use `"UniformCircular()"`) |
| `Omega` | Longitude of ascending node (use `"UniformCircular()"`) |
| `theta` | Mean anomaly at epoch (use `"UniformCircular()"`) |

#### `[priors.overrides.<STAR>]`  *(optional)*
Override defaults for individual stars. Only list parameters that differ.

```toml
[priors.overrides.B]
P = "Uniform(1, 2_000_000)"

[priors.overrides.G]
P = "Uniform(1, 200_000)"
```

#### Prior specification syntax

Prior strings are parsed at runtime. Supported forms:

```
"Uniform(lo, hi)"
"Normal(mu, sigma)"
"truncated(Normal(mu, sigma), lower=L)"
"truncated(Normal(mu, sigma), lower=L, upper=U)"
"Sine()"
"UniformCircular()"
```

Underscores in numbers are allowed (`"Uniform(10, 2_000_000)"`).

#### `[sampling]`
Controls Pigeons parallel-tempering run.

```toml
[sampling]
n_rounds             = 18     # 2^n_rounds total iterations
n_chains             = 192    # parallel chains (should match CPU count)
n_chains_variational = 192
checkpoint           = false
```

#### `[slurm]`
Slurm resource requests. Adjust for the target cluster.

```toml
[slurm]
account       = "def-vhenault"
job_name      = "octo_imbh"
nodes         = 1
cpus_per_task = 192
mem_per_cpu   = "3G"
time          = "72:00:00"
julia_module  = "julia/1.10.10"
julia_threads = 192
mail_type     = "ALL"
mail_user     = "you@institution.ca"
```

#### `[paths]`
Paths are resolved relative to the config file's directory.

```toml
[paths]
project    = "../../Octofitter_imbh.jl"   # --project= for Julia
output_dir = "run_outputs"                 # chain files, plots, summaries
log_dir    = "logs"                        # Slurm stdout/stderr and tee logs
```

---

## Running Tests

The three test scripts can be run individually or via the Slurm batch script. Since `test_small_fit.jl` runs a short MCMC it benefits from multiple CPUs and is best submitted as a cluster job.

### Run on HPC (recommended)

```bash
cd tests/
sbatch run_tests.sh
```

Before submitting, edit `run_tests.sh` to set the correct `OCTOFITTER_PROJECT` path:

```bash
OCTOFITTER_PROJECT=/path/to/Octofitter_imbh.jl
```

Output is written to `tests/logs/octo_tests_<JOBID>.out` and `.err`.

### Run locally (fast tests only)

`test_likelihoods.jl` and `test_ad_compatibility.jl` are lightweight and finish in under a minute:

```bash
julia --project=../Octofitter_imbh.jl tests/test_likelihoods.jl
julia --project=../Octofitter_imbh.jl tests/test_ad_compatibility.jl
```

`test_small_fit.jl` runs a real 2-star fit and requires multiple threads:

```bash
julia --project=../Octofitter_imbh.jl -t 4 tests/test_small_fit.jl
```

### What the tests check

| Script | What it tests |
|--------|--------------|
| `test_likelihoods.jl` | `PlanetPMObs` and `PlanetAccelObs` evaluate correctly on a known orbit; position/PM/acceleration recover injected values |
| `test_ad_compatibility.jl` | Model compiles as type-stable; ForwardDiff can compute gradients without errors |
| `test_small_fit.jl` | End-to-end: 2-star MCMC sampling completes; posterior plots are generated |

---

## Submitting a Production Fit

### 1. Create (or choose) a config file

```bash
cp configs/default.toml configs/my_run.toml
# edit configs/my_run.toml as needed
```

Adjust at minimum: `[stars] selected`, `[slurm] account`, `[slurm] mail_user`, and `[paths] project`.

### 2. Preview the generated Slurm script (dry run)

```bash
cd launch_scripts/
julia submit_job.jl ../configs/my_run.toml --dry-run
```

This writes a timestamped `.sh` file to the log directory and prints the `sbatch` command to run, but does **not** submit.

### 3. Submit

```bash
julia submit_job.jl ../configs/my_run.toml
```

`submit_job.jl` will:
1. Parse the config
2. Generate a Slurm batch script in `<log_dir>/job_<timestamp>.sh`
3. Call `sbatch` to submit it

The Slurm job runs `octo_orbit_direct_likelihoods.jl` with the config as its only argument.

### 4. Monitor

```bash
squeue -u $USER
tail -f logs/octo_imbh_<JOBID>.out
```

### 5. Outputs

All outputs land in `<output_dir>` (default: `run_outputs/` relative to the config file):

| File | Contents |
|------|---------|
| `stars<XYZ>_<N>c_<R>r_<JOBID>_summary.md` | Run metadata, priors, full config |
| `stars<XYZ>_<N>c_<R>r_<JOBID>_chain.fits` | Full posterior chain (load with `Octofitter.loadchain`) |
| `stars<XYZ>_<N>c_<R>r_<JOBID>_corner.png` | Corner plot of all parameters |
| `stars<XYZ>_<N>c_<R>r_<JOBID>_orbit_panels.png` | Sky-plane orbit panels per star |
| `stars<XYZ>_<N>c_<R>r_<JOBID>_posteriors.png` | Marginal posterior histograms |

---

## Running the Fitting Script Directly

For local testing or interactive use, bypass `submit_job.jl` and call the fitting script directly:

```bash
julia --project=../Octofitter_imbh.jl -t <threads> \
    launch_scripts/octo_orbit_direct_likelihoods.jl \
    configs/my_run.toml
```

If no config path is given, the script falls back to `configs/default.toml`.

---

## Utility Module

[`launch_scripts/octo_utils.jl`](launch_scripts/octo_utils.jl) is a Julia module providing:

- `StarData` struct — position (RA/Dec), proper motion, and acceleration with uncertainties for each star
- `stars` dictionary — labelled `"A"` through `"G"` with measured observables
- `build_star_observations(star, epoch_mjd)` — returns `(astrom_obs, pm_obs, acc_obs)` ready to pass to `Planet(observations=[...])`

This module is included automatically by `octo_orbit_direct_likelihoods.jl` and all test scripts.

---

## Legacy Script

[`launch_scripts/old_scripts/octo_orbit_192c_18r.jl`](launch_scripts/old_scripts/octo_orbit_192c_18r.jl) uses the v7 Octofitter API with synthetic 3-epoch astrometry rather than direct PM/acceleration likelihoods. It is kept for comparison only and is not actively maintained.
