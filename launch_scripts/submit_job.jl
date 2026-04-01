#!/usr/bin/env julia
"""
    submit_job.jl — Generate and submit a Slurm job from a TOML config

Usage:
    julia submit_job.jl configs/my_run.toml              # generate + submit
    julia submit_job.jl configs/my_run.toml --dry-run     # generate only (inspect before submitting)

The generated Slurm script is saved to the log directory for reproducibility.
"""

using TOML
using Dates

# ── Parse arguments ─────────────────────────────────────────────────────

if isempty(ARGS)
    println(stderr, "Usage: julia submit_job.jl <config.toml> [--dry-run]")
    exit(1)
end

config_path = ARGS[1]
dry_run = "--dry-run" in ARGS

isfile(config_path) || error("Config file not found: $config_path")
cfg = TOML.parsefile(config_path)

# ── Extract sections ────────────────────────────────────────────────────

slurm = cfg["slurm"]
paths = cfg["paths"]

# Resolve paths relative to the config file's directory
config_dir = dirname(abspath(config_path))
abs_config = abspath(config_path)

log_dir     = isabspath(paths["log_dir"])     ? paths["log_dir"]     : joinpath(config_dir, paths["log_dir"])
output_dir  = isabspath(paths["output_dir"])  ? paths["output_dir"]  : joinpath(config_dir, paths["output_dir"])
project_dir = isabspath(paths["project"])     ? paths["project"]     : joinpath(config_dir, paths["project"])

# The fitting script lives next to this launcher
fitting_script = joinpath(@__DIR__, "octo_orbit_direct_likelihoods.jl")

# ── Generate Slurm script ──────────────────────────────────────────────

job_name = slurm["job_name"]

script = """
#!/bin/bash
#SBATCH --account=$(slurm["account"])
#SBATCH --job-name=$(job_name)
#SBATCH --nodes=$(slurm["nodes"])
#SBATCH --cpus-per-task=$(slurm["cpus_per_task"])
#SBATCH --mem-per-cpu=$(slurm["mem_per_cpu"])
#SBATCH --time=$(slurm["time"])
#SBATCH --output=$(log_dir)/$(job_name)_%j.out
#SBATCH --error=$(log_dir)/$(job_name)_%j.err
#SBATCH --mail-type=$(slurm["mail_type"])
#SBATCH --mail-user=$(slurm["mail_user"])

mkdir -p $(log_dir)
mkdir -p $(output_dir)

module load $(slurm["julia_module"])

export JULIA_CONDAPKG_BACKEND=Null

julia --project=$(project_dir) -e 'using Pkg; Pkg.instantiate(); Pkg.add(["CairoMakie", "PairPlots", "Distributions", "Unitful", "UnitfulAstro", "Pigeons"])' 2>&1 | tee $(log_dir)/instantiate_\${SLURM_JOB_ID}.log

julia --project=$(project_dir) -t $(slurm["julia_threads"]) \\
    $(fitting_script) \\
    $(abs_config) 2>&1 | tee $(log_dir)/output_\${SLURM_JOB_ID}.log
"""

# ── Write the script ───────────────────────────────────────────────────

mkpath(log_dir)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
script_path = joinpath(log_dir, "job_$(timestamp).sh")
write(script_path, script)
println("Generated Slurm script: $script_path")

# ── Submit (or not) ────────────────────────────────────────────────────

if dry_run
    println("Dry run — not submitting. Inspect the script above, then run:")
    println("  sbatch $script_path")
else
    run(`sbatch $script_path`)
end
