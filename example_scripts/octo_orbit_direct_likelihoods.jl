"""
Orbital Models of High Velocity Stars in Omega Centauri
Using Octofitter — Direct PM & Acceleration Likelihoods

Uses direct single-epoch position, proper motion, and acceleration
likelihoods instead of synthetic multi-epoch astrometry.

Usage:
    julia --project=<Octofitter_imbh.jl> -t <threads> octo_orbit_direct_likelihoods.jl <config.toml>

If no config path is given, falls back to ../configs/default.toml.
"""

ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

using Octofitter
using Octofitter: @variables, System
using CairoMakie
using PairPlots
using Distributions
using Unitful
using UnitfulAstro
using LinearAlgebra
using Statistics
using Dates
using Pigeons

# Add the directory to LOAD_PATH
push!(LOAD_PATH, @__DIR__)
using octo_utils_julia_MCMC_centre  # local module

# Load configuration helpers and config file
include(joinpath(@__DIR__, "parse_config.jl"))
config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "configs", "default.toml")
cfg = load_config(config_path)
println("Loaded config: $config_path")

# === 1. Select stars and time config ===
star_names = cfg["stars"]["selected"]
epoch_mjd  = get_epoch_mjd(cfg)
epoch_year = cfg["epoch"]["year"]

# === 2. Build observation objects for each star ===
astrom_obs = Dict{String, Any}()
pm_obs = Dict{String, Any}()
acc_obs = Dict{String, Any}()

for name in star_names
    star = octo_utils_julia_MCMC_centre.stars[name]
    a, p, ac = octo_utils_julia_MCMC_centre.build_star_observations(star, epoch_mjd)
    astrom_obs[name] = a
    pm_obs[name] = p
    acc_obs[name] = ac
end

# === 3. Define companions ===
companions = Planet[]
for name in star_names
    # Parse priors from config (with per-star overrides)
    P_prior = parse_prior(get_companion_prior(cfg, name, "P"))
    e_prior = parse_prior(get_companion_prior(cfg, name, "e"))
    i_prior = parse_prior(get_companion_prior(cfg, name, "i"))
    ω_prior = parse_prior(get_companion_prior(cfg, name, "omega"))
    Ω_prior = parse_prior(get_companion_prior(cfg, name, "Omega"))
    θ_prior = parse_prior(get_companion_prior(cfg, name, "theta"))

    star = Planet(
        name = name,
        basis = Visual{KepOrbit},
        observations = [ObsPriorAstromONeil2019(astrom_obs[name]), pm_obs[name], acc_obs[name]],
        variables = @variables begin
            M = system.M
            P ~ $P_prior                 # Period [yrs]
            a = cbrt(M * P^2)            # Semi-major axis [AU]
            e ~ $e_prior                 # Eccentricity
            i ~ $i_prior                 # Inclination [rad]
            ω ~ $ω_prior                 # Argument of periastron [rad]
            Ω ~ $Ω_prior                 # Longitude of ascending node [rad]
            θ ~ $θ_prior                 # Mean anomaly at reference epoch [rad]
            tp = θ_at_epoch_to_tperi(θ, $epoch_mjd; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
        end
    )
    push!(companions, star)
end

# === 4. Define the full system ===
sys_priors = cfg["priors"]["system"]
plx_prior     = parse_prior(sys_priors["plx"])
M_prior       = parse_prior(sys_priors["M"])
offsetx_prior = parse_prior(sys_priors["offsetx"])
offsety_prior = parse_prior(sys_priors["offsety"])

sys = System(
    name = get(cfg["meta"], "system_name", "Omega_Cen"),
    observations = [],
    companions = companions,
    variables = @variables begin
        plx ~ $plx_prior             # Parallax [mas]
        M ~ $M_prior                 # Host mass [solar masses]
        offsetx ~ $offsetx_prior     # IMBH RA offset from assumed center [mas]
        offsety ~ $offsety_prior     # IMBH Dec offset from assumed center [mas]
    end
)

# === 5. Model ===
model = Octofitter.LogDensityModel(sys)

# === 6. Sampling config ===
sampling_cfg         = cfg["sampling"]
n_rounds             = sampling_cfg["n_rounds"]
n_chains             = sampling_cfg["n_chains"]
n_chains_variational = sampling_cfg["n_chains_variational"]
checkpoint           = get(sampling_cfg, "checkpoint", false)

# === 7. Output config ===
slurm_job_id = get(ENV, "SLURM_JOB_ID", "none")
paths_cfg    = cfg["paths"]
output_dir   = isabspath(paths_cfg["output_dir"]) ? paths_cfg["output_dir"] : joinpath(dirname(abspath(config_path)), paths_cfg["output_dir"])
stars_tag    = join(star_names, "")
run_prefix   = "stars$(stars_tag)_$(n_chains)c_$(n_rounds)r_$(slurm_job_id)"
mkpath(output_dir)

# === 8. Write run summary ===
summary_path = joinpath(output_dir, "$(run_prefix)_summary.md")
open(summary_path, "w") do io
    println(io, "# Run Summary")
    println(io)
    println(io, "- **Date:** $(Dates.now())")
    println(io, "- **Slurm Job ID:** $(slurm_job_id)")
    println(io, "- **Stars:** $(join(star_names, ", "))")
    println(io, "- **Reference epoch:** $(epoch_mjd) MJD ($(epoch_year) yr)")
    println(io, "- **Config file:** $(abspath(config_path))")
    println(io)
    println(io, "## Sampling Parameters")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|---|---|")
    println(io, "| n_rounds | $(n_rounds) |")
    println(io, "| n_chains | $(n_chains) |")
    println(io, "| n_chains_variational | $(n_chains_variational) |")
    println(io, "| checkpoint | $(checkpoint) |")
    println(io)
    println(io, "## System Priors")
    println(io)
    println(io, "| Parameter | Prior |")
    println(io, "|---|---|")
    for (k, v) in sys_priors
        println(io, "| $(k) | $(v) |")
    end
    println(io)
    println(io, "## Companion Priors (defaults)")
    println(io)
    println(io, "| Parameter | Prior |")
    println(io, "|---|---|")
    for (k, v) in cfg["priors"]["companion_defaults"]
        println(io, "| $(k) | $(v) |")
    end
    # Show per-star overrides if any
    overrides = get(get(cfg, "priors", Dict()), "overrides", Dict())
    if !isempty(overrides)
        println(io)
        println(io, "## Per-Star Prior Overrides")
        println(io)
        for (star, params) in overrides
            for (k, v) in params
                println(io, "- **Star $(star)**: $(k) = $(v)")
            end
        end
    end
    println(io)
    println(io, "## Full Configuration")
    println(io)
    println(io, "```toml")
    println(io, read(config_path, String))
    println(io, "```")
end
println("Run summary written to $(summary_path)")

# === 9. Fit with Pigeons ===
chain, pt = octofit_pigeons(model; n_rounds, n_chains, n_chains_variational, checkpoint)
println(chain)

# === 10. Save Chain ===
Octofitter.savechain(joinpath(output_dir, "$(run_prefix)_chain.fits"), chain)

# === 11. Corner Plot ===
corner_plot = octocorner(model, chain; small=true)
save(joinpath(output_dir, "$(run_prefix)_corner.png"), corner_plot)

# === 12. Orbit Plot ===
orbit_plot = octoplot(model, chain; show_physical_orbit=true, colorbar=true)
save(joinpath(output_dir, "$(run_prefix)_orbit.png"), orbit_plot)

# === 13. Orbit Plot (zoomed) ===
ts = Octofitter.range(54600, 55700, length=200)
orbit_plot_2 = octoplot(model, chain; show_physical_orbit=true, colorbar=false, figscale=1.5, ts=ts)

ax_orbit = orbit_plot_2.content[1]
xlims!(ax_orbit, -200, 200)
ylims!(ax_orbit, -100, 100)
ax_orbit.title = "Orbits of Fast-Moving Stars in ω Cen (Direct Likelihoods)"

save(joinpath(output_dir, "$(run_prefix)_orbit_zoomed.png"), orbit_plot_2)
