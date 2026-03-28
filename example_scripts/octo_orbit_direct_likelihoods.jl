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
using Printf
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

# === 12. Extract posterior samples ===
M_samples   = vec(chain[:M])
plx_samples = vec(chain[:plx])
ox_samples  = vec(chain[:offsetx])
oy_samples  = vec(chain[:offsety])

star_samples = Dict{String, NamedTuple}()
for name in star_names
    star_samples[name] = (
        a  = vec(chain[Symbol("$(name)_a")]),
        e  = vec(chain[Symbol("$(name)_e")]),
        i  = vec(chain[Symbol("$(name)_i")]),
        ω  = vec(chain[Symbol("$(name)_ω")]),
        Ω  = vec(chain[Symbol("$(name)_Ω")]),
        tp = vec(chain[Symbol("$(name)_tp")]),
    )
end

# === 13. Posterior summaries ===
println("\n=== Posterior summaries (median, 68% CI) ===")
@printf("%-20s  %10s  [%8s, %8s]\n", "Param", "Median", "16%", "84%")
function print_stat(label, samples; scale=1.0)
    med = median(samples) * scale
    lo  = quantile(samples, 0.16) * scale
    hi  = quantile(samples, 0.84) * scale
    @printf("%-20s  %10.3f  [%8.3f, %8.3f]\n", label, med, lo, hi)
end
print_stat("M_IMBH [10⁴ M☉]", M_samples; scale=1e-4)
print_stat("plx [mas]",        plx_samples)
print_stat("offsetx [mas]",    ox_samples)
print_stat("offsety [mas]",    oy_samples)
for name in star_names
    s = star_samples[name]
    print_stat("$(name): a [AU]", s.a)
    print_stat("$(name): e",      s.e)
    print_stat("$(name): i [°]",  s.i; scale=180/π)
    print_stat("$(name): ω [°]",  s.ω; scale=180/π)
    print_stat("$(name): Ω [°]",  s.Ω; scale=180/π)
end

# === 14. Sky-plane orbit panels (one per star) ===
println("\nGenerating orbit panels...")

sample_idx = round.(Int, range(1, length(M_samples), length=100))

function star_orbit_panel!(ax, s, M_samp, plx_samp, ox_samp, oy_samp,
                            obs_ra, obs_dec, epoch_mjd, sample_idx;
                            scale_pm=50.0, scale_acc=5000.0)
    ox_med_loc = median(ox_samp)
    oy_med_loc = median(oy_samp)
    # Time grid spanning one median period
    P_med_yr = median(@. s.a ^ 1.5 / sqrt(M_samp))
    ts = range(epoch_mjd - P_med_yr * 365.25 / 2,
               epoch_mjd + P_med_yr * 365.25 / 2; length=300)
    # Median orbit for vector computation
    orb_med = Visual{KepOrbit}(;
        a=median(s.a), e=median(s.e), i=median(s.i),
        ω=median(s.ω), Ω=median(s.Ω), tp=median(s.tp),
        M=median(M_samp), plx=median(plx_samp))
    sol_med = orbitsolve(orb_med, epoch_mjd)
    # Posterior orbit samples in sky frame (orbit + per-sample IMBH offset)
    for idx in sample_idx
        orb_s = Visual{KepOrbit}(;
            a=s.a[idx], e=s.e[idx], i=s.i[idx],
            ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
            M=M_samp[idx], plx=plx_samp[idx])
        ra_s  = [raoff(orbitsolve(orb_s, t)) + ox_samp[idx] for t in ts]
        dec_s = [decoff(orbitsolve(orb_s, t)) + oy_samp[idx] for t in ts]
        lines!(ax, ra_s, dec_s; color=(:gray, 0.5), linewidth=0.5)
    end
    # Coordinate origin — black cross
    scatter!(ax, [0.0], [0.0]; marker=:cross, markersize=16, color=:black)
    # Inferred IMBH position — filled black circle
    scatter!(ax, [ox_med_loc], [oy_med_loc]; marker=:circle, markersize=12, color=:black)
    # Instantaneous PM and acceleration vectors from posterior median
    arrows!(ax, [obs_ra], [obs_dec],
        [pmra(sol_med) * scale_pm], [pmdec(sol_med) * scale_pm];
        color=:royalblue, linewidth=2.0, arrowsize=10)
    arrows!(ax, [obs_ra], [obs_dec],
        [accra(sol_med) * scale_acc], [accdec(sol_med) * scale_acc];
        color=:firebrick, linewidth=2.0, arrowsize=10)
    # Observed star position — drawn last so star sits on top
    scatter!(ax, [obs_ra], [obs_dec];
        marker='★', color=Makie.wong_colors()[2], markersize=14,
        strokecolor=:black, strokewidth=0.5, label="Observed position")
    axislegend(ax; position=:rt, framevisible=false)
end

n_stars    = length(star_names)
n_cols_orb = min(n_stars, 3)
n_rows_orb = ceil(Int, n_stars / n_cols_orb)
fig_orbits = Figure(size=(n_cols_orb * 420, n_rows_orb * 440), fontsize=18)

for (k, name) in enumerate(star_names)
    row = ceil(Int, k / n_cols_orb)
    col = mod1(k, n_cols_orb)
    ax  = Axis(fig_orbits[row, col];
        xlabel="Δα* [mas]", ylabel="Δδ [mas]", title="Star $name",
        xreversed=true, autolimitaspect=1,
        xgridvisible=false, ygridvisible=false)
    star_orbit_panel!(ax, star_samples[name], M_samples, plx_samples,
        ox_samples, oy_samples,
        astrom_obs[name].table.ra[1], astrom_obs[name].table.dec[1],
        epoch_mjd, sample_idx)
end
save(joinpath(output_dir, "$(run_prefix)_orbit_panels.png"), fig_orbits, px_per_unit=3)
println("Orbit panels saved to $(run_prefix)_orbit_panels.png")

# === 15. Posterior panels (one per free parameter) ===
println("Generating posterior panels...")

function param_panel!(layout, row, col, cidx, samples, xlabel; show_legend=false)
    ax = Axis(layout[row, col]; xlabel=xlabel, ylabel="Probability Density",
              xgridvisible=false, ygridvisible=false)
    med = median(samples)
    hist!(ax, samples; normalization=:pdf, bins=30,
          color=(Makie.wong_colors()[cidx], 0.7))
    vlines!(ax, [med]; color=Makie.wong_colors()[2], linestyle=:solid, label="Median")
    show_legend && axislegend(ax; position=:rt, framevisible=false)
end

# Layout: row 1 = system params, rows 2..N+1 = one per star (5 orbital params each)
fig_post = Figure(size=(1600, (1 + n_stars) * 260), fontsize=18)

# Row 1: system-level parameters
param_panel!(fig_post, 1, 1, 1, M_samples ./ 1e4,
    Makie.rich("M", Makie.subscript("IMBH"), " [10⁴ M", Makie.subscript("☉"), "]");
    show_legend=true)
param_panel!(fig_post, 1, 2, 1, plx_samples, "plx [mas]")
param_panel!(fig_post, 1, 3, 1, ox_samples,
    Makie.rich("Δα*", Makie.subscript("IMBH"), " [mas]"))
param_panel!(fig_post, 1, 4, 1, oy_samples,
    Makie.rich("Δδ", Makie.subscript("IMBH"), " [mas]"))

# Per-star rows: a, e, i, ω, Ω
star_cidx = [3, 4, 5, 6, 7]
for (k, name) in enumerate(star_names)
    cidx = star_cidx[mod1(k, length(star_cidx))]
    row  = k + 1
    s    = star_samples[name]
    param_panel!(fig_post, row, 1, cidx, s.a,           "$(name): a [AU]")
    param_panel!(fig_post, row, 2, cidx, s.e,           "$(name): e")
    param_panel!(fig_post, row, 3, cidx, rad2deg.(s.i), "$(name): i [°]")
    param_panel!(fig_post, row, 4, cidx, rad2deg.(s.ω), "$(name): ω [°]")
    param_panel!(fig_post, row, 5, cidx, rad2deg.(s.Ω), "$(name): Ω [°]")
end
save(joinpath(output_dir, "$(run_prefix)_posteriors.png"), fig_post, px_per_unit=3)
println("Posterior panels saved to $(run_prefix)_posteriors.png")
