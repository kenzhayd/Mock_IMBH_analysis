"""
Orbital Models of High Velocity Stars in Omega Centauri
Using Octofitter — Direct PM & Acceleration Likelihoods

Uses direct single-epoch position, proper motion, and acceleration
likelihoods instead of synthetic multi-epoch astrometry.
"""

# Environment variables
ENV["JULIA_NUM_THREADS"] = "192"
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
push!(LOAD_PATH, raw"/home/vhenault/projects/def-vhenault/vhenault/octo/")
using octo_utils_julia_MCMC_centre  # local module

# === 1. Select stars and time config ===
star_names = ["A", "C", "D", "E", "F"]  # B and G excluded by default (lower quality data)
# star_names = ["A", "B", "C", "D", "E", "F", "G"]  # uncomment to include all stars
epoch_mjd = Octofitter.years2mjd(2010.0)

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
    star = Planet(
        name = name,
        basis = Visual{KepOrbit},
        observations = [ObsPriorAstromONeil2019(astrom_obs[name]), pm_obs[name], acc_obs[name]],
        variables = @variables begin
            M = system.M
            P ~ Uniform(10, 2_000_000)       # Period [yrs]
            a = cbrt(M * P^2)                # Semi-major axis [AU]
            e ~ Uniform(0.0, 0.99)           # Eccentricity
            i ~ Sine()                       # Inclination [rad]
            ω ~ UniformCircular()            # Argument of periastron [rad]
            Ω ~ UniformCircular()            # Longitude of ascending node [rad]
            θ ~ UniformCircular()            # Mean anomaly at reference epoch [rad]
            tp = θ_at_epoch_to_tperi(θ, $epoch_mjd; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
        end
    )
    push!(companions, star)
end

# === 4. Define the full system ===
sys = System(
    name = "Omega_Cen",
    observations = [],
    companions = companions,
    variables = @variables begin
        plx ~ truncated(Normal(0.19, 0.004), lower=0)  # Parallax [mas]
        M ~ Uniform(100, 120000)                        # Host mass [solar masses]
        offsetx ~ Normal(0, 10)                         # IMBH RA offset from assumed center [mas]
        offsety ~ Normal(0, 10)                         # IMBH Dec offset from assumed center [mas]
    end
)

# === 5. Model ===
model = Octofitter.LogDensityModel(sys)

# === 6. Sampling config ===
n_rounds = 18
n_chains = 192
n_chains_variational = 192

# === 7. Output config ===
slurm_job_id = get(ENV, "SLURM_JOB_ID", "none")
output_dir = "/home/vhenault/projects/def-vhenault/vhenault/Ocen_IMBH_analysis/run_outputs"
stars_tag = join(star_names, "")
run_prefix = "stars$(stars_tag)_$(n_chains)c_$(n_rounds)r_$(slurm_job_id)"
mkpath(output_dir)

# === 8. Write run summary ===
summary_path = "$(output_dir)/$(run_prefix)_summary.md"
open(summary_path, "w") do io
    println(io, "# Run Summary")
    println(io)
    println(io, "- **Date:** $(Dates.now())")
    println(io, "- **Slurm Job ID:** $(slurm_job_id)")
    println(io, "- **Stars:** $(join(star_names, ", "))")
    println(io, "- **Reference epoch:** $(epoch_mjd) MJD ($(2010.0) yr)")
    println(io)
    println(io, "## Sampling Parameters")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|---|---|")
    println(io, "| n_rounds | $(n_rounds) |")
    println(io, "| n_chains | $(n_chains) |")
    println(io, "| n_chains_variational | $(n_chains_variational) |")
    println(io)
    println(io, "## System Priors")
    println(io)
    println(io, "| Parameter | Prior |")
    println(io, "|---|---|")
    println(io, "| plx | truncated(Normal(0.19, 0.004), lower=0) |")
    println(io, "| M | Uniform(100, 120000) |")
    println(io, "| offsetx | Normal(0, 10) |")
    println(io, "| offsety | Normal(0, 10) |")
    println(io)
    println(io, "## Companion Priors (same for all stars)")
    println(io)
    println(io, "| Parameter | Prior |")
    println(io, "|---|---|")
    println(io, "| P | Uniform(10, 2_000_000) |")
    println(io, "| e | Uniform(0.0, 0.99) |")
    println(io, "| i | Sine() |")
    println(io, "| ω | UniformCircular() |")
    println(io, "| Ω | UniformCircular() |")
    println(io, "| θ | UniformCircular() |")
end
println("Run summary written to $(summary_path)")

# === 9. Fit with Pigeons ===
chain, pt = octofit_pigeons(model; n_rounds, n_chains, n_chains_variational, checkpoint=false)
println(chain)

# === 10. Save Chain ===
Octofitter.savechain("$(output_dir)/$(run_prefix)_chain.fits", chain)

# === 11. Corner Plot ===
corner_plot = octocorner(model, chain; small=true)
save("$(output_dir)/$(run_prefix)_corner.png", corner_plot)

# === 12. Orbit Plot ===
orbit_plot = octoplot(model, chain; show_physical_orbit=true, colorbar=true)
save("$(output_dir)/$(run_prefix)_orbit.png", orbit_plot)

# === 13. Orbit Plot (zoomed) ===
ts = Octofitter.range(54600, 55700, length=200)
orbit_plot_2 = octoplot(model, chain; show_physical_orbit=true, colorbar=false, figscale=1.5, ts=ts)

ax_orbit = orbit_plot_2.content[1]
xlims!(ax_orbit, -200, 200)
ylims!(ax_orbit, -100, 100)
ax_orbit.title = "Orbits of Fast-Moving Stars in ω Cen (Direct Likelihoods)"

save("$(output_dir)/$(run_prefix)_orbit_zoomed.png", orbit_plot_2)
