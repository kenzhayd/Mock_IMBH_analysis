"""
Small-scale fit test: 2 stars, short chain, verify sampling completes
and produces reasonable posteriors.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_small_fit.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using CairoMakie

# Load utility module with canonical star data and build_star_observations
push!(LOAD_PATH, joinpath(@__DIR__, "..", "example_scripts"))
using octo_utils_julia_MCMC_centre

epoch_mjd = 55197.0  # ~2010

astrom_A, pm_A, acc_A = octo_utils_julia_MCMC_centre.build_star_observations(
    octo_utils_julia_MCMC_centre.stars["A"], epoch_mjd)
astrom_C, pm_C, acc_C = octo_utils_julia_MCMC_centre.build_star_observations(
    octo_utils_julia_MCMC_centre.stars["C"], epoch_mjd)

# === Define planets ===
star_A = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [ObsPriorAstromONeil2019(astrom_A), pm_A, acc_A],
    variables = @variables begin
        M = system.M
        P ~ Uniform(10, 2_000_000)
        a = cbrt(M * P^2)
        e ~ Uniform(0.0, 0.99)
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, $epoch_mjd; a, e, i, ω, Ω, M)
    end
)

star_C = Planet(
    name = "C",
    basis = Visual{KepOrbit},
    observations = [ObsPriorAstromONeil2019(astrom_C), pm_C, acc_C],
    variables = @variables begin
        M = system.M
        P ~ Uniform(10, 2_000_000)
        a = cbrt(M * P^2)
        e ~ Uniform(0.0, 0.99)
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, $epoch_mjd; a, e, i, ω, Ω, M)
    end
)

# === Define system ===
sys = System(
    name = "Omega_Cen_2star",
    companions = [star_A, star_C],
    variables = @variables begin
        plx ~ truncated(Normal(0.19, 0.004), lower=0)
        M ~ Uniform(100, 120_000)
        offsetx ~ Normal(0, 10)
        offsety ~ Normal(0, 10)
    end
)

# === Compile and fit ===
model = Octofitter.LogDensityModel(sys)
println("Model compiled. Type stable: $(model.type_stable)")
println("Number of parameters: $(model.num_params)")

# Short MCMC run (HMC/NUTS, not Pigeons)
println("\nStarting short MCMC run (200 steps, 1 chain)...")
chain = octofit(model; iterations=200, adaptation=100, n_chains=1)

# === Basic diagnostics ===
println("\n=== Chain summary ===")
println(chain)

# Check key parameters
println("\n=== Key parameter posteriors ===")
M_samples = chain[:M][:]
println("M (IMBH mass): median=$(round(median(M_samples), digits=1)), " *
        "95% CI = [$(round(quantile(M_samples, 0.025), digits=1)), $(round(quantile(M_samples, 0.975), digits=1))]")

offsetx_samples = chain[:offsetx][:]
println("offsetx: median=$(round(median(offsetx_samples), digits=2)) mas")

offsety_samples = chain[:offsety][:]
println("offsety: median=$(round(median(offsety_samples), digits=2)) mas")

# === Quick orbit plot ===
println("\nGenerating orbit plot...")
fig = octoplot(model, chain; show_physical_orbit=true, colorbar=false)
save("test_small_fit_orbits.png", fig, px_per_unit=3)
println("Orbit plot saved to test_small_fit_orbits.png")

println("\n=== Small fit test completed successfully ===")
