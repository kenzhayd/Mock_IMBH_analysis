"""
Small-scale fit test: 2 stars, short chain, verify sampling completes
and produces reasonable posteriors.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_small_fit.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using CairoMakie

# === Data for stars A and C (from omega Cen) ===
# Cluster center
ra_cm_deg  = 201.6970988
dec_cm_deg = -47.4794533
epoch_mjd  = 55197.0  # ~2010

# Star A
ra_A_rel  = (201.6967263 - ra_cm_deg) * 3600 * 1000  # mas
dec_A_rel = (-47.4795835 - dec_cm_deg) * 3600 * 1000

astrom_A = PlanetRelAstromObs(
    (epoch=epoch_mjd, ra=ra_A_rel, dec=dec_A_rel, σ_ra=0.5, σ_dec=0.5, cor=0.0);
    name="A_pos"
)
pm_A = PlanetPMObs(
    (epoch=epoch_mjd, pmra=3.563, pmdec=2.564, σ_pmra=0.038, σ_pmdec=0.055, cor=0.0);
    name="A_pm"
)
acc_A = PlanetAccelObs(
    (epoch=epoch_mjd, accra=-0.0069, accdec=0.0085, σ_accra=0.0083, σ_accdec=0.0098, cor=0.0);
    name="A_acc"
)

# Star C
ra_C_rel  = (201.6966378 - ra_cm_deg) * 3600 * 1000
dec_C_rel = (-47.4793672 - dec_cm_deg) * 3600 * 1000

astrom_C = PlanetRelAstromObs(
    (epoch=epoch_mjd, ra=ra_C_rel, dec=dec_C_rel, σ_ra=0.5, σ_dec=0.5, cor=0.0);
    name="C_pos"
)
pm_C = PlanetPMObs(
    (epoch=epoch_mjd, pmra=1.117, pmdec=3.514, σ_pmra=0.127, σ_pmdec=0.056, cor=0.0);
    name="C_pm"
)
acc_C = PlanetAccelObs(
    (epoch=epoch_mjd, accra=0.0028, accdec=-0.0060, σ_accra=0.0333, σ_accdec=0.0123, cor=0.0);
    name="C_acc"
)

# === Define planets ===
planet_A = Planet(
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

planet_C = Planet(
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
    companions = [planet_A, planet_C],
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
