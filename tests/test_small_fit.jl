"""
Small-scale fit test: 2 stars, short chain, verify sampling completes
and produces reasonable posteriors.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_small_fit.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using CairoMakie
using Statistics
using Printf

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
println("Model compiled. Number of free parameters: $(model.D)")

println("\nStarting short MCMC run (500 iterations, 200 adaptation)...")
chain = octofit(model; iterations=500, adaptation=200)

# === Extract samples ===
M_samples   = vec(chain[:M])
plx_samples = vec(chain[:plx])
ox_samples  = vec(chain[:offsetx])
oy_samples  = vec(chain[:offsety])
# Star A
aA_samples = vec(chain[:A_a]);  eA_samples = vec(chain[:A_e])
iA_samples = vec(chain[:A_i]);  ωA_samples = vec(chain[:A_ω]);  ΩA_samples = vec(chain[:A_Ω])
# Star C
aC_samples = vec(chain[:C_a]);  eC_samples = vec(chain[:C_e])
iC_samples = vec(chain[:C_i]);  ωC_samples = vec(chain[:C_ω]);  ΩC_samples = vec(chain[:C_Ω])

# === Print diagnostics ===
println("\n=== Posterior summaries (median, 68% CI) ===")
@printf("%-18s  %10s  [%8s, %8s]\n", "Param", "Median", "16%", "84%")
function print_stat(name, samples; scale=1.0, fmt="%10.3f")
    med = median(samples) * scale
    lo  = quantile(samples, 0.16) * scale
    hi  = quantile(samples, 0.84) * scale
    @printf("%-18s  %10.3f  [%8.3f, %8.3f]\n", name, med, lo, hi)
end
print_stat("M_IMBH [10⁴ M☉]", M_samples; scale=1e-4)
print_stat("plx [mas]",        plx_samples)
print_stat("offsetx [mas]",    ox_samples)
print_stat("offsety [mas]",    oy_samples)
print_stat("A: a [AU]",        aA_samples)
print_stat("A: e",             eA_samples)
print_stat("A: i [°]",         iA_samples; scale=180/π)
print_stat("A: ω [°]",         ωA_samples; scale=180/π)
print_stat("A: Ω [°]",         ΩA_samples; scale=180/π)
print_stat("C: a [AU]",        aC_samples)
print_stat("C: e",             eC_samples)
print_stat("C: i [°]",         iC_samples; scale=180/π)
print_stat("C: ω [°]",         ωC_samples; scale=180/π)
print_stat("C: Ω [°]",         ΩC_samples; scale=180/π)

# === Orbit plot ===
println("\nGenerating orbit plot...")
fig_orbit = octoplot(model, chain; show_physical_orbit=true, colorbar=false)
save("test_small_fit_orbits.png", fig_orbit, px_per_unit=3)
println("Orbit plot saved to test_small_fit_orbits.png")

# === Posterior figure: one panel per free parameter ===
println("Generating posterior figure...")

function param_panel!(layout, row, col, cidx, samples, xlabel; show_legend=false)
    ax = Axis(layout[row, col]; xlabel=xlabel, ylabel="Probability Density",
              xgridvisible=false, ygridvisible=false)
    med = median(samples)
    hist!(ax, samples; normalization=:pdf, bins=30,
          color=(Makie.wong_colors()[cidx], 0.7))
    vlines!(ax, [med]; color=Makie.wong_colors()[2], linestyle=:solid, label="Median")
    show_legend && axislegend(ax; position=:rt, framevisible=false)
end

# 3×5 layout: system params (row 1), star A (row 2), star C (row 3)
fig_post = Figure(size=(1600, 780), fontsize=18)
# Row 1: system-level parameters (4 panels; col 5 left empty)
param_panel!(fig_post, 1, 1, 1, M_samples ./ 1e4,
    Makie.rich("M", Makie.subscript("IMBH"), " [10⁴ M", Makie.subscript("☉"), "]");
    show_legend=true)
param_panel!(fig_post, 1, 2, 1, plx_samples, "plx [mas]")
param_panel!(fig_post, 1, 3, 1, ox_samples,
    Makie.rich("Δα*", Makie.subscript("IMBH"), " [mas]"))
param_panel!(fig_post, 1, 4, 1, oy_samples,
    Makie.rich("Δδ", Makie.subscript("IMBH"), " [mas]"))
# Row 2: star A orbital parameters
param_panel!(fig_post, 2, 1, 3, aA_samples,           "A: a [AU]")
param_panel!(fig_post, 2, 2, 3, eA_samples,           "A: e")
param_panel!(fig_post, 2, 3, 3, rad2deg.(iA_samples), "A: i [°]")
param_panel!(fig_post, 2, 4, 3, rad2deg.(ωA_samples), "A: ω [°]")
param_panel!(fig_post, 2, 5, 3, rad2deg.(ΩA_samples), "A: Ω [°]")
# Row 3: star C orbital parameters
param_panel!(fig_post, 3, 1, 4, aC_samples,           "C: a [AU]")
param_panel!(fig_post, 3, 2, 4, eC_samples,           "C: e")
param_panel!(fig_post, 3, 3, 4, rad2deg.(iC_samples), "C: i [°]")
param_panel!(fig_post, 3, 4, 4, rad2deg.(ωC_samples), "C: ω [°]")
param_panel!(fig_post, 3, 5, 4, rad2deg.(ΩC_samples), "C: Ω [°]")
save("test_small_fit_posteriors.png", fig_post, px_per_unit=3)
println("Posterior figure saved to test_small_fit_posteriors.png")

println("\n=== Small fit test completed successfully ===")
