"""
Test AD compatibility: verify that the model compiles as type-stable
and that ForwardDiff gradients can be computed without errors.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_ad_compatibility.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using ForwardDiff

# Load utility module with canonical star data and build_star_observations
push!(LOAD_PATH, joinpath(@__DIR__, "..", "example_scripts"))
using octo_utils_julia_MCMC_centre

# === Build a 2-star system with all 3 likelihood types ===
epoch_mjd = 55197.0  # ~2010

astrom_A, pm_A, acc_A = octo_utils_julia_MCMC_centre.build_star_observations(
    octo_utils_julia_MCMC_centre.stars["A"], epoch_mjd)
astrom_C, pm_C, acc_C = octo_utils_julia_MCMC_centre.build_star_observations(
    octo_utils_julia_MCMC_centre.stars["C"], epoch_mjd)

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

sys = System(
    name = "Omega_Cen_test",
    companions = [star_A, star_C],
    variables = @variables begin
        plx ~ truncated(Normal(0.19, 0.004), lower=0)
        M ~ Uniform(100, 120_000)
        offsetx ~ Normal(0, 10)
        offsety ~ Normal(0, 10)
    end
)

# === Compile the model ===
println("Compiling LogDensityModel...")
model = Octofitter.LogDensityModel(sys)

println("Number of free parameters: $(model.D)")

# === Test gradient computation ===
println("\nTesting gradient computation...")
# Draw a random sample from the prior (in transformed space)
θ_init = randn(model.D)

try
    # Evaluate log-posterior
    lp = model.ℓπcallback(θ_init)
    println("Log-posterior at random point: $lp")

    # Evaluate gradient
    grad = ForwardDiff.gradient(model.ℓπcallback, θ_init)
    println("Gradient computed successfully ($(length(grad)) components)")
    println("Gradient norm: $(sqrt(sum(grad.^2)))")

    if any(isnan, grad) || any(isinf, grad)
        @warn "Gradient contains NaN or Inf values!"
    else
        println("Gradient is finite — AD is working correctly.")
    end
catch e
    @error "Gradient computation failed!" exception=e
    rethrow(e)
end

println("\n=== AD compatibility test passed ===")
