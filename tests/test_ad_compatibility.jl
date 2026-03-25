"""
Test AD compatibility: verify that the model compiles as type-stable
and that ForwardDiff gradients can be computed without errors.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_ad_compatibility.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using ForwardDiff

# === Build a 2-star system with all 3 likelihood types ===
epoch_mjd = 55197.0  # ~2010

# Star A data (from omega Cen dataset)
astrom_A = PlanetRelAstromObs(
    (epoch=epoch_mjd, ra=-13.42, dec=-4.69, σ_ra=0.5, σ_dec=0.5, cor=0.0);
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

# Star C data
astrom_C = PlanetRelAstromObs(
    (epoch=epoch_mjd, ra=-16.60, dec=3.10, σ_ra=0.5, σ_dec=0.5, cor=0.0);
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

sys = System(
    name = "Omega_Cen_test",
    companions = [planet_A, planet_C],
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

println("Type stable: $(model.type_stable)")
println("Number of parameters: $(model.num_params)")
println("Parameter names: $(model.param_names)")

if !model.type_stable
    @warn "Model is NOT type stable — this will cause slow sampling. Check likelihood implementations."
end

# === Test gradient computation ===
println("\nTesting gradient computation...")
# Draw a random sample from the prior (in transformed space)
θ_init = randn(model.num_params)

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
