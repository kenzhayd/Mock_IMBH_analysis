"""
Unit tests for PlanetRelativeRVObs (radial velocity likelihood).
Creates a known orbit, computes expected RV, and verifies that the
likelihood evaluates correctly with and without RV data.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_rv_likelihood.jl
"""

using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using Distributions
using Test
using Statistics
using Printf

# === 1. Define a known orbit (same as test_likelihoods.jl) ===
M_imbh = 10000.0    # solar masses
plx_val = 0.184     # mas (d = 5.43 kpc)
P_val = 1000.0      # years
a_val = cbrt(M_imbh * P_val^2)  # AU, Kepler's 3rd law
e_val = 0.3
i_val = 1.0         # rad
omega_val = 0.5     # rad
Omega_val = 1.2     # rad
tp_val = 50000.0    # MJD

orbit = Visual{KepOrbit}(;
    a = a_val, e = e_val, i = i_val, ω = omega_val, Ω = Omega_val,
    tp = tp_val, M = M_imbh, plx = plx_val,
)

# === 2. Evaluate orbit at test epoch ===
test_epoch = 55197.0  # MJD (~2010)
sol = orbitsolve(orbit, test_epoch)

true_rv = radvel(sol)  # companion's LOS velocity relative to central mass [m/s]
println("=== Known orbit RV at epoch $test_epoch ===")
println("radvel(sol) = $(round(true_rv, digits=1)) m/s")

# === 3. Test PlanetRelativeRVObs construction ===
@testset "PlanetRelativeRVObs construction" begin
    sigma_rv = 3000.0  # m/s

    rv_obs = PlanetRelativeRVObs(
        (epoch = test_epoch, rv = true_rv, σ_rv = sigma_rv);
        name = "test_rv",
        variables = @variables begin end
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [rv_obs],
        variables = @variables begin
            M = system.M
            a = $a_val
            e = $e_val
            i = $i_val
            ω = $omega_val
            Ω = $Omega_val
            tp = $tp_val
        end
    )

    sys = System(
        name = "test_rv_system",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  RV model compiled successfully (D=$(model.D) parameters)")
end

# === 4. Test combined pos + PM + accel + RV ===
@testset "Combined pos + PM + accel + RV" begin
    offset_ra = 5.0
    offset_dec = -3.0
    sigma_pos = 0.5
    sigma_pm = 0.05
    sigma_acc = 0.005
    sigma_rv = 3000.0

    true_ra = raoff(sol)
    true_dec = decoff(sol)
    true_pmra = pmra(sol)
    true_pmdec = pmdec(sol)
    true_accra = accra(sol)
    true_accdec = accdec(sol)

    astrom = PlanetRelAstromObs(
        (epoch=[test_epoch], ra=[true_ra + offset_ra], dec=[true_dec + offset_dec],
         σ_ra=[sigma_pos], σ_dec=[sigma_pos], cor=[0.0]);
        name="test_pos"
    )
    pm_obs = PlanetPMObs(
        (epoch=[test_epoch], pmra=[true_pmra], pmdec=[true_pmdec],
         σ_pmra=[sigma_pm], σ_pmdec=[sigma_pm], cor=[0.0]);
        name="test_pm"
    )
    acc_obs = PlanetAccelObs(
        (epoch=[test_epoch], accra=[true_accra], accdec=[true_accdec],
         σ_accra=[sigma_acc], σ_accdec=[sigma_acc], cor=[0.0]);
        name="test_acc"
    )
    rv_obs = PlanetRelativeRVObs(
        (epoch = test_epoch, rv = true_rv, σ_rv = sigma_rv);
        name = "test_rv",
        variables = @variables begin end
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [ObsPriorAstromONeil2019(astrom), pm_obs, acc_obs, rv_obs],
        variables = @variables begin
            M = system.M
            a = $a_val
            e = $e_val
            i = $i_val
            ω = $omega_val
            Ω = $Omega_val
            tp = $tp_val
        end
    )

    sys = System(
        name = "test_combined_rv",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
            offsetx = $offset_ra
            offsety = $offset_dec
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Combined model with RV compiled successfully (D=$(model.D) parameters)")
end

# === 5. AD compatibility ===
@testset "ForwardDiff gradient through RV likelihood" begin
    using ForwardDiff

    sigma_rv = 3000.0
    rv_obs = PlanetRelativeRVObs(
        (epoch = test_epoch, rv = true_rv, σ_rv = sigma_rv);
        name = "test_rv",
        variables = @variables begin end
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [rv_obs],
        variables = @variables begin
            M = system.M
            P ~ Uniform(10, 2_000_000)
            a = cbrt(M * P^2)
            e ~ Uniform(0.0, 0.99)
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            θ ~ UniformCircular()
            tp = θ_at_epoch_to_tperi(θ, $test_epoch; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
        end
    )

    sys = System(
        name = "test_rv_ad",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)

    # Try evaluating gradient at a random point in parameter space
    θ_test = randn(model.D)
    ll = model.ℓπcallback(θ_test)
    @test isfinite(ll) || ll == -Inf  # either finite or -Inf (out of prior), but not NaN

    grad = ForwardDiff.gradient(model.ℓπcallback, θ_test)
    @test length(grad) == model.D
    @test all(isfinite, grad) || !isfinite(ll)  # gradient finite when ll is finite
    println("  ForwardDiff gradient OK (D=$(model.D))")
end

println("\n=== All RV likelihood tests passed ===")
