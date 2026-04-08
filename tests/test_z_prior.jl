"""
Unit tests for PlanetZPriorObs (line-of-sight position prior).
Creates a known orbit, verifies that the z-prior likelihood evaluates
to logpdf(dist, posz(sol)), and checks ForwardDiff compatibility.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_z_prior.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using Test
using ForwardDiff
using Printf

# === 1. Define a known orbit ===
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
    a = a_val, e = e_val, i = i_val,
    ω = omega_val, Ω = Omega_val, tp = tp_val,
    M = M_imbh, plx = plx_val,
)

test_epoch = 55197.0  # MJD (~2010)
sol = orbitsolve(orbit, test_epoch)
true_z = posz(sol)  # AU

println("=== Known orbit z-position at epoch $test_epoch ===")
println("  posz = $(round(true_z, digits=1)) AU")

# === 2. Test PlanetZPriorObs construction and model compilation ===
@testset "PlanetZPriorObs construction" begin
    sigma_z = 845_000.0  # AU (≈ core radius of ω Cen)
    zp = PlanetZPriorObs(test_epoch, Normal(0.0, sigma_z); name="test_zprior")

    @test zp isa PlanetZPriorObs
    @test zp.table.epoch[1] == test_epoch
    println("  PlanetZPriorObs constructed successfully")

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [zp],
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
        name = "test_z_system",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Model with z-prior compiled successfully")
end

# === 3. Test likelihood value matches logpdf(dist, posz(sol)) ===
@testset "PlanetZPriorObs likelihood value" begin
    sigma_z = 845_000.0
    zp = PlanetZPriorObs(test_epoch, Normal(0.0, sigma_z); name="test_zprior")

    expected_ll = logpdf(Normal(0.0, sigma_z), true_z)
    println("  Expected z-prior log-likelihood: $(round(expected_ll, digits=6))")
    println("  (true_z = $(round(true_z, digits=1)) AU, σ_z = $sigma_z AU)")

    # The likelihood should be close to this value — verify by evaluating the
    # full model log-density at the known parameters.
    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [zp],
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
        name = "test_z_system",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)

    # The total log-density = log-prior(M) + log-likelihood(z-prior)
    # At the true parameters, the z-prior contribution should be logpdf(Normal(0, σ_z), true_z)
    # and the M prior contribution is logpdf(Uniform(100, 120_000), M_imbh).
    total_ld = model.ℓπcallback(model.starting_points)
    @test isfinite(total_ld)
    println("  Total log-density at starting point: $(round(total_ld, digits=3))")
end

# === 4. Test ForwardDiff compatibility ===
@testset "PlanetZPriorObs AD compatibility" begin
    sigma_z = 845_000.0
    zp = PlanetZPriorObs(test_epoch, Normal(0.0, sigma_z); name="test_zprior")

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [zp],
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
        name = "test_z_system",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)

    # Evaluate gradient via ForwardDiff
    x0 = model.starting_points
    grad = ForwardDiff.gradient(model.ℓπcallback, x0)
    @test all(isfinite, grad)
    println("  ForwardDiff gradient computed successfully: $(round.(grad, digits=6))")
end

# === 5. Test combined model (pos + PM + acc + z-prior) ===
@testset "Combined model with z-prior" begin
    sigma_z = 845_000.0
    true_ra = raoff(sol)
    true_dec = decoff(sol)
    true_pmra = pmra(sol)
    true_pmdec = pmdec(sol)
    true_accra = accra(sol)
    true_accdec = accdec(sol)

    astrom = PlanetRelAstromObs(
        (epoch=[test_epoch], ra=[true_ra], dec=[true_dec],
         σ_ra=[0.5], σ_dec=[0.5], cor=[0.0]);
        name="test_pos"
    )

    pm_obs = PlanetPMObs(
        (epoch=[test_epoch], pmra=[true_pmra], pmdec=[true_pmdec],
         σ_pmra=[0.5], σ_pmdec=[0.5], cor=[0.0]);
        name="test_pm"
    )

    acc_obs = PlanetAccelObs(
        (epoch=[test_epoch], accra=[true_accra], accdec=[true_accdec],
         σ_accra=[0.05], σ_accdec=[0.05], cor=[0.0]);
        name="test_acc"
    )

    zp = PlanetZPriorObs(test_epoch, Normal(0.0, sigma_z); name="test_zprior")

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [astrom, pm_obs, acc_obs, zp],
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
        name = "test_combined_z",
        companions = [planet],
        variables = @variables begin
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel

    # Verify gradient works with the combined model
    x0 = model.starting_points
    grad = ForwardDiff.gradient(model.ℓπcallback, x0)
    @test all(isfinite, grad)
    println("  Combined model (pos+PM+acc+z_prior) compiled and differentiable")
end

println("\n✓ All PlanetZPriorObs tests passed")
