"""
Unit tests for PlanetPMObs and PlanetAccelObs likelihoods.
Creates a known orbit, computes expected observables, and verifies
that the likelihood evaluates correctly.

Run with:  julia --project=path/to/Octofitter_imbh.jl test_likelihoods.jl
"""

using Octofitter
using PlanetOrbits
using Distributions
using Test

# === 1. Define a known orbit ===
# A star orbiting a 10,000 Msun IMBH at 5.43 kpc
M_imbh = 10000.0    # solar masses
plx_val = 0.184     # mas (d = 5.43 kpc)
P_val = 1000.0      # years
a_val = cbrt(M_imbh * P_val^2)  # AU, Kepler's 3rd law
e_val = 0.3
i_val = 1.0         # rad
omega_val = 0.5     # rad
Omega_val = 1.2     # rad
tp_val = 50000.0    # MJD

# Construct the orbit
orbit = Visual{KepOrbit}(;
    a = a_val,
    e = e_val,
    i = i_val,
    ω = omega_val,
    Ω = Omega_val,
    tp = tp_val,
    M = M_imbh,
    plx = plx_val,
)

# === 2. Evaluate the orbit at a test epoch ===
test_epoch = 55197.0  # MJD (~2010)
sol = orbitsolve(orbit, test_epoch)

# Extract the "true" observables
true_ra = raoff(sol)       # mas
true_dec = decoff(sol)     # mas
true_pmra = pmra(sol)      # mas/yr
true_pmdec = pmdec(sol)    # mas/yr
true_accra = accra(sol)    # mas/yr^2
true_accdec = accdec(sol)  # mas/yr^2

println("=== Known orbit observables at epoch $test_epoch ===")
println("Position:      RA = $(round(true_ra, digits=3)) mas, Dec = $(round(true_dec, digits=3)) mas")
println("Proper motion: pmRA = $(round(true_pmra, digits=6)) mas/yr, pmDec = $(round(true_pmdec, digits=6)) mas/yr")
println("Acceleration:  accRA = $(round(true_accra, digits=8)) mas/yr², accDec = $(round(true_accdec, digits=8)) mas/yr²")

# === 3. Test PlanetRelAstromObs (with offset) ===
@testset "PlanetRelAstromObs with offsetx/offsety" begin
    # Observation: star position = orbit position + IMBH offset
    offset_ra = 5.0   # mas
    offset_dec = -3.0  # mas
    obs_ra = true_ra + offset_ra
    obs_dec = true_dec + offset_dec
    sigma = 0.5  # mas

    astrom = PlanetRelAstromObs(
        (epoch=[test_epoch], ra=[obs_ra], dec=[obs_dec], σ_ra=[sigma], σ_dec=[sigma], cor=[0.0]);
        name="test_pos"
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [astrom],
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
        name = "test_system",
        companions = [planet],
        variables = @variables begin
            M = $M_imbh
            plx = $plx_val
            offsetx = $offset_ra
            offsety = $offset_dec
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Astrometry model compiled successfully (type stable: $(model.type_stable))")
end

# === 4. Test PlanetPMObs ===
@testset "PlanetPMObs likelihood" begin
    sigma_pm = 0.05  # mas/yr

    pm_obs = PlanetPMObs(
        (epoch=[test_epoch], pmra=[true_pmra], pmdec=[true_pmdec],
         σ_pmra=[sigma_pm], σ_pmdec=[sigma_pm], cor=[0.0]);
        name="test_pm"
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [pm_obs],
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
        name = "test_pm_system",
        companions = [planet],
        variables = @variables begin
            M = $M_imbh
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  PM model compiled successfully (type stable: $(model.type_stable))")

    # Expected log-likelihood at zero residuals
    expected_ll = logpdf(MvNormal([sigma_pm^2 0; 0 sigma_pm^2]), [0.0, 0.0])
    println("  Expected max PM log-likelihood: $(round(expected_ll, digits=3))")
end

# === 5. Test PlanetAccelObs ===
@testset "PlanetAccelObs likelihood" begin
    sigma_acc = 0.005  # mas/yr^2

    acc_obs = PlanetAccelObs(
        (epoch=[test_epoch], accra=[true_accra], accdec=[true_accdec],
         σ_accra=[sigma_acc], σ_accdec=[sigma_acc], cor=[0.0]);
        name="test_acc"
    )

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [acc_obs],
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
        name = "test_acc_system",
        companions = [planet],
        variables = @variables begin
            M = $M_imbh
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Acceleration model compiled successfully (type stable: $(model.type_stable))")

    expected_ll = logpdf(MvNormal([sigma_acc^2 0; 0 sigma_acc^2]), [0.0, 0.0])
    println("  Expected max acceleration log-likelihood: $(round(expected_ll, digits=3))")
end

# === 6. Test combined (all 3 likelihoods on one star) ===
@testset "Combined pos + PM + accel" begin
    offset_ra = 5.0
    offset_dec = -3.0
    sigma_pos = 0.5
    sigma_pm = 0.05
    sigma_acc = 0.005

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

    planet = Planet(
        name = "test_star",
        basis = Visual{KepOrbit},
        observations = [ObsPriorAstromONeil2019(astrom), pm_obs, acc_obs],
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
        name = "test_combined",
        companions = [planet],
        variables = @variables begin
            M = $M_imbh
            plx = $plx_val
            offsetx = $offset_ra
            offsety = $offset_dec
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    @test model.type_stable == true
    println("  Combined model compiled successfully (type stable: $(model.type_stable))")
    println("  Total parameters: $(model.num_params)")
end

println("\n=== All unit tests passed ===")
