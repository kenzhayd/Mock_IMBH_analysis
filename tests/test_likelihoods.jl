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
using CairoMakie
using Statistics
using Printf

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
            M ~ Uniform(100, 120_000)
            plx = $plx_val
            offsetx = $offset_ra
            offsety = $offset_dec
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Astrometry model compiled successfully")
end

# === 4. Test PlanetPMObs ===
@testset "PlanetPMObs likelihood" begin
    sigma_pm = 0.5  # mas/yr

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
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  PM model compiled successfully")

    # Expected log-likelihood at zero residuals
    expected_ll = logpdf(MvNormal([sigma_pm^2 0; 0 sigma_pm^2]), [0.0, 0.0])
    println("  Expected max PM log-likelihood: $(round(expected_ll, digits=3))")
end

# === 5. Test PlanetAccelObs ===
@testset "PlanetAccelObs likelihood" begin
    sigma_acc = 0.05  # mas/yr^2

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
            M ~ Uniform(100, 120_000)
            plx = $plx_val
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Acceleration model compiled successfully")

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
            M ~ Uniform(100, 120_000)
            plx = $plx_val
            offsetx = $offset_ra
            offsety = $offset_dec
        end
    )

    model = Octofitter.LogDensityModel(sys)
    @test model isa Octofitter.LogDensityModel
    println("  Combined model compiled successfully")
    println("  Total parameters: $(model.D)")
end

println("\n=== All unit tests passed ===")

# === 7. Fit and recovery check ===
println("\n=== Fitting: should recover M ≈ $M_imbh M☉ ===")

offset_ra_fit  = 5.0
offset_dec_fit = -3.0
sigma_pos_fit  = 0.5
sigma_pm_fit   = 0.05
sigma_acc_fit  = 0.005

astrom_rec = PlanetRelAstromObs(
    (epoch=[test_epoch], ra=[true_ra + offset_ra_fit], dec=[true_dec + offset_dec_fit],
     σ_ra=[sigma_pos_fit], σ_dec=[sigma_pos_fit], cor=[0.0]);
    name="rec_pos"
)
pm_rec = PlanetPMObs(
    (epoch=[test_epoch], pmra=[true_pmra], pmdec=[true_pmdec],
     σ_pmra=[sigma_pm_fit], σ_pmdec=[sigma_pm_fit], cor=[0.0]);
    name="rec_pm"
)
acc_rec = PlanetAccelObs(
    (epoch=[test_epoch], accra=[true_accra], accdec=[true_accdec],
     σ_accra=[sigma_acc_fit], σ_accdec=[sigma_acc_fit], cor=[0.0]);
    name="rec_acc"
)

planet_rec = Planet(
    name = "test_star",
    basis = Visual{KepOrbit},
    observations = [ObsPriorAstromONeil2019(astrom_rec), pm_rec, acc_rec],
    variables = @variables begin
        M = system.M
        a ~ Uniform(100, 10_000)
        e ~ Uniform(0.0, 0.99)
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, $test_epoch; a, e, i, ω, Ω, M)
    end
)

sys_rec = System(
    name = "test_recovery",
    companions = [planet_rec],
    variables = @variables begin
        M ~ Uniform(100, 120_000)
        plx = $plx_val
        offsetx ~ Normal(0, 10)
        offsety ~ Normal(0, 10)
    end
)

model_rec = Octofitter.LogDensityModel(sys_rec)
println("Recovery model compiled (D=$(model_rec.D) free parameters)")

println("Running MCMC (500 iterations, 200 adaptation)...")
chain_rec = octofit(model_rec; iterations=500, adaptation=200)

# === Print recovery statistics ===
M_samples = vec(chain_rec[:M])
a_samples = vec(chain_rec[:test_star_a])
e_samples = vec(chain_rec[:test_star_e])
i_samples = vec(chain_rec[:test_star_i])
ω_samples = vec(chain_rec[:test_star_ω])
Ω_samples  = vec(chain_rec[:test_star_Ω])
tp_samples = vec(chain_rec[:test_star_tp])
ox_samples = vec(chain_rec[:offsetx])
oy_samples = vec(chain_rec[:offsety])
M_med = median(M_samples);  M_lo = quantile(M_samples, 0.16);  M_hi = quantile(M_samples, 0.84)
a_med = median(a_samples);  a_lo = quantile(a_samples, 0.16);  a_hi = quantile(a_samples, 0.84)
e_med = median(e_samples);  e_lo = quantile(e_samples, 0.16);  e_hi = quantile(e_samples, 0.84)
i_med = median(i_samples);  i_lo = quantile(i_samples, 0.16);  i_hi = quantile(i_samples, 0.84)
ω_med = median(ω_samples);  ω_lo = quantile(ω_samples, 0.16);  ω_hi = quantile(ω_samples, 0.84)
Ω_med  = median(Ω_samples);  Ω_lo  = quantile(Ω_samples,  0.16);  Ω_hi  = quantile(Ω_samples,  0.84)
ox_med = median(ox_samples); ox_lo = quantile(ox_samples, 0.16);  ox_hi = quantile(ox_samples, 0.84)
oy_med = median(oy_samples); oy_lo = quantile(oy_samples, 0.16);  oy_hi = quantile(oy_samples, 0.84)
println("\n=== Recovery results ===")
@printf("%-6s  %10s  %10s  %10s  %s\n", "Param", "True", "Median", "68%% CI", "In CI?")
@printf("%-6s  %10.1f  %10.1f  [%8.1f, %8.1f]  %s\n",
    "M", M_imbh, M_med, M_lo, M_hi, M_lo ≤ M_imbh ≤ M_hi ? "✓" : "✗")
@printf("%-6s  %10.1f  %10.1f  [%8.1f, %8.1f]  %s\n",
    "a", a_val, a_med, a_lo, a_hi, a_lo ≤ a_val ≤ a_hi ? "✓" : "✗")
@printf("%-6s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "e", e_val, e_med, e_lo, e_hi, e_lo ≤ e_val ≤ e_hi ? "✓" : "✗")
@printf("%-6s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "i", i_val, i_med, i_lo, i_hi, i_lo ≤ i_val ≤ i_hi ? "✓" : "✗")
@printf("%-6s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "ω", omega_val, ω_med, ω_lo, ω_hi, ω_lo ≤ omega_val ≤ ω_hi ? "✓" : "✗")
@printf("%-6s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "Ω", Omega_val, Ω_med, Ω_lo, Ω_hi, Ω_lo ≤ Omega_val ≤ Ω_hi ? "✓" : "✗")
@printf("%-9s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "offsetx", offset_ra_fit, ox_med, ox_lo, ox_hi, ox_lo ≤ offset_ra_fit ≤ ox_hi ? "✓" : "✗")
@printf("%-9s  %10.3f  %10.3f  [%8.3f, %8.3f]  %s\n",
    "offsety", offset_dec_fit, oy_med, oy_lo, oy_hi, oy_lo ≤ offset_dec_fit ≤ oy_hi ? "✓" : "✗")

# === Figure: sky-plane orbit samples + M posterior ===
println("\nGenerating recovery figure...")

# Orbit time grid spanning one full period
ts_plot = range(test_epoch - P_val * 365.25 / 2,
                test_epoch + P_val * 365.25 / 2, length=300)

# Draw 100 posterior orbit samples
sample_idx = round.(Int, range(1, length(M_samples), length=100))

fig = Figure(size=(1300, 870))

# Left panel: sky-plane orbits (spans all three rows)
ax1 = Axis(fig[1:3, 1];
    xlabel="Δα* [mas]", ylabel="Δδ [mas]",
    title="Orbit recovery (sky plane)",
    xreversed=true, autolimitaspect=1,
    xgridvisible=false, ygridvisible=false,
)
# True orbit — drawn first so posterior samples render on top
ra_true  = [raoff(orbitsolve(orbit, t))  for t in ts_plot]
dec_true = [decoff(orbitsolve(orbit, t)) for t in ts_plot]
lines!(ax1, ra_true, dec_true; color=:black, linewidth=1.5, label="True orbit")
# Posterior orbit samples — thin gray transparent lines (on top of true orbit)
for idx in sample_idx
    M_s  = M_samples[idx];  a_s  = a_samples[idx]
    e_s  = e_samples[idx];  i_s  = i_samples[idx]
    ω_s  = ω_samples[idx];  Ω_s  = Ω_samples[idx];  tp_s = tp_samples[idx]
    orb_s = Visual{KepOrbit}(; a=a_s, e=e_s, i=i_s, ω=ω_s, Ω=Ω_s, tp=tp_s,
                               M=M_s, plx=plx_val)
    ra_s  = [raoff(orbitsolve(orb_s, t))  for t in ts_plot]
    dec_s = [decoff(orbitsolve(orb_s, t)) for t in ts_plot]
    lines!(ax1, ra_s, dec_s; color=(:gray, 0.5), linewidth=0.5)
end
# IMBH at its true offset position — filled black circle
scatter!(ax1, [offset_ra_fit], [offset_dec_fit];
    marker=:circle, markersize=12, color=:black)
# Instantaneous PM vector (scaled for visibility; no legend entry)
scale_pm = 50.0   # yr
arrows!(ax1, [true_ra], [true_dec], [true_pmra * scale_pm], [true_pmdec * scale_pm];
    color=:royalblue, linewidth=2.0, arrowsize=10)
# Instantaneous acceleration vector (scaled for visibility; no legend entry)
scale_acc = 5000.0  # yr²
arrows!(ax1, [true_ra], [true_dec], [true_accra * scale_acc], [true_accdec * scale_acc];
    color=:firebrick, linewidth=2.0, arrowsize=10)
# Observed position — drawn last so star symbol sits on top
scatter!(ax1, [true_ra], [true_dec];
    marker='★', color=Makie.wong_colors()[2], markersize=14,
    strokecolor=:black, strokewidth=0.5, label="Observed position")
axislegend(ax1; position=:rt, framevisible=false)

# Helper: one-parameter posterior panel
function param_panel!(layout, row, col, samples, true_val, xlabel, title)
    ax = Axis(layout[row, col]; xlabel=xlabel, ylabel="Density", title=title,
              xgridvisible=false, ygridvisible=false)
    med = median(samples)
    hist!(ax, samples; normalization=:pdf, bins=30,
          color=(Makie.wong_colors()[col - 1], 0.7))
    vlines!(ax, [true_val]; color=:black, linestyle=:dash,  label="True")
    vlines!(ax, [med];      color=Makie.wong_colors()[2], linestyle=:solid, label="Median")
    axislegend(ax; position=:rt, framevisible=false)
end

# Row 1: M, a, e
param_panel!(fig, 1, 2, M_samples, M_imbh,  "M [M☉]", "IMBH mass")
param_panel!(fig, 1, 3, a_samples, a_val,   "a [AU]",  "Semi-major axis")
param_panel!(fig, 1, 4, e_samples, e_val,   "e",       "Eccentricity")
# Row 2: i, ω, Ω
param_panel!(fig, 2, 2, i_samples, i_val,     "i [rad]", "Inclination")
param_panel!(fig, 2, 3, ω_samples, omega_val, "ω [rad]", "Arg. of periapsis")
param_panel!(fig, 2, 4, Ω_samples, Omega_val, "Ω [rad]", "Longitude of node")
# Row 3: IMBH position offsets
param_panel!(fig, 3, 2, ox_samples, offset_ra_fit,  "offsetx [mas]", "IMBH RA offset")
param_panel!(fig, 3, 3, oy_samples, offset_dec_fit, "offsety [mas]", "IMBH Dec offset")

save("test_likelihoods_recovery.png", fig, px_per_unit=3)
println("Recovery figure saved to test_likelihoods_recovery.png")
