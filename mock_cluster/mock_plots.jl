"""
Generate plots for mock cluster inference runs.

Works with outputs from mock_inference.jl.
"""

# === Environment variables 
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

# === Imports 
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
using Printf
using PlanetOrbits


# Load module 
include(joinpath(@__DIR__, "mock_utils.jl"))
using .mock_utils

# === CONFIGURATION 

cfg = mock_utils.run_config_plots(ARGS)

mock_name = cfg.mock_name
M_IMBH = cfg.M_IMBH
z_prior_sigma = cfg.z_prior_sigma
plx = cfg.plx

n_rounds = cfg.n_rounds
n_chains = cfg.n_chains
n_chains_variational = cfg.n_chains_variational

DISTANCE_KPC = cfg.distance_kpc
TREF = cfg.tref

SIGMA_RA_OFF = cfg.sigma_ra_off
SIGMA_DEC_OFF = cfg.sigma_dec_off
SIGMA_PM_RA = cfg.sigma_pm_ra
SIGMA_PM_DEC = cfg.sigma_pm_dec
SIGMA_ACC_RA = cfg.sigma_acc_ra
SIGMA_ACC_DEC = cfg.sigma_acc_dec
SIGMA_RV = cfg.sigma_rv

imbh_ra = cfg.imbh_ra
imbh_dec = cfg.imbh_dec

rv_cluster = cfg.rv_cluster
rv_cluster_err = cfg.rv_cluster_err

INCLUDE_ACC = cfg.include_acc
job_name = cfg.job_name
results_dir = cfg.results_dir

run_id = mock_utils.make_run_id(job_name, n_rounds, n_chains)

# Output directory 
outdir = results_dir


# === REBUILD MOCK SYSTEM
# Load star parameters
star_params = mock_utils.load_mock_params(mock_name)

# Build mock parameter set
stars = mock_utils.build_mock_paramset(star_params;
    M = M_IMBH,
    plx = plx,
    epoch = TREF,
    t_ref = TREF,
    sigma_ra_off = SIGMA_RA_OFF,
    sigma_dec_off = SIGMA_DEC_OFF,
    sigma_pm_ra = SIGMA_PM_RA,
    sigma_pm_dec = SIGMA_PM_DEC,
    sigma_acc_ra = SIGMA_ACC_RA,
    sigma_acc_dec = SIGMA_ACC_DEC,
    sigma_rv = SIGMA_RV
)

star_names = sort(collect(keys(stars)))
n_stars    = length(star_names)

# Load chain
# === Load chain (robust to naming issues)

files = filter(f -> endswith(f, "_chain.fits"), readdir(outdir))

println("Chain candidates found: ", files)

@assert length(files) == 1 "Expected exactly one chain file, found $(length(files)) in $(outdir)"

chain_path = joinpath(outdir, files[1])

println("Using chain file: ", chain_path)
println("Exists? ", isfile(chain_path))

chain = Octofitter.loadchain(chain_path)

# Extract samples
M_samples   = vec(chain[:M])
plx_samples = vec(chain[:plx])
ox_samples  = vec(chain[:offsetx])
oy_samples  = vec(chain[:offsety])

star_samples = Dict{String, NamedTuple}()
for name in star_names
    star_samples[name] = (
        a  = vec(chain[Symbol(name * "_a")]),
        e  = vec(chain[Symbol(name * "_e")]),
        i  = vec(chain[Symbol(name * "_i")]),
        ω  = vec(chain[Symbol(name * "_ω")]),
        Ω  = vec(chain[Symbol(name * "_Ω")]),
        tp = vec(chain[Symbol(name * "_tp")]),
    )
end

sample_idx = round.(Int, range(1, length(M_samples), length=100))


# Convert each StarData object to Octofitter compatible observations
obs = Dict{String, Any}()
for (name, star) in stars
    obs[name] = mock_utils.build_star_observations(
        star,
        float(TREF);
        include_rv = true,
        z_prior_sigma = z_prior_sigma
    )
end

astrom_obs = Dict{String, Any}()
pm_obs     = Dict{String, Any}()
acc_obs    = Dict{String, Any}()
rv_obs     = Dict{String, Any}()

for (name, (a, p, ac, rv, _zp)) in obs
    astrom_obs[name] = a
    pm_obs[name]     = p
    acc_obs[name]    = ac
    rv_obs[name]     = rv
end

# Create planets
# Priors matching starsACDEF_192c_16r_10836874
# Config file: /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/accel_escvel_16r_position.toml

planets = Planet[]

for (name, (astrom, pm, acc, rv, zp)) in obs

    obs_list = Any[
        ObsPriorAstromONeil2019(astrom),
        astrom,
        pm,
        acc
    ]

    if rv !== nothing
        push!(obs_list, rv)
    end

    if zp !== nothing
        push!(obs_list, zp)
    end

    planet = Planet(
        name = name,
        basis = Visual{KepOrbit},
        observations = obs_list,

        variables = @variables begin
            # Shared cluster parameter
            M = system.M

            # Orbital parameters
            P ~ Uniform(10, 2000000) # Period in yrs
            a = cbrt(M * P^2)      # Semi-Major axis in AU

            e ~ Uniform(0.0, 0.99) # Eccentricity
            i ~ Sine()             # Inclination [rad]
            ω ~ UniformCircular()  # Argument of periastron [rad]
            Ω ~ UniformCircular()  # Longitude of ascending node [rad]
            θ ~ UniformCircular()  # Mean anomaly at reference epoch [rad]

            # Convert phase to periastron passage time
            tp = θ_at_epoch_to_tperi(
                θ, 55197.0;
                a=a, e=e, i=i, ω=ω, Ω=Ω, M=M
            )
        end
    )

    push!(planets, planet)
end


# System model
# Priors matching starsACDEF_192c_16r_10836874
# Config file: /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/accel_escvel_16r_position.toml
sys = System(
    name = "mock_Omega_Centauri",
    companions = planets,

    variables = @variables begin

        M ~ Uniform(100, 100000)

        plx ~ truncated(Normal(0.19, 0.004), lower=0)

        # offsets
        offsetx ~ Uniform(-3000, 3000) 
        offsety ~ Uniform(-3000, 3000)
    end
)

model = Octofitter.LogDensityModel(sys)

# Colors
star_colors = Dict{String, Any}()
wong = Makie.wong_colors()
for (i, name) in enumerate(star_names)
    star_colors[name] = wong[mod1(i, length(wong))]
end

# === CORNER PLOT

max_corner_samples = 10_000
if size(chain, 1) > max_corner_samples
    thin_idx = round.(Int, range(1, size(chain, 1), length=max_corner_samples))
    chain_thin = chain[thin_idx, :, :]
else
    chain_thin = chain
end
corner_plot = octocorner(model, chain_thin; small=true,
    includecols=["M", "offsetx", "offsety"],
    labels=Dict{Symbol,Any}(
        :offsetx => "Δα*_IMBH [mas]",
        :offsety => "Δδ_IMBH [mas]",
    )
)

save(joinpath(outdir, "$(run_id)_corner_plot.png"), corner_plot, px_per_unit=3)

# === Posterior Summaries

function format_stat(label, samples; scale=1.0)
    med = median(samples) * scale
    lo  = quantile(samples, 0.16) * scale
    hi  = quantile(samples, 0.84) * scale
    return @sprintf("%-20s  %10.3f  [%8.3f, %8.3f]", label, med, lo, hi)
end

stat_lines = String[]
push!(stat_lines, @sprintf("%-20s  %10s  [%8s, %8s]", "Param", "Median", "16%", "84%"))
push!(stat_lines, format_stat("M_IMBH [10⁴ M☉]", M_samples; scale=1e-4))
push!(stat_lines, format_stat("plx [mas]",        plx_samples))
push!(stat_lines, format_stat("offsetx [mas]",    ox_samples))
push!(stat_lines, format_stat("offsety [mas]",    oy_samples))
for name in star_names
    s = star_samples[name]
    push!(stat_lines, format_stat("$(name): a [AU]", s.a))
    push!(stat_lines, format_stat("$(name): e",      s.e))
    push!(stat_lines, format_stat("$(name): i [°]",  s.i; scale=180/π))
    push!(stat_lines, format_stat("$(name): ω [°]",  s.ω; scale=180/π))
    push!(stat_lines, format_stat("$(name): Ω [°]",  s.Ω; scale=180/π))
end

println("\n=== Posterior summaries (median, 68% CI) ===")
for line in stat_lines
    println(line)
end
# Note: _posterior_stats.txt is written at the very end of the script so that
# physical-plausibility diagnostics (Sections 11.5–11.7) can append to
# stat_lines before the file is produced.
stats_path = joinpath(outdir, "$(run_id)_posterior_stats.txt")


# === POSTERIOR HISTOGRAM

function param_panel!(layout, row, col, color, samples, xlabel; show_legend=false)
    ax = Axis(layout[row, col]; xlabel=xlabel, ylabel="Probability Density",
              xgridvisible=false, ygridvisible=false)
    med = median(samples)
    hist!(ax, samples; normalization=:pdf, bins=30,
          color=(color, 0.7))
    vlines!(ax, [med]; color=Makie.wong_colors()[2], linestyle=:solid, label="Median")
    show_legend && axislegend(ax; position=:rt, framevisible=false)
end

fig_post = Figure(size=(1600, (1 + n_stars) * 260), fontsize=18)

sys_color = Makie.wong_colors()[1]
param_panel!(fig_post, 1, 1, sys_color, M_samples ./ 1e4,
    Makie.rich("M", Makie.subscript("IMBH"), " [10⁴ M", Makie.subscript("☉"), "]");
    show_legend=true)
param_panel!(fig_post, 1, 2, sys_color, plx_samples, "plx [mas]")
param_panel!(fig_post, 1, 3, sys_color, ox_samples,
    Makie.rich("Δα*", Makie.subscript("IMBH"), " [mas]"))
param_panel!(fig_post, 1, 4, sys_color, oy_samples,
    Makie.rich("Δδ", Makie.subscript("IMBH"), " [mas]"))

for (k, name) in enumerate(star_names)
    row  = k + 1
    s    = star_samples[name]
    c    = star_colors[name]
    param_panel!(fig_post, row, 1, c, s.a,           "$(name): a [AU]")
    param_panel!(fig_post, row, 2, c, s.e,           "$(name): e")
    param_panel!(fig_post, row, 3, c, rad2deg.(s.i), "$(name): i [°]")
    param_panel!(fig_post, row, 4, c, rad2deg.(s.ω), "$(name): ω [°]")
    param_panel!(fig_post, row, 5, c, rad2deg.(s.Ω), "$(name): Ω [°]")
end

# === SKY PLANE ORBIT PANELS 

function star_orbit_panel!(ax, s, M_samp, plx_samp, ox_samp, oy_samp,
                            obs_ra, obs_dec, obs_pmra, obs_pmdec,
                            TREF, sample_idx, color;
                            scale_pm=250.0)
    ox_med_loc = median(ox_samp)
    oy_med_loc = median(oy_samp)
    for idx in sample_idx
        orb_s = Visual{KepOrbit}(;
            a=s.a[idx], e=s.e[idx], i=s.i[idx],
            ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
            M=M_samp[idx], plx=plx_samp[idx])
        P_s = s.a[idx]^1.5 / sqrt(M_samp[idx])  # period in years for this sample
        ts  = range(TREF, TREF + P_s * 365.25; length=300)
        ra_s  = [raoff(orbitsolve(orb_s, t)) + ox_samp[idx] for t in ts]
        dec_s = [decoff(orbitsolve(orb_s, t)) + oy_samp[idx] for t in ts]
        lines!(ax, ra_s, dec_s; color=(color, 0.5), linewidth=0.5)
    end
    scatter!(ax, [0.0], [0.0]; marker=:cross, markersize=16, color=:black)
    scatter!(ax, [ox_med_loc], [oy_med_loc]; marker=:circle, markersize=12, color=:black)
    arrows2d!(ax, [obs_ra], [obs_dec],
        [obs_pmra * scale_pm], [obs_pmdec * scale_pm];
        color=:royalblue, shaftwidth=2.0, tipwidth=10, tiplength=10)
    scatter!(ax, [obs_ra], [obs_dec];
        marker='★', color=Makie.wong_colors()[2], markersize=14,
        strokecolor=:black, strokewidth=0.5)
end


n_cols_orb = min(n_stars, 3)
n_rows_orb = ceil(Int, n_stars / n_cols_orb)
# If the last row of individual panels has empty cells, reuse them for the
# combined panel; otherwise place it on a new row.
n_filled_last  = mod(n_stars, n_cols_orb)   # 0 means last row is full
combined_row   = n_filled_last == 0 ? n_rows_orb + 1 : n_rows_orb
combined_cols  = n_filled_last == 0 ? (1:n_cols_orb) : ((n_filled_last + 1):n_cols_orb)
n_rows_fig     = n_filled_last == 0 ? n_rows_orb + 1 : n_rows_orb

fig_orbits = Figure(size=(n_cols_orb * 420, n_rows_fig * 440), fontsize=18)

for (k, name) in enumerate(star_names)
    row   = ceil(Int, k / n_cols_orb)
    col   = mod1(k, n_cols_orb)
    color = star_colors[name]
    ax    = Axis(fig_orbits[row, col];
        xlabel="Δα* [mas]", ylabel="Δδ [mas]",
        xreversed=true, autolimitaspect=1,
        xgridvisible=false, ygridvisible=false)
    star_orbit_panel!(ax, star_samples[name], M_samples, plx_samples,
        ox_samples, oy_samples,
        astrom_obs[name].table.ra[1], astrom_obs[name].table.dec[1],
        pm_obs[name].table.pmra[1], pm_obs[name].table.pmdec[1],
        TREF, sample_idx, color)
    text!(ax, "Star $name"; position=(0.05, 0.95), align=(:left, :top),
          space=:relative, fontsize=18)
end

# ── Combined panel: all stars on one plate ───────────────────────────────────
ax_all = Axis(fig_orbits[combined_row, combined_cols];
    xlabel="Δα* [mas]", ylabel="Δδ [mas]",
    xreversed=true, autolimitaspect=1,
    xgridvisible=false, ygridvisible=false)

for (k, name) in enumerate(star_names)
    color    = star_colors[name]
    s        = star_samples[name]
    obs_ra   = astrom_obs[name].table.ra[1]
    obs_dec  = astrom_obs[name].table.dec[1]
    for idx in sample_idx
        orb_s = Visual{KepOrbit}(;
            a=s.a[idx], e=s.e[idx], i=s.i[idx],
            ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
            M=M_samples[idx], plx=plx_samples[idx])
        P_s = s.a[idx]^1.5 / sqrt(M_samples[idx])  # period in years for this sample
        ts  = range(TREF, TREF + P_s * 365.25; length=300)
        ra_s  = [raoff(orbitsolve(orb_s, t)) + ox_samples[idx] for t in ts]
        dec_s = [decoff(orbitsolve(orb_s, t)) + oy_samples[idx] for t in ts]
        lines!(ax_all, ra_s, dec_s; color=(color, 0.3), linewidth=0.5)
    end
    scatter!(ax_all, [obs_ra], [obs_dec];
        marker='★', color=Makie.wong_colors()[2], markersize=14,
        strokecolor=:black, strokewidth=0.5)
    # Solid-color dummy line used only for the legend entry
    lines!(ax_all, [NaN], [NaN]; color=color, linewidth=2, label="Star $name")
end

scatter!(ax_all, [0.0], [0.0]; marker=:cross, markersize=16, color=:black)
scatter!(ax_all, [median(ox_samples)], [median(oy_samples)];
    marker=:circle, markersize=12, color=:black)
axislegend(ax_all; position=:rt, framevisible=false)

save(joinpath(outdir, "$(run_id)_mock_orbits.png"), fig_orbits)

# === PHYSICAL PLAUSIBILITY 
# 1 AU/yr in km/s (1 AU / 1 yr in SI)
AU_YR_TO_KMS = 4.7404705
# G·M_sun in AU³/yr² (Kepler's third law with a in AU, P in yr, M in M_sun)
FOUR_PI2 = 4 * π^2

# Reference stellar templates (not priors — for tidal radius lines only)
R_SUN_AU   = 1 / 215.032           # 1 R_sun in AU
R_GIANT_AU = 30 * R_SUN_AU         # ~30 R_sun for a red giant
tidal_radius(R_star_au, m_star_Msun, M_BH_Msun) =
    R_star_au * cbrt(M_BH_Msun / m_star_Msun)

"""
Per-sample orbital scalars for one star: pericenter/apocenter distances (AU),
pericenter/apocenter speeds (km/s, via vis-viva), and period (yr).
"""
function orbital_scalars(s, M_samples)
    a = s.a; e = s.e
    r_peri = a .* (1 .- e)
    r_apo  = a .* (1 .+ e)
    v_peri = @. sqrt(FOUR_PI2 * M_samples * (1 + e) / (a * (1 - e))) * AU_YR_TO_KMS
    v_apo  = @. sqrt(FOUR_PI2 * M_samples * (1 - e) / (a * (1 + e))) * AU_YR_TO_KMS
    P_yr   = @. sqrt(a^3 / M_samples)
    return (; r_peri, r_apo, v_peri, v_apo, P_yr)
end

# Mass-dependent reference radii (one value per posterior draw)
rt_ms_samples  = tidal_radius.(R_SUN_AU,   1.0, M_samples)
rt_rg_samples  = tidal_radius.(R_GIANT_AU, 0.8, M_samples)
# Schwarzschild radius in AU:  r_s = 2GM/c² ≈ M[M_sun] · 1.909e-8 AU
r_schw_samples = M_samples .* 1.909e-8

# Append scalars to the text summary
for name in star_names
    scal = orbital_scalars(star_samples[name], M_samples)
    push!(stat_lines, format_stat("$(name): r_peri [AU]", scal.r_peri))
    push!(stat_lines, format_stat("$(name): r_apo  [AU]", scal.r_apo))
    push!(stat_lines, format_stat("$(name): v_peri [km/s]", scal.v_peri))
    push!(stat_lines, format_stat("$(name): v_apo  [km/s]", scal.v_apo))
    push!(stat_lines, format_stat("$(name): P      [yr]",   scal.P_yr))
end
push!(stat_lines, format_stat("r_tidal MS  [AU]", rt_ms_samples))
push!(stat_lines, format_stat("r_tidal RG  [AU]", rt_rg_samples))
push!(stat_lines, format_stat("r_Schw      [AU]", r_schw_samples))

fig_phys = Figure(size=(1200, n_stars * 260), fontsize=18)
for (k, name) in enumerate(star_names)
    s    = star_samples[name]
    scal = orbital_scalars(s, M_samples)
    c    = star_colors[name]

    ax_r = Axis(fig_phys[k, 1];
        xlabel="$(name): r_peri [AU]", ylabel="Probability Density",
        xscale=log10,
        xgridvisible=false, ygridvisible=false)
    # Build log-spaced bins so the histogram renders correctly on a log axis
    r_lo = max(minimum(scal.r_peri),
               0.5 * min(median(rt_ms_samples), median(r_schw_samples)))
    r_hi = maximum(scal.r_peri)
    r_bins = 10 .^ range(log10(r_lo), log10(r_hi); length=31)
    hist!(ax_r, scal.r_peri; normalization=:pdf, bins=r_bins, color=(c, 0.7))
    vlines!(ax_r, [median(rt_ms_samples)];
            color=:steelblue, linestyle=:dash, label="r_t (MS)")
    vlines!(ax_r, [median(rt_rg_samples)];
            color=:firebrick, linestyle=:dash, label="r_t (RG)")
    vlines!(ax_r, [median(r_schw_samples)];
            color=:black, linestyle=:dot, label="r_Schw")
    k == 1 && axislegend(ax_r; position=:lt, framevisible=false)

    ax_v = Axis(fig_phys[k, 2];
        xlabel="$(name): v_peri [km/s]", ylabel="Probability Density",
        xgridvisible=false, ygridvisible=false)
    hist!(ax_v, scal.v_peri; normalization=:pdf, bins=30, color=(c, 0.7))
end

save(joinpath(outdir, "$(run_id)_mock_plausibility.png"), fig_phys)


# === IMBH POSITION  DENSITY MAP

# Work in arcsec offsets from AvdM10 centre (primary axes).
dra_arcsec  = ox_samples ./ 1000.0   # Δα* [arcsec]
ddec_arcsec = oy_samples ./ 1000.0   # Δδ  [arcsec]

# 2D histogram
n_bins = 120
hist_x = range(extrema(dra_arcsec)..., length = n_bins + 1)
hist_y = range(extrema(ddec_arcsec)..., length = n_bins + 1)
hist_counts = zeros(n_bins, n_bins)
dx = step(hist_x)
dy = step(hist_y)
for (x, y) in zip(dra_arcsec, ddec_arcsec)
    ix = clamp(floor(Int, (x - first(hist_x)) / dx) + 1, 1, n_bins)
    iy = clamp(floor(Int, (y - first(hist_y)) / dy) + 1, 1, n_bins)
    hist_counts[ix, iy] += 1
end
# Normalize to peak = 1
hist_norm = hist_counts ./ maximum(hist_counts)

# Bin centres
cx = [first(hist_x) + (i - 0.5) * dx for i in 1:n_bins]
cy = [first(hist_y) + (j - 0.5) * dy for j in 1:n_bins]

# Greyscale colormap: white (0) → black (1)
cmap_grey = cgrad([:white, :black])

fig_imbh = Figure(size = (750, 650), fontsize = 16)

# Primary axes: Δα* and Δδ in arcsec (bottom / left)
ax_imbh = Axis(fig_imbh[1, 1];
    xlabel = "Δα* [arcsec]", ylabel = "Δδ [arcsec]",
    xreversed = true,
    autolimitaspect = 1,
    xgridvisible = false, ygridvisible = false,
    backgroundcolor = :white,
)

hm = heatmap!(ax_imbh, cx, cy, hist_norm;
    colormap = cmap_grey, colorrange = (0, 1), rasterize = 4)
Colorbar(fig_imbh[1, 2], hm; label = "Normalized density")

# AvdM10 centre at (0, 0) in offset coordinates
scatter!(ax_imbh, [0.0], [0.0];
    marker = '+', markersize = 20, color = :red, label = "AvdM10 centre")

# Overlay star positions and observed PM vectors (converted mas → arcsec)
scale_pm = 0.25   # arcsec per mas/yr for arrow length
for (k, name) in enumerate(star_names)
    color = star_colors[name]
    obs_ra_as   = astrom_obs[name].table.ra[1]  / 1000.0
    obs_dec_as  = astrom_obs[name].table.dec[1]  / 1000.0
    obs_pmra    = pm_obs[name].table.pmra[1]
    obs_pmdec   = pm_obs[name].table.pmdec[1]
    arrows2d!(ax_imbh, [obs_ra_as], [obs_dec_as],
        [obs_pmra * scale_pm], [obs_pmdec * scale_pm];
        color = :royalblue, shaftwidth = 2.0, tipwidth = 10, tiplength = 10)
    scatter!(ax_imbh, [obs_ra_as], [obs_dec_as];
        marker = '★', color = Makie.wong_colors()[2], markersize = 14,
        strokecolor = :black, strokewidth = 0.5)
end

axislegend(ax_imbh; position = :rt, framevisible = false)

# Secondary axes: absolute RA (top) and Dec (right) in degrees.

dra_lo, dra_hi   = extrema(cx)
ddec_lo, ddec_hi = extrema(cy)

# Convert arcsec → degrees
dra_lo_deg = dra_lo / 3600.0
dra_hi_deg = dra_hi / 3600.0
ddec_lo_deg = ddec_lo / 3600.0
ddec_hi_deg = ddec_hi / 3600.0

# Build absolute sky coordinates
ra_lo  = imbh_ra  + dra_lo_deg
ra_hi  = imbh_ra  + dra_hi_deg
dec_lo = imbh_dec + ddec_lo_deg
dec_hi = imbh_dec + ddec_hi_deg

# Fix ordering (IMPORTANT for Makie)
ra_min, ra_max   = extrema((ra_lo, ra_hi))
dec_min, dec_max = extrema((dec_lo, dec_hi))

ax_deg = Axis(fig_imbh[1, 1];
    xaxisposition = :top,
    yaxisposition = :right,
    xlabel = "RA [°]",
    ylabel = "Dec [°]",
    xreversed = true,
    xgridvisible = false,
    ygridvisible = false,
    limits = ((ra_min, ra_max), (dec_min, dec_max)),
)

hidespines!(ax_deg)

save(joinpath(outdir, "$(run_id)_mock_imbh_map.png"), fig_imbh)

# === SAVE POSTERIOR STATISTICS
chain_path = joinpath(outdir, "chain.fits")
open(stats_path, "w") do io
    println(io, "Posterior summaries (median, 68% CI)")
    println(io, "Chain: $chain_path")
    println(io, "Epoch: $TREF MJD")
    println(io, "Stars: $(join(star_names, ", "))")
    println(io)
    for line in stat_lines
        println(io, line)
    end
end

# ── Acceleration posterior predictive check ────────────────────────────
#
# Goal: treat the acceleration measurements as an independent validation of the
# orbit model rather than a fitting constraint.  For each posterior draw we
# compute the sky-plane acceleration that the fitted Keplerian orbit predicts
# at the observation epoch, then compare the full predictive distribution to the
# measured values.
#
# This is a standard posterior predictive check (PPC): if the model is
# consistent with the acceleration data, the measured acceleration should fall
# in the bulk of the predictive cloud.  If it sits in a clear tail, the
# acceleration is in tension with the model — which may indicate either a
# genuine physical inconsistency or a systematic error in the measurement.
#
# Two complementary diagnostics per star:
#
#   Left panel — 2D predictive scatter in (accra, accdec) space [mas/yr²]:
#     Each point is the sky-plane acceleration predicted by one posterior draw,
#     computed via accra(sol) and accdec(sol) from PlanetOrbits.  The measured
#     value is shown as a black cross with ±1σ error bars.  If the cross falls
#     in the bulk of the scatter cloud the model is consistent with the
#     acceleration data; if it is a clear outlier the two are in tension.
#
#   Right panel — Distribution of 2D chi-squared residuals:
#     For each posterior draw i the 2D chi-squared distance between the
#     predicted and measured acceleration is:
#
#         χ²_i = (accra_meas  - accra_pred_i )² / σ_ra²
#               + (accdec_meas - accdec_pred_i)² / σ_dec²
#
#     The vertical dashed line marks χ² = 2.30, which is the 68th percentile
#     of a chi-squared distribution with 2 degrees of freedom — i.e. the
#     boundary of the 1σ error ellipse in 2D.  The fraction of posterior draws
#     with χ²_i ≤ 2.30 is labelled f₆₈ on the plot:
#       • f₆₈ ≈ 0.68 → the measurement lies in the typical 1σ bulk of the
#         predictive distribution (fully consistent)
#       • f₆₈ << 0.68 → the measurement is in the tail; the model cannot
#         easily reproduce the observed acceleration, indicating tension
#
# Note: accra(sol) and accdec(sol) from PlanetOrbits return the gravitational
# acceleration of the star toward the IMBH in the sky plane (mas/yr²), with
# the same sign convention as PlanetAccelObs.  The IMBH offset (offsetx,
# offsety) is already encoded in the orbital geometry, so no additional
# correction is needed.

println("Generating acceleration posterior predictive check...")
if !INCLUDE_ACC
    println("Skipping acceleration PPC (acceleration not used in fit).")

else
    fig_acc_ppc = Figure(size=(1000, n_stars * 280), fontsize=18)

    for (k, name) in enumerate(star_names)
        s    = star_samples[name]
        c    = star_colors[name]

        # Measured acceleration and uncertainties for this star [mas/yr²]
        ax_meas = acc_obs[name].table.accra[1]
        ay_meas = acc_obs[name].table.accdec[1]
        σ_ra    = acc_obs[name].table.σ_accra[1]
        σ_dec   = acc_obs[name].table.σ_accdec[1]

        # --- Compute predicted acceleration for every posterior draw ---
        # accra(sol) and accdec(sol) evaluate the Keplerian gravitational
        # acceleration at the solved orbital position (mas/yr²).  This is the
        # model quantity that PlanetAccelObs compares to the measurement in the
        # likelihood — here we compute it purely as a prediction check.
        n_samp      = length(M_samples)
        accra_pred  = Vector{Float64}(undef, n_samp)
        accdec_pred = Vector{Float64}(undef, n_samp)
        @inbounds for idx in 1:n_samp
            orb = Visual{KepOrbit}(;
                a=s.a[idx], e=s.e[idx], i=s.i[idx],
                ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
                M=M_samples[idx], plx=plx_samples[idx])
            sol = orbitsolve(orb, TREF)
            accra_pred[idx]  = accra(sol)
            accdec_pred[idx] = accdec(sol)
        end

        # --- 2D chi-squared residual per draw ---
        # Measures the distance (in units of measurement uncertainty) between
        # each predicted acceleration and the measured value.
        chi2 = @. (ax_meas - accra_pred)^2 / σ_ra^2 +
                (ay_meas - accdec_pred)^2 / σ_dec^2

        # Fraction of draws that predict an acceleration within the 1σ error
        # ellipse of the measurement (chi-squared 2-DOF threshold = 2.30).
        chi2_1sigma = 2.30
        f68 = mean(chi2 .<= chi2_1sigma)

        # Append consistency metrics to the text summary
        push!(stat_lines,
            @sprintf("%-20s  %10.3f  (f68=%.2f)",
                    "$(name): acc χ²_med", median(chi2), f68))

        # --- Left panel: 2D predictive scatter ---
        ax_2d = Axis(fig_acc_ppc[k, 1];
            xlabel="$(name): predicted accra [mas/yr²]",
            ylabel="$(name): predicted accdec [mas/yr²]",
            xgridvisible=false, ygridvisible=false)

        # Subsample for visual clarity; rasterize to keep file size small.
        scatter!(ax_2d, accra_pred[sample_idx], accdec_pred[sample_idx];
            color=(c, 0.3), markersize=4, rasterize=4,
            label="Posterior predictions")

        # Measured acceleration with ±1σ error bars
        errorbars!(ax_2d, [ax_meas], [ay_meas], [σ_ra];
            direction=:x, color=:black, linewidth=2)
        errorbars!(ax_2d, [ax_meas], [ay_meas], [σ_dec];
            direction=:y, color=:black, linewidth=2)
        scatter!(ax_2d, [ax_meas], [ay_meas];
            marker=:xcross, markersize=14, color=:black, strokewidth=2,
            label="Measured ± 1σ")

        k == 1 && axislegend(ax_2d; position=:rt, framevisible=false, labelsize=12)

        # --- Right panel: chi-squared residual distribution ---
        ax_chi = Axis(fig_acc_ppc[k, 2];
            xlabel="$(name): χ² (predicted vs measured, 2-DOF)",
            ylabel="Probability Density",
            xgridvisible=false, ygridvisible=false)

        hist!(ax_chi, chi2; normalization=:pdf, bins=40, color=(c, 0.7))

        # Mark the 1σ ellipse boundary.  The fraction of draws to the left of
        # this line (f₆₈) is the key consistency metric printed in the legend.
        vlines!(ax_chi, [chi2_1sigma];
            color=:black, linestyle=:dash,
            label=@sprintf("χ²=2.30 (1σ ellipse), f₆₈=%.2f", f68))
        axislegend(ax_chi; position=:rt, framevisible=false, labelsize=12)
        end

        save(joinpath(outdir, "$(run_id)_accel_check.png"), fig_acc_ppc, px_per_unit=3)
        println("Acceleration posterior predictive check saved.")

        fig_acc_ppc = nothing; GC.gc()
    end 
