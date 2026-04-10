"""
Generate plots from a saved chain FITS file.

Re-creates the corner plot, sky-plane orbit panels, and posterior histogram
panels that are normally produced at the end of octo_orbit_direct_likelihoods.jl.

Usage (standalone, safe on a login node):
    julia --project=<Octofitter_imbh.jl> plot_chain.jl <chain.fits>

When called via include() from octo_orbit_direct_likelihoods.jl, the thread
count of the parent process is inherited automatically.

The companion *_summary.md file (same directory, same run prefix) is loaded
automatically to recover the epoch, priors, and star list.  The full TOML
configuration is embedded in that file, so no separate config.toml is needed.

Star names are also cross-checked against the chain column names (columns
ending in _a with matching _e, _i, _ω, _Ω, _tp companions).

Output files are written to the same directory as *chain.fits, using the same
run_prefix (everything before _chain.fits in the filename).
"""

ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

using Octofitter
using Octofitter: @variables, System
using CairoMakie
using PairPlots
using Distributions
using Unitful
using UnitfulAstro
using LinearAlgebra
using Statistics
using Printf
using TOML

push!(LOAD_PATH, @__DIR__)
using octo_utils

include(joinpath(@__DIR__, "parse_config.jl"))

# ── 1. Parse arguments ───────────────────────────────────────────────────────

length(ARGS) >= 1 || error("Usage: julia plot_chain.jl <chain.fits>")
chain_path = ARGS[1]
isfile(chain_path) || error("Chain file not found: $chain_path")

output_dir = dirname(abspath(chain_path))
chain_basename = basename(chain_path)
run_prefix = endswith(chain_basename, "_chain.fits") ?
    chain_basename[1:end-length("_chain.fits")] : splitext(chain_basename)[1]

# ── 2. Load configuration from the companion summary.md ──────────────────────

summary_path = joinpath(output_dir, "$(run_prefix)_summary.md")
isfile(summary_path) || error(
    "Summary file not found: $summary_path\n" *
    "Expected a *_summary.md alongside the chain file.")

summary_text = read(summary_path, String)

# Extract the TOML block embedded between ```toml and ``` fences
toml_match = match(r"```toml\r?\n(.*?)```"s, summary_text)
toml_match === nothing && error(
    "Could not find a ```toml ... ``` block in $summary_path")
cfg = TOML.parse(toml_match[1])
println("Loaded configuration from: $summary_path")

epoch_mjd  = get_epoch_mjd(cfg)
epoch_year = cfg["epoch"]["year"]
println("Reference epoch: $epoch_mjd MJD ($epoch_year yr)")

# ── 3. Load chain ────────────────────────────────────────────────────────────

chain = Octofitter.loadchain(chain_path)
println("Loaded chain: $chain_path")
println(chain)

# ── 4. Get star names from summary, verify against chain columns ──────────────
# FITS column names are ASCII-only; Unicode characters (ω, Ω) may be encoded
# differently on save/load.  Use the summary as the authoritative source for
# star names, then discover the actual column names for each orbital element.

col_names = Set(Symbol.(names(chain)))

# Primary: parse from summary
summary_stars_line = match(r"\*\*Stars:\*\*\s*([^\n]+)", summary_text)
summary_stars_line !== nothing ||
    error("Could not find '**Stars:**' line in $summary_path")
star_names = sort!(String.(strip.(split(summary_stars_line[1], ","))))
println("Star names from summary: $(join(star_names, ", "))")

# Helper: find a chain column for a given star and orbital element, trying
# multiple name variants to handle FITS ASCII encoding of Unicode symbols.
const _col_variants = Dict(
    "a"  => ["a"],
    "e"  => ["e"],
    "i"  => ["i"],
    "ω"  => ["ω",  "omega", "w"],
    "Ω"  => ["Ω",  "Omega", "W"],
    "tp" => ["tp"],
)
function find_col(col_names, star, element)
    for v in _col_variants[element]
        sym = Symbol("$(star)_$(v)")
        sym in col_names && return sym
    end
    error("Cannot find chain column for $(star)_$(element). " *
          "Available columns with prefix '$(star)_': " *
          join(filter(c -> startswith(String(c), "$(star)_"), collect(col_names)), ", "))
end

# Verify all expected columns exist
for name in star_names, el in keys(_col_variants)
    find_col(col_names, name, el)   # errors early with a clear message if missing
end

# ── 5. Rebuild observation objects ───────────────────────────────────────────

astrom_obs = Dict{String, Any}()
pm_obs     = Dict{String, Any}()
acc_obs    = Dict{String, Any}()
rv_obs     = Dict{String, Any}()   # may hold nothing for stars without RV
for name in star_names
    haskey(octo_utils.stars, name) ||
        error("Star '$name' not found in octo_utils.stars.")
    star = octo_utils.stars[name]
    a, p, ac, rv, _zp = octo_utils.build_star_observations(star, epoch_mjd)
    astrom_obs[name] = a
    pm_obs[name]     = p
    acc_obs[name]    = ac
    rv_obs[name]     = rv
end

# ── 6. Rebuild the Octofitter model (required by octocorner) ─────────────────

companions = Planet[]
for name in star_names
    P_prior = parse_prior(get_companion_prior(cfg, name, "P"))
    e_prior = parse_prior(get_companion_prior(cfg, name, "e"))
    i_prior = parse_prior(get_companion_prior(cfg, name, "i"))
    ω_prior = parse_prior(get_companion_prior(cfg, name, "omega"))
    Ω_prior = parse_prior(get_companion_prior(cfg, name, "Omega"))
    θ_prior = parse_prior(get_companion_prior(cfg, name, "theta"))

    star = Planet(
        name = name,
        basis = Visual{KepOrbit},
        observations = [ObsPriorAstromONeil2019(astrom_obs[name]), pm_obs[name], acc_obs[name]],
        variables = @variables begin
            M = system.M
            P ~ P_prior
            a = cbrt(M * P^2)
            e ~ e_prior
            i ~ i_prior
            ω ~ ω_prior
            Ω ~ Ω_prior
            θ ~ θ_prior
            tp = θ_at_epoch_to_tperi(θ, $epoch_mjd; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
        end
    )
    push!(companions, star)
end

sys_priors    = cfg["priors"]["system"]
plx_prior     = parse_prior(sys_priors["plx"])
M_prior       = parse_prior(sys_priors["M"])
offsetx_prior = parse_prior(sys_priors["offsetx"])
offsety_prior = parse_prior(sys_priors["offsety"])

sys = System(
    name = get(cfg["meta"], "system_name", "Omega_Cen"),
    observations = [],
    companions = companions,
    variables = @variables begin
        plx ~ plx_prior
        M ~ M_prior
        offsetx ~ offsetx_prior
        offsety ~ offsety_prior
    end
)

model = Octofitter.LogDensityModel(sys)

# ── 7. Extract posterior samples ─────────────────────────────────────────────

M_samples   = vec(chain[:M])
plx_samples = vec(chain[:plx])
ox_samples  = vec(chain[:offsetx])
oy_samples  = vec(chain[:offsety])

star_samples = Dict{String, NamedTuple}()
for name in star_names
    star_samples[name] = (
        a  = vec(chain[find_col(col_names, name, "a")]),
        e  = vec(chain[find_col(col_names, name, "e")]),
        i  = vec(chain[find_col(col_names, name, "i")]),
        ω  = vec(chain[find_col(col_names, name, "ω")]),
        Ω  = vec(chain[find_col(col_names, name, "Ω")]),
        tp = vec(chain[find_col(col_names, name, "tp")]),
    )
end

# ── 8. Posterior summaries ───────────────────────────────────────────────────

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
stats_path = joinpath(output_dir, "$(run_prefix)_posterior_stats.txt")

# ── 9. Corner plot ───────────────────────────────────────────────────────────

println("\nGenerating corner plot...")
# Subsample chain for corner plot to limit memory usage
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
save(joinpath(output_dir, "$(run_prefix)_corner.png"), corner_plot, px_per_unit=3)
corner_plot = nothing; chain_thin = nothing; GC.gc()
println("Corner plot saved.")

# ── 10. Sky-plane orbit panels ───────────────────────────────────────────────

println("Generating orbit panels...")

# Per-star color: grey for star A, Wong palette (starting at index 1) for all others.
star_colors = Dict{String, Any}()
wong_i = 0
for name in star_names
    if name == "A"
        star_colors[name] = colorant"grey60"
    else
        global wong_i += 1
        star_colors[name] = Makie.wong_colors()[mod1(wong_i, length(Makie.wong_colors()))]
    end
end

sample_idx = round.(Int, range(1, length(M_samples), length=100))

function star_orbit_panel!(ax, s, M_samp, plx_samp, ox_samp, oy_samp,
                            obs_ra, obs_dec, obs_pmra, obs_pmdec,
                            epoch_mjd, sample_idx, color;
                            scale_pm=250.0)
    ox_med_loc = median(ox_samp)
    oy_med_loc = median(oy_samp)
    for idx in sample_idx
        orb_s = Visual{KepOrbit}(;
            a=s.a[idx], e=s.e[idx], i=s.i[idx],
            ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
            M=M_samp[idx], plx=plx_samp[idx])
        P_s = s.a[idx]^1.5 / sqrt(M_samp[idx])  # period in years for this sample
        ts  = range(epoch_mjd, epoch_mjd + P_s * 365.25; length=300)
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

n_stars    = length(star_names)
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
        epoch_mjd, sample_idx, color)
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
        ts  = range(epoch_mjd, epoch_mjd + P_s * 365.25; length=300)
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

save(joinpath(output_dir, "$(run_prefix)_orbit_panels.png"), fig_orbits, px_per_unit=3)
fig_orbits = nothing; GC.gc()
println("Orbit panels saved.")

# ── 11. Posterior histogram panels ───────────────────────────────────────────

println("Generating posterior panels...")

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
save(joinpath(output_dir, "$(run_prefix)_posteriors.png"), fig_post, px_per_unit=3)
fig_post = nothing; GC.gc()
println("Posterior panels saved.")

# ── 11.5. Physical plausibility: pericenter, velocity, tidal radii ───────────

println("Generating plausibility diagnostics...")

# 1 AU/yr in km/s (1 AU / 1 yr in SI)
const AU_YR_TO_KMS = 4.7404705
# G·M_sun in AU³/yr² (Kepler's third law with a in AU, P in yr, M in M_sun)
const FOUR_PI2 = 4 * π^2

# Reference stellar templates (not priors — for tidal radius lines only)
const R_SUN_AU   = 1 / 215.032           # 1 R_sun in AU
const R_GIANT_AU = 30 * R_SUN_AU         # ~30 R_sun for a red giant
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
save(joinpath(output_dir, "$(run_prefix)_plausibility.png"), fig_phys, px_per_unit=3)
fig_phys = nothing; GC.gc()
println("Plausibility diagnostics saved.")

# ── 11.6. True anomaly at obs epoch + acceleration-vector alignment ──────────

println("Generating phase / acceleration-alignment diagnostics...")

using PlanetOrbits: trueanom, radvel

"True anomaly (deg, wrapped to (-180, 180]) at epoch_mjd, one per chain draw."
function true_anomaly_at_epoch(s, M_samp, plx_samp, epoch_mjd)
    ν = Vector{Float64}(undef, length(M_samp))
    @inbounds for idx in eachindex(M_samp)
        orb = Visual{KepOrbit}(;
            a=s.a[idx], e=s.e[idx], i=s.i[idx],
            ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
            M=M_samp[idx], plx=plx_samp[idx])
        ν[idx] = rad2deg(trueanom(orbitsolve(orb, epoch_mjd)))
    end
    return @. mod(ν + 180, 360) - 180
end

"Angle (deg, 0–180) between measured accel vector and star→IMBH direction."
function accel_alignment_angle(name, ox_samp, oy_samp)
    obs_ra  = astrom_obs[name].table.ra[1]
    obs_dec = astrom_obs[name].table.dec[1]
    ax_meas = acc_obs[name].table.accra[1]
    ay_meas = acc_obs[name].table.accdec[1]
    a_norm  = hypot(ax_meas, ay_meas)
    ax_hat  = ax_meas / a_norm
    ay_hat  = ay_meas / a_norm
    dx = ox_samp .- obs_ra
    dy = oy_samp .- obs_dec
    r  = hypot.(dx, dy)
    dxh = dx ./ r
    dyh = dy ./ r
    cosφ = clamp.(ax_hat .* dxh .+ ay_hat .* dyh, -1.0, 1.0)
    return rad2deg.(acos.(cosφ))
end

"""
    accel_toward_imbh_zscore(name, ox_samp, oy_samp)

Per-posterior-sample z-score: the component of the measured acceleration in
the direction of the star→IMBH unit vector, divided by the propagated
uncertainty of that component.  z > 0 means the measured acceleration has a
component pointing toward the IMBH; z ≈ 0 means the measurement cannot
distinguish the IMBH direction; z < 0 means it points away.

Also returns the angular uncertainty on the measured acceleration direction
(in degrees), which is constant across the posterior.
"""
function accel_toward_imbh_zscore(name, ox_samp, oy_samp)
    obs_ra  = astrom_obs[name].table.ra[1]
    obs_dec = astrom_obs[name].table.dec[1]
    ax_meas = acc_obs[name].table.accra[1]
    ay_meas = acc_obs[name].table.accdec[1]
    σx      = acc_obs[name].table.σ_accra[1]
    σy      = acc_obs[name].table.σ_accdec[1]

    # Angular uncertainty of the measured direction via error propagation on atan2
    σ_φ_deg = rad2deg(sqrt((ay_meas * σx)^2 + (ax_meas * σy)^2) / (ax_meas^2 + ay_meas^2))

    # Per-sample unit vector from star toward IMBH
    dx = ox_samp .- obs_ra
    dy = oy_samp .- obs_dec
    r  = hypot.(dx, dy)
    dxh = dx ./ r
    dyh = dy ./ r

    # Component of measured acceleration in the IMBH direction and its uncertainty
    a_toward  = ax_meas .* dxh .+ ay_meas .* dyh
    σ_toward  = sqrt.((σx .* dxh).^2 .+ (σy .* dyh).^2)

    z = a_toward ./ σ_toward
    return z, σ_φ_deg
end

fig_pa = Figure(size=(1500, n_stars * 240), fontsize=18)
for (k, name) in enumerate(star_names)
    c  = star_colors[name]
    ν  = true_anomaly_at_epoch(star_samples[name], M_samples, plx_samples, epoch_mjd)
    Δφ = accel_alignment_angle(name, ox_samples, oy_samples)
    z_accel, σ_φ_deg = accel_toward_imbh_zscore(name, ox_samples, oy_samples)

    ax_ν = Axis(fig_pa[k, 1];
        xlabel="$(name): ν(t_obs) [°]", ylabel="Probability Density",
        xticks=-180:90:180,
        xgridvisible=false, ygridvisible=false)
    hist!(ax_ν, ν; normalization=:pdf, bins=40, color=(c, 0.7))
    vlines!(ax_ν, [0.0]; color=:black, linestyle=:dot)

    ax_φ = Axis(fig_pa[k, 2];
        xlabel="$(name): accel misalignment Δφ [°]",
        ylabel="Probability Density",
        xgridvisible=false, ygridvisible=false)
    hist!(ax_φ, Δφ; normalization=:pdf, bins=40, color=(c, 0.7))
    # Dashed line at 0° (perfect alignment) and shaded band showing the
    # angular uncertainty of the measured acceleration direction
    vlines!(ax_φ, [0.0]; color=:black, linestyle=:dot, label="Perfect alignment")
    vspan!(ax_φ, 0.0, σ_φ_deg; color=(:grey, 0.25), label="Meas. dir. uncertainty (1σ)")
    k == 1 && axislegend(ax_φ; position=:rt, framevisible=false, labelsize=12)

    # Column 3: z-score (component of measured acc toward IMBH / uncertainty)
    ax_z = Axis(fig_pa[k, 3];
        xlabel="$(name): accel z-score toward IMBH",
        ylabel="Probability Density",
        xgridvisible=false, ygridvisible=false)
    hist!(ax_z, z_accel; normalization=:pdf, bins=40, color=(c, 0.7))
    vlines!(ax_z, [0.0]; color=:black, linestyle=:dot)
    vlines!(ax_z, [-1.0, 1.0]; color=:grey, linestyle=:dash)
    k == 1 && text!(ax_z, "z>0: acc toward IMBH"; position=(0.55, 0.90),
                    align=(:left, :top), space=:relative, fontsize=11, color=:grey40)

    push!(stat_lines, format_stat("$(name): ν(t_obs) [°]", ν))
    push!(stat_lines, format_stat("$(name): Δφ_accel [°]", Δφ))
    push!(stat_lines,
          @sprintf("%-20s  %10.1f °", "$(name): σ_φ_acc (meas)", σ_φ_deg))
    push!(stat_lines, format_stat("$(name): z_acc→IMBH", z_accel))
end
save(joinpath(output_dir, "$(run_prefix)_phase_accel.png"), fig_pa, px_per_unit=3)
fig_pa = nothing; GC.gc()
println("Phase / acceleration alignment diagnostics saved.")

# ── 11.7. Radial-velocity consistency check (stars with RV data only) ────────

rv_stars = [n for n in star_names if rv_obs[n] !== nothing]
if !isempty(rv_stars)
    println("Generating RV consistency check...")
    fig_rv = Figure(size=(500 * length(rv_stars), 400), fontsize=18)
    for (k, name) in enumerate(rv_stars)
        s = star_samples[name]
        rv_pred = Vector{Float64}(undef, length(M_samples))
        @inbounds for idx in eachindex(M_samples)
            orb = Visual{KepOrbit}(;
                a=s.a[idx], e=s.e[idx], i=s.i[idx],
                ω=s.ω[idx], Ω=s.Ω[idx], tp=s.tp[idx],
                M=M_samples[idx], plx=plx_samples[idx])
            rv_pred[idx] = radvel(orbitsolve(orb, epoch_mjd))  # m/s, peculiar
        end
        rv_pred_kms = rv_pred ./ 1000.0

        rv_meas_kms  = rv_obs[name].table.rv[1]   / 1000.0
        rv_sigma_kms = rv_obs[name].table.σ_rv[1] / 1000.0

        ax = Axis(fig_rv[1, k];
            xlabel="$(name): peculiar RV [km/s]",
            ylabel="Probability Density",
            xgridvisible=false, ygridvisible=false)
        hist!(ax, rv_pred_kms; normalization=:pdf, bins=40,
              color=(star_colors[name], 0.7), label="Posterior prediction")
        vspan!(ax, rv_meas_kms - rv_sigma_kms, rv_meas_kms + rv_sigma_kms;
               color=(:grey, 0.35), label="Measured ± 1σ")
        vlines!(ax, [rv_meas_kms]; color=:black, linewidth=2, label="Measured")
        k == 1 && axislegend(ax; position=:rt, framevisible=false)

        z = (median(rv_pred_kms) - rv_meas_kms) / rv_sigma_kms
        push!(stat_lines,
              @sprintf("%-20s  %+10.2f σ", "$(name): RV residual", z))
    end
    save(joinpath(output_dir, "$(run_prefix)_rv_check.png"), fig_rv, px_per_unit=3)
    fig_rv = nothing; GC.gc()
    println("RV consistency check saved.")
else
    println("No stars with RV data; skipping RV consistency check.")
end

# ── 12. IMBH position density map ────────────────────────────────────────

println("Generating IMBH position density map...")

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
cos_dec0 = cosd(octo_utils.dec_cm_deg)
dra_lo, dra_hi   = extrema(cx)
ddec_lo, ddec_hi = extrema(cy)
ra_lo  = octo_utils.ra_cm_deg  + dra_lo  / (3600.0 * cos_dec0)
ra_hi  = octo_utils.ra_cm_deg  + dra_hi  / (3600.0 * cos_dec0)
dec_lo = octo_utils.dec_cm_deg + ddec_lo / 3600.0
dec_hi = octo_utils.dec_cm_deg + ddec_hi / 3600.0

ax_deg = Axis(fig_imbh[1, 1];
    xaxisposition = :top, yaxisposition = :right,
    xlabel = "RA [°]", ylabel = "Dec [°]",
    xreversed = true,
    xgridvisible = false, ygridvisible = false,
    limits = ((ra_lo, ra_hi), (dec_lo, dec_hi)),
)
hidespines!(ax_deg)

save(joinpath(output_dir, "$(run_prefix)_imbh_position.png"), fig_imbh, px_per_unit = 3)
fig_imbh = nothing; GC.gc()
println("IMBH position density map saved.")

# ── 13. 3D orbit animation (360° pan, IMBH-centric) ───────────────────────

println("Generating 3D orbit animation...")

using PlanetOrbits: posx, posy, posz

const AU_PER_PC = 206265.0

# Pre-compute 3D orbit trajectories for each posterior sample (reuses
# sample_idx from the orbit panels).  All positions are relative to the
# IMBH and converted from AU to parsec.
orbit_3d_samples = Dict{String, Vector{NamedTuple}}()
star_pos_3d      = Dict{String, Vector{NamedTuple}}()

for name in star_names
    s = star_samples[name]
    orbits_list = NamedTuple[]
    pos_list    = NamedTuple[]
    for idx in sample_idx
        orb = Visual{KepOrbit}(;
            a = s.a[idx], e = s.e[idx], i = s.i[idx],
            ω = s.ω[idx], Ω = s.Ω[idx], tp = s.tp[idx],
            M = M_samples[idx], plx = plx_samples[idx])
        P_yr = s.a[idx]^1.5 / sqrt(M_samples[idx])
        ts   = range(epoch_mjd, epoch_mjd + P_yr * 365.25; length=300)
        sols = [orbitsolve(orb, t) for t in ts]
        push!(orbits_list, (
            x = [posx(sl) for sl in sols] ./ AU_PER_PC,
            y = [posy(sl) for sl in sols] ./ AU_PER_PC,
            z = [posz(sl) for sl in sols] ./ AU_PER_PC,
        ))
        sol_now = orbitsolve(orb, epoch_mjd)
        push!(pos_list, (
            x = posx(sol_now) / AU_PER_PC,
            y = posy(sol_now) / AU_PER_PC,
            z = posz(sol_now) / AU_PER_PC,
        ))
    end
    orbit_3d_samples[name] = orbits_list
    star_pos_3d[name]      = pos_list
end

# Symmetric axis limits from all orbit samples
all_coords = Float64[]
for name in star_names
    for o in orbit_3d_samples[name]
        append!(all_coords, o.x)
        append!(all_coords, o.y)
        append!(all_coords, o.z)
    end
end
lim = 0.5 * maximum(abs.(all_coords))   # shrink axis range — larger orbits clip

# Animation view-angle parameters (also used for the initial Axis3 view).
azim_start = -3 * π / 4    # starting azimuth (three-quarter view)
elev_max   = 50 * π / 180  # start/end elevation
elev_min   = 10 * π / 180  # low angle, reveals LOS depth

# Build figure with Axis3.  Starting azimuth (-3π/4) places x to the right.
# viewmode = :fit keeps ticks/labels inside the viewport as the camera rotates
# (the default :fitzoom lets labels drift outside the frame).
fig3d = Figure(size = (800, 800), fontsize = 16, figure_padding = 30)
ax3 = Axis3(fig3d[1, 1];
    xlabel = "x [pc]", ylabel = "y [pc]", zlabel = "z (LOS) [pc]",
    limits = (-lim, lim, -lim, lim, -lim, lim),
    aspect = :data,
    viewmode = :fit,
    azimuth   = azim_start,
    elevation = elev_max,
)

# Per-sample IMBH mass (used to scale star marker sizes below).
sample_masses = M_samples[sample_idx]
mass_ref      = median(sample_masses)
base_size     = lim * 0.012

# Draw orbit samples and current-epoch star positions
for (k, name) in enumerate(star_names)
    color = star_colors[name]
    for o in orbit_3d_samples[name]
        lines!(ax3, o.x, o.y, o.z; color = (color, 0.3), linewidth = 0.5)
    end
    px = [p.x for p in star_pos_3d[name]]
    py = [p.y for p in star_pos_3d[name]]
    pz = [p.z for p in star_pos_3d[name]]
    # Marker radius proportional to the IMBH mass of each chain sample,
    # normalized so the median-mass sample matches the previous default size.
    marker_sizes = base_size .* (sample_masses ./ mass_ref)
    meshscatter!(ax3, px, py, pz;
        markersize = marker_sizes, color = color)
    lines!(ax3, [NaN], [NaN], [NaN]; color = color, linewidth = 2, label = "Star $name")
end

# IMBH at origin
meshscatter!(ax3, [0.0], [0.0], [0.0];
    markersize = lim * 0.015, color = :black, label = "IMBH")

axislegend(ax3; position = :rt, framevisible = false)

# Animate: azimuth pans 360° while elevation oscillates between ~50°
# and 20° using a cosine profile so both the first and last frames
# match for seamless looping.
n_frames   = 240
framerate  = 15
anim_path  = joinpath(output_dir, "$(run_prefix)_orbits_3d.mp4")

record(fig3d, anim_path, 0:(n_frames - 1); framerate) do frame
    t = frame / n_frames                                   # 0 → 1 (exclusive)
    ax3.azimuth[]   = azim_start + 2π * t
    ax3.elevation[] = (elev_max + elev_min) / 2 +
                      (elev_max - elev_min) / 2 * cos(2π * t)  # starts high, dips low at t=0.5
end

println("3D orbit animation saved to: $anim_path")

# ── 14. Write collected posterior stats (after all diagnostic sections) ──────

open(stats_path, "w") do io
    println(io, "Posterior summaries (median, 68% CI)")
    println(io, "Chain: $chain_path")
    println(io, "Epoch: $epoch_mjd MJD ($epoch_year yr)")
    println(io, "Stars: $(join(star_names, ", "))")
    println(io)
    for line in stat_lines
        println(io, line)
    end
end
println("Posterior stats saved to: $stats_path")

println("\nDone. All plots written to: $output_dir")
