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
star_names = sort!(strip.(split(summary_stars_line[1], ",")))
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
for name in star_names
    haskey(octo_utils.stars, name) ||
        error("Star '$name' not found in octo_utils.stars.")
    star = octo_utils.stars[name]
    a, p, ac = octo_utils.build_star_observations(star, epoch_mjd)
    astrom_obs[name] = a
    pm_obs[name]     = p
    acc_obs[name]    = ac
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

println("\n=== Posterior summaries (median, 68% CI) ===")
@printf("%-20s  %10s  [%8s, %8s]\n", "Param", "Median", "16%", "84%")
function print_stat(label, samples; scale=1.0)
    med = median(samples) * scale
    lo  = quantile(samples, 0.16) * scale
    hi  = quantile(samples, 0.84) * scale
    @printf("%-20s  %10.3f  [%8.3f, %8.3f]\n", label, med, lo, hi)
end
print_stat("M_IMBH [10⁴ M☉]", M_samples; scale=1e-4)
print_stat("plx [mas]",        plx_samples)
print_stat("offsetx [mas]",    ox_samples)
print_stat("offsety [mas]",    oy_samples)
for name in star_names
    s = star_samples[name]
    print_stat("$(name): a [AU]", s.a)
    print_stat("$(name): e",      s.e)
    print_stat("$(name): i [°]",  s.i; scale=180/π)
    print_stat("$(name): ω [°]",  s.ω; scale=180/π)
    print_stat("$(name): Ω [°]",  s.Ω; scale=180/π)
end

# ── 9. Corner plot ───────────────────────────────────────────────────────────

println("\nGenerating corner plot...")
corner_plot = octocorner(model, chain; small=true,
    includecols=["M", "offsetx", "offsety"],
    labels=Dict{Symbol,Any}(
        :offsetx => "Δα*_IMBH [mas]",
        :offsety => "Δδ_IMBH [mas]",
    )
)
save(joinpath(output_dir, "$(run_prefix)_corner.png"), corner_plot, px_per_unit=3)
println("Corner plot saved.")

# ── 10. Sky-plane orbit panels ───────────────────────────────────────────────

println("Generating orbit panels...")

sample_idx = round.(Int, range(1, length(M_samples), length=100))

function star_orbit_panel!(ax, s, M_samp, plx_samp, ox_samp, oy_samp,
                            obs_ra, obs_dec, obs_pmra, obs_pmdec,
                            epoch_mjd, sample_idx, color;
                            scale_pm=250.0, scale_acc=25000.0)
    ox_med_loc = median(ox_samp)
    oy_med_loc = median(oy_samp)
    orb_med = Visual{KepOrbit}(;
        a=median(s.a), e=median(s.e), i=median(s.i),
        ω=median(s.ω), Ω=median(s.Ω), tp=median(s.tp),
        M=median(M_samp), plx=median(plx_samp))
    sol_med = orbitsolve(orb_med, epoch_mjd)
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
        color=:royalblue, linewidth=2.0, tipwidth=10, tiplength=10)
    arrows2d!(ax, [obs_ra], [obs_dec],
        [pmra(sol_med) * scale_pm], [pmdec(sol_med) * scale_pm];
        color=:royalblue, linewidth=2.0, tipwidth=10, tiplength=10, linestyle=:dash)
    arrows2d!(ax, [obs_ra], [obs_dec],
        [accra(sol_med) * scale_acc], [accdec(sol_med) * scale_acc];
        color=:firebrick, linewidth=2.0, tipwidth=10, tiplength=10, linestyle=:dash)
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
    color = Makie.wong_colors()[mod1(k, length(Makie.wong_colors()))]
    ax    = Axis(fig_orbits[row, col];
        xlabel="Δα* [mas]", ylabel="Δδ [mas]",
        xreversed=true, autolimitaspect=1,
        xgridvisible=false, ygridvisible=false)
    star_orbit_panel!(ax, star_samples[name], M_samples, plx_samples,
        ox_samples, oy_samples,
        astrom_obs[name].table.ra[1], astrom_obs[name].table.dec[1],
        pm_obs[name].table.pmra[1], pm_obs[name].table.pmdec[1],
        epoch_mjd, sample_idx, color)
    text!(ax, "Star $name"; position=(:left, :top), align=(:left, :top),
          space=:relative, offset=(4, -4), fontsize=18)
end

# ── Combined panel: all stars on one plate ───────────────────────────────────
ax_all = Axis(fig_orbits[combined_row, combined_cols];
    xlabel="Δα* [mas]", ylabel="Δδ [mas]",
    xreversed=true, autolimitaspect=1,
    xgridvisible=false, ygridvisible=false)

for (k, name) in enumerate(star_names)
    color    = Makie.wong_colors()[mod1(k, length(Makie.wong_colors()))]
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
println("Orbit panels saved.")

# ── 11. Posterior histogram panels ───────────────────────────────────────────

println("Generating posterior panels...")

function param_panel!(layout, row, col, cidx, samples, xlabel; show_legend=false)
    ax = Axis(layout[row, col]; xlabel=xlabel, ylabel="Probability Density",
              xgridvisible=false, ygridvisible=false)
    med = median(samples)
    hist!(ax, samples; normalization=:pdf, bins=30,
          color=(Makie.wong_colors()[cidx], 0.7))
    vlines!(ax, [med]; color=Makie.wong_colors()[2], linestyle=:solid, label="Median")
    show_legend && axislegend(ax; position=:rt, framevisible=false)
end

fig_post = Figure(size=(1600, (1 + n_stars) * 260), fontsize=18)

param_panel!(fig_post, 1, 1, 1, M_samples ./ 1e4,
    Makie.rich("M", Makie.subscript("IMBH"), " [10⁴ M", Makie.subscript("☉"), "]");
    show_legend=true)
param_panel!(fig_post, 1, 2, 1, plx_samples, "plx [mas]")
param_panel!(fig_post, 1, 3, 1, ox_samples,
    Makie.rich("Δα*", Makie.subscript("IMBH"), " [mas]"))
param_panel!(fig_post, 1, 4, 1, oy_samples,
    Makie.rich("Δδ", Makie.subscript("IMBH"), " [mas]"))

star_cidx = [3, 4, 5, 6, 7]
for (k, name) in enumerate(star_names)
    cidx = star_cidx[mod1(k, length(star_cidx))]
    row  = k + 1
    s    = star_samples[name]
    param_panel!(fig_post, row, 1, cidx, s.a,           "$(name): a [AU]")
    param_panel!(fig_post, row, 2, cidx, s.e,           "$(name): e")
    param_panel!(fig_post, row, 3, cidx, rad2deg.(s.i), "$(name): i [°]")
    param_panel!(fig_post, row, 4, cidx, rad2deg.(s.ω), "$(name): ω [°]")
    param_panel!(fig_post, row, 5, cidx, rad2deg.(s.Ω), "$(name): Ω [°]")
end
save(joinpath(output_dir, "$(run_prefix)_posteriors.png"), fig_post, px_per_unit=3)
println("Posterior panels saved.")

println("\nDone. All plots written to: $output_dir")
