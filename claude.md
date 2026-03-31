# Ocen_IMBH Analysis

## Project Purpose

This repo contains scripts for fitting orbits of stars around an IMBH candidate in the globular cluster ω Centauri (ω Cen), and for analysing and visualising the results. Orbit fitting uses `Octofitter_imbh.jl`, a development fork of [Octofitter](https://github.com/sefffal/Octofitter.jl), which performs Bayesian inference on orbits of fast stars via HMC/NUTS sampling.

**Key model features (implemented in `Octofitter_imbh.jl`):**
- RA and DEC position of the central mass are free parameters
- Likelihood for 2D proper motion velocities and accelerations (RA and DEC components separately)

---

## Repo Structure

- `launch_scripts/octo_utils.jl` — utility module: `StarData` struct, star dictionary (A–G with positions, PMs, accelerations), `build_star_observations` helper, `simulate_astrometry` (legacy), unit conversions
- `launch_scripts/octo_orbit_direct_likelihoods.jl` — **main fitting script** using direct PM/acceleration likelihoods (v8 API, Pigeons sampling, 192 chains, 18 rounds)
- `launch_scripts/old_scripts/octo_orbit_192c_18r.jl` — legacy script using synthetic 3-epoch astrometry (v7 API, kept for comparison)
- `tests/test_likelihoods.jl` — unit tests for `PlanetRelAstromObs` (with offsets), `PlanetPMObs`, `PlanetAccelObs` using a known orbit
- `tests/test_ad_compatibility.jl` — type stability and ForwardDiff gradient verification
- `tests/test_small_fit.jl` — short 2-star MCMC fit for end-to-end validation

---

## Octofitter v8 Usage

Models use the v8 API (`observations=`, `*Obs` naming). Each star gets 3 observation types: position + proper motion + acceleration.

```julia
using Octofitter

# Build observations from star data
astrom, pm, acc = build_star_observations(star, epoch_mjd)

planet = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [ObsPriorAstromONeil2019(astrom), pm, acc],
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
    name = "Omega_Cen",
    companions = [planet, ...],
    observations = [],
    variables = @variables begin
        plx ~ truncated(Normal(0.19, 0.004), lower=0)
        M ~ Uniform(100, 120_000)
        offsetx ~ Normal(0, 10)   # IMBH RA offset from assumed center [mas]
        offsety ~ Normal(0, 10)   # IMBH Dec offset from assumed center [mas]
    end
)

model = Octofitter.LogDensityModel(sys)
chain, pt = octofit_pigeons(model; n_rounds=18, n_chains=192)
```

Chains are saved as FITS files via `Octofitter.savechain` and loaded for analysis.

---

## Slurm / DAC Conventions

- Job scripts should use `#SBATCH` headers with explicit resource requests (nodes, CPUs, memory, time)
- Load Julia via the DAC module system: `module load julia`
- Use `julia --project=path/to/Octofitter_imbh.jl` to point at the local package
- Output logs to a `logs/` subdirectory; chain files to `chains/`
- Parameterise scripts (star ID, prior bounds, etc.) via command-line arguments where possible

---

## Plot Formatting Conventions (Makie / CairoMakie)

### Figure Sizes and Saving
- Main figures: `size=(700, 600)`
- Compact panels: `size=(500, 300)` to `(500, 400)`
- Always save with `px_per_unit=3` for publication quality

### Colors
- Per-object colormap: `Makie.cgrad([Makie.wong_colors()[i], "#FAFAFA"])`
- For multiple objects: cycle through `Makie.wong_colors()` by index

### Axes
- No grid lines: `xgridvisible=false, ygridvisible=false`
- Astrometry plots: `autolimitaspect=1` (square), `xreversed=true` (RA increases right-to-left)
- Zero-crossing reference: `vlines!(ax, 0, color=:grey, linestyle=:dash)`

### Axis Labels
- Rich text for sub/superscripts: `Makie.rich("μ", Makie.subscript("α*"), " [mas/yr]")`
- Mathematical notation: `Makie.latexstring(...)` where needed
- Unicode deltas in coordinate labels: Δα*, Δδ

### Markers and Lines
- Data points: `markersize=8`
- Posterior sample scatter: `markersize=2`, adaptive alpha `alpha=min.(1, 100/length(ii))`
- Central mass marker: `marker='★', markersize=20, color=:white, strokecolor=:black, strokewidth=1.5`
- Dense scatter: `rasterize=4` to reduce output file size
