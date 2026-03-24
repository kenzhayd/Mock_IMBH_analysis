# Ocen_IMBH Analysis

## Project Purpose

This repo contains scripts for fitting orbits of stars around an IMBH candidate in the globular cluster ω Centauri (ω Cen), and for analysing and visualising the results. Orbit fitting uses `Octofitter_imbh.jl`, a development fork of [Octofitter](https://github.com/sefffal/Octofitter.jl), which performs Bayesian inference on orbits of fast stars via HMC/NUTS sampling.

**Key model features (implemented in `Octofitter_imbh.jl`):**
- RA and DEC position of the central mass are free parameters
- Likelihood for 2D proper motion velocities and accelerations (RA and DEC components separately)

---

## Repo Structure

This repo holds:
- **Slurm job scripts** — submit orbit fitting runs to Digital Alliance of Canada (DAC) HPC clusters
- **Analysis scripts** — load posterior chains, compute derived quantities, run diagnostics
- **Figure scripts** — produce publication-quality plots from fit results

---

## Octofitter Usage

Models are defined in Julia using the `@variables` DSL and run via `Octofitter_imbh.jl`. Key pattern:

```julia
system = System(
    @variables begin
        M ~ LogUniform(1e3, 1e8)   # IMBH mass in solar masses
        ra ~ Normal(0.0, 1.0)      # RA offset [mas]
        dec ~ Normal(0.0, 1.0)     # DEC offset [mas]
        plx = 0.198                # fixed parallax [mas]
    end,
    SomeLikelihood(...),
    star1, star2, ...              # Planet objects for each tracked star
)
model = LogDensityModel(system)
chain = octofit(model)
```

Chains are saved as HDF5 files and loaded for analysis. Use `Octofitter.loadchain` to read them.

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
