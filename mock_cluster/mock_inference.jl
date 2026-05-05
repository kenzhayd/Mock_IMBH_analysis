"""
mock_inference.jl

Run Octofitter on a mock system to test recovery of true parameters
"""


# ========== Environment variables ==========
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

# ========== Imports ==========
using Octofitter
using Octofitter: @variables, System
using CairoMakie
using PairPlots
using Distributions
using Unitful
using UnitfulAstro
using LinearAlgebra
using Statistics
using Pigeons
using Printf
using PlanetOrbits
using Dates


include(joinpath(@__DIR__, "mock_utils.jl"))
using .mock_utils


# === CONFIGURATION 

cfg = mock_utils.run_config(ARGS)

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

resume_path = nothing

for i in 1:length(ARGS)-1
    if ARGS[i] == "--resume"
        global resume_path = ARGS[i+1]
    end
end

#output directory 
outdir = results_dir

mkpath(outdir)


# Run summary 

summary_path = joinpath(outdir, "summary.txt")

open(summary_path, "w") do io

    println(io, "========================")
    println(io, "RUN SUMMARY")
    println(io, "========================\n")

    # --- Identity ---
    println(io, "=== RUN INFO ===")
    println(io, "Job name: $job_name")
    println(io, "Run ID: $run_id")
    println(io, "Mock dataset: $mock_name")
    println(io, "Output dir: $(abspath(outdir))")
    println(io, "Resume path: $resume_path")
    println(io)

    # --- True system parameters ---
    println(io, "=== TRUE SIMULATION PARAMETERS ===")
    println(io, "M_IMBH [M☉]: $M_IMBH")
    println(io, "plx [mas]: $plx")
    println(io, "z_prior_sigma [AU]: $z_prior_sigma")
    println(io, "distance [kpc]: $DISTANCE_KPC")
    println(io)

    # --- Noise model ---
    println(io, "=== OBSERVATIONAL MODEL ===")
    println(io, "σ_RA_off [mas]: $SIGMA_RA_OFF")
    println(io, "σ_DEC_off [mas]: $SIGMA_DEC_OFF")
    println(io, "σ_PM_RA [mas/yr]: $SIGMA_PM_RA")
    println(io, "σ_PM_DEC [mas/yr]: $SIGMA_PM_DEC")
    println(io, "σ_ACC_RA [mas/yr²]: $SIGMA_ACC_RA")
    println(io, "σ_ACC_DEC [mas/yr²]: $SIGMA_ACC_DEC")
    println(io, "σ_RV [m/s]: $SIGMA_RV")
    println(io)

    # --- Sampling setup ---
    println(io, "=== SAMPLING CONFIG ===")
    println(io, "n_rounds: $n_rounds")
    println(io, "n_chains: $n_chains")
    println(io, "n_chains_variational: $n_chains_variational")
    println(io, "include acceleration: $INCLUDE_ACC")
    println(io)

    # --- Model structure ---
    println(io, "=== MODEL STRUCTURE ===")
    println(io, "n_stars: $(length(stars))")
    println(io)

    println(io)
end


println("Smmary written to $(summary_path)")

# Load mock dataset
star_params = mock_utils.load_mock_params(mock_name)

# Build mock dataset
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

# Convert each StarData object to Octofitter compatible observations
obs = Dict{String, Any}()

for (name, star) in stars
    obs[name] = mock_utils.build_star_observations(
        star,
        TREF;
        include_rv = true,
        include_acc = INCLUDE_ACC,
        z_prior_sigma = z_prior_sigma,
        rv_cluster = rv_cluster,
        rv_cluster_err = rv_cluster_err
)
end


# === CREATE PLANETS
# Priors matching starsACDEF_192c_16r_10836874
# Config file: /lustre09/project/6039459/vhenault/OCen_IMBH/Ocen_IMBH_analysis/configs/accel_escvel_16r_position.toml

planets = Planet[]

for (name, (astrom, pm, acc, rv, zp)) in obs

    obs_list = Any[
        ObsPriorAstromONeil2019(astrom),
        astrom,
        pm,
    ]
    if acc !== nothing
    push!(obs_list, acc)
    end

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


# === SYSTEM MODEL
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

# === BUILD LIKELIHOOD MODEL
model = Octofitter.LogDensityModel(sys)

println("Model compiled. Number of free parameters: $(model.D)")


# === INFERENCE WITH PIGEONS

kwargs = Dict(
    :n_rounds => n_rounds,
    :n_chains => n_chains,
    :n_chains_variational => n_chains_variational,
    :checkpoint => true,
)

is_resume = resume_path !== nothing

if is_resume
    println("Resume mode: loading from $resume_path")

    isdir(resume_path) || error("Resume path not found: $resume_path")

    pt_prev = Pigeons.PT(resume_path)

    n_done = pt_prev.inputs.n_rounds
    n_additional = n_rounds - n_done

    n_additional > 0 || error("n_rounds must be > completed rounds")

    println("Already done: $n_done → running additional: $n_additional")

    Pigeons.increment_n_rounds!(pt_prev, n_additional)

    chain_pt = octofit_pigeons(pt_prev)
    chain = chain_pt.chain
    pt = chain_pt.pt

else
    println("Fresh run")

    chain, pt = octofit_pigeons(model; kwargs...)
end


# === SAVE PT CHECKPOINT LOCATION

pt_exec_folder = try
    pt.exec_folder
catch
    "unknown"
end

write(joinpath(outdir, "pt_location.txt"), pt_exec_folder)

println("PT folder: $pt_exec_folder")


# === SAVE CHAIN

chain_path = joinpath(outdir, "$(run_id)_chain.fits")

Octofitter.savechain(chain_path, chain)


# === MASS RECOVERY?
println("\n=== IMBH MASS RECOVERY? ===")

M_samples = chain[:M]
M_median = median(M_samples)

println("True mass: ", M_IMBH)
println("Recovered median M [M☉]: ", quantile(M_samples, [0.16, 0.84]))

        