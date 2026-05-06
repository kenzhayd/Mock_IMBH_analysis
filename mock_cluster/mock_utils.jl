"""
mock_utils.jl

Module with functions for mock_inference.jl

    ssh kenzhayd@narval.alliancecan.ca
    
mock_cluster overview 
    mock_utils.jl (module)
        1. Define core structs (MockOrbit, StarData) for storing orbits and mock data
        2. Generate orbits for stars around a central mass (make_star)
        3. Extract (noise-free) mock data from MockOrbits (mock_data)
        4. Convert extracted mock data into StarData struct with default uncertainties (stardata_struct)
        5. Build a mock cluster dictionary from orbital parameter list (build_mock_paramset) 
        6. Convert mock data into noisy observation objects for Octofitter (build_star_observations)
        7. Load mock system parameter sets by name (load_mock_params)
    mock_inference.jl
        1. Define mock system parameters mock stellar orbit parameters (based on median values from starsACDEF_192c_18r_cont_10836842)
        2. Build mock dataset 
        3. Convert mock dataset into Octofitter observation objects (astrometry, PM, acceleration, RV, optional LOS prior)
        4. Build Planets (orbiting stars) with priors matching starsACDEF_192c_18r_cont_10836842
        5. Create system model with priors matching starsACDEF_192c_18r_cont_10836842
        6. Create likelihood model and run inference using Pigeons 
        7. Extract posterior statistics and plot results (Code from Ocen_IMBH_analysis)
    mock_job.sh
        1. Set mock system parameters and specify stellar orbital parameter set
        2. Send orbit-fitting job to a cluster
"""

module mock_utils

# === IMPORTS
using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using Distributions
using Test
using Statistics
using Printf
using Random
using Unitful
using Dates
using JSON3


# === ORBIT STRUCT
"""
Links a star name to its computed Keplerian orbit.
"""

struct MockOrbit
    name::String
    orbit::Any
end

"""
Container for a star's ovservable data and uncertainties.
"""
struct StarData
    name::String

    raoff::Float64      # RA offset [mas]
    decoff::Float64     # Dec offset [mas]

    pm_ra::Float64      # Proper motion RA [mas/yr]
    pm_dec::Float64     # Proper motion Dec [mas/yr]

    acc_ra::Float64     # Acceleration RA [mas/yr^2]
    acc_dec::Float64    # Acceleration Dec [mas/yr^2]

    sigma_raoff::Float64   # RA offset uncertainty [mas]
    sigma_decoff::Float64  # Dec offset uncertainty [mas]

    sigma_pm_ra::Float64   # PM RA uncertainty [mas/yr]
    sigma_pm_dec::Float64  # PM Dec uncertainty [mas/yr]

    sigma_acc_ra::Float64   # Accel RA uncertainty [mas/yr^2]
    sigma_acc_dec::Float64  # Accel Dec uncertainty [mas/yr^2]

    rv::Float64          # radial velocity [m/s]
    rv_err::Float64      # radial velocity uncertainty [m/s]
end

"""
RunConfig

Container for all configuration parameters used in mock_inference.jl and mock_plots.jl.

This struct is constructed  from `ARGS` passed by a Slurm job script.
"""
struct RunConfig

    mock_name::String
    M_IMBH::Float64
    z_prior_sigma::Float64
    plx::Float64

    n_rounds::Int
    n_chains::Int
    n_chains_variational::Int

    distance_kpc::Float64
    tref::Real

    sigma_ra_off::Float64
    sigma_dec_off::Float64
    sigma_pm_ra::Float64
    sigma_pm_dec::Float64
    sigma_acc_ra::Float64
    sigma_acc_dec::Float64
    sigma_rv::Float64

    imbh_ra::Float64
    imbh_dec::Float64

    rv_cluster::Float64
    rv_cluster_err::Float64

    include_acc::Bool
    job_name::String
    results_dir::String
end

# === ORBIT GENERATOR
"""
Constructs a mock orbit around the central mass.

Parameters:
- name : Identifier for the star
- a    : Semi-major axis [AU]
- e    : Eccentricity
- i    : Inclination [rad]
- ω    : Argument of periastron [rad]
- Ω    : Longitude of ascending node [rad]
- M    : Central mass [solar masses]
- plx  : Parallax [mas]
- t_ref: Reference epoch (MJD)

Returns:
- MockOrbit containing a Keplerian orbit 
"""

function make_star(name; a,e,i,ω,Ω, M, plx, t_ref)

    θ = 2π * rand()

    tp = θ_at_epoch_to_tperi(
    θ, 55197.0;
    a=a, e=e, i=i, ω=ω, Ω=Ω, M=M
    )

    orbit = Visual{KepOrbit}(;
        a=a,
        e=e,
        i=i,
        ω=ω,
        Ω=Ω,
        M=M,
        tp = tp,
        plx=plx
    )

    return MockOrbit(name, orbit)
end


# === MOCK OBSERVATIONAL DATA FUNCTION
"""
Calculates true observations at a given epoch.

Parameters:
- star  : MockOrbit containing true Keplerian orbit
- epoch : Observation epoch (MJD)

Returns:
- raoff     : Sky-projected RA offset [mas]
- decoff    : Sky-projected Dec offset [mas]
- pmra   : Proper motion in RA [mas/yr]
- pmdec  : Proper motion in Dec [mas/yr]
- accra  : Plane-of-sky acceleration in RA [mas/yr²]
- accdec : Plane-of-sky acceleration in Dec [mas/yr²]
- rv     : Line-of-sight radial velocity [m/s]
"""
function mock_data(star::MockOrbit, epoch::Real)

    sol = orbitsolve(star.orbit, epoch)

    return (
        raoff     = raoff(sol),
        decoff    = decoff(sol),
        pmra   = pmra(sol),
        pmdec  = pmdec(sol),
        accra  = accra(sol),
        accdec = accdec(sol),
        rv     = radvel(sol)
    )
end

# === MOCK DATASET
"""
Convert a MockOrbit observables into StarData struct at a given epoch
"""
function stardata_struct(name;
    a, e, i, ω, Ω,
    M,
    plx,
    t_ref,
    epoch,
    sigma_ra_off,
    sigma_dec_off,
    sigma_pm_ra,
    sigma_pm_dec,
    sigma_acc_ra,
    sigma_acc_dec,
    sigma_rv
)

    orbit = make_star(name; a, e, i, ω, Ω, M=M, plx=plx, t_ref=t_ref)
    obs = mock_data(orbit, epoch)

    return StarData(
        name,

        obs.raoff,
        obs.decoff,

        obs.pmra,
        obs.pmdec,

        obs.accra,
        obs.accdec,

        sigma_ra_off,
        sigma_dec_off,

        sigma_pm_ra,
        sigma_pm_dec,

        sigma_acc_ra,
        sigma_acc_dec,

        obs.rv,
        sigma_rv
    )
end


# === BUILD MOCK PARAMETER SET 
"""
Build a dictionary of mock `StarData` objects from orbital parameters.

Each entry in `star_params` should be `(name, a, e, i, ω, Ω)`. Global
system parameters (`M`, `plx`, `epoch`) are applied to all stars.
"""
function build_mock_paramset(star_params;
    epoch,
    M,
    plx,
    t_ref,
    sigma_ra_off,
    sigma_dec_off,
    sigma_pm_ra,
    sigma_pm_dec,
    sigma_acc_ra,
    sigma_acc_dec,
    sigma_rv
)

    stars = Dict{String, StarData}()

    for (name, a, e, i, ω, Ω) in star_params
        stars[name] = stardata_struct(
            name;
            a=a, e=e, i=i, ω=ω, Ω=Ω,
            M=M,
            plx=plx,
            t_ref=t_ref,
            epoch=epoch,
            sigma_ra_off=sigma_ra_off,
            sigma_dec_off=sigma_dec_off,
            sigma_pm_ra=sigma_pm_ra,
            sigma_pm_dec=sigma_pm_dec,
            sigma_acc_ra=sigma_acc_ra,
            sigma_acc_dec=sigma_acc_dec,
            sigma_rv=sigma_rv
        )
    end

    return stars
end

# === OBSERVATION OBJECTS FOR OCTOFITTER V8
"""
Parameters:
- star: StarData object with position, PM, acceleration, and (optional) RV data
- epoch_mjd: Observation epoch in Modified Julian Date
- include_rv: Whether to build RV observation (default true; only used if star has RV data)
- z_prior_sigma: If not nothing, build a PlanetZPriorObs with Normal(0, z_prior_sigma) in AU

Returns:
- astrom: PlanetRelAstromObs (single-epoch position)
- pm: PlanetPMObs (single-epoch proper motion)
- acc: PlanetAccelObs (single-epoch acceleration)
- rv: PlanetRelativeRVObs or nothing (single-epoch peculiar radial velocity)
- zp: PlanetZPriorObs or nothing (LOS position prior)
"""

function build_star_observations(star::StarData, epoch_mjd::Real;
    include_rv::Bool=true,
    include_acc::Bool=true,
    z_prior_sigma::Union{Nothing,Float64}=nothing,
    rv_cluster::Float64=0.0,
    rv_cluster_err::Float64=0.0
)
   # Noise is added to mock observations by sampling a gaussian distribution
    # === ASTROMETRY
    astrom = PlanetRelAstromObs(
        (epoch = [epoch_mjd],
        ra    = [rand(Normal(star.raoff,  star.sigma_raoff))],
        dec   = [rand(Normal(star.decoff, star.sigma_decoff))],
        σ_ra  = [star.sigma_raoff],
        σ_dec = [star.sigma_decoff],
        cor   = [0.0]);
        name = "$(star.name)_pos"
    )

    # === PROPER MOTION
    pm = PlanetPMObs(
        (epoch = [epoch_mjd],
        pmra  = [rand(Normal(star.pm_ra,  star.sigma_pm_ra))],
        pmdec = [rand(Normal(star.pm_dec, star.sigma_pm_dec))],
        σ_pmra  = [star.sigma_pm_ra],
        σ_pmdec = [star.sigma_pm_dec],
        cor = [0.0]);
        name = "$(star.name)_pm"
    )

    # === ACCELERATION
    acc = nothing
    if include_acc
        acc = PlanetAccelObs(
            (epoch = [epoch_mjd],
            accra  = [rand(Normal(star.acc_ra,  star.sigma_acc_ra))],
            accdec = [rand(Normal(star.acc_dec, star.sigma_acc_dec))],
            σ_accra  = [star.sigma_acc_ra],
            σ_accdec = [star.sigma_acc_dec],
            cor = [0.0]);
            name = "$(star.name)_acc"
        )
    end

    # === RADIAL VELOCITY
    rv = nothing
    if include_rv && !isnan(star.rv) && !isnan(star.rv_err)

        rv_peculiar = star.rv - rv_cluster
        σ_rv_total  = hypot(star.rv_err, rv_cluster_err)

        rv = PlanetRelativeRVObs(
            (epoch = epoch_mjd,
            rv    = rand(Normal(rv_peculiar, σ_rv_total)),
            σ_rv  = σ_rv_total);
            name = "$(star.name)_rv",
            variables = @variables begin end
        )
    end

    # LOS PRIOR 
    zp = nothing
    if z_prior_sigma !== nothing
        zp = PlanetZPriorObs(
            epoch_mjd,
            Normal(0.0, z_prior_sigma); #AU
            name = "$(star.name)_zprior"
        )
    end

    return astrom, pm, acc, rv, zp
end

# === Mock system parameters
"""
Returns a list of mock stellar orbital parameters.
Angular parameters are converted from degrees to radians internally.
"""

function load_mock_params(mock_name::String)
    
    deg2rad(x) = x * π / 180 # Octofitter output give angular parameters in degrees

    if mock_name == "mock_10836842"
        # === MOCK SYSTEM OBSERVATIONS BASED ON starsACDEF_192c_18r_cont_10836842
        # Median orbital parameter values from Ocen_IMBH_analysis: results/run_outputs/starsACDEF_192c_18r_cont_10836842_posterior_statstartxt 
        # Each tuple defines ONE star:
        # (name, semi-major axis [AU], eccentricity, inclination [rad],
        # argument of periastron [rad], longitude of ascending node [rad])
        return [
                ("A", 5566.495, 0.620, deg2rad(135.778), deg2rad(-6.152), deg2rad(16.608)),
                ("C", 4943.388, 0.173, deg2rad(68.993),  deg2rad(-49.672), deg2rad(-124.782)),
                ("D", 7195.788, 0.209, deg2rad(82.842),  deg2rad(57.655),  deg2rad(120.300)),
                ("E", 10795.598,0.754, deg2rad(75.385),  deg2rad(152.369), deg2rad(133.559)),
                ("F", 12100.584,0.054, deg2rad(14.530),  deg2rad(-23.719), deg2rad(-6.109))
        ]

    elseif mock_name == "mock_test"
        return [
            ("A", 6000.0, 0.5, deg2rad(120.0), deg2rad(10.0), deg2rad(30.0)),
            ("B", 8000.0, 0.3, deg2rad(70.0),  deg2rad(-20.0), deg2rad(100.0)),
        ]
    
    else
        error("Unknown mock dataset: $mock_name")
    
    end
end  

# === Configuration 

# Make run ID
function make_run_id(job_name, n_rounds, n_chains)
    return "$(job_name)_$(n_rounds)r_$(n_chains)c"
end

"""
parse_args(args)

Parse `key=value` CLI arguments into a dictionary.
"""
function parse_args(args::Vector{String})
    out = Dict{String, String}()

    for arg in args
        occursin("=", arg) || error("Invalid argument: $arg (expected key=value)")
        k, v = split(arg, "=", limit=2)
        out[k] = v
    end

    return out
end

"""
run_config(args)

Build `RunConfig` from keyword arguments.
"""
function run_config(args::Vector{String})
    d = parse_args(args)

    get(k) = haskey(d, k) ? d[k] : error("Missing argument: $k")

    return RunConfig(
        get("mock_name"),
        parse(Float64, get("M_IMBH")),
        parse(Float64, get("z_prior_sigma")),
        parse(Float64, get("plx")),

        parse(Int, get("n_rounds")),
        parse(Int, get("n_chains")),
        parse(Int, get("n_chains_variational")),

        parse(Float64, get("distance_kpc")),
        parse(Float64, get("tref")),

        parse(Float64, get("sigma_ra_off")),
        parse(Float64, get("sigma_dec_off")),
        parse(Float64, get("sigma_pm_ra")),
        parse(Float64, get("sigma_pm_dec")),
        parse(Float64, get("sigma_acc_ra")),
        parse(Float64, get("sigma_acc_dec")),
        parse(Float64, get("sigma_rv")),

        parse(Float64, get("imbh_ra")),
        parse(Float64, get("imbh_dec")),

        parse(Float64, get("rv_cluster")),
        parse(Float64, get("rv_cluster_err")),

        lowercase(get("include_acc")) == "true",
        get("job_name"),
        get("results_dir")
    )
end

"""
run_config(args)

Load run configuration from either a JSON file or CLI-style arguments.

- If `args` contains a single `.json` path, the configuration is read from
  that file using JSON3.
- Otherwise, `args` is parsed as `key=value` pairs via `parse_args`.
"""
function run_config_plots(args::Vector{String})

    # CASE 1: single JSON file
    if length(args) == 1 && endswith(args[1], ".json")
        return JSON3.read(read(args[1], String), RunConfig)
    end

    # CASE 2: key=value CLI args
    cfg = parse_args(args)
    return cfg
end

end
