module octo_utils_julia_MCMC_centre

# ========== Environment variables ==========
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

# ========== Imports ==========

using Unitful
using UnitfulAstro
using LinearAlgebra
using Octofitter  
using Statistics


# ========== Define StarData Type ==========
"""
Utilities for organizing and fitting multi-star astrometric data with Octofitter.
Includes:
- StarData struct for storing motion and error data
- stars dictionary with input for each star
- Functions to build models, propagate errors, and run Octofitter fits and plots
"""
struct StarData
    name::String
    ra::Float64         # Right Ascension position in degrees
    dec::Float64        # Declination position in degrees
    pm_ra::Float64      # Proper motion in RA (mas/year)
    pm_dec::Float64     # Proper motion in Dec (mas/year)
    acc_ra::Float64     # Acceleration in RA (mas/year²)
    acc_dec::Float64    # Acceleration in Dec (mas/year²)
    sigma_pm_ra::Float64
    sigma_pm_dec::Float64
    sigma_acc_ra::Float64
    sigma_acc_dec::Float64
    v2D::Float64
    v2D_err::Float64
    rv::Float64
    rv_err::Float64
end

# ========== Omega Centauri Data ==========
# We find the center to be at (α,δ) = (13:26:47.24, −47:28:46.45).
# from https://iopscience.iop.org/article/10.1088/0004-637X/710/2/1032
# reference 6 in Haberle et al. (2024; Nature, Vol. 631) 
# We assume the IMBH is located at the cluster center

# Center of mass RA and Dec in deg
ra_cm_deg  = 201.6970988
dec_cm_deg = -47.4794533

# Distance to Omega Centauri center (kiloparsecs and converted to km)
distance_kpc = 5.43u"kpc"
distance_km = uconvert(u"km", distance_kpc)

# Assumed errors in position (mas)
ra_err = 0.5u"mas"
dec_err = 0.5u"mas"

# ========== Define Error Propagation Function ==========
"""
Propagates uncertainty in position due to uncertainties in
proper motion and acceleration over time. No uncertainty in initial position is available.

Parameters:
- sigma_pos : Initial uncertainty in position (mas)
- sigma_pm  : Uncertainty in proper motion (mas/yr)
- sigma_acc : Uncertainty in acceleration (mas/yr²)
- dt        : Time from reference epoch (in years)

Returns:
- sigma_pos : Total uncertainty in predicted position at time dt (mas)
"""
function propagate_error(sigma_pos, sigma_pm, sigma_acc, dt)
    term_pos = sigma_pos^2                      
    term_pm = (dt * sigma_pm)^2                 
    term_acc = (0.5 * dt^2 * sigma_acc)^2      

    sqrt(term_pos + term_pm + term_acc)
end

# ========== Define Position Projection Function ==========
"""
Gives a fake observed position using an observed angular position, velocity, and acceleration

Parameters:
- pos : Initial position (in mas)
- pm : Proper motion (in mas/yr)
- acc : Acceleration (in mas/yr²)
- dt : Time offset(s) from the reference epoch in years

Returns:
- pos_final : calculated position (in mas)
"""
function fake_pos(pos, pm, acc, dt)
    pos + pm*dt + 0.5*acc*dt^2
end

# ========== Define Star Dictionary ==========
stars = Dict{String,StarData}(
    "A" => StarData(
        "A",
        201.6967263, # deg
        -47.4795835, # deg  |
        3.563, # mas/year
        2.564, # mas/year
        -0.0069, # mas/year²
        0.0085,  # mas/year²
        0.038,   # mas/year
        0.055,   # mas/year
        0.0083,  # mas/year²
        0.0098,  # mas/year²
        113.0,   # km/s
        1.1,      # km/s error
        NaN,
        NaN
    ),
    "B" => StarData(
        "B",
        201.6968888,
        -47.4797138,
        2.167,
        1.415,
        0.0702,
        0.0228,
        0.182,
        0.081,
        0.0239,
        0.0157,
        66.6,
        4.1,
        NaN,
        NaN
    ),
    "C" => StarData(
        "C",
        201.6966378,
        -47.4793672,
        1.117,
        3.514,
        0.0028,
        -0.0060,
        0.127,
        0.056,
        0.0333,
        0.0123,
        94.9,
        1.7,
        NaN,
        NaN
    ),
    "D" => StarData(
        "D",
        201.6968346,
        -47.4793233,
        2.559,
        -1.617,
        0.0357,
        -0.0194,
        0.082,
        0.061,
        0.0177,
        0.0162,
        77.9,
        2.0,
        NaN,
        NaN
    ),
    "E" => StarData(
        "E",
        201.6973080,
        -47.4797545,
        -2.149,
        1.638,
        0.0072,
        -0.0009,
        0.025,
        0.037,
        0.0042,
        0.0075,
        69.6,
        0.8,
        261700.0,
        2700.0
    ),
    "F" => StarData(
        "F",
        201.6977125,
        -47.4792625,
        0.436,
        -2.584,
        0.0052,
        -0.0015,
        0.017,
        0.016,
        0.0038,
        0.0038,
        67.4,
        0.4,
        232500.0,
        4000.0
        ),
    "G" => StarData(
        "G",
        201.6961340,
        -47.4790585,
        -1.317,
        2.207,
        -0.0197,
        0.0173,
        0.098,
        0.062,
        0.0267,
        0.0170,
        66.2,
        1.9,
        NaN,
        NaN
    ),
)

# ========== Astrometry Input for Octofitter ==========
"""
Simulates astrometric positions for a star at three epochs: past, present, and future.

Parameters:
- star: StarData object containing motion and error data
- epoch: Central epoch in calendar years (e.g., 2010)
- dt: Time offset from the central epoch in years (e.g., 10)

Returns:
- epochs_mjd: Modified Julian Dates for the three epochs
- ra_rel: Relative right ascension values at the three epochs (mas)
- dec_rel: Relative declination values at the three epochs (mas)
- ra_errs: Measurement errors for RA at the three epochs (mas)
- dec_errs: Measurement errors for Dec at the three epochs (mas)
"""

function simulate_astrometry(star::StarData, epoch::Real, dt::Real)
    # Epochs in years and MJD
    epochs_years = [epoch - dt, epoch, epoch + dt]
    epochs_mjd = [Octofitter.years2mjd(y) for y in epochs_years]

    # --- Absolute positions in mas ---
    ra0_mas  = star.ra * 3600 * 1000    
    dec0_mas = star.dec * 3600 * 1000

    # --- Propagate positions using proper motion & acceleration  ---
    past_ra_mas   = fake_pos(ra0_mas, star.pm_ra, star.acc_ra, -dt)
    past_dec_mas  = fake_pos(dec0_mas, star.pm_dec, star.acc_dec, -dt)
    future_ra_mas = fake_pos(ra0_mas, star.pm_ra, star.acc_ra, dt)
    future_dec_mas= fake_pos(dec0_mas, star.pm_dec, star.acc_dec, dt)

    ra_abs = [past_ra_mas, ra0_mas, future_ra_mas]
    dec_abs = [past_dec_mas, dec0_mas, future_dec_mas]

    # --- Convert to relative RA/Dec  ---
    ra_rel  = ra_abs .- (ra_cm_deg * 3600 * 1000)
    dec_rel = dec_abs .- (dec_cm_deg * 3600 * 1000)

    # --- Error propagation ---
    past_ra_err   = propagate_error(ustrip(ra_err), star.sigma_pm_ra, star.sigma_acc_ra, -dt)
    past_dec_err  = propagate_error(ustrip(dec_err), star.sigma_pm_dec, star.sigma_acc_dec, -dt)
    future_ra_err = propagate_error(ustrip(ra_err), star.sigma_pm_ra, star.sigma_acc_ra, dt)
    future_dec_err= propagate_error(ustrip(dec_err), star.sigma_pm_dec, star.sigma_acc_dec, dt)

    ra_errs  = [past_ra_err, ustrip(ra_err), future_ra_err]
    dec_errs = [past_dec_err, ustrip(dec_err), future_dec_err]

    return epochs_mjd, ra_rel, dec_rel, ra_errs, dec_errs
end






# ========================================================
#  Angular to Linear Acceleration Conversion
# ========================================================
"""
Convert angular acceleration from milliarcseconds per year squared (mas/yr²)
to linear acceleration in kilometers per second squared (km/s²).

# Arguments
- a_masyr2::Quantity : Angular acceleration with units mas/yr²
- distance_km::Quantity : Distance to the object with units km

# Returns
- Linear acceleration in km/s² as a Quantity
"""
function masyr2_to_kms2(a_masyr2::Unitful.Quantity, distance_km::Unitful.Quantity)
    # 1 mas = 1e-3 arcsec = (1e-3 / 3600) deg = (1e-3 / 3600)*(π/180) rad
    # convert manually:
    a_radyr2 = uconvert(u"rad"/u"yr"^2, a_masyr2 * (1e-3 / 3600) * (π / 180))

    # Linear acceleration = angular acceleration × distance
    # a_radyr2 [rad/yr²] * distance_km [km] = km/yr²
    a_kmyr2 = a_radyr2 * distance_km

    # Convert km/yr² to km/s²
    a_kms2 = uconvert(u"km"/u"s"^2, a_kmyr2)

    return a_kms2
end

# ========================================================
#  Total Angular Accelerations and Uncertainty 
# ========================================================
"""
Compute total plane-of-sky angular and physical acceleration
(and their uncertainties) for a given star.

Parameters
----------
star : StarData
    A star object with acc_ra, acc_dec, sigma_acc_ra, and sigma_acc_dec attributes.

Returns
-------
a_total_masyr2 : Float64
    Total angular acceleration in mas/yr².
a_total_masyr2_err : Float64
    Uncertainty in total angular acceleration.
a_total_kms2 : Float64
    Total physical acceleration in km/s².
a_total_kms2_err : Float64
    Uncertainty in physical acceleration.
"""
function total_accelerations(star::StarData)
    a_ra = star.acc_ra
    a_dec = star.acc_dec
    a_ra_err = star.sigma_acc_ra
    a_dec_err = star.sigma_acc_dec

    # Angular acceleration magnitude
    a_total_masyr2 = sqrt(a_ra^2 + a_dec^2)

    # Uncertainty propagation
    a_total_masyr2_err = sqrt(
        (a_ra * a_ra_err / a_total_masyr2)^2 +
        (a_dec * a_dec_err / a_total_masyr2)^2
    )

    # Convert to physical acceleration [km/s²]
    a_total_kms2 = masyr2_to_kms2(a_total_masyr2 * u"mas/yr^2", distance_km)
    a_total_kms2_err = masyr2_to_kms2(a_total_masyr2_err * u"mas/yr^2", distance_km)

    return a_total_masyr2, a_total_masyr2_err, a_total_kms2, a_total_kms2_err
end

# ========================================================
#  Build Direct Observation Objects for Octofitter v8
# ========================================================
"""
Build PlanetRelAstromObs, PlanetPMObs, and PlanetAccelObs from a StarData object
at a single epoch. Uses the direct kinematic observables rather than synthetic
multi-epoch astrometry.

Parameters:
- star: StarData object with position, PM, and acceleration data
- epoch_mjd: Observation epoch in Modified Julian Date

Returns:
- astrom: PlanetRelAstromObs (single-epoch relative position)
- pm: PlanetPMObs (single-epoch proper motion)
- acc: PlanetAccelObs (single-epoch acceleration)
"""
function build_star_observations(star::StarData, epoch_mjd::Float64)
    # 1. Single-epoch position relative to cluster center
    ra_rel_mas = (star.ra - ra_cm_deg) * 3600 * 1000
    dec_rel_mas = (star.dec - dec_cm_deg) * 3600 * 1000
    astrom = PlanetRelAstromObs(
        (epoch=[epoch_mjd], ra=[ra_rel_mas], dec=[dec_rel_mas],
         σ_ra=[ustrip(ra_err)], σ_dec=[ustrip(dec_err)], cor=[0.0]);
        name="$(star.name)_pos"
    )

    # 2. Proper motion at same epoch
    pm = PlanetPMObs(
        (epoch=[epoch_mjd], pmra=[star.pm_ra], pmdec=[star.pm_dec],
         σ_pmra=[star.sigma_pm_ra], σ_pmdec=[star.sigma_pm_dec], cor=[0.0]);
        name="$(star.name)_pm"
    )

    # 3. Acceleration at same epoch
    acc = PlanetAccelObs(
        (epoch=[epoch_mjd], accra=[star.acc_ra], accdec=[star.acc_dec],
         σ_accra=[star.sigma_acc_ra], σ_accdec=[star.sigma_acc_dec], cor=[0.0]);
        name="$(star.name)_acc"
    )

    return astrom, pm, acc
end

end

