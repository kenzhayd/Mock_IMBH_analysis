Mock IMBH Orbit Analysis

This repository contains a full simulation and inference pipeline for testing intermediate-mass black hole (IMBH) parameter recovery. The workflow uses Bayesian orbit-fitting package Octofitter (sefffal.github.io/Octofitter.jl/dev/) with additions specific to the orbits of stars around a black hole (https://github.com/vincent-hb/Octofitter_imbh.jl.git).

Workflow overview:

- Generate mock stellar orbits around a central IMBH based on a given orbital parameter set
- Simulate observational data (astrometry, proper motion, acceleration, radial velocity)
- Incorperate realistic measurement uncertainties
- Recover system parameters using Bayesian inference with Octofitter to validate real data fits

Repository Structure: 

mock_cluster/
│
├── mock_utils.jl          # Function library 
├── mock_inference.jl      # Main inference scipt
├── mock_plots.jl          # Posterior plots and diagnostics 
├── create_mock_job.jl     # SLURM job generator / runner
├── mock_oops.jl 			# Builds a config.json file for the runs with the oops
├── mock_oops.jl 
└── results/
    └── (run_id)/
        ├── *_chain.fits
        ├── config.json
        ├── args.txt
        ├── summary.txt
        ├── *_posterior_stats.txt
        ├── *_mock_orbits.png
        ├── *_imbh_map.png
        └── *_accel_check.png

Key Assumptions:

 - Stars follow Keplerian orbits around a central IMBH
 - System parameters defined in create_mock_job.jl

Dependencies:

CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Octofitter = "daf3887e-d01a-44a1-9d7e-98f15c5d69c9"
OctofitterRadialVelocity = "c6a353d9-c9c1-48aa-9c23-64f4679bd07d"
PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
Pigeons = "0eb8d820-af6a-4919-95ae-11206f830c31"
PlanetOrbits = "fd6f9641-d78f-43ce-a379-ceb0bddb468a"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"
UnitfulAstro = "6112ee07-acf9-5e0f-b108-d242c714bf9f"

