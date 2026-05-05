using JSON3

# On cluster
# base = "/home/kenzhayd/projects/def-vhenault/kenzhayd/mock_cluster/results"

base = raw"C:\Users\macke\Clusters\Ocen_IMBH_analysis\mock_cluster\results"

run_id = "AccDiff-40M_sun_16r_192c"

results_dir = joinpath(base, run_id)
config_path = joinpath(results_dir, "config.json")

config = Dict(
    "mock_name" => "mock_10836842",
    "M_IMBH" => 6.373e4,
    "z_prior_sigma" => 4558.0,
    "plx" => 0.191,
    "n_rounds" => 16,
    "n_chains" => 192,
    "n_chains_variational" => 192,
    "distance_kpc" => 5.43,
    "tref" => 55197.0,
    "sigma_ra_off" => 0.5,
    "sigma_dec_off" => 0.5,
    "sigma_pm_ra" => 0.081,
    "sigma_pm_dec" => 0.053,
    "sigma_acc_ra" => 0.0168,
    "sigma_acc_dec" => 0.0118,
    "sigma_rv" => 3350.0,
    "imbh_ra" => 201.6968333,
    "imbh_dec" => -47.4795694,
    "rv_cluster" => 232780.0,
    "rv_cluster_err" => 210.0,
    "include_acc" => true,
    "job_name" => run_id,
    "results_dir" => results_dir
)

write(config_path, JSON3.write(config; indent=4))

println("Wrote config for $run_id")