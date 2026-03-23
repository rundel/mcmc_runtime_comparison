import sys
import os
from fetch_data import create_arrays
from time import time
import nutpie

if __name__ == "__main__":
    start_year = int(sys.argv[1])
    target_dir = sys.argv[2] + "/nutpie_stan"
    seed = int(sys.argv[3])
    cores = int(sys.argv[4])
    chains = int(sys.argv[5])

    os.makedirs(target_dir, exist_ok=True)

    arrays = create_arrays(start_year=start_year)

    start_time = time()

    winner_ids = arrays["winner_ids"]
    loser_ids = arrays["loser_ids"]
    player_encoder = arrays["player_encoder"]

    compiled = nutpie.compile_stan_model(filename="stan_model_optimised.stan")

    compiled = compiled.with_data(
        n_matches=len(winner_ids),
        n_players=len(player_encoder.classes_),
        winner_ids=(winner_ids + 1).tolist(),
        loser_ids=(loser_ids + 1).tolist(),
    )

    trace = nutpie.sample(compiled, draws=1000, tune=1000, chains=chains, cores=cores, seed=seed)

    runtime = time() - start_time

    trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
    print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
