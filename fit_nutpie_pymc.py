import sys
import os
from fetch_data import get_pymc_model
from time import time
import nutpie

if __name__ == "__main__":
    start_year = int(sys.argv[1])
    target_dir = sys.argv[2] + "/nutpie_pymc"
    seed = int(sys.argv[3])
    cores = int(sys.argv[4])

    os.makedirs(target_dir, exist_ok=True)

    model = get_pymc_model(start_year=start_year)

    start_time = time()

    compiled = nutpie.compile_pymc_model(model)
    trace = nutpie.sample(compiled, draws=1000, tune=1000, chains=cores, cores=cores, seed=seed)

    runtime = time() - start_time

    trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
    print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
