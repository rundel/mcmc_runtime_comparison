import sys
import os

# Must be set before JAX is imported
if len(sys.argv) > 2:
    if sys.argv[2] == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        # Expose enough virtual CPU devices for parallel chains
        if len(sys.argv) > 6 and sys.argv[5] == "parallel":
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={sys.argv[6]}"

from fetch_data import get_pymc_model
from time import time
import jax
import pymc as pm

if __name__ == "__main__":
    start_year = int(sys.argv[1])
    platform = sys.argv[2]
    base_dir = sys.argv[3]
    seed = int(sys.argv[4])
    chain_method = sys.argv[5]
    cores = int(sys.argv[6])
    chains = int(sys.argv[7])

    assert platform in ["cpu", "gpu"]

    # For GPU parallel, cap chains to available device count
    if platform == "gpu" and chain_method == "parallel":
        available = jax.local_device_count()
        if chains > available:
            print(f"Warning: requested {chains} chains but only {available} GPU(s) available; capping to {available}.")
            chains = available

    target_dir = f"{base_dir}/pymc_blackjax_{platform}_{chain_method}"

    os.makedirs(target_dir, exist_ok=True)

    model = get_pymc_model(start_year=start_year)

    start_time = time()

    with model:
        hierarchical_trace = pm.sample(
            nuts_sampler="blackjax", chains=chains, random_seed=seed,
            nuts_sampler_kwargs={"chain_method": chain_method},
            idata_kwargs={'log_likelihood': False},
            progressbar=chain_method != "vectorized")

    runtime = time() - start_time

    hierarchical_trace.to_netcdf(os.path.join(target_dir, f"samples_{start_year}.netcdf"))
    print(runtime, file=open(os.path.join(target_dir, f"runtime_{start_year}.txt"), "w"))
