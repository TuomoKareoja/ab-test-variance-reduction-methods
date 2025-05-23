# %%

import os
import pandas as pd
import logging
import dvc.api
from src.simulate import simulate_experiments_batch
from tqdm import tqdm
import multiprocessing
import concurrent.futures
import numpy as np

# %%

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting simulation script")

# %%

# load the simulation parameters from dvc
params = dvc.api.params_show("params.yml")
experiments = params["experiments"]

# %%

# run each of the scenarios and use the params related to that scenario and save the results to disk
simulation_configurations = [
    {
        "has_covariate": False,
        "has_selection_bias": False,
        "scenario_name": "no_covariate",
    },
    {
        "has_covariate": True,
        "has_selection_bias": False,
        "scenario_name": "covariate",
    },
    {
        "has_covariate": True,
        "has_selection_bias": True,
        "scenario_name": "selection_bias",
    },
]


# %%

# Create experiments folder if it does not exist
if not os.path.exists("experiments"):
    os.makedirs("experiments")


# %%


def run_simulation_batch(experiment_numbers, config, params):
    """Run a batch of simulations and return structured array."""
    return simulate_experiments_batch(experiment_numbers, config, params)


# %%

# Leave one CPU core free for other processes
num_workers = max(1, multiprocessing.cpu_count() - 1)

# Calculate optimal batch size based on number of workers
batch_size = max(1, experiments // (num_workers * 4))  # 4 batches per worker

for config in simulation_configurations:
    # Create batches of experiment numbers
    experiment_batches = []
    for i in range(0, experiments, batch_size):
        batch = list(range(i, min(i + batch_size, experiments)))
        experiment_batches.append(batch)

    # Pre-allocate list for results
    all_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        with tqdm(
            total=len(experiment_batches) * batch_size,
            desc=f"Simulating {config['scenario_name']}",
        ) as pbar:
            for batch in experiment_batches:
                future = executor.submit(run_simulation_batch, batch, config, params)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                result_array = future.result()
                all_results.append(result_array)
                pbar.update(batch_size)

    # Concatenate all structured arrays
    combined_array = np.concatenate(all_results)

    # Convert to DataFrame only once at the end
    simulations_df = pd.DataFrame(combined_array)

    output_path = os.path.join("experiments", f'{config["scenario_name"]}.parquet')
    simulations_df.to_parquet(output_path)

# %%
