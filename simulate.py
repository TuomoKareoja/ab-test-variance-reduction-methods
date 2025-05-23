# %%

import os
import pandas as pd
import logging
import dvc.api
from src.simulate import simulate_experiment
from tqdm import tqdm
import concurrent.futures

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


def run_simulation(experiment_number, config, params):
    df = simulate_experiment(
        seed=experiment_number,
        has_covariate=config["has_covariate"],
        has_selection_bias=config["has_selection_bias"],
        params=params,
    )
    df["experiment_number"] = experiment_number
    return df


# %%

for config in simulation_configurations:
    simulations = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        with tqdm(
            total=experiments, desc=f"Simulating {config['scenario_name']}"
        ) as pbar:
            for experiment_number in range(experiments):
                future = executor.submit(
                    run_simulation, experiment_number, config, params
                )
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                simulations.append(future.result())
                pbar.update(1)

    simulations_df = pd.concat(simulations, ignore_index=True)

    output_path = os.path.join("experiments", f'{config["scenario_name"]}.parquet')
    simulations_df.to_parquet(output_path)

# %%
