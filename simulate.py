# %%

import os
import pandas as pd
import logging
import dvc.api
from src.simulate import simulate_experiment
from tqdm import tqdm

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
# We need to do this as DVC removes the folder when we run dvc repro
# because we have set the whole folder as a dvc output
if not os.path.exists("experiments"):
    os.makedirs("experiments")

for config in simulation_configurations:

    simulations = []

    # Simulate the experiment with progress bar
    with tqdm(total=experiments, desc=f"Simulating {config['scenario_name']}") as pbar:
        for experiment_number in range(experiments):
            df = simulate_experiment(
                seed=experiment_number,
                has_covariate=config["has_covariate"],
                has_selection_bias=config["has_selection_bias"],
                params=params,
            )
            df["experiment_number"] = experiment_number

            simulations.append(df)
            pbar.update(1)

    simulations_df = pd.concat(simulations, ignore_index=True)

    # Save the experiment data to a parquet file
    output_path = os.path.join("experiments", f'{config["scenario_name"]}.parquet')
    simulations_df.to_parquet(output_path)

# %%
