# %%

import os
import pandas as pd
import numpy as np
import dvc.api
from tqdm import tqdm

import logging
from tqdm import trange
from src.evaluate import (
    t_test,
    t_test_on_change,
    autoregression,
    cuped,
    diff_in_diff,
)

# import the analytical packages needed

# Enable autoreload for Jupyter Notebook
# %load_ext autoreload
# %autoreload 2

# %%

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting evaluation script")

# %%

# load in the experiment datasets
df_no_covariate = pd.read_parquet(os.path.join("experiments", "no_covariate.parquet"))
df_covariate = pd.read_parquet(os.path.join("experiments", "covariate.parquet"))
df_selection_bias = pd.read_parquet(
    os.path.join("experiments", "selection_bias.parquet")
)

# %%


# define which methods to run on each dataset
scenario_config = [
    {
        "scenario_name": "post_only",
        "data": df_no_covariate,
        "experiments": df_no_covariate["experiment_number"].max() + 1,
        "methods": [
            t_test,
        ],
        "output_path": os.path.join("results", "post_only.parquet"),
    },
    {
        "scenario_name": "pre_and_post",
        "data": df_no_covariate,
        "experiments": df_no_covariate["experiment_number"].max() + 1,
        "methods": [
            t_test,
            t_test_on_change,
            autoregression,
            diff_in_diff,
            cuped,
        ],
        "output_path": os.path.join("results", "pre_and_post.parquet"),
    },
]

# %%

# Create results folder if it does not exist
# We need to do this as DVC removes the folder when we run dvc repro
# because we have set the whole folder as a dvc output
if not os.path.exists("results"):
    os.makedirs("results")

for config in scenario_config:

    print(f"Running scenario: {config['scenario_name']}")

    results = []

    # Run each method and save the results to disk
    for estimation_method in config["methods"]:

        for experiment_number in trange(
            config["experiments"], desc=estimation_method.__name__
        ):
            # Filter the data for the current experiment number
            filtered_data = config["data"].loc[
                config["data"]["experiment_number"] == experiment_number
            ]

            # Run the method on the filtered data
            result = estimation_method(filtered_data)
            result["experiment_number"] = experiment_number
            result["method"] = estimation_method.__name__

            results.append(result)

    # create results df from the result dictionaries
    results_df = pd.DataFrame(results)

    results_df.to_parquet(config["output_path"])

# %%
