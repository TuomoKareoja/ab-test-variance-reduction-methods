# %%

import os
import pandas as pd

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
        "scenario_name": "pre_and_post",
        "data": df_no_covariate,
        "experiments": df_no_covariate["experiment_number"].max() + 1,
        "methods": [
            {"func": t_test, "use_covariate": False},
            {"func": t_test_on_change, "use_covariate": False},
            {"func": autoregression, "use_covariate": False},
            {"func": diff_in_diff, "use_covariate": False},
            {"func": cuped, "use_covariate": False},
        ],
        "output_path": os.path.join("results", "pre_and_post.parquet"),
    },
    {
        "scenario_name": "covariate",
        "data": df_covariate,
        "experiments": df_covariate["experiment_number"].max() + 1,
        "methods": [
            {"func": t_test, "use_covariate": False},
            {"func": t_test, "use_covariate": True},
            {"func": t_test_on_change, "use_covariate": False},
            {"func": t_test_on_change, "use_covariate": True},
            {"func": autoregression, "use_covariate": False},
            {"func": autoregression, "use_covariate": True},
            {"func": diff_in_diff, "use_covariate": False},
            {"func": diff_in_diff, "use_covariate": True},
            {"func": cuped, "use_covariate": False},
            {"func": cuped, "use_covariate": True},
        ],
        "output_path": os.path.join("results", "covariate.parquet"),
    },
    {
        "scenario_name": "selection_bias",
        "data": df_selection_bias,
        "experiments": df_selection_bias["experiment_number"].max() + 1,
        "methods": [
            {"func": t_test, "use_covariate": False},
            {"func": t_test, "use_covariate": True},
            {"func": t_test_on_change, "use_covariate": False},
            {"func": t_test_on_change, "use_covariate": True},
            {"func": autoregression, "use_covariate": False},
            {"func": autoregression, "use_covariate": True},
            {"func": diff_in_diff, "use_covariate": False},
            {"func": diff_in_diff, "use_covariate": True},
            {"func": cuped, "use_covariate": False},
            {"func": cuped, "use_covariate": True},
        ],
        "output_path": os.path.join("results", "selection_bias.parquet"),
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
    for method_config in config["methods"]:
        estimation_method = method_config["func"]
        use_covariate = method_config["use_covariate"]

        # Create method name with covariate suffix if applicable
        method_name = estimation_method.__name__
        if use_covariate:
            method_name += "_covariate"

        logger.info(f"Running method: {method_name}")

        for experiment_number in trange(config["experiments"], desc=method_name):
            # Filter the data for the current experiment number
            filtered_data = config["data"].loc[
                config["data"]["experiment_number"] == experiment_number
            ]

            # Skip if trying to use covariate but the dataset doesn't have it
            if use_covariate and "covariate" not in filtered_data.columns:
                logger.warning(
                    f"Cannot use covariate with {config['scenario_name']} as it doesn't contain covariate data"
                )
                break

            # Run the method on the filtered data
            result = estimation_method(filtered_data, covariate=use_covariate)
            result["experiment_number"] = experiment_number
            result["method"] = method_name
            result["true_effect"] = filtered_data["true_effect"].mean()

            results.append(result)

    # create results df from the result dictionaries
    results_df = pd.DataFrame(results)

    # Save the results
    results_df.to_parquet(config["output_path"])
    logger.info(f"Saved results to {config['output_path']}")

# %%
