# %%

import os
import pandas as pd

import logging
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from src.evaluate import (
    t_test,
    autoregression,
    cuped,
    diff_in_diff,
    evaluate_experiments_batch,
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
if not os.path.exists("results"):
    os.makedirs("results")

# %%


def run_experiments_batch(args):
    """Process a batch of experiments for better performance."""
    grouped_data, experiment_numbers, method_configs = args
    return evaluate_experiments_batch(grouped_data, experiment_numbers, method_configs)


# %%

# Leave one CPU core free for other processes
max_workers = max(1, multiprocessing.cpu_count() - 1)


def run_scenario(config):
    print(f"Running scenario: {config['scenario_name']}")

    # Pre-group data once for efficient access
    grouped_data = config["data"].groupby("experiment_number")

    # Calculate optimal batch size
    batch_size = max(1, config["experiments"] // (max_workers * 4))

    # Create batches of experiment numbers
    experiment_batches = []
    for i in range(0, config["experiments"], batch_size):
        batch = list(range(i, min(i + batch_size, config["experiments"])))
        experiment_batches.append(batch)

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in experiment_batches:
            future = executor.submit(
                run_experiments_batch, (grouped_data, batch, config["methods"])
            )
            futures.append(future)

        for result in concurrent.futures.as_completed(futures):
            batch_results = result.result()
            results.extend(batch_results)

    results_df = pd.DataFrame(results)
    results_df.to_parquet(config["output_path"])
    logger.info(f"Saved results to {config['output_path']}")


# %%

# Run scenarios in parallel for better performance
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    scenario_futures = [
        executor.submit(run_scenario, config) for config in scenario_config
    ]

    for future in tqdm(
        concurrent.futures.as_completed(scenario_futures),
        total=len(scenario_futures),
        desc="Overall progress",
    ):
        future.result()  # Wait for completion

# %%
