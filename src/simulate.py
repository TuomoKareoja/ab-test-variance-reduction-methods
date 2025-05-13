import pandas as pd
import numpy as np


def simulate_experiment(seed, has_covariate, has_selection_bias, params):
    """
    Simulate an experiment with different configurations based on the experiment type.

    Parameters:
        seed (int): Random seed for reproducibility.
        has_covariate (bool): Whether to include a covariate in the simulation.
        has_selection_bias (bool): Whether to include selection bias in the simulation.
        params (dict): Parameters for the simulation.

    Returns:
        pd.DataFrame: Simulated experiment data.
    """

    assert not (
        has_selection_bias and not has_covariate
    ), "Selection bias cannot be simulated without a covariate."

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Common Parameters
    n = params["n"]
    true_effect = params["true_effect"]
    target_pre_experiment_mean = params["target_pre_experiment_mean"]
    target_std = params["target_std"]

    # Pre-experiment mean
    pre_experiment = np.random.normal(target_pre_experiment_mean, target_std, size=n)

    # Initialize covariate and selection impact
    covariate = 0
    covariate_effect = 0
    selection_impact = 0

    if has_covariate:
        # Additional parameters for covariates
        covariate_min = params["covariate_min"]
        covariate_max = params["covariate_max"]
        covariate_effect = params["covariate_effect"]

        covariate = np.random.uniform(covariate_min, covariate_max, size=n)

        if has_selection_bias:
            # Parameters for selection bias
            covariate_selection_bias = params["covariate_selection_bias"]
            # Calculate selection impact
            selection_impact = covariate * covariate_selection_bias

    is_treatment = np.where(np.random.rand(n) > 0.5 + selection_impact, 1, 0)

    # Generate the post-only measurements
    post_experiment = np.random.normal(
        pre_experiment + true_effect * is_treatment + covariate * covariate_effect,
        target_std,
        size=n,
    )

    # Create the dataframe
    data = {
        "pre_experiment": pre_experiment,
        "post_experiment": post_experiment,
        "change": post_experiment - pre_experiment,
        "is_treatment": is_treatment,
        "true_effect": true_effect,
    }

    if has_covariate:
        data["covariate"] = covariate

    df = pd.DataFrame(data)

    return df
