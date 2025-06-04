import pandas as pd
import numpy as np


def simulate_experiments_batch(experiment_numbers, config, params):
    """
    Simulate multiple experiments in a vectorized batch for better performance.
    """
    assert not (
        config["has_selection_bias"] and not config["has_covariate"]
    ), "Selection bias cannot be simulated without a covariate."

    # Extract parameters
    n = params["n"]
    true_effect = params["true_effect"]
    target_pre_experiment_mean = params["target_pre_experiment_mean"]
    target_std = params["target_std"]
    has_covariate = config["has_covariate"]
    has_selection_bias = config["has_selection_bias"]

    num_experiments = len(experiment_numbers)
    total_samples = num_experiments * n

    # Pre-allocate structured array for all data
    dtype = [
        ("experiment_number", "i4"),
        ("pre_experiment", "f8"),
        ("post_experiment", "f8"),
        ("change", "f8"),
        ("is_treatment", "i4"),
        ("true_effect", "f8"),
    ]

    if has_covariate:
        dtype.append(("covariate", "f8"))

    data_array = np.zeros(total_samples, dtype=dtype)

    # Vectorized random number generation for all experiments at once
    np.random.seed(experiment_numbers[0])  # Use first experiment number as base seed

    # Generate all random numbers at once
    pre_experiment_all = np.random.normal(
        target_pre_experiment_mean, target_std, size=total_samples
    )
    post_noise_all = np.random.normal(0, target_std, size=total_samples)
    treatment_rand_all = np.random.rand(total_samples)

    covariate_all = None
    if has_covariate:
        covariate_min = params["covariate_min"]
        covariate_max = params["covariate_max"]
        covariate_all = np.random.uniform(
            covariate_min, covariate_max, size=total_samples
        )

    # Process each experiment's slice
    for i, exp_num in enumerate(experiment_numbers):
        start_idx = i * n
        end_idx = start_idx + n

        # Set unique seed for this experiment to maintain reproducibility
        np.random.seed(exp_num)

        # Extract slices for this experiment
        pre_experiment = pre_experiment_all[start_idx:end_idx]
        post_noise = post_noise_all[start_idx:end_idx]
        treatment_rand = treatment_rand_all[start_idx:end_idx]

        # Initialize variables
        covariate_effect = 0
        selection_impact = 0
        covariate = np.zeros(n)

        if has_covariate:
            covariate = covariate_all[start_idx:end_idx]
            covariate_effect = params["covariate_effect"]

            if has_selection_bias:
                covariate_selection_bias = params["covariate_selection_bias"]
                selection_impact = covariate * covariate_selection_bias

        is_treatment = (treatment_rand < 0.5 + selection_impact).astype(np.int32)
        # NOTE: we are not using the pre-experiment values but their mean because
        # we don't want the noise in pre-experiment to be cumulative to the post-experiment.
        # Post-experiment has its own noise.
        post_experiment = (
            target_pre_experiment_mean
            + true_effect * is_treatment
            + covariate * covariate_effect
            + post_noise
        )

        # Fill structured array
        data_array["experiment_number"][start_idx:end_idx] = exp_num
        data_array["pre_experiment"][start_idx:end_idx] = pre_experiment
        data_array["post_experiment"][start_idx:end_idx] = post_experiment
        data_array["change"][start_idx:end_idx] = post_experiment - pre_experiment
        data_array["is_treatment"][start_idx:end_idx] = is_treatment
        data_array["true_effect"][start_idx:end_idx] = true_effect

        if has_covariate:
            data_array["covariate"][start_idx:end_idx] = covariate

    return data_array


def simulate_experiment(seed, has_covariate, has_selection_bias, params):
    """
    Legacy function for single experiment simulation - kept for compatibility.
    """
    config = {"has_covariate": has_covariate, "has_selection_bias": has_selection_bias}

    data_array = simulate_experiments_batch([seed], config, params)

    # Convert to DataFrame
    df = pd.DataFrame(data_array)
    return df
