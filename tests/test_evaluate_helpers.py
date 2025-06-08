import numpy as np
import pandas as pd
import pytest

from src.evaluate_helpers import autoregression, cuped, diff_in_diff, t_test

np.random.seed(0)
n = 10000  # Large sample for accurate estimation
true_beta_pre = 0.5
true_beta_treatment = 2.0
true_beta_covariate = -1.0
true_intercept = 3.0


def create_test_dataframe():
    """
    Create a test DataFrame
    """
    pre_experiment = np.random.normal(size=n)
    is_treatment = np.random.binomial(1, 0.5, size=n)
    covariate = np.random.normal(size=n)
    noise = np.random.normal(scale=0.5, size=n)

    post_experiment = (
        true_intercept
        + true_beta_pre * pre_experiment
        + true_beta_treatment * is_treatment
        + true_beta_covariate * covariate
        + noise
    )

    return pd.DataFrame(
        {
            "pre_experiment": pre_experiment,
            "post_experiment": post_experiment,
            "change": post_experiment - pre_experiment,
            "is_treatment": is_treatment,
            "covariate": covariate,
        }
    )


def evaluate_results(model):
    df = create_test_dataframe()
    results = model(df, covariate=True)

    # The estimated effect should be close to the true parameter
    assert results["estimate"] == pytest.approx(true_beta_treatment, abs=0.05)


def test_t_test():
    evaluate_results(t_test)


def test_diff_in_diff():
    evaluate_results(diff_in_diff)


def test_autoregression():
    evaluate_results(autoregression)


def test_cuped():
    evaluate_results(cuped)
