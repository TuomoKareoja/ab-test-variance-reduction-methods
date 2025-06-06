import pytest
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from src.evaluate import autoregression


def test_autoregression_confidence_interval_matches_statsmodels():
    np.random.seed(0)
    n = 20
    df = pd.DataFrame({
        "pre_experiment": np.random.normal(size=n),
        "post_experiment": np.random.normal(size=n),
        "is_treatment": [0, 1] * (n // 2),
        "covariate": np.random.normal(size=n),
    })

    # Results from library function
    results = autoregression(df, covariate=True)

    # Fit the same model directly
    formula = "post_experiment ~ pre_experiment + is_treatment + covariate"
    model = smf.ols(formula, data=df).fit()
    ci = model.conf_int().loc["is_treatment"]

    assert results["ci_lower"] == pytest.approx(ci[0])
    assert results["ci_upper"] == pytest.approx(ci[1])
