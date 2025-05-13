import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def t_test(df):
    """
    Perform a t-test on the post-experiment data.
    """

    model = smf.ols("post_experiment ~ is_treatment", data=df).fit()
    p_value = model.pvalues["is_treatment"]
    estimate = model.params["is_treatment"]
    ci = model.conf_int().iloc[1].values

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def t_test_on_change(df):
    """
    Perform a t-test on the change in post-experiment data.
    """

    model = smf.ols("change ~ is_treatment", data=df).fit()
    p_value = model.pvalues["is_treatment"]
    estimate = model.params["is_treatment"]
    ci = model.conf_int().iloc[1].values

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def autoregression(df):
    """
    Perform an old value adjusted t-test on the post-experiment data.
    """

    model = smf.ols("post_experiment ~ pre_experiment + is_treatment", data=df).fit()
    p_value = model.pvalues["is_treatment"]
    estimate = model.params["is_treatment"]
    ci = model.conf_int().iloc[1].values
    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def cuped(df):
    """
    Perform a CUPED t-test on the post-experiment data.
    """

    theta = (
        smf.ols("post_experiment ~ pre_experiment", data=df)
        .fit()
        .params["pre_experiment"]
    )

    model = smf.ols(
        "post_experiment_cuped ~ is_treatment",
        data=df.assign(
            post_experiment_cuped=df["post_experiment"]
            - theta * (df["pre_experiment"] - np.mean(df["pre_experiment"]))
        ),
    ).fit()
    p_value = model.pvalues["is_treatment"]
    estimate = model.params["is_treatment"]
    ci = model.conf_int().iloc[1].values
    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def diff_in_diff(df):
    """
    Perform a difference-in-differences t-test on the post-experiment data.
    """

    # change the data to long format where for each observation we have two rows:
    # one for pre-experiment and one for post-experiment
    df_long = pd.concat(
        [
            pd.DataFrame(
                {
                    "participant": df.index,
                    "time": 0,
                    "target": df["pre_experiment"],
                    "is_treatment": df["is_treatment"],
                }
            ),
            pd.DataFrame(
                {
                    "participant": df.index,
                    "time": 1,
                    "target": df["post_experiment"],
                    "is_treatment": df["is_treatment"],
                }
            ),
        ],
        ignore_index=True,
    )

    # NOTE: Adjusting the model to take into account that the observations from the
    # same participant are not independent (this reduces the standard errors)
    model = smf.ols("target ~ time * is_treatment", data=df_long).fit(
        cov_type="cluster", cov_kwds={"groups": df_long["participant"]}
    )
    p_value = model.pvalues["time:is_treatment"]
    estimate = model.params["time:is_treatment"]
    ci = model.conf_int().iloc[3].values
    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }
