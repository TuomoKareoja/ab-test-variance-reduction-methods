import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def build_formula(base_formula, covariate=False):
    """
    Build regression formula with optional covariate.
    """
    formula = base_formula
    if covariate:
        formula += " + covariate"
    return formula


def extract_treatment_results(model):
    """
    Extract p-value, estimate, and confidence interval for treatment effect.
    """
    p_value = model.pvalues["is_treatment"]
    estimate = model.params["is_treatment"]
    ci = model.conf_int().iloc[1].values

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def extract_did_results(model):
    """
    Extract results for difference-in-differences interaction term.
    """
    p_value = model.pvalues["time:is_treatment"]
    estimate = model.params["time:is_treatment"]
    ci = model.conf_int().iloc[3].values

    return {
        "p_value": p_value,
        "estimate": estimate,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }


def t_test(df, covariate=False):
    """
    Perform a t-test on the post-experiment data with or without a covariate.
    """
    formula = build_formula("post_experiment ~ is_treatment", covariate)
    model = smf.ols(formula, data=df).fit()
    return extract_treatment_results(model)


def t_test_on_change(df, covariate=False):
    """
    Perform a t-test on the change in post-experiment data with or without a covariate.
    """
    formula = build_formula("change ~ is_treatment", covariate)
    model = smf.ols(formula, data=df).fit()
    return extract_treatment_results(model)


def autoregression(df, covariate=False):
    """
    Perform an old value adjusted t-test on the post-experiment data with or without a covariate.
    """
    formula = build_formula(
        "post_experiment ~ pre_experiment + is_treatment", covariate
    )
    model = smf.ols(formula, data=df).fit()
    return extract_treatment_results(model)


def _calculate_cuped_adjustment(df, covariate=False):
    """
    Calculate CUPED adjustment for the post-experiment values.
    """
    # Cache means to avoid recalculation
    pre_mean = np.mean(df["pre_experiment"])

    if covariate:
        thetas = (
            smf.ols("post_experiment ~ pre_experiment + covariate", data=df)
            .fit()
            .params
        )
        cov_mean = np.mean(df["covariate"])

        adjustment = thetas["pre_experiment"] * (
            df["pre_experiment"] - pre_mean
        ) + thetas["covariate"] * (df["covariate"] - cov_mean)
    else:
        theta = (
            smf.ols("post_experiment ~ pre_experiment", data=df)
            .fit()
            .params["pre_experiment"]
        )
        adjustment = theta * (df["pre_experiment"] - pre_mean)

    return df["post_experiment"] - adjustment


def cuped(df, covariate=False):
    """
    Perform a t-test on the CUPED adjusted post-experiment data.
    """
    cuped_values = _calculate_cuped_adjustment(df, covariate)
    model = smf.ols(
        "post_experiment_cuped ~ is_treatment",
        data=df.assign(post_experiment_cuped=cuped_values),
    ).fit()

    return extract_treatment_results(model)


def diff_in_diff(df, covariate=False):
    """
    Perform a difference-in-differences t-test on the post-experiment data.
    """
    n = len(df)

    df_long = pd.DataFrame(
        {
            "participant": np.tile(df.index, 2),
            "time": np.repeat([0, 1], n),
            "target": np.concatenate([df["pre_experiment"], df["post_experiment"]]),
            "is_treatment": np.tile(df["is_treatment"], 2),
        }
    )

    formula = build_formula("target ~ time * is_treatment", covariate)

    if covariate:
        df_long["covariate"] = np.tile(df["covariate"], 2)

    # NOTE: Adjusting the model to take into account that the observations from the
    # same participant are not independent (this reduces the standard errors)
    model = smf.ols(formula, data=df_long).fit(
        cov_type="cluster", cov_kwds={"groups": df_long["participant"]}
    )

    return extract_did_results(model)


def evaluate_experiments_batch(grouped_data, experiment_numbers, method_configs):
    """
    Evaluate multiple experiments in a batch for better performance.
    """
    results = []

    for exp_num in experiment_numbers:
        if exp_num not in grouped_data.groups:
            continue

        exp_data = grouped_data.get_group(exp_num)

        for method_config in method_configs:
            estimation_method = method_config["func"]
            use_covariate = method_config["use_covariate"]

            method_name = estimation_method.__name__
            if use_covariate:
                method_name += "_covariate"

            # Skip if trying to use covariate but dataset doesn't have it
            if use_covariate and "covariate" not in exp_data.columns:
                continue

            result = estimation_method(exp_data, covariate=use_covariate)
            result["experiment_number"] = exp_num
            result["method"] = method_name
            result["true_effect"] = exp_data["true_effect"].iloc[0]
            results.append(result)

    return results
