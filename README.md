# Variance‑Reduction in A/B Testing

*Bias & Variance of different analysis methods across progressively complex experiments*

---

## 1 ▕ What this repo is about

This project is a **simulation‑driven playground** that shows how different analysis methods behave—in terms of **bias, variance, mean‑squared‑error and confidence‑interval coverage**—when faced with increasing experiment complexity:

| Scenario                         | Data‑generating logic                      | Causal twist                                |
| -------------------------------- | ------------------------------------------ | ------------------------------------------- |
| **S1. Post‑only**                | Treat‑vs‑control, measure outcome once     | Baseline A/B test                           |
| **S2. Pre + Post**               | Baseline metric X<sub>pre</sub> → Y (post) | Exploits correlation for variance‑reduction |
| **S3. Outcome‑only covariate**   | Independent Z → Y                          | ANCOVA usefulness                           |
| **S4. Selection‑bias covariate** | Z → T *and* Z → Y                          | Confounding; need causal adjustment         |

For each scenario the repo benchmarks **six estimators**:

1. Plain post‑experiment **t‑test** (E1)
2. **Change‑score** / delta model (E2)
3. **Diff‑in‑Diff** (E3)
4. **CUPED** (E4)
5. **Simple covariate adjustment** (E5)
6. **Causal‑graph (IPW) adjustment** (E6)

For each experiment variation we run thousands of experiment and analyze the distribution of estimates against true values.
For more hands on understanding of the methods and their difference, there is also a streamlit app that lets you quickly simulate
different experiments and see how the methods fare against each other.

---

## 2 ▕ Folder layout

```text
.
├── .dvc/                # DVC config
├── app/                 # Streamlit UI (main.py, components/)
├── experiments/         # Simulated experiment data
├── notebooks/
│   ├── S0_introduction_to_the_methods.ipynb
│   ├── S1_only_post_experiment_measurement.ipynb
│   ├── S2_pre_and_post_measurements.ipynb
│   ├── S3_covariates.ipynb
│   └── S2_selection_bias_covariate.ipynb
├── plots/               # Plots of the bias and variance of different methods in simulated experiments
├── results/             # Point estimates, confidence intervals and p-values of different methods per experiment
├── src/                 # Helper functions
├── tests/               # Pytest unit & property tests
├── .dvcignore           # What files should DVC ignore
├── .gitignore
├── dvc.lock             # Used by DVC to check which pipeline steps need to be rerun when making changes
├── dvc.yaml             # DVC processing pipeline definition
├── environment.yml      # Conda environment definition
├── evaluate.py          # Apply statistic methods to simulated experiments
├── LICENSE
├── params.yml           # DVC parameter file defining the experiments
├── plot.py              # Plotting the bias and variance of the estimates from different methods
├── README.md
└── simulate.py          # Create simulated experiments
```

> **Note** heavy artifacts (e.g. results/) are excluded via `.gitignore`; use `dvc repro` to regenerate.

---

## 3 ▕ Quick‑start

### 3.1 Install Environment

Install Conda environment
```bash
conda env create -f environment.yml
```

Activate the environment
```bash
conda activate ab-test-variance-reduction-methods
```

### 3.2 Launch the interactive app

```bash
$ streamlit run app/main.py
```

Then open `http://localhost:8501` to play with experiment definitions and how they affect the estimates with different methods. This does not require the that we run the full simulations as the data and estimates on the app are created on the fly.

---

### 3.3 Run the simulations and calculate estimates

```bash
dvc repro
```

This runs all the simulations and applies the methods for each of them. Outputs raw simulation data and estimates to `results` folder.

### 3.4 Run tests

```bash
pytest
```

