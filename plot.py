# %%

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# %%

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting plotting script")

# %%

# For each experiment variation load the results and plot them saving the outcome to plots folder
df_post_only = pd.read_parquet(os.path.join("results", "post_only.parquet"))
df_pre_and_post = pd.read_parquet(os.path.join("results", "pre_and_post.parquet"))

# %%

for df in [df_post_only, df_pre_and_post]:
    df["estimate_error"] = df["estimate"] - df["true_effect"]

# %%

os.makedirs("plots", exist_ok=True)

for df in [df_post_only, df_pre_and_post]:

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="estimate_error", y="method", orient="h")
    plt.title("Estimate Error by Method")
    plt.xlabel("Estimate Error")
    plt.ylabel("Method")

    plt.show()

# %%
