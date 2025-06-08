# %%

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting plotting script")

# %%

# Load all result files
scenario_files = {
    "pre_and_post": "pre_and_post.parquet",
    "covariate": "covariate.parquet",
    "selection_bias": "selection_bias.parquet",
}

data_frames = {}
for scenario, filename in scenario_files.items():
    try:
        file_path = os.path.join("results", filename)
        logger.info(f"Loading results from {file_path}")
        data_frames[scenario] = pd.read_parquet(file_path)
    except FileNotFoundError:
        logger.warning(f"File {file_path} not found. Skipping.")

# %%

# Calculate estimate error for all data frames
for scenario, df in data_frames.items():
    df["estimate_error"] = df["estimate"] - df["true_effect"]
    df["scenario"] = scenario  # Add scenario name for combined plots

# %%

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)
logger.info("Created plots directory")

# %%

# Generate individual scenario plots
for scenario, df in data_frames.items():
    logger.info(f"Creating plot for {scenario}")

    # Create the boxplot
    plt.figure(figsize=(12, 8))

    # Create a boxplot
    sns.boxplot(data=df, x="estimate_error", y="method", orient="h")

    # Add a vertical line at zero for reference
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    # Add labels and title
    plt.title(f"Estimate Error by Method - {scenario.replace('_', ' ').title()}")
    plt.xlabel("Estimate Error (Estimate - True Effect)")
    plt.ylabel("Method")

    # Save the plot
    output_path = os.path.join("plots", f"{scenario}_error_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")

# %%

# Generate a combined plot for all scenarios
if len(data_frames) > 1:
    logger.info("Creating combined plot for all scenarios")

    # Combine all data frames
    combined_df = pd.concat(data_frames.values())

    # Get available scenarios in order
    scenario_order = ["pre_and_post", "covariate", "selection_bias"]
    available_scenarios = [s for s in scenario_order if s in data_frames]

    # Create subplots in a single row
    fig, axes = plt.subplots(
        1,
        len(available_scenarios),
        figsize=(6 * len(available_scenarios), 6),
        sharey=True,
    )
    if len(available_scenarios) == 1:
        axes = [axes]  # Make it iterable for single subplot

    for i, scenario in enumerate(available_scenarios):
        scenario_df = data_frames[scenario]

        # Create boxplot for this scenario
        sns.boxplot(data=scenario_df, x="estimate_error", y="method", orient="h", ax=axes[i])

        # Add vertical line at zero
        axes[i].axvline(x=0, color="r", linestyle="--", alpha=0.5)

        # Set titles and labels
        axes[i].set_title(scenario.replace("_", " ").title())
        axes[i].set_xlabel("Estimate Error (Estimate - True Effect)")
        if i == 0:
            axes[i].set_ylabel("Method")
        else:
            axes[i].set_ylabel("")

    # Add main title
    fig.suptitle("Estimate Error by Method Across Scenarios", size=16)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join("plots", "combined_error_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined plot to {output_path}")

# %%

# Create KDE plot of errors in single row with methods overlapping
if len(data_frames) > 0:
    logger.info("Creating KDE error distribution plot with subplots")

    # Combine all data frames
    combined_df = pd.concat(data_frames.values())

    # Get available scenarios in order
    scenario_order = ["pre_and_post", "covariate", "selection_bias"]
    available_scenarios = [s for s in scenario_order if s in data_frames]

    if available_scenarios:
        # Create subplots in a single row
        fig, axes = plt.subplots(
            1,
            len(available_scenarios),
            figsize=(7 * len(available_scenarios), 6),
            sharex=False,
            sharey=True,
        )
        if len(available_scenarios) == 1:
            axes = [axes]  # Make it iterable for single subplot

        # Get all unique methods for consistent colors
        all_methods = combined_df["method"].unique()
        colors = sns.color_palette("husl", len(all_methods))
        method_colors = dict(zip(all_methods, colors))

        for i, scenario in enumerate(available_scenarios):
            scenario_df = data_frames[scenario]

            # Plot KDE for each method in this scenario
            for method in scenario_df["method"].unique():
                method_data = scenario_df[scenario_df["method"] == method]
                sns.kdeplot(
                    data=method_data,
                    x="estimate_error",
                    label=method,
                    fill=False,
                    alpha=0.7,
                    color=method_colors[method],
                    ax=axes[i],
                )

            # Add vertical line at zero
            axes[i].axvline(x=0, color="r", linestyle="--", alpha=0.7)

            # Set titles and labels
            axes[i].set_title(scenario.replace("_", " ").title())
            axes[i].set_xlabel("Estimate Error")
            if i == 0:
                axes[i].set_ylabel("Density")
            else:
                axes[i].set_ylabel("")

            # Add method mean annotations
            for j, method in enumerate(scenario_df["method"].unique()):
                method_data = scenario_df[scenario_df["method"] == method]
                mean_error = method_data["estimate_error"].mean()

                # Add mean annotation
                text_y = 0.95 - (j * 0.05)
                axes[i].text(
                    0.95,
                    text_y,
                    f"{method}: μ={mean_error:.4f}",
                    transform=axes[i].transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7),
                    color=method_colors[method],
                    fontsize=9,
                )

        # Add main title
        fig.suptitle("Error Distributions by Scenario with Method Comparison", fontsize=16)

        # Adjust spacing
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join("plots", "error_distribution_facet.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved error distribution facet plot to {output_path}")


# %%

logger.info("Plotting completed successfully")

# %%
# %%
