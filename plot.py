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

# Load all result files
scenario_files = {
    "post_only": "post_only.parquet",
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

    # Create a faceted boxplot
    plt.figure(figsize=(18, 12))
    g = sns.FacetGrid(combined_df, col="scenario", col_wrap=2, height=6)
    g.map_dataframe(sns.boxplot, x="estimate_error", y="method", orient="h")

    # Add a vertical line at zero for each subplot
    for ax in g.axes.flat:
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Estimate Error (Estimate - True Effect)")
        ax.set_ylabel("Method")

    # Adjust title and layout
    g.fig.suptitle("Estimate Error by Method Across Scenarios", size=16)
    g.fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot
    output_path = os.path.join("plots", "combined_error_boxplot.png")
    g.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined plot to {output_path}")

# %%

# Create KDE plot of errors in 2x2 grid with methods overlapping
if len(data_frames) > 0:
    logger.info("Creating KDE error distribution plot with FacetGrid")

    # Combine all data frames
    combined_df = pd.concat(data_frames.values())

    # Get unique scenarios
    scenarios = ["post_only", "pre_and_post", "covariate", "selection_bias"]
    available_scenarios = [s for s in scenarios if s in data_frames]

    if available_scenarios:
        # Create a 2x2 FacetGrid
        g = sns.FacetGrid(
            combined_df,
            col="scenario",
            col_order=scenarios,
            col_wrap=2,
            height=7,
            aspect=1.2,
            sharex=True,
            sharey=True,
        )

        # Plot KDE for each method within each scenario facet
        g.map_dataframe(
            sns.kdeplot,
            x="estimate_error",
            hue="method",
            fill=False,
            alpha=0.6,
            common_norm=False,
            palette="husl",
        )

        # Add a vertical line at zero for each subplot
        for ax in g.axes.flat:
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.7)

            # Set more readable scenario title
            col_var = ax.get_title().split(" = ")[1]
            ax.set_title(col_var.replace("'", "").replace("_", " ").title())

            # Add legend with mean values
            if ax.get_legend():
                ax.get_legend().remove()

        # Add a common legend at the bottom
        g.add_legend(
            title="Method", bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=3
        )

        # Set axis labels
        g.set_axis_labels("Estimate Error", "Density")

        # Add method means as vertical lines
        for scenario in available_scenarios:
            scenario_df = data_frames[scenario]
            ax_idx = (
                scenarios.index(scenario)
                if scenarios.index(scenario) < len(g.axes)
                else 0
            )

            for method in scenario_df["method"].unique():
                method_data = scenario_df[scenario_df["method"] == method]
                mean_error = method_data["estimate_error"].mean()

                # Get the color from the current palette
                method_idx = combined_df["method"].unique().tolist().index(method)
                method_color = sns.color_palette(
                    "husl", len(combined_df["method"].unique())
                )[method_idx]

                # Add mean annotation
                text_y = 0.95 - (method_idx * 0.05)  # Stack annotations
                g.axes[ax_idx].text(
                    0.95,
                    text_y,
                    f"{method}: μ={mean_error:.4f}",
                    transform=g.axes[ax_idx].transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7),
                    color=method_color,
                    fontsize=9,
                )

        # Add main title
        plt.suptitle(
            "Error Distributions by Scenario with Method Comparison",
            fontsize=16,
            y=1.02,
        )

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
