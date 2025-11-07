#!/usr/bin/env python3
"""
Simple script to quickly visualize learning rates from GAN experiments.
Usage: python draw_lr_simple.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Configuration constants
MAX_STEPS = 5000  # Maximum number of training steps to display
FIGURE_SIZE = (14, 8)  # Figure size (width, height)
DPI = 300  # Resolution for saved plots
INDIVIDUAL_PLOTS = True  # Generate individual plots for each experiment
OVERVIEW_PLOT = True  # Generate overview plot with all experiments
FIGURE_PATH = Path("figures")  # Directory to save figures
LR_PATH = FIGURE_PATH / "LR"  # Directory to save learning rate plots
os.makedirs(LR_PATH, exist_ok=True)


def get_experiment_info(log_file_path):
    """Extract experiment information from file path."""
    # Path format: GAN/optimizer/dataset/timestamp/optimizer_log_x.csv
    path_parts = log_file_path.replace("GAN/", "").split("/")
    if len(path_parts) >= 3:
        optimizer = path_parts[0]
        dataset = path_parts[1]
        timestamp = path_parts[2]
        return f"{optimizer}_{dataset}_{timestamp}"
    return "unknown_experiment"


def plot_individual_experiment(exp_name, log_file_x, log_file_y):
    """Plot learning rates for a single experiment."""
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    # Plot generator learning rate
    if log_file_x:
        try:
            df = pd.read_csv(log_file_x)
            df_first = df.groupby("step").first().reset_index()
            df_filtered = df_first[df_first["step"] <= MAX_STEPS]

            ax.plot(
                df_filtered["step"],
                df_filtered["learning_rate"],
                label="Generator",
                linewidth=2,
                alpha=0.8,
                linestyle="-",
                color="blue",
            )
        except Exception as e:
            print(f"Error reading generator log for {exp_name}: {e}")

    # Plot discriminator learning rate
    if log_file_y:
        try:
            df = pd.read_csv(log_file_y)
            df_first = df.groupby("step").first().reset_index()
            df_filtered = df_first[df_first["step"] <= MAX_STEPS]

            ax.plot(
                df_filtered["step"],
                df_filtered["learning_rate"],
                label="Discriminator",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
                color="red",
            )
        except Exception as e:
            print(f"Error reading discriminator log for {exp_name}: {e}")

    # Customize plot
    ax.set_title(
        f'Learning Rate Evolution - {exp_name.replace("_", " ").title()} (First {MAX_STEPS//1000}K Steps)',
        fontsize=16,
    )
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    # Create LR_PATH directory if it doesn't exist
    LR_PATH.mkdir(parents=True, exist_ok=True)

    # Save individual plot
    output_file = LR_PATH / f"learning_rate_{exp_name}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight")
    print(f"Saved individual plot to {output_file}")
    plt.show()
    plt.close()


def plot_overview():
    """Plot overview with all experiments."""
    # Find all optimizer log files (CSV format)
    log_files_x = glob.glob("GAN/**/optimizer_log_x.csv", recursive=True)
    log_files_y = glob.glob("GAN/**/optimizer_log_y.csv", recursive=True)

    print(
        f"Found {len(log_files_x)} generator logs and {len(log_files_y)} discriminator logs"
    )

    if len(log_files_x) == 0 and len(log_files_y) == 0:
        print("No optimizer log files found in GAN directory")
        return

    # Create single plot for overview
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    # Plot generator learning rates
    for log_file in log_files_x:
        try:
            df = pd.read_csv(log_file)
            experiment_name = get_experiment_info(log_file)

            # Select the first learning rate for each step
            df_first = df.groupby("step").first().reset_index()
            df_filtered = df_first[df_first["step"] <= MAX_STEPS]

            ax.plot(
                df_filtered["step"],
                df_filtered["learning_rate"],
                label=f"{experiment_name} (Generator)",
                linewidth=2,
                alpha=0.8,
                linestyle="-",
            )
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    # Plot discriminator learning rates
    for log_file in log_files_y:
        try:
            df = pd.read_csv(log_file)
            experiment_name = get_experiment_info(log_file)

            # Select the first learning rate for each step
            df_first = df.groupby("step").first().reset_index()
            df_filtered = df_first[df_first["step"] <= MAX_STEPS]

            ax.plot(
                df_filtered["step"],
                df_filtered["learning_rate"],
                label=f"{experiment_name} (Discriminator)",
                linewidth=2,
                alpha=0.8,
                linestyle="--",
            )
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    # Customize overview plot
    ax.set_title(
        f"Learning Rate Evolution - All Experiments (First {MAX_STEPS//1000}K Steps)",
        fontsize=16,
    )
    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Learning Rate", fontsize=14)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    # Create LR_PATH directory if it doesn't exist
    LR_PATH.mkdir(parents=True, exist_ok=True)

    # Save overview plot
    overview_file = LR_PATH / "learning_rate_overview.png"
    plt.savefig(overview_file, dpi=DPI, bbox_inches="tight")
    print(f"Saved overview plot to {overview_file}")
    plt.show()
    plt.close()


def quick_plot():
    """Generate learning rate plots for GAN experiments."""

    # Find all optimizer log files
    log_files_x = glob.glob("GAN/**/optimizer_log_x.csv", recursive=True)
    log_files_y = glob.glob("GAN/**/optimizer_log_y.csv", recursive=True)

    if len(log_files_x) == 0 and len(log_files_y) == 0:
        print("No optimizer log files found in GAN directory")
        return

    # Group files by experiment
    experiments = {}

    # Process generator files
    for log_file in log_files_x:
        exp_name = get_experiment_info(log_file)
        if exp_name not in experiments:
            experiments[exp_name] = {"x": None, "y": None}
        experiments[exp_name]["x"] = log_file

    # Process discriminator files
    for log_file in log_files_y:
        exp_name = get_experiment_info(log_file)
        if exp_name not in experiments:
            experiments[exp_name] = {"x": None, "y": None}
        experiments[exp_name]["y"] = log_file

    print(f"Found {len(experiments)} experiments")

    # Generate individual plots
    if INDIVIDUAL_PLOTS:
        for exp_name, files in experiments.items():
            if (
                files["x"] is not None or files["y"] is not None
            ):  # At least one file exists
                print(f"Generating plot for experiment: {exp_name}")
                plot_individual_experiment(exp_name, files["x"], files["y"])

    # Generate overview plot
    if OVERVIEW_PLOT:
        plot_overview()


if __name__ == "__main__":
    quick_plot()
