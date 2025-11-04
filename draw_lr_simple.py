#!/usr/bin/env python3
"""
Simple script to quickly visualize learning rates from GAN experiments.
Usage: python draw_lr_simple.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Configuration constants
MAX_STEPS = 5000  # Maximum number of training steps to display
FIGURE_SIZE = (14, 8)  # Figure size (width, height)
DPI = 300  # Resolution for saved plots
OUTPUT_FILE = "learning_rate_comparison.png"  # Output filename

def quick_plot():
    """Quick plot of all learning rate logs found in GAN directory."""
    
    # Find all optimizer log files (CSV format)
    log_files_x = glob.glob("GAN/**/optimizer_log_x.csv", recursive=True)
    log_files_y = glob.glob("GAN/**/optimizer_log_y.csv", recursive=True)
    
    print(f"Found {len(log_files_x)} generator logs and {len(log_files_y)} discriminator logs")
    
    if len(log_files_x) == 0 and len(log_files_y) == 0:
        print("No optimizer log files found in GAN directory")
        return
    
    # Create single plot for both optimizers
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    
    # Plot generator learning rates
    for log_file in log_files_x:
        try:
            df = pd.read_csv(log_file)
            experiment_name = os.path.dirname(log_file).replace("GAN/", "")
            
            # Select the first learning rate for each step
            df_first = df.groupby('step').first().reset_index()
            
            # Filter to specified maximum steps
            df_filtered = df_first[df_first['step'] <= MAX_STEPS]
            
            ax.plot(df_filtered['step'], df_filtered['learning_rate'], 
                   label=f"{experiment_name} (Generator)", linewidth=2, alpha=0.8, linestyle='-')
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Plot discriminator learning rates
    for log_file in log_files_y:
        try:
            df = pd.read_csv(log_file)
            experiment_name = os.path.dirname(log_file).replace("GAN/", "")
            
            # Select the first learning rate for each step
            df_first = df.groupby('step').first().reset_index()
            # Filter to specified maximum steps
            df_filtered = df_first[df_first['step'] <= MAX_STEPS]

            ax.plot(df_filtered['step'], df_filtered['learning_rate'],
                   label=f"{experiment_name} (Discriminator)", linewidth=2, alpha=0.8, linestyle='--')
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    # Customize plot
    ax.set_title(f'Generator vs Discriminator Learning Rate Evolution (First {MAX_STEPS//1000}K Steps)', fontsize=16)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    
    # Save and show
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    quick_plot()