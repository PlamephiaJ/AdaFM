import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d

def find_inception_score_files(gan_dir="GAN"):
    """
    Find all real_inception_scores.pkl files in the GAN directory
    
    Args:
        gan_dir (str): Directory containing GAN experiments
        
    Returns:
        dict: Dictionary mapping experiment names to file paths
    """
    inception_files = {}
    
    if not os.path.exists(gan_dir):
        print(f"GAN directory '{gan_dir}' not found!")
        return inception_files
    
    # Search for all real_inception_scores.pkl files recursively
    pkl_pattern = os.path.join(gan_dir, "**", "real_inception_scores.pkl")
    pkl_files = glob.glob(pkl_pattern, recursive=True)
    
    for pkl_file in pkl_files:
        # Extract experiment name from path
        # Path structure: GAN/optimizer_name/dataset_name/timestamp/real_inception_scores.pkl
        path_parts = Path(pkl_file).parts
        if len(path_parts) >= 4:
            optimizer_name = path_parts[-4]
            dataset_name = path_parts[-3] 
            timestamp = path_parts[-2]
            experiment_name = f"{optimizer_name}_{dataset_name}_{timestamp}"
        else:
            experiment_name = os.path.basename(os.path.dirname(pkl_file))
        
        inception_files[experiment_name] = pkl_file
        print(f"Found experiment: {experiment_name}")
    
    return inception_files

def smooth_data(data, method='moving_average', window_size=5, alpha=0.3):
    """
    Smooth the data using different methods
    
    Args:
        data (np.array): Input data to smooth
        method (str): Smoothing method ('moving_average', 'exponential', 'savgol')
        window_size (int): Window size for smoothing
        alpha (float): Alpha parameter for exponential smoothing
        
    Returns:
        np.array: Smoothed data
    """
    if len(data) < 3:
        return data
    
    if method == 'moving_average':
        # Simple moving average
        return uniform_filter1d(data.astype(float), size=window_size, mode='nearest')
    
    elif method == 'exponential':
        # Exponential smoothing
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    elif method == 'savgol':
        # Savitzky-Golay filter
        window_size = min(window_size, len(data))
        if window_size % 2 == 0:
            window_size -= 1  # Must be odd
        if window_size < 3:
            window_size = 3
        return signal.savgol_filter(data, window_size, 2)
    
    return data

def compute_envelope(data, window_size=10):
    """
    Compute upper and lower envelope of the data
    
    Args:
        data (np.array): Input data
        window_size (int): Window size for envelope calculation
        
    Returns:
        tuple: (upper_envelope, lower_envelope)
    """
    if len(data) < window_size:
        return data, data
    
    # Find local maxima and minima
    from scipy.signal import argrelextrema
    
    # Extend data at edges to avoid boundary effects
    extended_data = np.concatenate([data[:1], data, data[-1:]])
    
    # Find peaks and troughs
    peaks = argrelextrema(extended_data, np.greater, order=window_size//2)[0] - 1
    troughs = argrelextrema(extended_data, np.less, order=window_size//2)[0] - 1
    
    # Ensure we have boundary points
    peaks = np.concatenate([[0], peaks[peaks < len(data)], [len(data)-1]])
    troughs = np.concatenate([[0], troughs[troughs < len(data)], [len(data)-1]])
    
    # Remove duplicates and sort
    peaks = np.unique(np.clip(peaks, 0, len(data)-1))
    troughs = np.unique(np.clip(troughs, 0, len(data)-1))
    
    # Interpolate envelopes
    x = np.arange(len(data))
    upper_envelope = np.interp(x, peaks, data[peaks])
    lower_envelope = np.interp(x, troughs, data[troughs])
    
    return upper_envelope, lower_envelope

def load_inception_scores(file_path):
    """
    Load inception scores from pickle file
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        np.array: Array of inception scores
    """
    try:
        with open(file_path, 'rb') as f:
            scores = pickle.load(f)
        return np.array(scores)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_inception_scores(inception_files, save_interval=1000, save_path="inception_scores_plot.png", 
                         smooth_method='exponential', show_envelope=True, show_raw=False):
    """
    Plot inception scores for all experiments with smoothing and envelope
    
    Args:
        inception_files (dict): Dictionary mapping experiment names to file paths
        save_interval (int): Interval between measurements (for x-axis)
        save_path (str): Path to save the plot
        smooth_method (str): Smoothing method ('moving_average', 'exponential', 'savgol')
        show_envelope (bool): Whether to show envelope bands
        show_raw (bool): Whether to show raw data points
    """
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(inception_files)))
    
    for i, (exp_name, file_path) in enumerate(inception_files.items()):
        scores = load_inception_scores(file_path)
        
        if scores is not None and len(scores) > 0:
            # Create x-axis (iterations)
            iterations = np.arange(1, len(scores) + 1) * save_interval
            
            # Smooth the data
            smoothed_scores = smooth_data(scores, method=smooth_method, window_size=5, alpha=0.2)
            
            # Plot smoothed line
            plt.plot(iterations, smoothed_scores, 
                    label=f"{exp_name} (smoothed)", 
                    color=colors[i], 
                    linewidth=3, 
                    alpha=0.9)
            
            # Show raw data if requested
            if show_raw:
                plt.plot(iterations, scores, 
                        color=colors[i], 
                        alpha=0.3, 
                        linewidth=1,
                        linestyle='--',
                        label=f"{exp_name} (raw)")
            
            # Add envelope if requested
            if show_envelope and len(scores) > 10:
                upper_env, lower_env = compute_envelope(scores, window_size=max(5, len(scores)//10))
                plt.fill_between(iterations, upper_env, lower_env, 
                               color=colors[i], 
                               alpha=0.2,
                               label=f"{exp_name} (envelope)")
            
            print(f"Plotted {exp_name}: {len(scores)} points, "
                  f"Max IS: {np.max(scores):.3f}, "
                  f"Final IS (smoothed): {smoothed_scores[-1]:.3f}, "
                  f"Final IS (raw): {scores[-1]:.3f}")
        else:
            print(f"Skipped {exp_name}: No valid data")
    
    # Customize the plot
    plt.xlabel('Training Iterations', fontsize=14)
    plt.ylabel('Inception Score', fontsize=14)
    plt.title('Inception Score Evolution During Training (Smoothed)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Group labels by experiment
    smoothed_handles = [h for h, l in zip(handles, labels) if 'smoothed' in l]
    smoothed_labels = [l for l in labels if 'smoothed' in l]
    
    plt.legend(smoothed_handles, smoothed_labels, 
              bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=10)
    
    # Set reasonable y-axis limits
    plt.ylim(bottom=0)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def plot_comparison_by_optimizer(inception_files, save_interval=1000, smooth_method='exponential'):
    """
    Create separate plots for each optimizer comparing different datasets with smoothing
    
    Args:
        inception_files (dict): Dictionary mapping experiment names to file paths
        save_interval (int): Interval between measurements
        smooth_method (str): Smoothing method to use
    """
    # Group experiments by optimizer
    optimizer_groups = {}
    
    for exp_name, file_path in inception_files.items():
        parts = exp_name.split('_')
        if len(parts) >= 2:
            optimizer = parts[0]
            if optimizer not in optimizer_groups:
                optimizer_groups[optimizer] = {}
            optimizer_groups[optimizer][exp_name] = file_path
    
    # Create a plot for each optimizer
    for optimizer, experiments in optimizer_groups.items():
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
        
        for i, (exp_name, file_path) in enumerate(experiments.items()):
            scores = load_inception_scores(file_path)
            
            if scores is not None and len(scores) > 0:
                iterations = np.arange(1, len(scores) + 1) * save_interval
                
                # Smooth the data
                smoothed_scores = smooth_data(scores, method=smooth_method, window_size=5, alpha=0.2)
                
                # Plot smoothed line
                plt.plot(iterations, smoothed_scores, 
                        label=f"{exp_name} (smoothed)", 
                        color=colors[i], 
                        linewidth=3,
                        alpha=0.9)
                
                # Add envelope
                if len(scores) > 10:
                    upper_env, lower_env = compute_envelope(scores, window_size=max(5, len(scores)//10))
                    plt.fill_between(iterations, upper_env, lower_env, 
                                   color=colors[i], 
                                   alpha=0.15)
                
                # Show raw data with low opacity
                plt.plot(iterations, scores, 
                        color=colors[i], 
                        alpha=0.2, 
                        linewidth=1,
                        linestyle=':')
        
        plt.xlabel('Training Iterations', fontsize=12)
        plt.ylabel('Inception Score', fontsize=12)
        plt.title(f'Inception Score Evolution - {optimizer.upper()} (Smoothed)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Filter legend to show only smoothed lines
        handles, labels = plt.gca().get_legend_handles_labels()
        smoothed_handles = [h for h, l in zip(handles, labels) if 'smoothed' in l]
        smoothed_labels = [l for l in labels if 'smoothed' in l]
        plt.legend(smoothed_handles, smoothed_labels)
        
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        # Save optimizer-specific plot
        save_path = f"inception_scores_{optimizer}_smoothed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimizer-specific smoothed plot saved to: {save_path}")
        
        plt.show()

def main():
    """
    Main function to process all experiments and create plots
    """
    print("Starting Inception Score Analysis...")
    print("=" * 50)
    
    # Find all inception score files
    inception_files = find_inception_score_files()
    
    if not inception_files:
        print("No inception score files found!")
        return
    
    print(f"\nFound {len(inception_files)} experiments:")
    for exp_name in inception_files.keys():
        print(f"  - {exp_name}")
    
    print("\n" + "=" * 50)
    
    # Create overall comparison plot
    print("Creating overall comparison plot with smoothing...")
    plot_inception_scores(inception_files, save_interval=1000, 
                         save_path="inception_scores_comparison_smoothed.png",
                         smooth_method='exponential', 
                         show_envelope=True, 
                         show_raw=False)
    
    # Create optimizer-specific plots
    print("\nCreating optimizer-specific smoothed plots...")
    plot_comparison_by_optimizer(inception_files, save_interval=1000, smooth_method='exponential')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
