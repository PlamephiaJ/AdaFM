import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d

FIGURE_FOLDER = Path("figures")
IS_FOLDER = FIGURE_FOLDER / "IS"
os.makedirs(IS_FOLDER, exist_ok=True)

def find_best_experiments_by_stats(gan_dir="GAN"):
    """
    Find the best experiments based on best_real_inception_score.csv files
    
    Args:
        gan_dir (str): Directory containing GAN experiments
        
    Returns:
        dict: Dictionary containing best experiments info
    """
    best_experiments = {}
    
    if not os.path.exists(gan_dir):
        print(f"GAN directory '{gan_dir}' not found!")
        return best_experiments
    
    # Find all best_real_inception_score.csv files
    best_files = glob.glob(os.path.join(gan_dir, "**", "best_real_inception_score.csv"), recursive=True)
    
    if not best_files:
        print("No best_real_inception_score.csv files found!")
        return best_experiments
    
    # Group by optimizer
    optimizer_experiments = {}
    
    for best_file in best_files:
        try:
            # Read the best scores
            df = pd.read_csv(best_file)
            if 'BestIS' not in df.columns or 'AvgIS' not in df.columns:
                continue
                
            best_is = float(df['BestIS'].iloc[0])
            avg_is = float(df['AvgIS'].iloc[0])
            
            # Extract experiment info from path
            path_parts = Path(best_file).parts
            if len(path_parts) >= 4:
                optimizer_name = path_parts[-4]
                dataset_name = path_parts[-3]
                timestamp = path_parts[-2]
                experiment_name = f"{optimizer_name}_{dataset_name}_{timestamp}"
                
                if optimizer_name not in optimizer_experiments:
                    optimizer_experiments[optimizer_name] = []
                
                optimizer_experiments[optimizer_name].append({
                    'experiment_name': experiment_name,
                    'best_is': best_is,
                    'avg_is': avg_is,
                    'folder_path': Path(best_file).parent,
                    'optimizer': optimizer_name,
                    'dataset': dataset_name,
                    'timestamp': timestamp
                })
                
        except Exception as e:
            print(f"Error processing {best_file}: {e}")
            continue
    
    # Find best experiments for each optimizer
    for optimizer, experiments in optimizer_experiments.items():
        if not experiments:
            continue
            
        # Find best by max IS and best by avg IS
        best_by_max = max(experiments, key=lambda x: x['best_is'])
        best_by_avg = max(experiments, key=lambda x: x['avg_is'])
        
        print(f"\nOptimizer: {optimizer}")
        print(f"  Best by Max IS: {best_by_max['experiment_name']} (Best: {best_by_max['best_is']:.3f}, Avg: {best_by_max['avg_is']:.3f})")
        print(f"  Best by Avg IS: {best_by_avg['experiment_name']} (Best: {best_by_avg['best_is']:.3f}, Avg: {best_by_avg['avg_is']:.3f})")
        
        # Add to results - avoid duplicates
        optimizer_best = {}
        optimizer_best[f"{optimizer}_best_max"] = best_by_max
        
        # Only add avg best if it's different from max best
        if best_by_max['experiment_name'] != best_by_avg['experiment_name']:
            optimizer_best[f"{optimizer}_best_avg"] = best_by_avg
            print(f"  -> Will plot both experiments (different)")
        else:
            print(f"  -> Will plot single experiment (same for both metrics)")
        
        best_experiments.update(optimizer_best)
    
    return best_experiments

def find_inception_score_files_for_best(best_experiments):
    """
    Find inception score files for the best experiments
    
    Args:
        best_experiments (dict): Dictionary of best experiments
        
    Returns:
        dict: Dictionary mapping experiment names to file paths
    """
    inception_files = {}
    
    for exp_key, exp_info in best_experiments.items():
        folder_path = exp_info['folder_path']
        experiment_name = exp_info['experiment_name']
        
        # Try to find inception score files in different formats
        score_file = None
        
        # Try CSV first
        csv_file = folder_path / "real_inception_scores.csv"
        if csv_file.exists():
            score_file = csv_file
            print(f"Found CSV for {experiment_name}")
        
        # Try pickle if no CSV
        if score_file is None:
            pkl_file = folder_path / "real_inception_scores.pkl"
            if pkl_file.exists():
                score_file = pkl_file
                print(f"Found PKL for {experiment_name}")
        
        # Try text if no pickle
        if score_file is None:
            txt_file = folder_path / "real_inception_scores.txt"
            if txt_file.exists():
                score_file = txt_file
                print(f"Found TXT for {experiment_name}")
        
        if score_file:
            # Use a descriptive name that includes the selection criterion
            display_name = f"{experiment_name} ({exp_key.split('_')[-1]} IS)"
            inception_files[display_name] = str(score_file)
        else:
            print(f"No inception score file found for {experiment_name}")
    
    return inception_files

def find_inception_score_files(gan_dir="GAN", use_best_only=True):
    """
    Find inception score files in the GAN directory
    
    Args:
        gan_dir (str): Directory containing GAN experiments
        use_best_only (bool): If True, only return files for best experiments
        
    Returns:
        dict: Dictionary mapping experiment names to file paths
    """
    if use_best_only:
        print("=== Finding Best Experiments by IS Statistics ===")
        best_experiments = find_best_experiments_by_stats(gan_dir)
        if best_experiments:
            return find_inception_score_files_for_best(best_experiments)
        else:
            print("No best experiments found, falling back to all experiments")
            use_best_only = False
    
    if not use_best_only:
        print("=== Finding All Experiments ===")
        inception_files = {}
        
        if not os.path.exists(gan_dir):
            print(f"GAN directory '{gan_dir}' not found!")
            return inception_files
        
        # Search for all real_inception_scores.pkl files recursively
        pkl_pattern = os.path.join(gan_dir, "**", "real_inception_scores.pkl")
        pkl_files = glob.glob(pkl_pattern, recursive=True)
        
        for pkl_file in pkl_files:
            # Extract experiment name from path
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
    Load inception scores from various file formats
    
    Args:
        file_path (str): Path to the score file (CSV, PKL, or TXT)
        
    Returns:
        np.array: Array of inception scores
    """
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.csv':
            # Load from CSV
            df = pd.read_csv(file_path)
            if 'inception_score' in df.columns:
                return np.array(df['inception_score'].values)
            elif 'IS' in df.columns:
                return np.array(df['IS'].values)
            else:
                print(f"No recognized inception score column in {file_path}")
                return None
                
        elif file_path.suffix.lower() == '.pkl':
            # Load from pickle
            with open(file_path, 'rb') as f:
                scores = pickle.load(f)
            return np.array(scores)
            
        elif file_path.suffix.lower() == '.txt':
            # Load from text file
            scores = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if line and 'Iteration' in line:
                    # Parse "Iteration 200: 1.234567"
                    parts = line.split(': ')
                    if len(parts) == 2:
                        score = float(parts[1])
                        scores.append(score)
            
            return np.array(scores) if scores else None
            
        else:
            print(f"Unsupported file format: {file_path.suffix}")
            return None
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_inception_scores(inception_files, save_interval=1000, save_path=IS_FOLDER / "inception_scores_plot.png", 
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
        save_path = IS_FOLDER / f"inception_scores_{optimizer}_smoothed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimizer-specific smoothed plot saved to: {save_path}")
        
        plt.show()

def main(use_best_only=True):
    """
    Main function to process experiments and create plots
    
    Args:
        use_best_only (bool): If True, only plot the best experiments based on IS statistics
    """
    print("Starting Inception Score Analysis...")
    print("=" * 50)
    
    # Find inception score files
    inception_files = find_inception_score_files(use_best_only=use_best_only)
    
    if not inception_files:
        print("No inception score files found!")
        return
    
    print(f"\nSelected {len(inception_files)} experiments for plotting:")
    for exp_name in inception_files.keys():
        print(f"  - {exp_name}")
    
    print("\n" + "=" * 50)
    
    # Create overall comparison plot
    plot_name = "best_inception_scores_comparison_smoothed.png" if use_best_only else "inception_scores_comparison_smoothed.png"
    print(f"Creating comparison plot: {plot_name}")
    plot_inception_scores(inception_files, save_interval=1000, 
                         save_path=IS_FOLDER / plot_name,
                         smooth_method='exponential', 
                         show_envelope=True, 
                         show_raw=False)
    
    # Create optimizer-specific plots
    print("\nCreating optimizer-specific smoothed plots...")
    plot_comparison_by_optimizer(inception_files, save_interval=1000, smooth_method='exponential')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
