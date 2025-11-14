import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d

# Figure styling constants
FIG_SIZE = (7.2, 4.5)  # Width, Height in inches for main plots
LINE_WIDTH = 6  # Line width for plots
LEGEND_FONTSIZE = 12  # Legend font size
Y_AXIS_BOTTOM = 1  # Y-axis starting point for plots
# Optimizer color mapping based on optimizer specs
OPTIMIZER_COLORS = {
    "adafm": "#d62728",      # red - AdaSTORM-M / AdaFM
    "adastorm-m": "#d62728",
    "adam": "#1f77b4",       # blue
    "rmsprop": "#ff7f0e",    # orange
    "adagrad": "#2ca02c",    # green
    "msgda": "#9467bd",      # purple
    "tiada": "#8c564b",      # brown
    "pesg": "#17becf",       # cyan
}
OPTIMIZER_REAL_NAME = {
    "adafm": "AdaSTORM-M",
    "adastorm-m": "AdaSTORM-M",
    "adam": "Adam",
    "rmsprop": "RMSProp",
    "adagrad": "Adagrad",
    "msgda": "MSGDA",
    "tiada": "TiAda",
    "pesg": "PESG"
}
LEGEND_POS_COMPARISON = "upper left"      # Overall comparison plot
LEGEND_POS_SINGLE_LOOP = "upper left"    # Single loop optimizers plot
LEGEND_POS_DOUBLE_LOOP = "lower right"    # Double loop optimizers plot  
LEGEND_POS_OTHER = "upper left"          # Other optimizers plot
NUM_DATA_POINTS = 400  # Number of data points to plot

SAVE_INTERVAL = 200
SHOW_ENVELOPE = False
MAX_ITERATIONS = 80000  # Maximum x-axis range for plotting
X_LIMIT = 40_000
NEW_X_LIMIT = (0, 40000)
Y_AXIS_BOTTOM = 1  # Y-axis starting point for plots
# Legend position settings for different plots
FIGURE_FOLDER = Path("figures")
IS_FOLDER = FIGURE_FOLDER / "IS"
os.makedirs(IS_FOLDER, exist_ok=True)

SINGLE_LOOP_COMPARISON_SETTING = ["adastorm-m", "msgda", "tiada", "pesg"]
DOUBLE_LOOP_COMPARISON_SETTING = ["adam", "rmsprop", "adagrad", "adastorm-m"]
SINGLE_LOOP_OPTIMIZERS = ["adastorm-m", "msgda", "tiada", "pesg"]
DOUBLE_LOOP_OPTIMIZERS = ["adam", "rmsprop", "adagrad"]



def find_best_experiments_by_stats(gan_dirs=["GAN", "GAN/model_factory"]):
    """
    Find the best experiments based on best_real_inception_score.csv files

    Args:
        gan_dirs (list): List of directories containing GAN experiments

    Returns:
        dict: Dictionary containing best experiments info
    """
    best_experiments = {}
    
    all_best_files = []
    
    # Search in multiple directories
    for gan_dir in gan_dirs:
        if not os.path.exists(gan_dir):
            print(f"GAN directory '{gan_dir}' not found, skipping...")
            continue
            
        print(f"Searching in directory: {gan_dir}")
        
        # Find all best_real_inception_score.csv files
        best_files = glob.glob(
            os.path.join(gan_dir, "**", "best_real_inception_score.csv"), recursive=True
        )

        # Filter out grid search results
        best_files = [
            f for f in best_files
            if "grid_search" not in f
        ]
        
        all_best_files.extend(best_files)
        print(f"Found {len(best_files)} files in {gan_dir}")

    if not all_best_files:
        print("No best_real_inception_score.csv files found in any directory!")
        return best_experiments

    print(f"Total files found: {len(all_best_files)}")

    # Group by optimizer
    optimizer_experiments = {}

    for best_file in all_best_files:
        try:
            # Read the best scores
            df = pd.read_csv(best_file)
            if "BestIS" not in df.columns or "AvgIS" not in df.columns:
                continue

            best_is = float(df["BestIS"].iloc[0])
            avg_is = float(df["AvgIS"].iloc[0])

            # Extract experiment info from path
            path_parts = Path(best_file).parts
            
            # Handle different path structures:
            # GAN/optimizer/dataset/timestamp/best_real_inception_score.csv
            # GAN/model_factory/backbone/optimizer/dataset/timestamp/best_real_inception_score.csv
            
            optimizer_name = None
            dataset_name = None
            timestamp = None
            backbone_name = None
            
            if "model_factory" in path_parts:
                # GAN/model_factory/backbone/optimizer/dataset/timestamp/
                model_factory_idx = path_parts.index("model_factory")
                if len(path_parts) >= model_factory_idx + 5:
                    backbone_name = path_parts[model_factory_idx + 1]
                    optimizer_name = path_parts[model_factory_idx + 2]
                    dataset_name = path_parts[model_factory_idx + 3]
                    timestamp = path_parts[model_factory_idx + 4]
            else:
                # GAN/optimizer/dataset/timestamp/
                if len(path_parts) >= 4:
                    optimizer_name = path_parts[-4]
                    dataset_name = path_parts[-3]
                    timestamp = path_parts[-2]
            
            if not optimizer_name or not dataset_name or not timestamp:
                print(f"Could not parse path: {best_file}")
                continue
                
            # Create experiment name including backbone if present
            if backbone_name:
                experiment_name = f"{backbone_name}_{optimizer_name}_{dataset_name}_{timestamp}"
                optimizer_key = f"{backbone_name}_{optimizer_name}"
            else:
                experiment_name = f"{optimizer_name}_{dataset_name}_{timestamp}"
                optimizer_key = optimizer_name

            if optimizer_key not in optimizer_experiments:
                optimizer_experiments[optimizer_key] = []

            optimizer_experiments[optimizer_key].append(
                {
                    "experiment_name": experiment_name,
                    "best_is": best_is,
                    "avg_is": avg_is,
                    "folder_path": Path(best_file).parent,
                    "optimizer": optimizer_name,
                    "dataset": dataset_name,
                    "timestamp": timestamp,
                    "backbone": backbone_name,
                    "optimizer_key": optimizer_key,
                }
            )

        except Exception as e:
            print(f"Error processing {best_file}: {e}")
            continue

    # Find best experiments for each optimizer
    for optimizer, experiments in optimizer_experiments.items():
        if not experiments:
            continue

        # Find best by max IS and best by avg IS
        best_by_max = max(experiments, key=lambda x: x["best_is"])
        best_by_avg = max(experiments, key=lambda x: x["avg_is"])

        print(f"\nOptimizer: {optimizer}")
        print(
            f"  Best by Max IS: {best_by_max['experiment_name']} (Best: {best_by_max['best_is']:.3f}, Avg: {best_by_max['avg_is']:.3f})"
        )
        print(
            f"  Best by Avg IS: {best_by_avg['experiment_name']} (Best: {best_by_avg['best_is']:.3f}, Avg: {best_by_avg['avg_is']:.3f})"
        )

        # Add to results - avoid duplicates
        optimizer_best = {}
        optimizer_best[f"{optimizer}_best_max"] = best_by_max

        # Only add avg best if it's different from max best
        # if best_by_max["experiment_name"] != best_by_avg["experiment_name"]:
        #     optimizer_best[f"{optimizer}_best_avg"] = best_by_avg
        #     print(f"  -> Will plot both experiments (different)")
        # else:
        #     print(f"  -> Will plot single experiment (same for both metrics)")

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
        folder_path = exp_info["folder_path"]
        experiment_name = exp_info["experiment_name"]

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


def find_inception_score_files(gan_dirs=["GAN", "GAN/model_factory"], use_best_only=True):
    """
    Find inception score files in the GAN directories

    Args:
        gan_dirs (list): List of directories containing GAN experiments
        use_best_only (bool): If True, only return files for best experiments

    Returns:
        dict: Dictionary mapping experiment names to file paths
    """
    if use_best_only:
        print("=== Finding Best Experiments by IS Statistics ===")
        best_experiments = find_best_experiments_by_stats(gan_dirs)
        if best_experiments:
            return find_inception_score_files_for_best(best_experiments)
        else:
            print("No best experiments found, falling back to all experiments")
            use_best_only = False

    if not use_best_only:
        print("=== Finding All Experiments ===")
        inception_files = {}

        for gan_dir in gan_dirs:
            if not os.path.exists(gan_dir):
                print(f"GAN directory '{gan_dir}' not found, skipping...")
                continue

            # Search for all real_inception_scores.pkl files recursively
            pkl_pattern = os.path.join(gan_dir, "**", "real_inception_scores.pkl")
            pkl_files = glob.glob(pkl_pattern, recursive=True)

            for pkl_file in pkl_files:
                # Extract experiment name from path
                path_parts = Path(pkl_file).parts
                
                # Handle different path structures similar to best_experiments
                optimizer_name = None
                dataset_name = None
                timestamp = None
                backbone_name = None
                
                if "model_factory" in path_parts:
                    # GAN/model_factory/backbone/optimizer/dataset/timestamp/
                    model_factory_idx = path_parts.index("model_factory")
                    if len(path_parts) >= model_factory_idx + 5:
                        backbone_name = path_parts[model_factory_idx + 1]
                        optimizer_name = path_parts[model_factory_idx + 2]
                        dataset_name = path_parts[model_factory_idx + 3]
                        timestamp = path_parts[model_factory_idx + 4]
                else:
                    # GAN/optimizer/dataset/timestamp/
                    if len(path_parts) >= 4:
                        optimizer_name = path_parts[-4]
                        dataset_name = path_parts[-3]
                        timestamp = path_parts[-2]
                
                if optimizer_name and dataset_name and timestamp:
                    if backbone_name:
                        experiment_name = f"{backbone_name}_{optimizer_name}_{dataset_name}_{timestamp}"
                    else:
                        experiment_name = f"{optimizer_name}_{dataset_name}_{timestamp}"
                else:
                    experiment_name = os.path.basename(os.path.dirname(pkl_file))

                inception_files[experiment_name] = pkl_file
                print(f"Found experiment: {experiment_name}")

        return inception_files


def smooth_data(data, method="moving_average", window_size=5, alpha=0.3):
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

    if method == "moving_average":
        # Simple moving average
        return uniform_filter1d(data.astype(float), size=window_size, mode="nearest")

    elif method == "exponential":
        # Exponential smoothing
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    elif method == "savgol":
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
    peaks = argrelextrema(extended_data, np.greater, order=window_size // 2)[0] - 1
    troughs = argrelextrema(extended_data, np.less, order=window_size // 2)[0] - 1

    # Ensure we have boundary points
    peaks = np.concatenate([[0], peaks[peaks < len(data)], [len(data) - 1]])
    troughs = np.concatenate([[0], troughs[troughs < len(data)], [len(data) - 1]])

    # Remove duplicates and sort
    peaks = np.unique(np.clip(peaks, 0, len(data) - 1))
    troughs = np.unique(np.clip(troughs, 0, len(data) - 1))

    # Interpolate envelopes
    x = np.arange(len(data))
    upper_envelope = np.interp(x, peaks, data[peaks])
    lower_envelope = np.interp(x, troughs, data[troughs])

    return upper_envelope, lower_envelope


def load_inception_scores(file_path, optimizer_name=None):
    """
    Load inception scores from various file formats

    Args:
        file_path (str): Path to the score file (CSV, PKL, or TXT)
        optimizer_name (str): Name of the optimizer (used for data limiting)

    Returns:
        np.array: Array of inception scores
    """
    file_path = Path(file_path)

    try:
        if file_path.suffix.lower() == ".csv":
            # Load from CSV
            df = pd.read_csv(file_path)
            if "inception_score" in df.columns:
                scores = np.array(df["inception_score"].values)
            elif "IS" in df.columns:
                scores = np.array(df["IS"].values)
            else:
                print(f"No recognized inception score column in {file_path}")
                return None

        elif file_path.suffix.lower() == ".pkl":
            # Load from pickle
            with open(file_path, "rb") as f:
                scores = pickle.load(f)
            scores = np.array(scores)

        elif file_path.suffix.lower() == ".txt":
            # Load from text file
            scores = []
            with open(file_path, "r") as f:
                lines = f.readlines()

            for line in lines[1:]:  # Skip header
                line = line.strip()
                if line and "Iteration" in line:
                    # Parse "Iteration 200: 1.234567"
                    parts = line.split(": ")
                    if len(parts) == 2:
                        score = float(parts[1])
                        scores.append(score)

            scores = np.array(scores) if scores else None

        else:
            print(f"Unsupported file format: {file_path.suffix}")
            return None

        # Limit data points for DOUBLE_LOOP_OPTIMIZERS
        if scores is not None and optimizer_name is not None:
            optimizer_key = optimizer_name.lower()
            if optimizer_key in [opt.lower() for opt in DOUBLE_LOOP_OPTIMIZERS]:
                max_points = NUM_DATA_POINTS // 3
                if len(scores) > max_points:
                    scores = scores[:max_points]
                    print(f"Limited {optimizer_name} to {max_points} data points (DOUBLE_LOOP_OPTIMIZER)")

        return scores

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def plot_inception_scores(
    inception_files,
    save_interval=SAVE_INTERVAL,
    save_path=IS_FOLDER / "inception_scores_plot.png",
    smooth_method="exponential",
    show_envelope=SHOW_ENVELOPE,
    show_raw=False,
    legend_pos="upper left",
):
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
    # Configure larger, clearer fonts for publication-quality figures (further enlarged)
    font_cfg = {
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 56,
        "axes.titlesize": 18,
    }
    for k, v in font_cfg.items():
        plt.rcParams[k] = v

    plt.figure(figsize=FIG_SIZE)
    # track which time indices were actually plotted (for legend)
    plotted_time_indices = set()

    # Sort experiments to put adastorm-m first in legend
    sorted_files = sorted(
        inception_files.items(),
        key=lambda x: (0 if "adastorm-m" in x[0].lower() else 1, x[0])
    )

    for i, (exp_name, file_path) in enumerate(sorted_files):
        # Extract optimizer name first for data limiting
        exp_parts = exp_name.split("_")
        if "model_factory" in exp_name or len(exp_parts) >= 4:
            # For model_factory experiments: backbone_optimizer_dataset_timestamp
            # The optimizer is the second part
            optimizer_name = exp_parts[1] if len(exp_parts) > 1 else exp_parts[0]
        else:
            # For traditional experiments: optimizer_dataset_timestamp
            # The optimizer is the first part
            optimizer_name = exp_parts[0]
        
        scores = load_inception_scores(file_path, optimizer_name)

        if scores is not None and len(scores) > 0:
            
            # Create x-axis (iterations)
            iterations = np.arange(1, len(scores) + 1) * save_interval
            
            # For DOUBLE_LOOP_OPTIMIZERS, multiply x-axis by 3 to compensate for reduced data points
            optimizer_key = optimizer_name.lower()
            if optimizer_key in [opt.lower() for opt in DOUBLE_LOOP_OPTIMIZERS]:
                iterations = iterations * 3
                print(f"Scaled x-axis by 3 for {optimizer_name} (DOUBLE_LOOP_OPTIMIZER)")
            
            # Filter data to keep only x <= MAX_ITERATIONS
            mask = iterations <= MAX_ITERATIONS
            iterations = iterations[mask]
            scores = scores[mask]

            iterations_scaled = iterations * (40_000 / MAX_ITERATIONS)
            
            if len(scores) == 0:
                print(f"Skipped {exp_name}: No data within x <= {MAX_ITERATIONS} range")
                continue

            # Smooth the data
            smoothed_scores = smooth_data(
                scores, method=smooth_method, window_size=5, alpha=0.2
            )

            # Get color from OPTIMIZER_COLORS mapping, fallback to default if not found
            optimizer_key = optimizer_name.lower()
            line_color = OPTIMIZER_COLORS.get(optimizer_key, f"C{i}")

            optimizer_real_name = OPTIMIZER_REAL_NAME.get(optimizer_key, optimizer_name)

            # Plot smoothed line
            plt.plot(
                iterations_scaled,
                smoothed_scores,
                label=f"{optimizer_real_name}",
                color=line_color,
                linewidth=LINE_WIDTH,
                alpha=0.9,
            )

            # Show raw data if requested
            if show_raw:
                plt.plot(
                    iterations_scaled,
                    scores,
                    color=line_color,
                    alpha=0.3,
                    linewidth=LINE_WIDTH/6,
                    linestyle="--",
                    label=None,  # Don't show in legend
                )

            # Add envelope if requested
            if show_envelope and len(scores) > 10:
                upper_env, lower_env = compute_envelope(
                    scores, window_size=max(5, len(scores) // 10)
                )
                plt.fill_between(
                    iterations,
                    upper_env,
                    lower_env,
                    color=line_color,
                    alpha=0.2,
                    label=None,  # Don't show in legend
                )

            # ----------- Add: plot physical time markers (wall-time) -----------
            # Look for real_inception_scores_with_wall_time.csv in the same folder
            try:
                folder_path = Path(file_path).parent
            except Exception:
                folder_path = Path(file_path)

            wall_time_file = folder_path / "real_inception_scores_with_wall_time.csv"
            if wall_time_file.exists():
                try:
                    df_wall = pd.read_csv(wall_time_file)
                    if (
                        "wall_time_seconds" in df_wall.columns
                        and "inception_score" in df_wall.columns
                        and "step" in df_wall.columns
                    ):
                        wall_times = np.array(df_wall["wall_time_seconds"].values)
                        wall_scores = np.array(df_wall["inception_score"].values)
                        wall_steps = np.array(df_wall["step"].values)
                        for t_idx, t in enumerate([2400]):
                            if len(wall_times) == 0:
                                continue
                            if np.max(wall_times) < t:
                                # skip if this experiment doesn't reach the requested wall time
                                continue
                            idx = np.argmin(np.abs(wall_times - t))
                            step_x = wall_steps[int(idx)]
                            # only plot markers if step_x is within the plotted iteration range
                            if step_x < iterations[0] or step_x > iterations[-1]:
                                continue
                            try:
                                y_on_smoothed = np.interp(step_x, iterations, smoothed_scores)
                            except Exception:
                                y_on_smoothed = wall_scores[int(idx)]

                            step_x_scaled = step_x * (40_000 / MAX_ITERATIONS)
                            plt.scatter(
                                step_x_scaled,
                                y_on_smoothed,
                                marker='o',
                                s=360,
                                color=line_color,
                                edgecolors='k',
                                zorder=10,
                            )
                            # record that this time index was plotted for legend
                            try:
                                plotted_time_indices.add(t_idx)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Error reading wall time csv for {exp_name}: {e}")

            print(
                f"Plotted {exp_name}: {len(scores)} points, "
                f"Max IS: {np.max(scores):.3f}, "
                f"Final IS (smoothed): {smoothed_scores[-1]:.3f}, "
                f"Final IS (raw): {scores[-1]:.3f}"
            )
        else:
            print(f"Skipped {exp_name}: No valid data")

    # Customize the plot (no title, per request)
    plt.xlabel("Generator Steps", fontsize=18)
    plt.ylabel("Inception Score", fontsize=18)
    # No title to keep the figure cleaner
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", labelsize=15)

    # Format x-axis to show "10k", "20k" instead of "10000", "20000"
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

    # Build legend: get existing handles/labels and append time-marker proxies
    ax = plt.gca()
    existing_handles, existing_labels = ax.get_legend_handles_labels()

    from matplotlib.lines import Line2D
    marker_handles = []
    marker_labels = []
    time_markers = ['o']
    time_points = [2400]
    for t_idx in sorted(plotted_time_indices):
        mh = Line2D([], [], color='k', marker=time_markers[t_idx], linestyle='', markersize=6)
        try:
            mh.set_markeredgewidth(0.6)
        except Exception:
            pass
        marker_handles.append(mh)
        marker_labels.append(f"runtime={int(time_points[t_idx]/60)} min")

    all_handles = existing_handles + marker_handles
    all_labels = existing_labels + marker_labels

    ax.legend(
        all_handles,
        all_labels,
        loc=legend_pos,
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.9,
        fancybox=True,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=2.0,
    )

    # Set reasonable axis limits
    plt.ylim(bottom=Y_AXIS_BOTTOM)
    plt.xlim(left=0, right=40000)
    plt.yticks([1, 2, 3, 4, 5])

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

    # Show the plot
    plt.show()


def create_optimizer_performance_table(best_experiments, save_path=None):
    """
    Create a performance comparison table for single loop vs double loop optimizers
    
    Args:
        best_experiments (dict): Dictionary containing best experiments info
        save_path (str): Path to save the table. If None, saves to IS_FOLDER
    """
    if save_path is None:
        save_path = IS_FOLDER / "optimizer_performance_comparison.csv"
    
    # Separate single loop and double loop experiments
    single_loop_stats = {}  # GAN experiments
    double_loop_stats = {}  # GAN/model_factory experiments
    
    for exp_key, exp_info in best_experiments.items():
        optimizer_name = exp_info["optimizer"]
        best_is = exp_info["best_is"]
        avg_is = exp_info["avg_is"]
        backbone_name = exp_info.get("backbone", None)
        
        if backbone_name:  # Double loop (model_factory)
            if optimizer_name not in double_loop_stats:
                double_loop_stats[optimizer_name] = {
                    "best_is_experiments": [],
                    "avg_is_experiments": []
                }
            double_loop_stats[optimizer_name]["best_is_experiments"].append((best_is, exp_info))
            double_loop_stats[optimizer_name]["avg_is_experiments"].append((avg_is, exp_info))
        else:  # Single loop (traditional GAN)
            if optimizer_name not in single_loop_stats:
                single_loop_stats[optimizer_name] = {
                    "best_is_experiments": [],
                    "avg_is_experiments": []
                }
            single_loop_stats[optimizer_name]["best_is_experiments"].append((best_is, exp_info))
            single_loop_stats[optimizer_name]["avg_is_experiments"].append((avg_is, exp_info))
    
    # Create table data
    table_data = []
    
    # Process single loop optimizers
    for optimizer, stats in single_loop_stats.items():
        best_is_max = max(stats["best_is_experiments"], key=lambda x: x[0])
        avg_is_max = max(stats["avg_is_experiments"], key=lambda x: x[0])
        
        table_data.append({
            "Type": "Single Loop",
            "Optimizer": optimizer,
            "Best IS": f"{best_is_max[0]:.4f}",
            "Max Avg IS": f"{avg_is_max[0]:.4f}"
        })
    
    # Process double loop optimizers
    for optimizer, stats in double_loop_stats.items():
        best_is_max = max(stats["best_is_experiments"], key=lambda x: x[0])
        avg_is_max = max(stats["avg_is_experiments"], key=lambda x: x[0])
        
        table_data.append({
            "Type": "Double Loop",
            "Optimizer": optimizer,
            "Best IS": f"{best_is_max[0]:.4f}",
            "Max Avg IS": f"{avg_is_max[0]:.4f}"
        })
    
    # Create DataFrame and sort by Type then Best IS
    df = pd.DataFrame(table_data)
    df["Best IS (numeric)"] = df["Best IS"].astype(float)
    df = df.sort_values(["Type", "Best IS (numeric)"], ascending=[True, False])
    df = df.drop("Best IS (numeric)", axis=1)  # Remove helper column
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Optimizer performance comparison table saved to: {save_path}")
    
    # Print table to console
    print("\n" + "=" * 60)
    print("OPTIMIZER PERFORMANCE COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    # Create and save table as image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style header row
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f2f2f2')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Color-code by type
    for i in range(1, len(df) + 1):
        if df.iloc[i-1]['Type'] == 'Single Loop':
            table[(i, 0)].set_facecolor('#e3f2fd')  # Light blue
        else:  # Double Loop
            table[(i, 0)].set_facecolor('#fff3e0')  # Light orange
    
    plt.title('Optimizer Performance Comparison\nSingle Loop vs Double Loop', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Save the table image
    image_save_path = save_path.parent / "optimizer_performance_comparison.png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Table image saved to: {image_save_path}")
    
    plt.show()
    
    return df


def main(use_best_only=True):
    """
    Main function to process experiments and create plots

    Args:
        use_best_only (bool): If True, only plot the best experiments based on IS statistics
    """
    print("Starting Inception Score Analysis...")
    print("=" * 50)

    # Find inception score files from multiple directories
    # Find inception score files from current directory (since script is in GAN_CIFAR100/)
    gan_dirs = [".", "model_factory"]
    inception_files = find_inception_score_files(gan_dirs=gan_dirs, use_best_only=use_best_only)

    if not inception_files:
        print("No inception score files found!")
        return

    print(f"\nSelected {len(inception_files)} experiments for plotting:")
    for exp_name in inception_files.keys():
        print(f"  - {exp_name}")

    print("\n" + "=" * 50)

    # Create overall comparison plot
    plot_name = (
        "best_inception_scores_comparison_smoothed.png"
        if use_best_only
        else "inception_scores_comparison_smoothed.png"
    )
    print(f"Creating comparison plot: {plot_name}")
    plot_inception_scores(
        inception_files,
        save_interval=SAVE_INTERVAL,
        save_path=IS_FOLDER / plot_name,
        smooth_method="exponential",
        show_envelope=SHOW_ENVELOPE,
        show_raw=False,
        legend_pos=LEGEND_POS_COMPARISON,
    )

    # Separate experiments by optimizer groups
    single_loop_experiments = {}
    double_loop_experiments = {}
    other_experiments = {}
    
    for exp_name, file_path in inception_files.items():
        # Extract optimizer name from experiment name
        exp_parts = exp_name.split("_")
        if "model_factory" in exp_name or len(exp_parts) >= 4:
            # For model_factory experiments: backbone_optimizer_dataset_timestamp
            optimizer_name = exp_parts[1] if len(exp_parts) > 1 else exp_parts[0]
        else:
            # For traditional experiments: optimizer_dataset_timestamp
            optimizer_name = exp_parts[0]
        
        optimizer_name = optimizer_name.lower()
        
        # Categorize by optimizer group (allow adastorm-m to be in both groups)
        is_in_single = optimizer_name in [opt.lower() for opt in SINGLE_LOOP_COMPARISON_SETTING]
        is_in_double = optimizer_name in [opt.lower() for opt in DOUBLE_LOOP_COMPARISON_SETTING]
        
        if is_in_single:
            single_loop_experiments[exp_name] = file_path
        if is_in_double:
            double_loop_experiments[exp_name] = file_path
        if not is_in_single and not is_in_double:
            other_experiments[exp_name] = file_path
            print(f"Warning: {optimizer_name} not found in predefined groups")
    
    # Create single loop optimizers plot
    if single_loop_experiments:
        print(f"\nCreating single loop optimizers plot...")
        print(f"Single loop experiments: {list(single_loop_experiments.keys())}")
        single_loop_plot_name = (
            "IS_CIFAR100_single.png"
            if use_best_only
            else "single_loop_optimizers_smoothed.png"
        )
        plot_inception_scores(
            single_loop_experiments,
            save_interval=SAVE_INTERVAL,
            save_path=IS_FOLDER / single_loop_plot_name,
            smooth_method="exponential",
            show_envelope=SHOW_ENVELOPE,
            show_raw=False,
            legend_pos=LEGEND_POS_SINGLE_LOOP,
        )
    
    # Create double loop optimizers plot
    if double_loop_experiments:
        print(f"\nCreating double loop optimizers plot...")
        print(f"Double loop experiments: {list(double_loop_experiments.keys())}")
        double_loop_plot_name = (
            "IS_CIFAR100_double.png"
            if use_best_only
            else "double_loop_optimizers_smoothed.png"
        )
        plot_inception_scores(
            double_loop_experiments,
            save_interval=SAVE_INTERVAL,
            save_path=IS_FOLDER / double_loop_plot_name,
            smooth_method="exponential",
            show_envelope=SHOW_ENVELOPE,
            show_raw=False,
            legend_pos=LEGEND_POS_DOUBLE_LOOP,
        )
    
    # Create plot for other optimizers if any
    if other_experiments:
        print(f"\nCreating other optimizers plot...")
        print(f"Other experiments: {list(other_experiments.keys())}")
        other_plot_name = (
            "best_other_optimizers_smoothed.png"
            if use_best_only
            else "other_optimizers_smoothed.png"
        )
        plot_inception_scores(
            other_experiments,
            save_interval=SAVE_INTERVAL,
            save_path=IS_FOLDER / other_plot_name,
            smooth_method="exponential",
            show_envelope=SHOW_ENVELOPE,
            show_raw=False,
            legend_pos=LEGEND_POS_OTHER,
        )

    # Create optimizer performance comparison table
    if use_best_only:
        print("\nCreating optimizer performance comparison table...")
        
        # Get best experiments data for table creation
        best_experiments = find_best_experiments_by_stats(gan_dirs)
        
        if best_experiments:
            create_optimizer_performance_table(best_experiments)
        else:
            print("No best experiments data found for table creation")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
