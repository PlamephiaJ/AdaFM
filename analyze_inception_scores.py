#!/usr/bin/env python3
"""
Script to extract and analyze Inception Scores from all GAN experiment folders.
This script finds all experiments with real_inception_scores data, extracts statistics,
and saves them in a standardized format.

Usage: python analyze_inception_scores.py
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_inception_scores_from_csv(csv_path):
    """Load inception scores from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'inception_score' in df.columns:
            return df['inception_score'].values
        elif 'IS' in df.columns:
            return df['IS'].values
        else:
            logger.warning(f"No recognized inception score column in {csv_path}")
            return None
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return None

def load_inception_scores_from_pkl(pkl_path):
    """Load inception scores from pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            scores = pickle.load(f)
        return np.array(scores)
    except Exception as e:
        logger.error(f"Error reading pickle {pkl_path}: {e}")
        return None

def load_inception_scores_from_txt(txt_path):
    """Load inception scores from text file."""
    try:
        scores = []
        with open(txt_path, 'r') as f:
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
    except Exception as e:
        logger.error(f"Error reading text {txt_path}: {e}")
        return None

def find_and_analyze_experiments():
    """Find all GAN experiments and analyze their Inception Scores."""
    
    # Find all possible experiment directories
    gan_base = Path("GAN")
    if not gan_base.exists():
        logger.error("GAN directory not found!")
        return
    
    experiment_folders = []
    
    # Search for experiment folders (typically 3 levels deep: optimizer/dataset/timestamp)
    for optimizer_dir in gan_base.iterdir():
        if optimizer_dir.is_dir():
            for dataset_dir in optimizer_dir.iterdir():
                if dataset_dir.is_dir():
                    for timestamp_dir in dataset_dir.iterdir():
                        if timestamp_dir.is_dir():
                            experiment_folders.append(timestamp_dir)
    
    logger.info(f"Found {len(experiment_folders)} potential experiment folders")
    
    results = []
    
    for exp_folder in experiment_folders:
        logger.info(f"Processing experiment: {exp_folder}")
        
        # Check if best_real_inception_score.csv already exists
        best_is_save_path = exp_folder / "best_real_inception_score.csv"
        if best_is_save_path.exists():
            logger.info(f"Skipping {exp_folder} - best_real_inception_score.csv already exists")
            
            # Still add to results for summary (read existing data)
            try:
                existing_df = pd.read_csv(best_is_save_path)
                if 'BestIS' in existing_df.columns and 'AvgIS' in existing_df.columns:
                    path_parts = exp_folder.parts
                    if len(path_parts) >= 4:
                        optimizer = path_parts[-3]
                        dataset = path_parts[-2] 
                        timestamp = path_parts[-1]
                        experiment_name = f"{optimizer}_{dataset}_{timestamp}"
                    else:
                        experiment_name = exp_folder.name
                    
                    results.append({
                        'experiment': experiment_name,
                        'optimizer': optimizer if len(path_parts) >= 4 else 'unknown',
                        'dataset': dataset if len(path_parts) >= 4 else 'unknown',
                        'timestamp': timestamp if len(path_parts) >= 4 else 'unknown',
                        'best_is': float(existing_df['BestIS'].iloc[0]),
                        'avg_is': float(existing_df['AvgIS'].iloc[0]),
                        'final_is': None,  # Not available from summary file
                        'std_is': None,    # Not available from summary file
                        'num_scores': None, # Not available from summary file
                        'data_source': 'EXISTING',
                        'folder_path': str(exp_folder)
                    })
            except Exception as e:
                logger.warning(f"Could not read existing best file {best_is_save_path}: {e}")
            
            continue
        
        # Try to find inception score files
        inception_scores = None
        data_source = None
        
        # Look for CSV file first
        csv_files = list(exp_folder.glob("real_inception_scores.csv"))
        if csv_files:
            inception_scores = load_inception_scores_from_csv(csv_files[0])
            data_source = "CSV"
        
        # If no CSV, try pickle file
        if inception_scores is None:
            pkl_files = list(exp_folder.glob("real_inception_scores.pkl"))
            if pkl_files:
                inception_scores = load_inception_scores_from_pkl(pkl_files[0])
                data_source = "PKL"
        
        # If no pickle, try text file
        if inception_scores is None:
            txt_files = list(exp_folder.glob("real_inception_scores.txt"))
            if txt_files:
                inception_scores = load_inception_scores_from_txt(txt_files[0])
                data_source = "TXT"
        
        if inception_scores is not None and len(inception_scores) > 0:
            # Calculate statistics
            best_is = float(inception_scores.max())
            avg_is = float(inception_scores.mean())
            final_is = float(inception_scores[-1])
            std_is = float(inception_scores.std())
            num_scores = len(inception_scores)
            
            # Extract experiment info
            path_parts = exp_folder.parts
            if len(path_parts) >= 4:
                optimizer = path_parts[-3]
                dataset = path_parts[-2] 
                timestamp = path_parts[-1]
                experiment_name = f"{optimizer}_{dataset}_{timestamp}"
            else:
                experiment_name = exp_folder.name
            
            # Save individual experiment statistics
            best_is_save_path = exp_folder / "best_real_inception_score.csv"
            with open(best_is_save_path, "w") as f:
                f.write("BestIS,AvgIS\n")
                f.write(f"{best_is},{avg_is}\n")
            
            logger.info(f"Saved statistics for {experiment_name} to {best_is_save_path}")
            
            # Store for summary
            results.append({
                'experiment': experiment_name,
                'optimizer': optimizer if len(path_parts) >= 4 else 'unknown',
                'dataset': dataset if len(path_parts) >= 4 else 'unknown',
                'timestamp': timestamp if len(path_parts) >= 4 else 'unknown',
                'best_is': best_is,
                'avg_is': avg_is,
                'final_is': final_is,
                'std_is': std_is,
                'num_scores': num_scores,
                'data_source': data_source,
                'folder_path': str(exp_folder)
            })
            
        else:
            logger.warning(f"No valid inception scores found in {exp_folder}")
    
    # Create summary CSV
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = Path("inception_scores_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("INCEPTION SCORE ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total experiments analyzed: {len(results)}")
        print(f"Optimizers found: {summary_df['optimizer'].unique()}")
        print(f"Datasets found: {summary_df['dataset'].unique()}")
        print("\nBest IS by optimizer:")
        best_by_optimizer = summary_df.groupby('optimizer')['best_is'].agg(['max', 'mean', 'count'])
        print(best_by_optimizer)
        print("\nAverage IS by optimizer:")
        avg_by_optimizer = summary_df.groupby('optimizer')['avg_is'].agg(['max', 'mean', 'count'])
        print(avg_by_optimizer)
        
        # Find overall best experiments
        print(f"\nTop 5 experiments by Best IS:")
        top_experiments = summary_df.nlargest(5, 'best_is')[['experiment', 'best_is', 'avg_is']]
        print(top_experiments.to_string(index=False))
        
    else:
        logger.warning("No experiments with inception scores found!")

if __name__ == "__main__":
    find_and_analyze_experiments()