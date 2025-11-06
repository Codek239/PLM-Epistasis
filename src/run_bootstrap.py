"""
run_bootstrap.py
-----------------
Script to perform a one-sided bootstrap hypothesis test to determine if
a model's performance is statistically significantly better than a baseline.

This script loads:
1. True labels from the full training/evaluation dataset.
2. Model predictions (as dict-like strings) from a prediction summary CSV.
3. Fits a DummyClassifier on the true labels to act as a baseline.

It then runs N bootstrap samples to compute a p-value for the null hypothesis
that the estimator's metric is less than or equal to the baseline's metric.

Typical usage:
    # Test if a model's metrics are better than 'stratified'
    python src/run_bootstrap.py \
        --task_type classification \
        --input_csv data/input_VRC01_IC80.csv \
        --pred_csv results/full/training_summary_results.csv \
        --output_file bootstrap_rep_1.json \
        --baseline_strategy stratified \
        --n_bootstrap 5000
"""

# ============================
# Imports
# ============================
import os
import sys
import argparse
import logging
import re
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from tqdm import tqdm

# Local imports
sys.path.insert(0, 'src')
import utils as ut

# ============================
# Defaults
# ============================
CLF_LABEL_NAME = "Label"
REG_LABEL_NAME = "Value"
LOG_DIR = "logs/bootstrap"
RESULT_DIR = "results/bootstrap"
# ============================

def parse_args():
    """Parse command-line arguments for the bootstrap test."""
    parser = argparse.ArgumentParser(description="Run bootstrap significance test.")

    # --- Task & Model Configuration ---
    parser.add_argument("--task_type", type=str, required=True, choices=["classification", "regression"],
                        help="The type of task the model was trained for.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (for classification model).")
    
    # --- Paths and I/O ---
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the *training/evaluation* dataset CSV (must have true labels).")
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to the prediction summary CSV (must have 'Prediction' column).")
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR, help="Directory to save result files.")
    parser.add_argument("--output_file", type=str, default="bootstrap_results.json", help="Name of the output JSON file to save results.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save log files.")

    # --- Test Hyperparameters ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap samples.")
    parser.add_argument("--baseline_strategy", type=str, default="most_frequent", choices=["most_frequent", "stratified", "uniform"], help="DummyClassifier strategy.")

    return parser.parse_args()

def main():
    """Main execution routine for running the bootstrap test."""
    args = parse_args()
    logger = ut.setup_logging(args.log_dir, "bootstrap_test")

    logger.info("=================================================")
    logger.info(f"   Starting Bootstrap Significance Test   ")
    logger.info("=================================================")
    logger.info("Running with the following configuration:")
    for key, value in vars(args).items():
        logger.info(f"  - {key}: {value}")

    # --- Setup ---
    os.makedirs(args.result_dir, exist_ok=True)
    ut.set_seed(args.seed)

    # --- Load Data ---
    logger.info(f"Loading true labels from: {args.input_csv}")
    eval_df = pd.read_csv(args.input_csv)
    
    logger.info(f"Loading predictions from: {args.pred_csv}")
    pred_df = pd.read_csv(args.pred_csv)

    if args.task_type == "regression":
        y_true = eval_df[REG_LABEL_NAME].astype(float).values
        clf_preds = pred_df["Prediction"].astype(float).values
    else: # classification
        y_true = eval_df[CLF_LABEL_NAME].astype(int).values
        if args.num_classes != 2:
            logger.error(f"This script only supports binary classification task (num_classes=2)")
            sys.exit(1)
        
        logger.info("Parsing classification predictions...")
        clf_preds_list = []
        for pred in tqdm(pred_df["Prediction"], desc="Parsing predictions"):
            try:
                match_0 = re.search(r"['\"]0['\"]:\s*([0-9\.]+)", str(pred))
                match_1 = re.search(r"['\"]1['\"]:\s*([0-9\.]+)", str(pred))
                
                if not match_0 or not match_1:
                    raise ValueError("Could not find keys '0' and '1' in pred string.")
                
                pred_0 = float(match_0.group(1))
                pred_1 = float(match_1.group(1))
                
                clf_preds_list.append(1 if pred_1 > pred_0 else 0)
            
            except Exception as e:
                logger.error(f"Failed to parse prediction string: '{pred}'. Error: {e}")
                sys.exit(1)
        
        clf_preds = np.array(clf_preds_list)
    
    X_train_dummy = np.zeros(len(y_true))
    
    if len(y_true) != len(clf_preds):
        logger.error(f"Mismatched lengths! True labels: {len(y_true)}, Predictions: {len(clf_preds)}")
        sys.exit(1)

    # --- Initialize and Fit Baseline Model ---
    logger.info(f"Fitting baseline DummyClassifier (strategy='{args.baseline_strategy}')...")
    baseline = DummyClassifier(strategy=args.baseline_strategy, random_state=args.seed)
    baseline.fit(X_train_dummy, y_true)
    baseline_preds = baseline.predict(X_train_dummy)

    # --- Calculate Observed Scores ---
    clf_score_metric = ut.evaluate_predictions(pred=clf_preds, true=y_true, task_type=args.task_type)
    baseline_score_metric = ut.evaluate_predictions(pred=baseline_preds, true=y_true, task_type=args.task_type)

    logger.info("--- Observed Scores (on full set) ---")
    for score in clf_score_metric:
        logger.info(f"  - Estimator {score.upper()}: {clf_score_metric[score]:.4f}")
        logger.info(f"  - Baseline  {score.upper()}: {baseline_score_metric[score]:.4f}")
        logger.info(f"  - Observed Difference: {clf_score_metric[score] - baseline_score_metric[score]:+.4f}")
    
    # --- Run Bootstrap Test ---
    logger.info(f"Running {args.n_bootstrap} bootstrap samples...")
    n_samples = len(y_true)
    better_metric_count = dict.fromkeys(clf_score_metric.keys(), 0)
    all_indices = np.arange(n_samples)

    for i in tqdm(range(args.n_bootstrap), desc="Bootstrapping"):
        bootstrap_indices = np.random.choice(all_indices, size=n_samples, replace=True)
        
        bootstrap_y_true = y_true[bootstrap_indices]
        bootstrap_clf_preds = clf_preds[bootstrap_indices]
        bootstrap_baseline_preds = baseline_preds[bootstrap_indices]

        bootstrap_clf_score_metric = ut.evaluate_predictions(
            pred=bootstrap_clf_preds, 
            true=bootstrap_y_true, 
            task_type=args.task_type
            )
        bootstrap_baseline_score_metric = ut.evaluate_predictions(
            pred=bootstrap_baseline_preds, 
            true=bootstrap_y_true, 
            task_type=args.task_type
            )
        
        for score in bootstrap_clf_score_metric:
            bootstrap_metric_diff = bootstrap_clf_score_metric[score] - bootstrap_baseline_score_metric[score]
            # Count times the estimator is strictly better
            if bootstrap_metric_diff > 0:
                better_metric_count[score] += 1
    
    logger.info("=================================================")
    logger.info("         Bootstrap Test Results         ")
    logger.info("=================================================")
    logger.info(f"  - H0: Estimator_Metric <= Baseline_Metric")
    logger.info(f"  - HA: Estimator_Metric > Baseline_Metric")
    for score in clf_score_metric:
        p_value = 1.0 - (better_metric_count[score] / float(args.n_bootstrap))
        percent_better = (better_metric_count[score] / float(args.n_bootstrap)) * 100.0
        logger.info(f"--- Metric: {score.upper()} ---")
        logger.info(f"  - Samples where Est > Base: {better_metric_count[score]} / {args.n_bootstrap} ({percent_better:.1f}%)")
        logger.info(f"  - p-value (1 - better_count/N): {p_value:.4f}")

        if p_value < 0.05:
            logger.info(f"  - RESULT: Statistically significant (p < 0.05).")
            logger.info("  - The estimator's performance is significantly better than the baseline.")
        else:
            logger.info(f"  - RESULT: Not statistically significant (p >= 0.05).")
            logger.info("  - Cannot conclude the estimator is better than the baseline.")
        logger.info("---------------------------------")

    # --- Save the results to JSON ---
    output_path = os.path.join(args.result_dir, args.output_file)
    logger.info(f"Saving final results to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save results to JSON. Error: {e}")

if __name__ == "__main__":
    main()
