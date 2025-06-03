#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Counts the number of unique Orthogroups (OGs) present in the
structural_metrics_tmalign_plddt_filtered.csv file.

Input:
- structural_metrics_tmalign_plddt_filtered.csv

Output:
- Prints the count of unique OGs to the console.
"""

import pandas as pd
from pathlib import Path
import logging

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
else:
    for handler in logger.handlers: 
        if handler.level > logging.INFO: handler.setLevel(logging.INFO)
    if logger.level > logging.INFO : logger.setLevel(logging.INFO)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent 
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

# Input CSV file from the RMSD calculation step
RMSD_RESULTS_CSV_FILE = ANALYSIS_OUTPUTS_DIR / "structural_metrics_tmalign_plddt_filtered.csv"

# Column name for OG ID in the RMSD results file
OG_ID_COL = "OG_ID" 

def main():
    logger.info("Starting OG count from RMSD results...")

    if not RMSD_RESULTS_CSV_FILE.exists():
        logger.error(f"RMSD results file '{RMSD_RESULTS_CSV_FILE}' not found. Exiting.")
        return
    
    try:
        df_rmsd_results = pd.read_csv(RMSD_RESULTS_CSV_FILE, low_memory=False)
        logger.info(f"Read {len(df_rmsd_results)} rows from '{RMSD_RESULTS_CSV_FILE}'.")
    except Exception as e:
        logger.error(f"Error reading RMSD results file '{RMSD_RESULTS_CSV_FILE}': {e}", exc_info=True)
        return

    if OG_ID_COL not in df_rmsd_results.columns:
        logger.error(f"Column '{OG_ID_COL}' not found in '{RMSD_RESULTS_CSV_FILE}'. Cannot count OGs. Exiting.")
        return

    if df_rmsd_results.empty:
        logger.info("The RMSD results file is empty. Number of unique OGs: 0")
        return

    unique_og_count = df_rmsd_results[OG_ID_COL].nunique()
    
    logger.info(f"Number of unique OGs with structural metrics calculated: {unique_og_count}")
    
    # You might also want to see how many pairs per OG, or other quick stats
    if unique_og_count > 0 :
        logger.info(f"Example OG IDs present (first 5 unique): {df_rmsd_results[OG_ID_COL].unique()[:5]}")
        og_pair_counts = df_rmsd_results.groupby(OG_ID_COL).size()
        logger.info(f"Average member-anchor pairs per OG: {og_pair_counts.mean():.2f}")
        logger.info(f"Median member-anchor pairs per OG: {og_pair_counts.median()}")


    logger.info("--- OG Count Script Finished ---")

if __name__ == "__main__":
    main()
