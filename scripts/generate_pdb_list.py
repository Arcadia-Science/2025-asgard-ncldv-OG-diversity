#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a master list of PDB IDs to fetch directly from the
proteome_database_v3.2.csv. (v3 - Enhanced diagnostic logging for PDB_HIT_COL)

This script identifies all unique proteins that have a PDB hit recorded
and extracts their ProteinID and the PDB ID.
"""

import pandas as pd
from pathlib import Path
import logging
from collections import Counter

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
else:
    logger.setLevel(logging.INFO)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent # Assumes script is in 'scripts' directory
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs" # Output directory

PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
OUTPUT_PDB_MASTER_FETCH_LIST_CSV = ANALYSIS_OUTPUTS_DIR / "pdb_master_fetch_list.csv"

PROTEIN_ID_COL = "ProteinID"
PDB_HIT_COL = "SeqSearch_PDB_Hit" # Column in proteome_db containing PDB IDs

# Columns to load from proteome_database_v3.2.csv
PROTEOME_COLS_TO_LOAD = [PROTEIN_ID_COL, PDB_HIT_COL, "Orthogroup"] # Orthogroup for context

def main():
    logger.info("Starting generation of master PDB fetch list...")
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROTEOME_DB_FILE.exists():
        logger.error(f"Proteome database file '{PROTEOME_DB_FILE}' not found. Exiting.")
        return
    try:
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=lambda c: c in PROTEOME_COLS_TO_LOAD, low_memory=False)
        logger.info(f"Read {len(df_proteome)} entries from '{PROTEOME_DB_FILE}'.")
    except ValueError:
        logger.warning(f"Could not load specified columns. Trying to load all from {PROTEOME_DB_FILE}.")
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, low_memory=False)
        if not all(col in df_proteome.columns for col in [PROTEIN_ID_COL, PDB_HIT_COL]):
            logger.error(f"Required columns ('{PROTEIN_ID_COL}', '{PDB_HIT_COL}') not found in '{PROTEOME_DB_FILE}'. Exiting.")
            return
        cols_to_keep_after_full_load = [col for col in PROTEOME_COLS_TO_LOAD if col in df_proteome.columns]
        df_proteome = df_proteome[cols_to_keep_after_full_load]

    if PDB_HIT_COL not in df_proteome.columns:
        logger.error(f"Critical column '{PDB_HIT_COL}' not found in loaded proteome data. Exiting.")
        return

    df_pdb_hits_initial = df_proteome[df_proteome[PDB_HIT_COL].notna()].copy()
    logger.info(f"Found {len(df_pdb_hits_initial)} rows with non-NaN values in '{PDB_HIT_COL}'.")

    if not df_pdb_hits_initial.empty:
        # --- ENHANCED DIAGNOSTIC LOGGING ---
        unique_pdb_hit_values = df_pdb_hits_initial[PDB_HIT_COL].unique()
        logger.info(f"Analyzing unique non-NaN values found in '{PDB_HIT_COL}':")
        
        value_types = Counter(type(val).__name__ for val in unique_pdb_hit_values)
        logger.info(f"  Data types found: {dict(value_types)}")

        str_values = [str(val).strip() for val in unique_pdb_hit_values if isinstance(val, str)]
        bool_true_count = sum(1 for val in unique_pdb_hit_values if isinstance(val, bool) and val is True)
        bool_false_count = sum(1 for val in unique_pdb_hit_values if isinstance(val, bool) and val is False)
        
        logger.info(f"  Number of unique string values: {len(str_values)}")
        logger.info(f"  Number of unique True boolean values: {bool_true_count}")
        logger.info(f"  Number of unique False boolean values: {bool_false_count}")

        logger.info(f"  Sample of unique string values (up to 10):")
        for i, val_str in enumerate(str_values):
            if i < 10:
                logger.info(f"    String Value: '{val_str}' (Length: {len(val_str)})")
            else:
                logger.info(f"    ... and {len(str_values) - 10} more unique string values.")
                break
        if not str_values:
            logger.info("    No string values found among unique non-NaN entries.")
        # --- END ENHANCED DIAGNOSTIC LOGGING ---

    def is_valid_pdb_id_format(pdb_id_val):
        if isinstance(pdb_id_val, str):
            pdb_id_str = pdb_id_val.strip()
            # Standard PDB IDs are 4 characters. Some entries might include chain (e.g., 1XYZ_A)
            # Allowing a bit more flexibility, e.g. up to 6-8 for PDB ID + chain or minor variations.
            is_valid = len(pdb_id_str) >= 4 and len(pdb_id_str) <= 8 
            return is_valid
        return False

    df_pdb_hits_validated = df_pdb_hits_initial[df_pdb_hits_initial[PDB_HIT_COL].apply(is_valid_pdb_id_format)].copy()
    logger.info(f"Found {len(df_pdb_hits_validated)} proteins with PDB IDs matching string & length criteria (4-8 chars) in '{PDB_HIT_COL}'.")

    if df_pdb_hits_validated.empty:
        logger.info("No proteins with PDB IDs matching string & length criteria found. Output file will be empty.")
        logger.info(f"If '{PDB_HIT_COL}' contains boolean True values, it indicates a PDB hit exists, but the actual PDB ID is missing from this column.")
        logger.info("This script requires actual PDB ID strings (e.g., '1XYZ') to generate a download list.")
        pd.DataFrame(columns=[PROTEIN_ID_COL, "PDB_ID", "Orthogroup"]).to_csv(OUTPUT_PDB_MASTER_FETCH_LIST_CSV, index=False)
        return

    cols_for_output_selection = [col for col in [PROTEIN_ID_COL, PDB_HIT_COL, "Orthogroup"] if col in df_pdb_hits_validated.columns]
    if PDB_HIT_COL not in cols_for_output_selection:
        logger.error(f"Critical column '{PDB_HIT_COL}' is missing before creating output. Aborting.")
        pd.DataFrame(columns=[PROTEIN_ID_COL, "PDB_ID", "Orthogroup"]).to_csv(OUTPUT_PDB_MASTER_FETCH_LIST_CSV, index=False)
        return

    df_output = df_pdb_hits_validated[cols_for_output_selection].copy()
    df_output.rename(columns={PDB_HIT_COL: "PDB_ID"}, inplace=True)
    
    if "PDB_ID" in df_output.columns:
        df_output.loc[:, 'PDB_ID'] = df_output['PDB_ID'].apply(lambda x: x.split(';')[0].strip() if isinstance(x, str) else x)
    else:
        logger.error("Column 'PDB_ID' (after rename) not found. Cannot process multiple PDB_IDs.")

    if "PDB_ID" in df_output.columns and PROTEIN_ID_COL in df_output.columns:
        df_output.drop_duplicates(subset=[PROTEIN_ID_COL, "PDB_ID"], inplace=True)
        df_output.sort_values(by=[PROTEIN_ID_COL], inplace=True)
    elif PROTEIN_ID_COL in df_output.columns:
        df_output.drop_duplicates(subset=[PROTEIN_ID_COL], inplace=True)
        df_output.sort_values(by=[PROTEIN_ID_COL], inplace=True)

    try:
        df_output.to_csv(OUTPUT_PDB_MASTER_FETCH_LIST_CSV, index=False)
        logger.info(f"Master PDB fetch list with {len(df_output)} entries saved to: {OUTPUT_PDB_MASTER_FETCH_LIST_CSV}")
        logger.info(f"This file can now be used as input for a PDB fetching script.")
    except Exception as e:
        logger.error(f"Error writing master PDB fetch list CSV: {e}", exc_info=True)

    logger.info("--- Generate Master PDB Fetch List Script Finished ---")

if __name__ == "__main__":
    main()
