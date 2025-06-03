#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a list of ProteinIDs and UniProtKB_ACs for proteins that are
flagged as having a PDB hit (where SeqSearch_PDB_Hit is True) in the
main proteome database.

This list can then be used as a starting point to find actual PDB IDs
associated with these UniProt ACs.
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
else:
    logger.setLevel(logging.INFO)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent # Assumes script is in 'scripts' directory
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
OUTPUT_UNIPROT_LIST_CSV = ANALYSIS_OUTPUTS_DIR / "uniprot_ids_for_pdb_candidates.csv"

PROTEIN_ID_COL = "ProteinID"
UNIPROT_AC_COL = "UniProtKB_AC"
PDB_HIT_FLAG_COL = "SeqSearch_PDB_Hit" # Column indicating a PDB hit (expected to be boolean True/False)

# Columns to load from proteome_database_v3.2.csv
PROTEOME_COLS_TO_LOAD = [PROTEIN_ID_COL, UNIPROT_AC_COL, PDB_HIT_FLAG_COL, "Orthogroup"]

def main():
    logger.info("Starting generation of UniProt AC list for PDB candidates...")
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROTEOME_DB_FILE.exists():
        logger.error(f"Proteome database file '{PROTEOME_DB_FILE}' not found. Exiting.")
        return
    try:
        # Pandas might infer boolean types correctly if they are True/False literals in CSV.
        # If they are strings "True"/"False", specific converters might be needed at read_csv.
        # For now, assume they are read as booleans or can be evaluated as such.
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=lambda c: c in PROTEOME_COLS_TO_LOAD, low_memory=False)
        logger.info(f"Read {len(df_proteome)} entries from '{PROTEOME_DB_FILE}'.")
    except ValueError:
        logger.warning(f"Could not load specified columns. Trying to load all from {PROTEOME_DB_FILE}.")
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, low_memory=False)
        if not all(col in df_proteome.columns for col in [PROTEIN_ID_COL, UNIPROT_AC_COL, PDB_HIT_FLAG_COL]):
            logger.error(f"Required columns ('{PROTEIN_ID_COL}', '{UNIPROT_AC_COL}', '{PDB_HIT_FLAG_COL}') not found in '{PROTEOME_DB_FILE}'. Exiting.")
            return
        cols_to_keep_after_full_load = [col for col in PROTEOME_COLS_TO_LOAD if col in df_proteome.columns]
        df_proteome = df_proteome[cols_to_keep_after_full_load]

    if PDB_HIT_FLAG_COL not in df_proteome.columns:
        logger.error(f"Critical column '{PDB_HIT_FLAG_COL}' not found in loaded proteome data. Exiting.")
        return
    if UNIPROT_AC_COL not in df_proteome.columns:
        logger.error(f"Critical column '{UNIPROT_AC_COL}' not found. Cannot extract UniProt ACs. Exiting.")
        return


    # Filter for rows where PDB_HIT_FLAG_COL is True
    # Pandas reads 'True'/'False' strings as strings, actual booleans as booleans.
    # If the column is string "True"/"False":
    # df_pdb_candidates = df_proteome[df_proteome[PDB_HIT_FLAG_COL].astype(str).str.lower() == 'true'].copy()
    # If the column is actual boolean True/False:
    if df_proteome[PDB_HIT_FLAG_COL].dtype == 'bool':
        df_pdb_candidates = df_proteome[df_proteome[PDB_HIT_FLAG_COL] == True].copy()
    else: # Attempt to handle strings 'True'/'False' or other representations
        # This flexible check handles 'True', 'true', True (boolean), 1 (int/float)
        # and 'False', 'false', False (boolean), 0 (int/float)
        def evaluate_as_true(val):
            if isinstance(val, bool): return val
            if isinstance(val, str): return val.strip().lower() == 'true'
            if isinstance(val, (int, float)): return bool(val) # 1 is True, 0 is False
            return False
        
        df_proteome['_eval_pdb_hit_'] = df_proteome[PDB_HIT_FLAG_COL].apply(evaluate_as_true)
        df_pdb_candidates = df_proteome[df_proteome['_eval_pdb_hit_'] == True].copy()
        df_pdb_candidates.drop(columns=['_eval_pdb_hit_'], inplace=True)

    logger.info(f"Found {len(df_pdb_candidates)} proteins flagged with '{PDB_HIT_FLAG_COL}' as True (or equivalent).")

    if df_pdb_candidates.empty:
        logger.info("No proteins flagged as PDB candidates found. Output file will be empty.")
        pd.DataFrame(columns=[PROTEIN_ID_COL, UNIPROT_AC_COL, "Orthogroup"]).to_csv(OUTPUT_UNIPROT_LIST_CSV, index=False)
        return

    # Select relevant columns and filter out those with missing UniProtKB_AC
    df_output = df_pdb_candidates[[PROTEIN_ID_COL, UNIPROT_AC_COL, "Orthogroup"]].copy()
    df_output.dropna(subset=[UNIPROT_AC_COL], inplace=True)
    
    # Ensure UniProtKB_AC is string and clean it
    df_output.loc[:, UNIPROT_AC_COL] = df_output[UNIPROT_AC_COL].astype(str).str.strip()
    # Filter out empty strings after stripping
    df_output = df_output[df_output[UNIPROT_AC_COL] != '']


    logger.info(f"Found {len(df_output)} PDB candidates with valid UniProtKB_ACs.")

    if df_output.empty:
        logger.info("No PDB candidates with valid UniProtKB_ACs found after filtering. Output file will be empty.")
        pd.DataFrame(columns=[PROTEIN_ID_COL, UNIPROT_AC_COL, "Orthogroup"]).to_csv(OUTPUT_UNIPROT_LIST_CSV, index=False)
        return

    df_output.drop_duplicates(subset=[PROTEIN_ID_COL, UNIPROT_AC_COL], inplace=True)
    df_output.sort_values(by=[PROTEIN_ID_COL], inplace=True)

    try:
        df_output.to_csv(OUTPUT_UNIPROT_LIST_CSV, index=False)
        logger.info(f"List of {len(df_output)} ProteinIDs and UniProtKB_ACs for PDB candidates saved to: {OUTPUT_UNIPROT_LIST_CSV}")
    except Exception as e:
        logger.error(f"Error writing UniProt AC list for PDB candidates: {e}", exc_info=True)

    logger.info("--- Generate UniProt List for PDB Candidates Script Finished ---")

if __name__ == "__main__":
    main()
