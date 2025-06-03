#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-checks the 'afdb_structures_missing_from_log_v6.csv' against the main
proteome database to add UniProtKB Accession Numbers and checks if the
corresponding AlphaFold structure file exists locally.

This helps verify the list of proteins that supposedly have AlphaFold structures
(based on Avg_pLDDT) but are not in the local structure download log,
and whether their files are already downloaded but perhaps unlogged.
"""

import pandas as pd
from pathlib import Path
import logging
import os # For os.path.join

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent # Assumes script is in 'scripts' directory
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"
DATA_DIR = BASE_DIR / "data"

MISSING_AFDB_LOG_FILE = ANALYSIS_OUTPUTS_DIR / "afdb_structures_missing_from_log_v6.csv"
PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
# Output filename remains the same, but will now have an extra column
OUTPUT_CROSSCHECKED_FILE = ANALYSIS_OUTPUTS_DIR / "afdb_missing_log_with_uniprot_ac_v6.csv"

# Directory where AlphaFold PDB files are stored
LOCAL_AFDB_STRUCTURE_DIR = DATA_DIR / "downloaded_structures" / "alphafold"

PROTEIN_ID_COL = "ProteinID"
UNIPROT_AC_COL = "UniProtKB_AC" # Column name in your main proteome database
AFDB_MODEL_VERSION = "v4" # Common AlphaFold model version

# Columns to load from proteome_database_v3.2.csv
PROTEOME_COLS_FOR_CROSSCHECK = [PROTEIN_ID_COL, UNIPROT_AC_COL]


def check_local_afdb_file_exists(uniprot_ac):
    """
    Checks if the AlphaFold DB structure file for a given UniProt AC exists locally.
    Returns True if found, False otherwise.
    """
    if pd.isna(uniprot_ac) or not isinstance(uniprot_ac, str) or not uniprot_ac.strip():
        return False
    
    # Construct the expected filename (adjust if your naming convention differs)
    # Standard AlphaFold DB filename: AF-{UniProt AC}-F1-model_v{version}.pdb
    expected_filename = f"AF-{uniprot_ac.strip()}-F1-model_{AFDB_MODEL_VERSION}.pdb"
    file_path = LOCAL_AFDB_STRUCTURE_DIR / expected_filename
    
    return file_path.exists()

def main():
    logger.info("Starting cross-check for missing AFDB structures and local file verification...")
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_AFDB_STRUCTURE_DIR.mkdir(parents=True, exist_ok=True) # Ensure local AFDB dir exists

    # Load the list of proteins with missing AFDB structures
    if not MISSING_AFDB_LOG_FILE.exists():
        logger.error(f"Input file '{MISSING_AFDB_LOG_FILE}' not found. Exiting.")
        return
    try:
        df_missing_afdb = pd.read_csv(MISSING_AFDB_LOG_FILE)
        logger.info(f"Read {len(df_missing_afdb)} entries from '{MISSING_AFDB_LOG_FILE}'.")
    except Exception as e:
        logger.error(f"Error reading '{MISSING_AFDB_LOG_FILE}': {e}")
        return

    output_columns = list(df_missing_afdb.columns) + [UNIPROT_AC_COL, 'Local_AFDB_File_Found']
    if df_missing_afdb.empty:
        logger.info("The missing AFDB log file is empty. No cross-checking needed.")
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_CROSSCHECKED_FILE, index=False)
        logger.info(f"Empty cross-checked file saved to: {OUTPUT_CROSSCHECKED_FILE}")
        return

    # Load the main proteome database to get UniProtKB ACs
    if not PROTEOME_DB_FILE.exists():
        logger.error(f"Proteome database file '{PROTEOME_DB_FILE}' not found. Exiting.")
        return
    try:
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=lambda c: c in PROTEOME_COLS_FOR_CROSSCHECK, low_memory=False)
        logger.info(f"Read {len(df_proteome)} entries from '{PROTEOME_DB_FILE}' for UniProtKB AC mapping.")
    except ValueError: 
        logger.warning(f"Could not load specified columns for UniProtKB AC mapping. Trying to load all from {PROTEOME_DB_FILE}.")
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, low_memory=False)
        if PROTEIN_ID_COL not in df_proteome.columns or UNIPROT_AC_COL not in df_proteome.columns:
            logger.error(f"Required columns ('{PROTEIN_ID_COL}', '{UNIPROT_AC_COL}') not found in '{PROTEOME_DB_FILE}'. Exiting.")
            return
        df_proteome = df_proteome[PROTEOME_COLS_FOR_CROSSCHECK]


    # Merge to add UniProtKB_AC to the missing AFDB list
    df_crosschecked = pd.merge(df_missing_afdb,
                               df_proteome[[PROTEIN_ID_COL, UNIPROT_AC_COL]].drop_duplicates(subset=[PROTEIN_ID_COL]), # Ensure unique ProteinIDs from proteome
                               on=PROTEIN_ID_COL,
                               how='left')

    num_after_merge = len(df_crosschecked)
    if num_after_merge != len(df_missing_afdb):
        logger.warning(f"Merge resulted in {num_after_merge} rows, but original missing log had {len(df_missing_afdb)}. This might indicate duplicate ProteinIDs in the proteome DB if rows increased (should be handled by drop_duplicates), or missing ProteinIDs if rows decreased (unlikely with left merge).")

    # Check for local file existence
    if UNIPROT_AC_COL in df_crosschecked.columns:
        df_crosschecked['Local_AFDB_File_Found'] = df_crosschecked[UNIPROT_AC_COL].apply(check_local_afdb_file_exists)
        num_local_files_found = df_crosschecked['Local_AFDB_File_Found'].sum()
        logger.info(f"Checked for local AFDB files: {num_local_files_found} were found in '{LOCAL_AFDB_STRUCTURE_DIR}'.")
    else:
        logger.warning(f"'{UNIPROT_AC_COL}' column not found after merge. Cannot check for local files.")
        df_crosschecked['Local_AFDB_File_Found'] = False # Add column with False if UniProt AC is missing


    num_with_uniprot = df_crosschecked[UNIPROT_AC_COL].notna().sum()
    num_missing_uniprot = df_crosschecked[UNIPROT_AC_COL].isna().sum()

    logger.info(f"Out of {len(df_crosschecked)} proteins from the missing AFDB log:")
    logger.info(f"  {num_with_uniprot} were successfully mapped to a UniProtKB_AC.")
    logger.info(f"  {num_missing_uniprot} could not be mapped to a UniProtKB_AC (UniProtKB_AC is NaN).")

    # Save the cross-checked list
    try:
        # Ensure correct column order if 'Local_AFDB_File_Found' was added
        final_columns = list(df_missing_afdb.columns) + [UNIPROT_AC_COL]
        if 'Local_AFDB_File_Found' in df_crosschecked.columns:
            final_columns.append('Local_AFDB_File_Found')
        
        # Reorder if necessary and drop any extra columns from merge
        df_crosschecked = df_crosschecked[[col for col in final_columns if col in df_crosschecked.columns]]

        df_crosschecked.sort_values(by=['OG_ID', PROTEIN_ID_COL], inplace=True)
        df_crosschecked.to_csv(OUTPUT_CROSSCHECKED_FILE, index=False, float_format='%.2f')
        logger.info(f"Cross-checked list with UniProtKB_AC and local file status saved to: {OUTPUT_CROSSCHECKED_FILE}")
        logger.info("Please review this file. You can use the UniProtKB_AC to manually check AlphaFold DB or adapt the fetch_structures_script.py.")
    except Exception as e:
        logger.error(f"Error writing cross-checked CSV file: {e}", exc_info=True)

    logger.info("--- Cross-check Missing AFDB Script Finished ---")

if __name__ == "__main__":
    main()
