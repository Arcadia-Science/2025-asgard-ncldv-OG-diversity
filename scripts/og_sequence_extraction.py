#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for Step 0.1 & 0.2 of Phase 1 Alignment Pipeline:
Extracts member sequences for selected Orthogroups (OGs) and saves them
into individual FASTA files.
(v2 - Adds UniProtKB_AC and OG_ID to FASTA headers)

Input:
- curated_og_anchor_list_vX.csv (or user-specified curated OG list)
- proteome_database_v3.2.csv (main database with sequences and UniProtKB_AC)

Output:
- FASTA files, one per OG, in DATA_DIR/raw_og_fastas/
  (e.g., DATA_DIR/raw_og_fastas/OG000XXXX.fasta)
  Headers will be like: >ProteinID UniProtKB_AC=P12345 OG_ID=OG000XXXX
"""

import pandas as pd
from pathlib import Path
import logging
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm.auto import tqdm

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
DATA_DIR = BASE_DIR / "data/data_esp_ogs"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs/analysis_esp_ogs"

# Make sure this points to your latest curated OG list from the anchor selection script
CURATED_OG_LIST_FILE = ANALYSIS_OUTPUTS_DIR / "curated_esp_og_anchor_list_v1.csv" # Example, update if needed
PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"

# Output directory for raw FASTA files per OG
RAW_OG_FASTAS_OUTPUT_DIR = DATA_DIR / "esp_raw_og_fastas"

# Column names in your input files
OG_ID_COL_CURATED = "OG_ID" 

PROTEIN_ID_COL_DB = "ProteinID"
OG_ID_COL_DB = "Orthogroup" 
SEQUENCE_COL_DB = "Sequence"
UNIPROT_AC_COL_DB = "UniProtKB_AC" # ADDED

# Columns to load from proteome_database_v3.2.csv
PROTEOME_COLS_TO_LOAD = [PROTEIN_ID_COL_DB, OG_ID_COL_DB, SEQUENCE_COL_DB, UNIPROT_AC_COL_DB] # ADDED UniProtKB_AC

def main():
    logger.info("Starting extraction of OG member sequences to FASTA files (v2 - with UniProtKB_AC in header)...")
    RAW_OG_FASTAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output FASTA files will be saved to: {RAW_OG_FASTAS_OUTPUT_DIR}")

    if not CURATED_OG_LIST_FILE.exists():
        logger.error(f"Curated OG list file '{CURATED_OG_LIST_FILE}' not found. Exiting.")
        return
    try:
        df_curated_ogs = pd.read_csv(CURATED_OG_LIST_FILE)
        if OG_ID_COL_CURATED not in df_curated_ogs.columns:
            logger.error(f"Column '{OG_ID_COL_CURATED}' not found in '{CURATED_OG_LIST_FILE}'. Exiting.")
            return
        ogs_to_process = df_curated_ogs[OG_ID_COL_CURATED].unique()
        logger.info(f"Found {len(ogs_to_process)} unique OGs to process from '{CURATED_OG_LIST_FILE}'.")
    except Exception as e:
        logger.error(f"Error reading curated OG list '{CURATED_OG_LIST_FILE}': {e}", exc_info=True)
        return

    if not ogs_to_process.size: 
        logger.warning("No OGs found in the curated list. Exiting.")
        return

    if not PROTEOME_DB_FILE.exists():
        logger.error(f"Proteome database file '{PROTEOME_DB_FILE}' not found. Exiting.")
        return
    try:
        logger.info(f"Loading proteome database (cols: {PROTEOME_COLS_TO_LOAD})... This might take a moment.")
        # Load all columns from PROTEOME_COLS_TO_LOAD that exist in the CSV
        # First, get actual columns from CSV to avoid error if a requested col is missing
        csv_cols = pd.read_csv(PROTEOME_DB_FILE, nrows=0).columns.tolist()
        cols_to_actually_load = [col for col in PROTEOME_COLS_TO_LOAD if col in csv_cols]
        missing_requested_cols = [col for col in PROTEOME_COLS_TO_LOAD if col not in csv_cols]
        if missing_requested_cols:
            logger.warning(f"Requested columns not found in {PROTEOME_DB_FILE} and will be ignored: {missing_requested_cols}")
        
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=cols_to_actually_load, low_memory=True)
        logger.info(f"Read {len(df_proteome)} total protein entries from '{PROTEOME_DB_FILE}'.")
        
        # Validate required columns are present after loading
        for col in [PROTEIN_ID_COL_DB, OG_ID_COL_DB, SEQUENCE_COL_DB]: # UniProtKB_AC is optional for this script's core function but preferred
            if col not in df_proteome.columns:
                logger.error(f"Required column '{col}' not found in loaded data from '{PROTEOME_DB_FILE}'. Exiting.")
                return
                
    except Exception as e:
        logger.error(f"Error reading proteome database '{PROTEOME_DB_FILE}': {e}", exc_info=True)
        return

    ogs_processed_count = 0
    ogs_empty_count = 0
    for og_id in tqdm(ogs_to_process, desc="Processing OGs for FASTA export"):
        if pd.isna(og_id):
            logger.warning("Encountered a NaN OG_ID in the curated list. Skipping.")
            continue

        og_members_df = df_proteome[df_proteome[OG_ID_COL_DB] == og_id]

        if og_members_df.empty:
            logger.warning(f"No members found in proteome database for OG: {og_id}. Skipping.")
            ogs_empty_count += 1
            continue

        fasta_records = []
        for _, row in og_members_df.iterrows():
            protein_id = str(row[PROTEIN_ID_COL_DB]) # Ensure string
            sequence_str = row[SEQUENCE_COL_DB]
            uniprot_ac = row.get(UNIPROT_AC_COL_DB) # Use .get() in case column was missing despite request

            if pd.isna(protein_id) or protein_id.strip() == "" or pd.isna(sequence_str):
                logger.warning(f"Skipping protein in OG {og_id} due to missing ID ('{protein_id}') or sequence ('{sequence_str}').")
                continue
            
            if not isinstance(sequence_str, str) or not sequence_str.strip():
                logger.warning(f"Protein {protein_id} in OG {og_id} has an invalid or empty sequence. Skipping.")
                continue
            
            # Construct FASTA header description
            description_parts = []
            if pd.notna(uniprot_ac) and str(uniprot_ac).strip():
                description_parts.append(f"UniProtKB_AC={str(uniprot_ac).strip()}")
            description_parts.append(f"OG_ID={og_id}")
            description_str = " ".join(description_parts)

            seq = Seq(str(sequence_str))
            record = SeqRecord(seq, id=protein_id, description=description_str)
            fasta_records.append(record)

        if fasta_records:
            # Sanitize OG_ID for filename if it contains problematic characters (e.g., '/')
            safe_og_id_filename = str(og_id).replace('/', '_')
            output_fasta_file = RAW_OG_FASTAS_OUTPUT_DIR / f"{safe_og_id_filename}.fasta"
            try:
                SeqIO.write(fasta_records, output_fasta_file, "fasta")
                logger.info(f"Successfully wrote {len(fasta_records)} sequences for OG {og_id} to {output_fasta_file}")
                ogs_processed_count += 1
            except Exception as e:
                logger.error(f"Error writing FASTA file for OG {og_id}: {e}", exc_info=True)
        else:
            logger.warning(f"No valid sequences found to write for OG: {og_id}.")
            ogs_empty_count +=1

    logger.info(f"--- OG Member Sequence Extraction Finished ---")
    logger.info(f"Successfully processed and wrote FASTA files for {ogs_processed_count} OGs.")
    if ogs_empty_count > 0:
        logger.warning(f"{ogs_empty_count} OGs had no members found or no valid sequences to write.")
    logger.info(f"FASTA files are located in: {RAW_OG_FASTAS_OUTPUT_DIR}")
    logger.info("Next step: Run initial MAFFT alignment on these FASTA files (Step 0.3 of the plan).")

if __name__ == "__main__":
    main()
