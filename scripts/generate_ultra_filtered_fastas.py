#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters members of large OGs based on sequence identity to their anchor,
then extracts sequences for these filtered members and their anchors to create
new FASTA files for a refined analysis pipeline.

Input:
- Curated OG anchor list (e.g., curated_og_anchor_list_v7.2.csv)
- OG member identity to anchor CSV (e.g., og_member_identity_to_anchor.csv)
- Proteome database (e.g., proteome_database_v3.2.csv)
- OG category summary for large OGs (e.g., og_protein_category_summary_large_ogs.csv)
  or the full OG category summary to identify large OGs.

Output:
- FASTA files in a new directory (e.g., data_large_ogs_gt100mem_gt15id/raw_og_fastas/)
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
    logger.propagate = False
else:
    for handler in logger.handlers: 
        if handler.level > logging.INFO: handler.setLevel(logging.INFO)
    if logger.level > logging.INFO : logger.setLevel(logging.INFO)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

# Input files
CURATED_OG_ANCHOR_FILE = ANALYSIS_OUTPUTS_DIR / "curated_og_anchor_list_v7.2.csv"
MEMBER_IDENTITY_FILE = ANALYSIS_OUTPUTS_DIR / "og_member_identity_to_anchor.csv"
PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
# Use the summary file for large OGs if available, otherwise the full summary and filter it.
OG_CATEGORY_SUMMARY_LARGE_OGS_FILE = ANALYSIS_OUTPUTS_DIR / "og_protein_category_summary_large_ogs.csv" 
OG_CATEGORY_SUMMARY_FULL_FILE = ANALYSIS_OUTPUTS_DIR / "og_protein_category_summary.csv" # Fallback

# New output directory for this "ultra-filtered" run
NEW_PIPELINE_BASE_DIR_NAME = "data_large_ogs_gt100mem_gt15id"
NEW_RAW_OG_FASTAS_OUTPUT_DIR = DATA_DIR / NEW_PIPELINE_BASE_DIR_NAME / "raw_og_fastas"

# Filtering thresholds
MIN_MEMBERS_FOR_LARGE_OG = 100 # Based on N_Members_High_pLDDT
MIN_IDENTITY_TO_ANCHOR_PERCENT = 15.0
MIN_MEMBERS_AFTER_FILTERING_FOR_OG = 3 # Minimum sequences (anchor + members) to keep an OG

# Column names
OG_ID_COL = "OG_ID"
ANCHOR_PID_COL = "Anchor_ProteinID"
MEMBER_PID_COL = "Member_ProteinID"
IDENTITY_COL_TO_FILTER = "Identity_To_Anchor_Ungapped" # Or FullLength, if preferred
N_MEMBERS_COL_FOR_LARGE_OG = "N_Members_High_pLDDT" # From category summary

PROTEIN_ID_COL_DB = "ProteinID"
OG_ID_COL_DB = "Orthogroup" 
SEQUENCE_COL_DB = "Sequence"
UNIPROT_AC_COL_DB = "UniProtKB_AC"

PROTEOME_COLS_TO_LOAD = [PROTEIN_ID_COL_DB, OG_ID_COL_DB, SEQUENCE_COL_DB, UNIPROT_AC_COL_DB]

def main():
    logger.info(f"Starting generation of FASTA files for ultra-filtered large OGs (>{MIN_MEMBERS_FOR_LARGE_OG} members, member ID to anchor >= {MIN_IDENTITY_TO_ANCHOR_PERCENT}%)...")
    NEW_RAW_OG_FASTAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output FASTA files will be saved to: {NEW_RAW_OG_FASTAS_OUTPUT_DIR}")

    # 1. Load necessary files
    try:
        df_curated_ogs = pd.read_csv(CURATED_OG_ANCHOR_FILE)
        df_member_identities = pd.read_csv(MEMBER_IDENTITY_FILE)
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=PROTEOME_COLS_TO_LOAD, low_memory=True)

        if OG_CATEGORY_SUMMARY_LARGE_OGS_FILE.exists():
            df_og_summary = pd.read_csv(OG_CATEGORY_SUMMARY_LARGE_OGS_FILE)
            logger.info(f"Using pre-filtered large OG summary: {OG_CATEGORY_SUMMARY_LARGE_OGS_FILE}")
        elif OG_CATEGORY_SUMMARY_FULL_FILE.exists():
            df_og_summary_full = pd.read_csv(OG_CATEGORY_SUMMARY_FULL_FILE)
            df_og_summary = df_og_summary_full[df_og_summary_full[N_MEMBERS_COL_FOR_LARGE_OG] > MIN_MEMBERS_FOR_LARGE_OG].copy()
            logger.info(f"Using full OG summary and filtering for OGs with >{MIN_MEMBERS_FOR_LARGE_OG} members.")
        else:
            logger.error(f"Neither large OG summary nor full OG summary file found. Cannot identify large OGs. Searched for: {OG_CATEGORY_SUMMARY_LARGE_OGS_FILE} and {OG_CATEGORY_SUMMARY_FULL_FILE}")
            return
        
        # Validate columns
        if OG_ID_COL not in df_curated_ogs.columns or ANCHOR_PID_COL not in df_curated_ogs.columns:
            logger.error(f"Curated OG list missing required columns. Needed: '{OG_ID_COL}', '{ANCHOR_PID_COL}'."); return
        if not all(c in df_member_identities.columns for c in [OG_ID_COL, MEMBER_PID_COL, ANCHOR_PID_COL, IDENTITY_COL_TO_FILTER]):
            logger.error(f"Member identity file missing required columns. Needed: '{OG_ID_COL}', '{MEMBER_PID_COL}', '{ANCHOR_PID_COL}', '{IDENTITY_COL_TO_FILTER}'."); return
        if not all(c in df_proteome.columns for c in PROTEOME_COLS_TO_LOAD):
            missing_p_cols = [c for c in PROTEOME_COLS_TO_LOAD if c not in df_proteome.columns]
            logger.error(f"Proteome DB missing required columns: {missing_p_cols}."); return
        if OG_ID_COL not in df_og_summary.columns or N_MEMBERS_COL_FOR_LARGE_OG not in df_og_summary.columns:
            logger.error(f"OG summary file missing required columns. Needed: '{OG_ID_COL}', '{N_MEMBERS_COL_FOR_LARGE_OG}'."); return

    except Exception as e:
        logger.error(f"Error loading input files: {e}", exc_info=True); return

    large_og_ids = set(df_og_summary[df_og_summary[N_MEMBERS_COL_FOR_LARGE_OG] > MIN_MEMBERS_FOR_LARGE_OG][OG_ID_COL].unique())
    logger.info(f"Identified {len(large_og_ids)} large OGs (>{MIN_MEMBERS_FOR_LARGE_OG} members based on N_Members_High_pLDDT).")

    if not large_og_ids:
        logger.warning("No large OGs identified based on the criteria. Exiting."); return

    # Filter curated OGs to only include these large OGs
    df_curated_large_ogs = df_curated_ogs[df_curated_ogs[OG_ID_COL].isin(large_og_ids)].copy()
    logger.info(f"Processing {len(df_curated_large_ogs)} large OGs for sequence extraction.")

    ogs_written_count = 0
    ogs_skipped_too_few_members_after_filter = 0

    for _, og_row in tqdm(df_curated_large_ogs.iterrows(), total=len(df_curated_large_ogs), desc="Extracting Filtered Sequences"):
        og_id = og_row[OG_ID_COL]
        anchor_pid = str(og_row[ANCHOR_PID_COL])

        # Get members for this OG from the identity file
        current_og_identities = df_member_identities[df_member_identities[OG_ID_COL] == og_id]
        if current_og_identities.empty:
            logger.warning(f"No identity data found for OG {og_id}. Skipping."); continue

        # Filter members by identity threshold
        filtered_members = current_og_identities[
            current_og_identities[IDENTITY_COL_TO_FILTER] >= MIN_IDENTITY_TO_ANCHOR_PERCENT
        ]
        
        member_pids_to_keep = set(filtered_members[MEMBER_PID_COL].astype(str).unique())
        # Always include the anchor
        member_pids_to_keep.add(anchor_pid) 

        if len(member_pids_to_keep) < MIN_MEMBERS_AFTER_FILTERING_FOR_OG:
            logger.warning(f"OG {og_id}: After identity filter (>= {MIN_IDENTITY_TO_ANCHOR_PERCENT}%) and adding anchor, only {len(member_pids_to_keep)} sequences remain (min required: {MIN_MEMBERS_AFTER_FILTERING_FOR_OG}). Skipping this OG.")
            ogs_skipped_too_few_members_after_filter += 1
            continue
            
        # Fetch sequences from proteome_db
        sequences_for_og_df = df_proteome[df_proteome[PROTEIN_ID_COL_DB].isin(member_pids_to_keep) & (df_proteome[OG_ID_COL_DB] == og_id)]
        
        fasta_records = []
        # Ensure anchor is present in the fetched sequences
        anchor_seq_fetched = False
        for _, seq_row in sequences_for_og_df.iterrows():
            protein_id_db = str(seq_row[PROTEIN_ID_COL_DB])
            sequence_str = seq_row[SEQUENCE_COL_DB]
            uniprot_ac = seq_row.get(UNIPROT_AC_COL_DB)

            if pd.isna(sequence_str) or not str(sequence_str).strip():
                logger.warning(f"Protein {protein_id_db} in OG {og_id} has missing/empty sequence. Skipping."); continue
            
            description_parts = []
            if pd.notna(uniprot_ac) and str(uniprot_ac).strip(): description_parts.append(f"UniProtKB_AC={str(uniprot_ac).strip()}")
            description_parts.append(f"OG_ID={og_id}")
            description_str = " ".join(description_parts)

            seq = Seq(str(sequence_str))
            record = SeqRecord(seq, id=protein_id_db, description=description_str)
            fasta_records.append(record)
            if protein_id_db == anchor_pid:
                anchor_seq_fetched = True
        
        if not anchor_seq_fetched: # Should not happen if anchor_pid was added to member_pids_to_keep and exists in proteome_db
            logger.error(f"CRITICAL: Anchor {anchor_pid} for OG {og_id} was not found in proteome_db after filtering. This indicates a data inconsistency. Skipping OG.")
            continue

        if fasta_records:
            safe_og_id_filename = str(og_id).replace('/', '_')
            output_fasta_file = NEW_RAW_OG_FASTAS_OUTPUT_DIR / f"{safe_og_id_filename}.fasta"
            try:
                SeqIO.write(fasta_records, output_fasta_file, "fasta")
                logger.info(f"Wrote {len(fasta_records)} sequences for ultra-filtered OG {og_id} to {output_fasta_file}")
                ogs_written_count += 1
            except Exception as e:
                logger.error(f"Error writing FASTA for ultra-filtered OG {og_id}: {e}", exc_info=True)
        else:
            logger.warning(f"No valid sequences to write for ultra-filtered OG: {og_id}.")

    logger.info(f"--- Ultra-Filtered Large OG FASTA Generation Finished ---")
    logger.info(f"Successfully wrote FASTA files for {ogs_written_count} ultra-filtered large OGs.")
    if ogs_skipped_too_few_members_after_filter > 0:
        logger.warning(f"{ogs_skipped_too_few_members_after_filter} large OGs were skipped as they had < {MIN_MEMBERS_AFTER_FILTERING_FOR_OG} members after identity filtering.")
    logger.info(f"FASTA files are in: {NEW_RAW_OG_FASTAS_OUTPUT_DIR}")
    logger.info(f"Next: Run initial MAFFT on files in {NEW_RAW_OG_FASTAS_OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
