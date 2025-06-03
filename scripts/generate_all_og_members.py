#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generates a list of all member proteins present in the final, ultra-filtered
MSAs for each selected OG, along with their corresponding anchor protein ID.
This serves as input for calculate_rmsds_parallel.py for the ultra-filtered track.

Input:
- Curated OG anchor list (e.g., curated_og_anchor_list_v7.2.csv)
- Directory containing final ultra-filtered MSAs (e.g., data/data_large_ogs_gt100mem_gt15id/trimmed_fastas_for_trees/)

Output:
- CSV file (e.g., analysis_outputs/analysis_large_ogs_gt100mem_gt15id/all_og_members_for_rmsd_ultra_filtered.csv)
  (Columns: OG_ID, Member_ProteinID, Anchor_ProteinID)
"""

import pandas as pd
from pathlib import Path
import logging
from Bio import AlignIO
from tqdm.auto import tqdm
import argparse

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

# --- Default Configuration (for ultra-filtered large OGs track) ---
SCRIPT_DIR_DEFAULT = Path(__file__).resolve().parent
BASE_DIR_DEFAULT = SCRIPT_DIR_DEFAULT.parent 
NEW_PIPELINE_DATA_DIR_DEFAULT = BASE_DIR_DEFAULT / "data" / "data_large_ogs_gt100mem_gt15id"
ANALYSIS_OUTPUTS_DIR_DEFAULT = BASE_DIR_DEFAULT / "analysis_outputs" # Main analysis outputs
NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "analysis_large_ogs_gt100mem_gt15id" # Specific for this track

CURATED_OG_ANCHOR_FILE_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "curated_og_anchor_list_v7.2.csv" 
MSA_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "trimmed_fastas_for_trees"
MSA_SUFFIX_DEFAULT = "_final_trimmed.fasta"

OUTPUT_MEMBERS_FILE_DEFAULT = NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT / "all_og_members_for_rmsd_ultra_filtered.csv"

# Column names
OG_ID_COL_CURATED = "OG_ID"
ANCHOR_PID_COL_CURATED = "Anchor_ProteinID"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate member list for RMSD calculation from ultra-filtered MSAs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--anchor_list_csv", type=Path, default=CURATED_OG_ANCHOR_FILE_DEFAULT,
                        help="CSV file mapping OG_ID to Anchor_ProteinID.")
    parser.add_argument("--msa_dir", type=Path, default=MSA_DIR_DEFAULT,
                        help="Directory containing final ultra-filtered MSAs.")
    parser.add_argument("--msa_suffix", type=str, default=MSA_SUFFIX_DEFAULT,
                        help="Suffix for MSA files.")
    parser.add_argument("--output_csv", type=Path, default=OUTPUT_MEMBERS_FILE_DEFAULT,
                        help="Output CSV file for all members for RMSD.")
    return parser.parse_args()

def main(args: argparse.Namespace):
    logger.info("Starting generation of 'all OG members for RMSD' list from ultra-filtered MSAs...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    if not args.anchor_list_csv.exists():
        logger.error(f"Curated OG anchor list '{args.anchor_list_csv}' not found. Exiting.")
        return
    try:
        df_curated_ogs = pd.read_csv(args.anchor_list_csv)
        if not all(col in df_curated_ogs.columns for col in [OG_ID_COL_CURATED, ANCHOR_PID_COL_CURATED]):
            logger.error(f"'{args.anchor_list_csv}' is missing required columns ('{OG_ID_COL_CURATED}', '{ANCHOR_PID_COL_CURATED}'). Exiting.")
            return
        logger.info(f"Read {len(df_curated_ogs)} OGs from '{args.anchor_list_csv}'.")
    except Exception as e:
        logger.error(f"Error reading '{args.anchor_list_csv}': {e}", exc_info=True); return

    if not args.msa_dir.exists() or not args.msa_dir.is_dir():
        logger.error(f"MSA directory '{args.msa_dir}' not found or is not a directory. Exiting.")
        return

    all_members_list = []
    ogs_with_no_msa_found = 0
    ogs_with_empty_msa = 0

    for _, row in tqdm(df_curated_ogs.iterrows(), total=len(df_curated_ogs), desc="Extracting Members from MSAs"):
        og_id = row[OG_ID_COL_CURATED]
        anchor_pid = str(row[ANCHOR_PID_COL_CURATED]) # Ensure anchor_pid is string for comparison

        if pd.isna(og_id) or pd.isna(anchor_pid):
            logger.warning(f"Skipping row in curated OG list due to missing OG_ID or Anchor_ProteinID: OG='{og_id}', Anchor='{anchor_pid}'")
            continue

        msa_file_path = args.msa_dir / f"{og_id}{args.msa_suffix}"
        if not msa_file_path.exists():
            logger.warning(f"MSA file not found for OG {og_id} at '{msa_file_path}'. Skipping this OG for member listing.")
            ogs_with_no_msa_found += 1
            continue
        
        try:
            alignment = AlignIO.read(str(msa_file_path), "fasta")
            if not alignment or len(alignment) == 0:
                logger.warning(f"MSA for OG {og_id} at '{msa_file_path}' is empty or unreadable. Skipping.")
                ogs_with_empty_msa += 1
                continue

            found_anchor_in_msa = False
            for record in alignment:
                member_pid = str(record.id) # Ensure member_pid is string
                all_members_list.append({
                    "OG_ID": og_id,
                    "Member_ProteinID": member_pid,
                    "Anchor_ProteinID": anchor_pid
                })
                if member_pid == anchor_pid:
                    found_anchor_in_msa = True
            
            if not found_anchor_in_msa:
                logger.warning(f"Designated anchor {anchor_pid} was NOT found among sequence IDs in MSA file {msa_file_path} for OG {og_id}.")

        except Exception as e:
            logger.error(f"Error processing MSA file '{msa_file_path}' for OG {og_id}: {e}", exc_info=True)
            # Optionally add an error entry or skip
            continue
    
    if ogs_with_no_msa_found > 0:
        logger.warning(f"{ogs_with_no_msa_found} OGs were skipped because their MSA file was not found in {args.msa_dir}.")
    if ogs_with_empty_msa > 0:
        logger.warning(f"{ogs_with_empty_msa} OGs were skipped because their MSA file was empty or unreadable.")

    if not all_members_list:
        logger.warning("No members extracted from any MSAs. Output file will be empty.")
        pd.DataFrame(columns=["OG_ID", "Member_ProteinID", "Anchor_ProteinID"]).to_csv(args.output_csv, index=False)
        return

    df_output = pd.DataFrame(all_members_list)
    df_output.sort_values(by=["OG_ID", "Member_ProteinID"], inplace=True)
    df_output.drop_duplicates(inplace=True) 
    
    try:
        df_output.to_csv(args.output_csv, index=False)
        logger.info(f"Successfully saved {len(df_output)} member entries for RMSD calculation to '{args.output_csv}'.")
    except Exception as e:
        logger.error(f"Error writing output members file: {e}", exc_info=True)

    logger.info("--- Generate All OG Members for RMSD (from Ultra-Filtered MSAs) Script Finished ---")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
