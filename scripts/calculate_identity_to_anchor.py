#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculates sequence identity of each member protein in an Orthogroup (OG)
to its designated anchor protein, using final **ultra-filtered** trimmed MSAs.
(v2.1 - Adapted for ultra-filtered large OGs track)
"""

import pandas as pd
from pathlib import Path
import logging
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from tqdm.auto import tqdm
import numpy as np 
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
NEW_PIPELINE_DATA_DIR_DEFAULT = BASE_DIR_DEFAULT / "data" / "data_esp_ogs"
ANALYSIS_OUTPUTS_DIR_DEFAULT = BASE_DIR_DEFAULT / "analysis_outputs" # Main analysis outputs
NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "analysis_esp_ogs"

CURATED_OG_ANCHOR_FILE_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "analysis_esp_ogs/curated_esp_og_anchor_list_v1.csv"
MSA_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "trimmed_fastas_for_trees_esp" 
MSA_SUFFIX_DEFAULT = "_final_trimmed.fasta"

OUTPUT_IDENTITY_CSV_DEFAULT = NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT / "og_member_identity_to_anchor_esp.csv"

OG_ID_COL = "OG_ID"
ANCHOR_PID_COL = "Anchor_ProteinID"
MEMBER_PID_COL = "Member_ProteinID"
GAP_CHAR = '-'

def calculate_pairwise_identities(member_seq_str: str, anchor_seq_str: str) -> tuple[float, float]:
    if len(member_seq_str) != len(anchor_seq_str): raise ValueError("Sequences must be same length.")
    alignment_length = len(member_seq_str)
    if alignment_length == 0: return 0.0, 0.0
    identical_ungapped, compared_ungapped = 0, 0
    for i in range(alignment_length):
        res_m, res_a = member_seq_str[i], anchor_seq_str[i]
        if res_m != GAP_CHAR and res_a != GAP_CHAR:
            compared_ungapped += 1
            if res_m == res_a: identical_ungapped += 1
    identity_ungapped = (identical_ungapped / compared_ungapped) * 100.0 if compared_ungapped > 0 else 0.0
    identical_non_gap_full = sum(1 for i in range(alignment_length) if member_seq_str[i] == anchor_seq_str[i] and member_seq_str[i] != GAP_CHAR)
    identity_full_length = (identical_non_gap_full / alignment_length) * 100.0 if alignment_length > 0 else 0.0
    return round(identity_ungapped, 4), round(identity_full_length, 4)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate member-to-anchor identity for ultra-filtered large OGs.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--anchor_list_csv", type=Path, default=CURATED_OG_ANCHOR_FILE_DEFAULT)
    parser.add_argument("--msa_dir", type=Path, default=MSA_DIR_DEFAULT)
    parser.add_argument("--msa_suffix", type=str, default=MSA_SUFFIX_DEFAULT)
    parser.add_argument("--output_csv", type=Path, default=OUTPUT_IDENTITY_CSV_DEFAULT)
    return parser.parse_args()

def main(args: argparse.Namespace):
    logger.info("Starting member-to-anchor sequence identity calculation (Ultra-Filtered Large OGs)...")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.anchor_list_csv.exists(): logger.error(f"Anchor list '{args.anchor_list_csv}' not found. Exiting."); return
    try:
        df_curated_ogs = pd.read_csv(args.anchor_list_csv)
        if not all(col in df_curated_ogs.columns for col in [OG_ID_COL, ANCHOR_PID_COL]):
            logger.error(f"'{args.anchor_list_csv}' missing '{OG_ID_COL}' or '{ANCHOR_PID_COL}'. Exiting."); return
        logger.info(f"Read {len(df_curated_ogs)} OGs from '{args.anchor_list_csv}'.")
    except Exception as e: logger.error(f"Error reading '{args.anchor_list_csv}': {e}", exc_info=True); return

    if not args.msa_dir.is_dir(): logger.error(f"MSA directory '{args.msa_dir}' not found. Exiting."); return

    all_identity_results = []
    for _, og_row in tqdm(df_curated_ogs.iterrows(), total=len(df_curated_ogs), desc="Processing OGs for Identity (Ultra-Filtered)"):
        og_id, anchor_pid = og_row[OG_ID_COL], str(og_row[ANCHOR_PID_COL])
        msa_file_path = args.msa_dir / f"{og_id}{args.msa_suffix}"
        if not msa_file_path.exists():
            logger.warning(f"MSA file not found for OG {og_id} at '{msa_file_path}'. Skipping.")
            all_identity_results.append({OG_ID_COL: og_id, MEMBER_PID_COL: None, ANCHOR_PID_COL: anchor_pid, "Identity_To_Anchor_Ungapped": np.nan, "Identity_To_Anchor_FullLength": np.nan, "Error": "MSA_File_Not_Found"}); continue
        try:
            alignment = AlignIO.read(str(msa_file_path), "fasta")
            if not alignment or len(alignment) == 0:
                logger.warning(f"MSA for OG {og_id} empty/unreadable. Skipping."); 
                all_identity_results.append({OG_ID_COL: og_id, MEMBER_PID_COL: None, ANCHOR_PID_COL: anchor_pid, "Identity_To_Anchor_Ungapped": np.nan, "Identity_To_Anchor_FullLength": np.nan, "Error": "MSA_Empty_Or_Unreadable"}); continue
            anchor_record = next((r for r in alignment if str(r.id) == anchor_pid), None)
            if anchor_record is None:
                logger.warning(f"Anchor '{anchor_pid}' not found in MSA for OG {og_id}. Skipping."); 
                all_identity_results.append({OG_ID_COL: og_id, MEMBER_PID_COL: None, ANCHOR_PID_COL: anchor_pid, "Identity_To_Anchor_Ungapped": np.nan, "Identity_To_Anchor_FullLength": np.nan, "Error": "Anchor_Not_In_MSA"}); continue
            anchor_seq_str = str(anchor_record.seq)
            for member_record in alignment:
                member_pid, member_seq_str = str(member_record.id), str(member_record.seq)
                id_ungapped, id_full = calculate_pairwise_identities(member_seq_str, anchor_seq_str)
                all_identity_results.append({OG_ID_COL: og_id, MEMBER_PID_COL: member_pid, ANCHOR_PID_COL: anchor_pid, "Identity_To_Anchor_Ungapped": id_ungapped, "Identity_To_Anchor_FullLength": id_full, "Error": None})
        except Exception as e:
            logger.error(f"Error processing OG {og_id} from MSA '{msa_file_path}': {e}", exc_info=True)
            all_identity_results.append({OG_ID_COL: og_id, MEMBER_PID_COL: None, ANCHOR_PID_COL: anchor_pid, "Identity_To_Anchor_Ungapped": np.nan, "Identity_To_Anchor_FullLength": np.nan, "Error": f"Processing_Error_{type(e).__name__}"})

    if not all_identity_results: logger.warning("No identity results generated."); return
    df_output = pd.DataFrame(all_identity_results)
    df_output.sort_values(by=[OG_ID_COL, MEMBER_PID_COL], inplace=True)
    try:
        df_output.to_csv(args.output_csv, index=False, float_format='%.4f')
        logger.info(f"Saved {len(df_output)} member-to-anchor identity records to '{args.output_csv}'.")
    except Exception as e: logger.error(f"Error writing output identity CSV: {e}", exc_info=True)
    logger.info("--- Member-to-Anchor Sequence Identity Calculation (Ultra-Filtered) Finished ---")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
