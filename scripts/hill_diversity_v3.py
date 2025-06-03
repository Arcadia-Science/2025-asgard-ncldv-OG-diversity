#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to calculate Hill's diversity, APSI, and MSA entropy.
# Version 3.4 (Adapted for "ultra-filtered large OGs" pipeline)

import math
import numpy as np
import os
import glob
from collections import Counter
from Bio import Phylo, AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree 
import pandas as pd
import logging
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Optional
import argparse 
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# --- Default Configuration (for esp OGs track) ---
SCRIPT_DIR_DEFAULT = Path(__file__).resolve().parent
BASE_DIR_DEFAULT = SCRIPT_DIR_DEFAULT.parent 
NEW_PIPELINE_DATA_DIR_DEFAULT = BASE_DIR_DEFAULT / "data" / "data_esp_ogs"
NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT = BASE_DIR_DEFAULT / "analysis_outputs" / "analysis_esp_ogs"


INPUT_MSA_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "trimmed_fastas_for_trees_esp"
MSA_FILE_SUFFIX_DEFAULT = "_final_trimmed.fasta" 

INPUT_TREE_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "fasttree_output_final_trees_esp"
TREE_FILE_SUFFIX_DEFAULT = "_tree.nwk" 

OUTPUT_RESULTS_CSV_FINAL_DEFAULT = NEW_PIPELINE_ANALYSIS_OUTPUTS_DIR_DEFAULT / "og_diversity_metrics_esp.csv"

APSI_GAP_CHAR = '-'
# --- End Default Configuration ---

def shannon_entropy_msa_per_column(alignment: MultipleSeqAlignment) -> List[float]:
    entropies = []
    if not alignment or len(alignment) == 0: return entropies
    for i in range(alignment.get_alignment_length()):
        column_str = alignment[:, i]
        counts = Counter(c for c in column_str if c != APSI_GAP_CHAR) 
        column_entropy = 0.0
        effective_n_this_col = sum(counts.values())
        if effective_n_this_col > 1: 
            for residue_count in counts.values():
                p = residue_count / effective_n_this_col
                if p > 0: column_entropy -= p * math.log2(p)
        entropies.append(column_entropy)
    return entropies

def average_pairwise_sequence_identity(alignment: MultipleSeqAlignment) -> Tuple[Optional[float], Optional[float]]:
    if not alignment or len(alignment) < 2: return None, None
    num_sequences, alignment_length = len(alignment), alignment.get_alignment_length()
    total_identity_ungapped_pairs, num_pairs_ungapped_pairs_method = 0.0, 0
    total_identity_full_length, num_pairs_full_length_method = 0.0, 0
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            seq1_str, seq2_str = str(alignment[i].seq), str(alignment[j].seq)
            matches_ungapped, positions_compared_ungapped = 0, 0
            for k in range(alignment_length):
                res1, res2 = seq1_str[k], seq2_str[k]
                if res1 != APSI_GAP_CHAR and res2 != APSI_GAP_CHAR: 
                    positions_compared_ungapped += 1
                    if res1 == res2: matches_ungapped += 1
            if positions_compared_ungapped > 0:
                total_identity_ungapped_pairs += (matches_ungapped / positions_compared_ungapped)
                num_pairs_ungapped_pairs_method += 1
            matches_full = 0
            for k in range(alignment_length):
                if seq1_str[k] == seq2_str[k] and seq1_str[k] != APSI_GAP_CHAR : matches_full +=1
            if alignment_length > 0: 
                total_identity_full_length += (matches_full / alignment_length)
            num_pairs_full_length_method +=1 
    apsi_ungapped_pairs = (total_identity_ungapped_pairs / num_pairs_ungapped_pairs_method) * 100 if num_pairs_ungapped_pairs_method > 0 else None
    apsi_full_length = (total_identity_full_length / num_pairs_full_length_method) * 100 if num_pairs_full_length_method > 0 else None
    return apsi_ungapped_pairs, apsi_full_length

def phylogenetic_diversity_branch_lengths(tree: Tree) -> float:
    total_branch_length = 0.0
    for clade in tree.find_clades(): # type: ignore
        if clade.branch_length and clade.branch_length > 0: total_branch_length += clade.branch_length
    return total_branch_length

def hill_diversity_tree(tree: Tree, q: int = 1) -> Tuple[Optional[float], Optional[float]]:
    if not tree or not tree.get_terminals(): return None, None
    terminals = tree.get_terminals()
    branch_lengths = np.array([term.branch_length for term in terminals if term.branch_length is not None and term.branch_length > 0.000001]) 
    if len(branch_lengths) == 0: return None, None 
    total_terminal_branch_length = np.sum(branch_lengths)
    if total_terminal_branch_length <= 0.000001: return None, None
    proportions = branch_lengths / total_terminal_branch_length 
    proportions = proportions[proportions > 1e-9] 
    if len(proportions) == 0: return None, None
    raw_shannon_entropy = -np.sum(proportions * np.log(proportions)) 
    if q == 1: 
        if pd.isna(raw_shannon_entropy): return None, None
        return np.exp(raw_shannon_entropy), raw_shannon_entropy
    else: 
        logger.warning(f"Hill diversity for q={q} not fully implemented, returning q=1 result.")
        if pd.isna(raw_shannon_entropy): return None, None
        return np.exp(raw_shannon_entropy), raw_shannon_entropy

def initialize_metrics_dict(og_id: str) -> Dict:
    metrics = {"Orthogroup": og_id, "Error": None}
    msa_keys = ["MSA_N_Seqs", "MSA_Length", "APSI_UngappedPairs", "APSI_FullLength", "MSA_Mean_Col_Entropy", "MSA_Median_Col_Entropy", "MSA_Num_Conserved_Cols_Entropy_leq_0.5"]
    tree_keys = ["Tree_PD_TotalBranchLength", "Tree_N_Tips", "Tree_Hill_Diversity_q1", "Tree_Raw_Shannon_Entropy", "Tree_Hill_Diversity_q1_NormByTips", "Tree_Hill_Diversity_q1_NormByPD"]
    for k in msa_keys + tree_keys: metrics[k] = np.nan
    return metrics

def process_og_files(og_id: str, msa_file: Path, tree_file: Optional[Path]) -> Dict:
    results = initialize_metrics_dict(og_id)
    logger.debug(f"Worker processing OG: {og_id}") 
    current_error_messages = []
    try: 
        alignment = AlignIO.read(str(msa_file), "fasta")
        if not alignment or len(alignment) == 0: raise ValueError("MSA is empty or unreadable.")
        results["MSA_N_Seqs"], results["MSA_Length"] = len(alignment), alignment.get_alignment_length()
        apsi_ungapped, apsi_full = average_pairwise_sequence_identity(alignment)
        results["APSI_UngappedPairs"], results["APSI_FullLength"] = apsi_ungapped, apsi_full
        col_entropies = shannon_entropy_msa_per_column(alignment)
        if col_entropies:
            results["MSA_Mean_Col_Entropy"], results["MSA_Median_Col_Entropy"] = np.mean(col_entropies), np.median(col_entropies)
            results["MSA_Num_Conserved_Cols_Entropy_leq_0.5"] = sum(1 for e in col_entropies if e <= 0.5)
    except Exception as e:
        logger.error(f"Error processing MSA {msa_file} for OG {og_id}: {e}", exc_info=False)
        current_error_messages.append(f"MSA_Error:{type(e).__name__}")

    if tree_file and tree_file.exists(): 
        try:
            tree = Phylo.read(str(tree_file), "newick")
            if tree is None: raise ValueError("Phylo.read returned None (empty/invalid tree file).")
            results["Tree_PD_TotalBranchLength"], results["Tree_N_Tips"] = phylogenetic_diversity_branch_lengths(tree), tree.count_terminals()
            hill_q1, raw_tree_entropy = hill_diversity_tree(tree, q=1)
            results["Tree_Hill_Diversity_q1"], results["Tree_Raw_Shannon_Entropy"] = hill_q1, raw_tree_entropy
            if pd.notna(hill_q1) and results["Tree_N_Tips"] is not None and results["Tree_N_Tips"] > 0: results["Tree_Hill_Diversity_q1_NormByTips"] = hill_q1 / results["Tree_N_Tips"]
            if pd.notna(hill_q1) and pd.notna(results["Tree_PD_TotalBranchLength"]) and results["Tree_PD_TotalBranchLength"] > 0:
                 results["Tree_Hill_Diversity_q1_NormByPD"] = hill_q1 / results["Tree_PD_TotalBranchLength"]
        except ValueError as ve:
            logger.error(f"Error (ValueError) processing tree {tree_file} for OG {og_id}: {ve}.")
            current_error_messages.append(f"Tree_Error:InvalidOrEmptyTreeFile")
        except Exception as e:
            logger.error(f"Error processing tree {tree_file} for OG {og_id}: {e}", exc_info=False)
            current_error_messages.append(f"Tree_Error:{type(e).__name__}")
    else: logger.warning(f"Tree file not found or not specified for OG {og_id} at {tree_file}. Skipping tree metrics.")
    
    if current_error_messages: results["Error"] = "; ".join(current_error_messages)
    return results

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate diversity metrics for OGs in parallel (ultra-filtered large OGs track).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--msa_dir", type=Path, default=INPUT_MSA_DIR_DEFAULT, help="Input directory for MSA files from ultra-filtered track.")
    parser.add_argument("--msa_suffix", type=str, default=MSA_FILE_SUFFIX_DEFAULT, help="Suffix for MSA files.")
    parser.add_argument("--tree_dir", type=Path, default=INPUT_TREE_DIR_DEFAULT, help="Input directory for tree files from ultra-filtered track.")
    parser.add_argument("--tree_suffix", type=str, default=TREE_FILE_SUFFIX_DEFAULT, help="Suffix for tree files.")
    parser.add_argument("--output_csv", type=Path, default=OUTPUT_RESULTS_CSV_FINAL_DEFAULT, help="Output CSV file for results from ultra-filtered track.")
    default_num_cores = max(1, os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 2 else 1)
    parser.add_argument("--num_cores", type=int, default=default_num_cores,
                        help=f"Number of CPU cores for parallel processing. Default: {default_num_cores}.")
    parser.add_argument("--force_rerun", action='store_true', help="Force re-processing of all OGs, ignoring existing output file.")
    return parser.parse_args()

def main(args: argparse.Namespace): 
    logger.info(f"--- Starting Diversity Calculation Script (v3.4 - Ultra-Filtered Large OGs) ---")
    logger.info(f"Script arguments: {args}")
    
    args.output_csv.parent.mkdir(parents=True, exist_ok=True) # Ensure analysis_outputs/analysis_large_ogs_gt100mem_gt15id/ exists

    msa_files = sorted(list(args.msa_dir.glob(f"*{args.msa_suffix}")))
    if not msa_files:
        logger.error(f"No MSA files found in '{args.msa_dir}' with suffix '{args.msa_suffix}'. Exiting."); return
    logger.info(f"Found {len(msa_files)} MSA files to process for ultra-filtered large OGs.")

    existing_results_df = pd.DataFrame() 
    processed_ogs_from_existing_output = set()

    if not args.force_rerun and args.output_csv.exists():
        try:
            existing_results_df = pd.read_csv(args.output_csv)
            if 'Orthogroup' in existing_results_df.columns:
                processed_ogs_from_existing_output = set(existing_results_df['Orthogroup'].unique())
                logger.info(f"Found {len(processed_ogs_from_existing_output)} OGs in existing output file: {args.output_csv}. These will be skipped.")
            else: existing_results_df = pd.DataFrame() 
        except Exception as e: logger.warning(f"Could not read existing output file {args.output_csv}, will process all. Error: {e}"); existing_results_df = pd.DataFrame() 

    tasks_to_submit = []
    for msa_filepath_obj in msa_files:
        msa_filename = msa_filepath_obj.name
        og_id = msa_filename.replace(args.msa_suffix, "")
        if og_id in processed_ogs_from_existing_output and not args.force_rerun: continue
        tree_filepath = args.tree_dir / f"{og_id}{args.tree_suffix}"
        tree_filepath_arg = tree_filepath if tree_filepath.exists() else None
        if not tree_filepath_arg: logger.warning(f"Tree file not found for OG {og_id} at {tree_filepath}. Tree metrics will be NaN.")
        tasks_to_submit.append((og_id, msa_filepath_obj, tree_filepath_arg))

    if not tasks_to_submit:
        logger.info("No new OGs to process based on existing output file and --force_rerun flag.")
        if not existing_results_df.empty: 
             logger.info(f"Re-saving existing {len(existing_results_df)} results to {args.output_csv}")
             existing_results_df.to_csv(args.output_csv, index=False, float_format='%.4f')
        return

    logger.info(f"Submitting {len(tasks_to_submit)} new OGs for processing using {args.num_cores} cores...")
    newly_processed_results_list = [] 
    with ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        futures = {executor.submit(process_og_files, og_id, msa_fp, tree_fp): og_id for og_id, msa_fp, tree_fp in tasks_to_submit}
        for future in tqdm(as_completed(futures), total=len(tasks_to_submit), desc="Calculating Diversity (Ultra-Filtered)"):
            og_id_completed = futures[future]
            try: newly_processed_results_list.append(future.result())
            except Exception as exc:
                logger.error(f"OG {og_id_completed} generated an unhandled exception in executor: {exc}", exc_info=True)
                error_result = initialize_metrics_dict(og_id_completed)
                error_result["Error"] = f"Executor_Unhandled_Exception:{type(exc).__name__}"
                newly_processed_results_list.append(error_result)

    if not args.force_rerun and not existing_results_df.empty:
        df_newly_processed = pd.DataFrame(newly_processed_results_list)
        ogs_in_new_results = set(df_newly_processed['Orthogroup']) if not df_newly_processed.empty else set()
        df_existing_to_keep = existing_results_df[~existing_results_df['Orthogroup'].isin(ogs_in_new_results)]
        df_final_results = pd.concat([df_existing_to_keep, df_newly_processed], ignore_index=True)
    else: df_final_results = pd.DataFrame(newly_processed_results_list)

    if df_final_results.empty: logger.warning("No diversity results generated/collected. Check logs."); return
    
    ordered_cols = ["Orthogroup", "MSA_N_Seqs", "MSA_Length", "APSI_UngappedPairs", "APSI_FullLength", "MSA_Mean_Col_Entropy", "MSA_Median_Col_Entropy", "MSA_Num_Conserved_Cols_Entropy_leq_0.5","Tree_PD_TotalBranchLength", "Tree_N_Tips", "Tree_Hill_Diversity_q1", "Tree_Raw_Shannon_Entropy","Tree_Hill_Diversity_q1_NormByTips", "Tree_Hill_Diversity_q1_NormByPD","Error"]
    final_cols_present = [col for col in ordered_cols if col in df_final_results.columns]
    if "Error" in df_final_results.columns and "Error" not in final_cols_present: final_cols_present.append("Error")
    df_final_results = df_final_results[final_cols_present]
    
    logger.info(f"\nSaving final diversity results for {len(df_final_results)} OGs (ultra-filtered) to '{args.output_csv}'")
    try:
        df_final_results.sort_values(by="Orthogroup", inplace=True)
        df_final_results.to_csv(args.output_csv, index=False, float_format='%.4f')
        logger.info("Successfully saved final diversity results for ultra-filtered large OGs.")
    except Exception as e: logger.error(f"ERROR: Failed to save final results to {args.output_csv}: {e}", exc_info=True)

    logger.info(f"--- Diversity Calculation Script (Ultra-Filtered Large OGs) Finished ---")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
