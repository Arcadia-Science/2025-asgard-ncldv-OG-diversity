#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculates structural similarity metrics (TM-score, RMSD) using TMalign
for member proteins against their designated anchors within Orthogroups (OGs).
Runs TMalign in parallel.
(v1.4 - Fixed UnboundLocalError, improved logging for missing anchor structures)
"""

import pandas as pd
from pathlib import Path
import logging
import sys
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import shutil # For finding TMalign executable
import os
import numpy as np 
from typing import Optional, Dict, Tuple 

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
else:
    for handler in logger.handlers:
        if handler.level > logging.INFO: handler.setLevel(logging.INFO)
    if logger.level > logging.INFO: logger.setLevel(logging.INFO)

# --- Default Configuration ---
SCRIPT_DIR_DEFAULT = Path(__file__).resolve().parent
BASE_DIR_DEFAULT = SCRIPT_DIR_DEFAULT.parent 

ESP_ANALYSIS_SUBDIR_NAME_DEFAULT = "analysis_esp_ogs" 
ANALYSIS_OUTPUTS_DIR_DEFAULT = BASE_DIR_DEFAULT / "analysis_outputs" / ESP_ANALYSIS_SUBDIR_NAME_DEFAULT

CURATED_OG_ANCHOR_FILE_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "curated_esp_og_anchor_list_v1.csv"
PROTEOME_DB_FILE_DEFAULT = BASE_DIR_DEFAULT / "data" / "proteome_database_v3.2.csv"
STRUCTURE_LOG_FILE_DEFAULT = BASE_DIR_DEFAULT / "data" / "structure_download_log.csv" 
STRUCTURE_FILES_ROOT_DIR_DEFAULT = BASE_DIR_DEFAULT / "data" / "downloaded_structures" 

OUTPUT_STRUCTURAL_METRICS_CSV_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "structural_metrics_tmalign_esp.csv"
TMALIGN_EXE_DEFAULT = "TMalign" 

DEFAULT_N_CORES = max(1, os.cpu_count() - 2 if os.cpu_count() and os.cpu_count() > 2 else 1)
DEFAULT_TIMEOUT_SECONDS = 600 

# --- Helper function to parse TMalign output ---
def parse_tmalign_output(output_str: str) -> dict: 
    metrics = {
        "TMscore_Chain1": np.nan, "TMscore_Chain2": np.nan, 
        "RMSD_TMalign": np.nan, "AlignedLength_TMalign": np.nan,
        "SeqID_TMalign": np.nan, "TMalign_Error_Parsing": None 
    }
    try:
        tm_score1_found, tm_score2_found = False, False
        for line in output_str.splitlines():
            if line.startswith("Aligned length="):
                parts = line.split(',')
                if len(parts) > 2: 
                    metrics["AlignedLength_TMalign"] = int(parts[0].split('=')[1].strip())
                    metrics["RMSD_TMalign"] = float(parts[1].split('=')[1].strip())
                if len(parts) > 3 and "SeqIDn=" in parts[3]: 
                    metrics["SeqID_TMalign"] = float(parts[3].split('=')[1].strip()) 
                elif len(parts) > 2 and "SeqID=" in parts[2]: 
                     metrics["SeqID_TMalign"] = float(parts[2].split('=')[1].strip())
            elif "TM-score=" in line and "Chain 1" in line:
                metrics["TMscore_Chain1"] = float(line.split("TM-score=")[1].split()[0].strip())
                tm_score1_found = True
            elif "TM-score=" in line and "Chain 2" in line:
                metrics["TMscore_Chain2"] = float(line.split("TM-score=")[1].split()[0].strip())
                tm_score2_found = True
        
        if not (tm_score1_found and tm_score2_found):
            if "Nothing Aligned" in output_str or "Aligned length=0" in output_str :
                 metrics["TMalign_Error_Parsing"] = "Nothing_Aligned"
                 metrics["TMscore_Chain1"] = 0.0 
                 metrics["TMscore_Chain2"] = 0.0
                 metrics["RMSD_TMalign"] = np.nan 
                 metrics["AlignedLength_TMalign"] = 0
                 metrics["SeqID_TMalign"] = 0.0
            else:
                 metrics["TMalign_Error_Parsing"] = "TMscore_Parsing_Failed_Or_Bad_Output"
                 logger.debug(f"TM-score parsing failed. Output snippet:\n{output_str[:500]}")
    except Exception as e:
        logger.error(f"Error parsing TMalign output: {e}\nOutput was:\n{output_str[:500]}")
        metrics["TMalign_Error_Parsing"] = f"ParsingException_{type(e).__name__}"
    return metrics

# --- Function to run TMalign on a pair ---
def run_tmalign_for_pair(args_tuple: tuple) -> dict: 
    og_id, member_pid, anchor_pid, member_struct_path_str, anchor_struct_path_str, \
    tmalign_exe_path, timeout_sec = args_tuple

    result = {
        "OG_ID": og_id, "Member_ProteinID": member_pid, "Anchor_ProteinID": anchor_pid,
        "TMscore_Chain1": np.nan, "TMscore_Chain2": np.nan, 
        "RMSD_TMalign": np.nan, "AlignedLength_TMalign": np.nan,
        "SeqID_TMalign": np.nan, "TMalign_Run_Error": None 
    }

    member_struct_path = Path(member_struct_path_str) if member_struct_path_str else None
    anchor_struct_path = Path(anchor_struct_path_str) if anchor_struct_path_str else None

    if not member_struct_path or not member_struct_path.exists():
        result["TMalign_Run_Error"] = "Member_Structure_NotFound"
        logger.debug(f"Member structure not found for {member_pid} at '{member_struct_path}'")
        return result
    if not anchor_struct_path or not anchor_struct_path.exists():
        result["TMalign_Run_Error"] = "Anchor_Structure_NotFound"
        logger.debug(f"Anchor structure not found for {anchor_pid} at '{anchor_struct_path}'")
        return result

    command = [tmalign_exe_path, str(member_struct_path), str(anchor_struct_path)]
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, 
                                 check=False, timeout=timeout_sec, encoding='utf-8', errors='ignore')
        
        if process.returncode != 0:
            result["TMalign_Run_Error"] = f"TMalign_Failed_Code{process.returncode}"
            logger.warning(f"TMalign failed for {member_pid} vs {anchor_pid} (OG: {og_id}). stderr: {process.stderr[:200]}")
            return result

        parsed_metrics = parse_tmalign_output(process.stdout)
        result.update(parsed_metrics) 
        
        if result.get("TMalign_Error_Parsing"): 
            result["TMalign_Run_Error"] = result["TMalign_Error_Parsing"]
            logger.warning(f"TMalign output parsing issue for {member_pid} vs {anchor_pid}. Error: {result['TMalign_Run_Error']}")
        if "TMalign_Error_Parsing" in result: del result["TMalign_Error_Parsing"]

    except subprocess.TimeoutExpired:
        result["TMalign_Run_Error"] = f"TMalign_Timeout_{timeout_sec}s"
        logger.warning(f"TMalign timed out for {member_pid} vs {anchor_pid} (OG: {og_id}).")
    except FileNotFoundError: 
        result["TMalign_Run_Error"] = "TMalign_Executable_NotFound_In_Worker"
        logger.error(f"TMalign executable not found at {tmalign_exe_path} when processing {member_pid}.")
    except Exception as e:
        result["TMalign_Run_Error"] = f"UnexpectedRunError_{type(e).__name__}"
        logger.error(f"Unexpected error running TMalign for {member_pid} vs {anchor_pid}: {e}", exc_info=False)
    
    return result

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TMalign in parallel for member-anchor pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--anchor_list_csv", type=Path, required=True, 
                        help="CSV file listing OGs and their Anchor_ProteinID.")
    parser.add_argument("--proteome_db_csv", type=Path, required=True,
                        help="Path to the main proteome database CSV (containing ProteinID, Orthogroup).")
    parser.add_argument("--structure_log_csv", type=Path, required=True,
                        help="CSV file mapping ProteinID to its structure filename and source. Expected columns: ProteinID, Saved_Filename, Retrieved_From")
    parser.add_argument("--structure_files_root_dir", type=Path, required=True,
                        help="Root directory where structure subdirectories (e.g., 'alphafold', 'rcsb_pdb') are located.")
    parser.add_argument("--output_csv", type=Path, required=True,
                        help="Output CSV file for structural comparison metrics.")
    parser.add_argument("--tmalign_exe", default=TMALIGN_EXE_DEFAULT,
                        help="Name or path to the TMalign executable.")
    parser.add_argument("--num_cores", type=int, default=DEFAULT_N_CORES,
                        help="Number of CPU cores for parallel TMalign runs.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Timeout in seconds for each TMalign job (0 or negative to disable).")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for this script's messages.")
    parser.add_argument("--force_rerun_og_ids", nargs='*', default=None,
                        help="List of OG_IDs to force re-run, even if results exist in output_csv. If 'ALL', re-runs all. Case-sensitive.")

    return parser.parse_args()

def get_structure_path(protein_id: str, saved_filename: Optional[str], retrieved_from: Optional[str], root_dir: Path) -> Optional[Path]:
    if pd.isna(saved_filename) or not saved_filename:
        logger.debug(f"No Saved_Filename for {protein_id}. Cannot determine structure path.")
        return None
    
    subdir_map = {
        "alphafolddb": "alphafold", 
        "alphafold": "alphafold",
        "rcsb_pdb": "rcsb_pdb",
        "pdb": "rcsb_pdb" 
    }
    
    source_subdir = None
    if pd.notna(retrieved_from):
        source_subdir = subdir_map.get(str(retrieved_from).lower())

    if not source_subdir:
        logger.warning(f"Unknown 'Retrieved_From' source ('{retrieved_from}') for {protein_id} or it's NaN. Cannot determine subdirectory. Will check root_dir/{saved_filename} directly.")
        potential_path = root_dir / saved_filename
        if potential_path.exists(): return potential_path
        logger.debug(f"Could not determine subdirectory for {protein_id}, and not found at {potential_path}")
        return None 

    return root_dir / source_subdir / saved_filename


def main(args: argparse.Namespace):
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(log_level_numeric)
    for handler in logger.handlers: handler.setLevel(log_level_numeric)

    logger.info(f"--- Starting Structural Similarity Calculation (TMalign v1.4) ---") # Updated version
    logger.info(f"Script arguments: {args}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    tmalign_exe_path = shutil.which(args.tmalign_exe)
    if not tmalign_exe_path:
        logger.error(f"TMalign executable '{args.tmalign_exe}' not found in PATH or as specified. Exiting.")
        sys.exit(1)
    logger.info(f"Using TMalign executable: {tmalign_exe_path}")

    structure_log_cols_needed = ['ProteinID', 'Saved_Filename', 'Retrieved_From']
    try:
        df_anchors = pd.read_csv(args.anchor_list_csv, dtype={'OG_ID': str, 'Anchor_ProteinID': str})
        df_proteome = pd.read_csv(args.proteome_db_csv, usecols=['ProteinID', 'Orthogroup'], dtype={'ProteinID': str, 'Orthogroup': str})
        df_struct_log = pd.read_csv(args.structure_log_csv, usecols=structure_log_cols_needed, dtype={'ProteinID':str, 'Saved_Filename':str, 'Retrieved_From':str})
        logger.info(f"Loaded {len(df_anchors)} anchors, {len(df_proteome)} proteins, {len(df_struct_log)} structure log entries.")
    except FileNotFoundError as e:
        logger.error(f"Error: Input CSV file not found: {e.filename}. Exiting.")
        sys.exit(1)
    except ValueError as e: 
        logger.error(f"Error: Column missing in one of the input CSVs (likely --structure_log_csv needs {structure_log_cols_needed}): {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input CSV files: {e}", exc_info=True); sys.exit(1)

    struct_info_map = {}
    for _, row in df_struct_log.iterrows():
        struct_info_map[row['ProteinID']] = (row['Saved_Filename'], row['Retrieved_From'])

    tasks = []
    # FIX: Initialize df_existing_results before the conditional block
    df_existing_results = pd.DataFrame() 
    processed_pairs_in_existing_output = set()
    ogs_with_missing_anchor_structures = set() # To log OGs for re-curation

    force_rerun_all = args.force_rerun_og_ids is not None and "ALL" in [str(x).upper() for x in args.force_rerun_og_ids]
    force_rerun_specific_ogs = set(map(str, args.force_rerun_og_ids)) if args.force_rerun_og_ids and not force_rerun_all else set()

    if not force_rerun_all and args.output_csv.exists():
        try:
            df_existing_results = pd.read_csv(args.output_csv, dtype={'OG_ID':str, 'Member_ProteinID':str, 'Anchor_ProteinID':str})
            if all(col in df_existing_results.columns for col in ['OG_ID','Member_ProteinID', 'Anchor_ProteinID']):
                for _, row in df_existing_results.iterrows():
                    if row['OG_ID'] not in force_rerun_specific_ogs:
                        processed_pairs_in_existing_output.add((row['Member_ProteinID'], row['Anchor_ProteinID']))
                logger.info(f"Found {len(processed_pairs_in_existing_output)} processed pairs (not in force_rerun_og_ids) in existing output: {args.output_csv}")
            else:
                logger.warning(f"Existing output file {args.output_csv} lacks key columns. Will re-run all applicable.")
                df_existing_results = pd.DataFrame() # Reset if problematic
        except Exception as e:
            logger.warning(f"Could not read or parse existing output file {args.output_csv}. Will re-run all applicable. Error: {e}")
            df_existing_results = pd.DataFrame() # Reset on error
    
    logger.info("Preparing tasks for TMalign...")
    for _, anchor_row in tqdm(df_anchors.iterrows(), total=len(df_anchors), desc="Preparing TMalign tasks"):
        og_id = anchor_row['OG_ID']
        anchor_pid = anchor_row['Anchor_ProteinID']
        
        anchor_file_info = struct_info_map.get(anchor_pid)
        if not anchor_file_info:
            logger.warning(f"Anchor {anchor_pid} (OG: {og_id}) not found in structure log. Skipping this OG for TMalign.")
            ogs_with_missing_anchor_structures.add(og_id)
            continue
        anchor_struct_path_str = str(get_structure_path(anchor_pid, anchor_file_info[0], anchor_file_info[1], args.structure_files_root_dir) or "")
        
        if not anchor_struct_path_str or not Path(anchor_struct_path_str).exists(): 
            logger.warning(f"Constructed anchor structure path for {anchor_pid} (OG: {og_id}) not found: '{anchor_struct_path_str}'. Skipping this OG for TMalign.")
            ogs_with_missing_anchor_structures.add(og_id)
            continue
            
        og_members = df_proteome[df_proteome['Orthogroup'] == og_id]['ProteinID'].unique()
        
        for member_pid in og_members:
            if member_pid == anchor_pid: continue 

            if (member_pid, anchor_pid) in processed_pairs_in_existing_output:
                logger.debug(f"Skipping already processed pair: {member_pid} vs {anchor_pid} for OG {og_id}")
                continue

            member_file_info = struct_info_map.get(member_pid)
            member_struct_path_str = "" 
            if member_file_info:
                member_struct_path_str = str(get_structure_path(member_pid, member_file_info[0], member_file_info[1], args.structure_files_root_dir) or "")
            
            tasks.append((og_id, member_pid, anchor_pid, member_struct_path_str, anchor_struct_path_str, 
                          tmalign_exe_path, args.timeout if args.timeout > 0 else None))

    if not tasks:
        logger.info("No new TMalign tasks to run.")
        # Ensure df_existing_results is defined before this block
        if args.output_csv.exists() and not df_existing_results.empty and not force_rerun_all and not force_rerun_specific_ogs:
            logger.info(f"Existing output at {args.output_csv} is considered up-to-date for non-forced OGs.")
        elif not df_existing_results.empty : # If existing results were loaded, re-save them
             try:
                df_existing_results.to_csv(args.output_csv, index=False, float_format='%.4f')
                logger.info(f"Re-saved existing structural metrics to {args.output_csv} as no new tasks were added.")
             except Exception as e:
                logger.error(f"Error re-saving existing results: {e}")
        # Log OGs skipped due to anchor issues even if no new tasks
        if ogs_with_missing_anchor_structures:
            logger.warning(f"Summary: {len(ogs_with_missing_anchor_structures)} OGs were skipped entirely due to missing anchor structure information: {sorted(list(ogs_with_missing_anchor_structures))}")
        return

    logger.info(f"Submitting {len(tasks)} TMalign tasks using up to {args.num_cores} cores...")
    
    newly_processed_results = []
    with ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        futures = {executor.submit(run_tmalign_for_pair, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            task_info = futures[future] 
            processed_count = i + 1
            try:
                result_dict = future.result()
                newly_processed_results.append(result_dict)
            except Exception as exc:
                logger.error(f"Task for OG {task_info[0]}, Member {task_info[1]} generated an unhandled exception in executor: {exc}", exc_info=True)
                newly_processed_results.append({"OG_ID": task_info[0], "Member_ProteinID": task_info[1], "Anchor_ProteinID": task_info[2], "TMalign_Run_Error": f"ExecutorError_{type(exc).__name__}"})
            
            if processed_count % 50 == 0 or processed_count == len(tasks): 
                logger.info(f"Completed {processed_count}/{len(tasks)} TMalign tasks...")

    df_new_results = pd.DataFrame(newly_processed_results) if newly_processed_results else pd.DataFrame()
    
    # Combine with existing results (those not forced to re-run)
    if not df_existing_results.empty: # Check if it was successfully loaded/initialized
        if force_rerun_all:
            df_final_results = df_new_results
        else:
            df_to_keep_from_existing = df_existing_results[~df_existing_results['OG_ID'].isin(force_rerun_specific_ogs)]
            df_final_results = pd.concat([df_to_keep_from_existing, df_new_results], ignore_index=True).drop_duplicates(subset=['Member_ProteinID', 'Anchor_ProteinID'], keep='last')
    else: # If df_existing_results is empty (e.g., file didn't exist or was problematic)
        df_final_results = df_new_results

    if df_final_results.empty:
        logger.warning("No structural metrics results generated or collected. Check logs.")
    else:
        cols_order = ["OG_ID", "Member_ProteinID", "Anchor_ProteinID", "TMscore_Chain1", "TMscore_Chain2", 
                      "RMSD_TMalign", "AlignedLength_TMalign", "SeqID_TMalign", "TMalign_Run_Error"]
        for col in cols_order: # Ensure all expected columns are present for consistency
            if col not in df_final_results.columns:
                df_final_results[col] = np.nan 
        df_final_results = df_final_results[cols_order]

        logger.info(f"\nSaving final structural metrics for {len(df_final_results)} pairs to '{args.output_csv}'")
        try:
            df_final_results.sort_values(by=["OG_ID", "Member_ProteinID"], inplace=True)
            df_final_results.to_csv(args.output_csv, index=False, float_format='%.4f')
            logger.info(f"Successfully saved structural metrics to {args.output_csv}")
        except Exception as e:
            logger.error(f"ERROR: Failed to save final results to {args.output_csv}: {e}", exc_info=True)

    if ogs_with_missing_anchor_structures:
        logger.warning(f"Summary: {len(ogs_with_missing_anchor_structures)} OGs were skipped entirely due to missing anchor structure information or path issues: {sorted(list(ogs_with_missing_anchor_structures))}")
        skipped_anchors_file = args.output_csv.parent / f"{args.output_csv.stem}_skipped_anchor_OGs.txt"
        try:
            with open(skipped_anchors_file, 'w') as f:
                for og_id in sorted(list(ogs_with_missing_anchor_structures)):
                    f.write(f"{og_id}\n")
            logger.info(f"List of OGs with missing anchor structures saved to: {skipped_anchors_file}")
        except Exception as e:
            logger.error(f"Could not save list of OGs with missing anchors: {e}")


    logger.info(f"--- Structural Similarity Calculation Finished ---")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
