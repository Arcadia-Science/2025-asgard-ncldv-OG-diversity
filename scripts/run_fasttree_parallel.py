#!/usr/bin/env python3

"""
Runs FastTree (preferably FastTreeMP) in parallel on multiple input alignment
files (FASTA format) using Python's concurrent.futures module.

Includes optional timeout per job and logs errors/timeouts. Skips files if the
output tree already exists and is non-empty.
"""

import os
import sys
import glob
import subprocess
import logging
import time
import argparse
import shutil # For checking executable path
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import freeze_support # See note in main guard

# --- Setup Logging ---
# BasicConfig will be set up in main based on args.log_level
# This initial logger is for any top-level script issues before main config.
script_logger = logging.getLogger(__name__) # Use __name__ for module-level logger
if not script_logger.handlers:
    script_logger.setLevel(logging.INFO) # Default to INFO for now
    _console_handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
    _console_handler.setFormatter(_formatter)
    script_logger.addHandler(_console_handler)


# --- Default Configuration ---
DEFAULT_FASTTREE_EXE = "veryfasttree" # Prefer multi-threaded version
DEFAULT_N_CORES = max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)
DEFAULT_FASTTREE_ARGS = ["-lg", "-gamma"] # Common args for protein, WAG/LG + CAT
DEFAULT_LOG_DIR_NAME = 'fasttree_run_logs' # Changed from 'fasttree_logs' to avoid clash with potential FastTree internal logs
DEFAULT_TIMEOUT_SECONDS = 300 # 1 hour timeout, set to 0 or None to disable

# --- Function to Run FastTree ---
def run_fasttree_on_file(args_tuple: tuple) -> tuple[str, str, str]:
    """
    Runs FastTree on a single input trimmed alignment file with an optional timeout.
    Accepts a tuple of arguments for use with ProcessPoolExecutor.

    Args:
        args_tuple: Contains (trimmed_fasta_path, output_dir, log_dir,
                             fasttree_exe_path, fasttree_args_list, timeout_sec,
                             input_suffix, output_suffix)

    Returns:
        Tuple (input_filename_str, status_message, output_nwk_path_str).
    """
    # Unpack arguments
    trimmed_fasta_path, output_dir, log_dir, fasttree_exe_path, \
    fasttree_args_list, timeout_sec, input_suffix, output_suffix = args_tuple

    base_name = trimmed_fasta_path.name
    # Derive identifier by removing suffix
    identifier = base_name
    if input_suffix and base_name.endswith(input_suffix):
        identifier = base_name[:-len(input_suffix)]
    else:
        identifier = trimmed_fasta_path.stem # Fallback if suffix not provided or doesn't match

    output_nwk_path = output_dir / f"{identifier}{output_suffix}"
    # This log path is for the stderr/stdout of the FastTree process itself
    process_log_path = log_dir / f"{identifier}_fasttree_process.log" 

    # Skip if output exists and is non-empty
    if output_nwk_path.exists() and output_nwk_path.stat().st_size > 0:
        # Use the main logger configured in main()
        logging.debug(f"Output tree {output_nwk_path} exists and is non-empty. Skipping.")
        return base_name, "Skipped (Output Exists)", str(output_nwk_path)
    elif output_nwk_path.exists():
        logging.warning(f"Output tree {output_nwk_path} exists but is empty. Will attempt to overwrite.")

    # Construct command
    command = [fasttree_exe_path] + fasttree_args_list
    # FastTree typically reads from stdin and writes tree to stdout.
    # Some versions might take input file as an argument. This script assumes stdin/stdout.
    # If your FastTree version needs input file arg: command.append(str(trimmed_fasta_path))
    
    logging.debug(f"Running command for {identifier}: {' '.join(command)} < {trimmed_fasta_path} > {output_nwk_path} (Timeout: {timeout_sec}s)")

    try:
        with open(trimmed_fasta_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(output_nwk_path, 'w', encoding='utf-8') as f_out, \
             open(process_log_path, 'w', encoding='utf-8') as f_err:

            process = subprocess.run(command, stdin=f_in, stdout=f_out, stderr=f_err,
                                     check=False, # Set to False to handle non-zero exit codes manually
                                     text=True, # encoding handled by open context managers
                                     timeout=timeout_sec) 

        if process.returncode != 0:
            logging.error(f"FastTree failed for {base_name} with exit code {process.returncode}. Check log: {process_log_path}")
            if output_nwk_path.exists(): # Remove potentially incomplete/empty output on error
                 if output_nwk_path.stat().st_size == 0: output_nwk_path.unlink(missing_ok=True)
            return base_name, f"Error: FastTree failed (Code {process.returncode})", str(output_nwk_path)

        # Verify output file was created and is not empty
        if not output_nwk_path.exists() or not output_nwk_path.stat().st_size > 0:
             error_info = "FastTree Error (Output missing or empty after successful run)"
             log_content = ""
             try: 
                  with open(process_log_path, 'r', encoding='utf-8', errors='ignore') as f_log_read:
                       log_content = f_log_read.read(500).strip() 
                       if log_content: error_info = f"FastTree Info/Error (see process log): {log_content}..."
             except Exception: pass
             if output_nwk_path.exists(): output_nwk_path.unlink(missing_ok=True)
             logging.warning(f"FastTree ran for {base_name} (exit 0) but produced no/empty output. {error_info}. Log: {process_log_path}")
             return base_name, error_info, str(output_nwk_path)

        return base_name, "Success", str(output_nwk_path)

    except FileNotFoundError:
        logging.error(f"FastTree executable not found at {fasttree_exe_path} when processing {base_name}")
        return base_name, f"Error: FastTree not found at {fasttree_exe_path}", str(output_nwk_path)
    except subprocess.TimeoutExpired:
        logging.warning(f"FastTree timed out for {base_name} after {timeout_sec} seconds. Check log: {process_log_path}")
        if output_nwk_path.exists(): output_nwk_path.unlink(missing_ok=True)
        return base_name, f"Error: Timeout ({timeout_sec}s)", str(output_nwk_path)
    except Exception as e:
        logging.exception(f"An unexpected error occurred while running FastTree for {base_name}: {e}")
        if output_nwk_path.exists(): output_nwk_path.unlink(missing_ok=True)
        return base_name, f"Error: Unexpected ({type(e).__name__})", str(output_nwk_path)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FastTree in parallel on multiple alignment files with timeout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input_dir", required=True, type=Path,
                        help="Directory containing input alignment files (FASTA format).")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Directory to save the output FastTree Newick files (.nwk).")
    parser.add_argument("--log_dir", default=None, type=Path, # Changed from -l to --log_dir for clarity
                        help=f"Directory to save FastTree process log files (stderr). Default: '<output_dir>/{DEFAULT_LOG_DIR_NAME}'")
    parser.add_argument("--input_suffix", default="_final_trimmed.fasta", # More specific default
                        help="Suffix of input alignment files to process (e.g., _trimmed.fasta, .aln).")
    parser.add_argument("--output_suffix", default="_final_tree.nwk", # More specific default
                        help="Suffix to use for output tree files (e.g., _fasttree.nwk).")
    parser.add_argument("-c", "--cores", type=int, default=DEFAULT_N_CORES,
                        help="Number of parallel FastTree jobs to run.")
    parser.add_argument("-f", "--fasttree_exe", default=DEFAULT_FASTTREE_EXE,
                        help="Path to the FastTree executable (FastTree or FastTreeMP).")
    parser.add_argument("--fasttree_args", default=" ".join(DEFAULT_FASTTREE_ARGS),
                        help="Arguments to pass to FastTree (as a single string, e.g., '-lg -gamma').")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                        help="Timeout in seconds for each FastTree job (0 or negative to disable).")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for this script's messages.")
    parser.add_argument("--force_rerun", action='store_true',
                        help="Force re-processing of all OGs, ignoring existing output tree files.")


    return parser.parse_args()

# --- Main Execution Logic ---
def main():
    """ Main logic to run FastTree in parallel. """
    args = parse_arguments()
    start_time = time.time()

    # Configure main logger based on args
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    # Get root logger or specific logger if preferred
    main_logger = logging.getLogger() # Get root logger
    main_logger.setLevel(log_level_numeric)
    # Ensure handlers are set up if not already (e.g. if script is imported)
    if not main_logger.handlers:
        _ch = logging.StreamHandler(sys.stdout)
        _ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'))
        main_logger.addHandler(_ch)
    else: # If handlers exist, just set level for all
        for handler in main_logger.handlers:
            handler.setLevel(log_level_numeric)


    logging.info("--- Starting FastTree Parallel Run (Python Script) ---")

    fasttree_exe_path = shutil.which(args.fasttree_exe)
    if not fasttree_exe_path:
        if args.fasttree_exe == DEFAULT_FASTTREE_EXE: # Try fallback if default (FastTreeMP) was used
             logging.warning(f"Executable '{args.fasttree_exe}' not found, trying 'FastTree'...")
             fasttree_exe_path = shutil.which("FastTree")
        if not fasttree_exe_path: # If still not found
             logging.error(f"FastTree executable ('{args.fasttree_exe}' or 'FastTree') not found in PATH or as specified.")
             sys.exit(1)
    logging.info(f"Using FastTree executable: {fasttree_exe_path}")

    fasttree_args_list = args.fasttree_args.split()
    logging.info(f"Using FastTree arguments: {fasttree_args_list}")

    timeout_value = args.timeout if args.timeout and args.timeout > 0 else None
    if timeout_value: logging.info(f"Setting timeout per job: {timeout_value} seconds")
    else: logging.info("No timeout set per job.")

    # Setup directories
    process_log_dir = args.log_dir if args.log_dir else args.output_dir / DEFAULT_LOG_DIR_NAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input alignment directory: {args.input_dir}")
    logging.info(f"Output tree directory: {args.output_dir}")
    logging.info(f"FastTree process log directory: {process_log_dir}")

    fasta_files = sorted(list(args.input_dir.glob(f"*{args.input_suffix}")))
    if not fasta_files:
        logging.error(f"No files matching '*{args.input_suffix}' found in {args.input_dir}. Stopping.")
        sys.exit(1)
    logging.info(f"Found {len(fasta_files)} alignment files to process.")
    logging.info(f"Running up to {args.cores} FastTree jobs in parallel...")

    tasks = []
    for f_path in fasta_files:
        # If force_rerun, delete existing output tree to ensure it's regenerated
        if args.force_rerun:
            base_name_temp = f_path.name
            identifier_temp = base_name_temp[:-len(args.input_suffix)] if args.input_suffix and base_name_temp.endswith(args.input_suffix) else f_path.stem
            potential_output_nwk_path = args.output_dir / f"{identifier_temp}{args.output_suffix}"
            if potential_output_nwk_path.exists():
                logging.debug(f"--force_rerun: Deleting existing tree {potential_output_nwk_path}")
                potential_output_nwk_path.unlink(missing_ok=True)
        
        tasks.append((
            f_path, args.output_dir, process_log_dir, fasttree_exe_path,
            fasttree_args_list, timeout_value,
            args.input_suffix, args.output_suffix
        ))

    success_count = 0
    skipped_count = 0
    error_count = 0
    results_summary = []
    
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        futures = {executor.submit(run_fasttree_on_file, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            input_filepath = futures[future]
            input_fname_str = str(input_filepath.name) # For logging
            processed_count = i + 1
            try:
                fname_returned, status, out_nwk_path_str = future.result()
                results_summary.append({'file': fname_returned, 'status': status, 'output_tree': out_nwk_path_str})
                if status == "Success": success_count += 1
                elif status == "Skipped (Output Exists)": skipped_count += 1
                else:
                    error_count += 1
                    logging.warning(f"Job for {fname_returned} had status: {status}") 
            except Exception as exc:
                logging.error(f"Main loop error processing result for {input_fname_str}: {exc}", exc_info=True)
                results_summary.append({'file': input_fname_str, 'status': f"Error: Future Exception ({type(exc).__name__})", 'output_tree': 'N/A'})
                error_count += 1

            if processed_count % 20 == 0 or processed_count == len(fasta_files): # Log progress less frequently
                 logging.info(f"Processed {processed_count}/{len(fasta_files)} files... (Success: {success_count}, Skipped: {skipped_count}, Errors: {error_count})")

    end_time = time.time()
    logging.info("--- FastTree Summary ---")
    logging.info(f"Successfully created/verified trees: {success_count + skipped_count}")
    logging.info(f"  - Newly created: {success_count}")
    logging.info(f"  - Skipped (output existed): {skipped_count}")
    logging.info(f"Errors during tree building (incl. timeouts): {error_count}")
    logging.info(f"Total alignment files considered: {len(fasta_files)}")
    logging.info(f"Tree files ({args.output_suffix}) saved in: {args.output_dir}")
    logging.info(f"FastTree process logs saved in: {process_log_dir}")

    if error_count > 0:
        logging.warning("--- Files with Errors/Timeouts ---")
        error_files_list = [r['file'] for r in results_summary if r['status'] not in ["Success", "Skipped (Output Exists)"]]
        for err_idx, err_file_name in enumerate(error_files_list):
             if err_idx < 20 : # Log first 20 errors fully
                logging.warning(f"  - {err_file_name} (Status: {next((item['status'] for item in results_summary if item['file'] == err_file_name), 'Unknown')})")
        if len(error_files_list) > 20: logging.warning(f"  ... ({len(error_files_list) - 20} additional errors not listed)")
        logging.warning(f"Check corresponding logs in {process_log_dir} for details on the {error_count} error(s)/timeout(s).")

    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    logging.info("FastTree Python script process completed.")

    if error_count > 0:
        logging.info("Exiting with error code 1 due to failures in tree building.")
        sys.exit(1) 

if __name__ == '__main__':
    # freeze_support() # For Windows compatibility if creating executables
    main()
