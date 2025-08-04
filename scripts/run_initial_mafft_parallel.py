#!/usr/bin/env python3

"""
Runs MAFFT alignment in parallel on multiple input FASTA files using Python's
concurrent.futures module.
This version is adapted for the initial alignment step of the "ultra-filtered large OGs" pipeline.

Input: Raw FASTA files from DATA_DIR/data_large_ogs_gt100mem_gt15id/raw_og_fastas/
Output: MAFFT alignments in DATA_DIR/data_large_ogs_gt100mem_gt15id/initial_mafft_alignments/
        MAFFT logs in DATA_DIR/data_large_ogs_gt100mem_gt15id/initial_mafft_logs/
"""

import os
import sys
import glob
import subprocess
import logging
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil 
from tqdm.auto import tqdm 

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

# --- Helper Function to Run MAFFT ---
def run_mafft_on_file(fasta_file_path: Path, output_dir: Path, log_dir: Path,
                      mafft_exe: str, mafft_args: list[str],
                      input_suffix: str, output_suffix: str) -> tuple[str, str, str]: 
    base_name = fasta_file_path.name
    if input_suffix and base_name.endswith(input_suffix):
        identifier = base_name[:-len(input_suffix)]
    else:
        identifier = fasta_file_path.stem 

    output_aln_path = output_dir / f"{identifier}{output_suffix}"
    output_log_path = log_dir / f"{identifier}_mafft.log"

    if output_aln_path.exists() and output_aln_path.stat().st_size > 0:
        logger.debug(f"Output alignment {output_aln_path} already exists and is non-empty. Skipping {base_name}.")
        return base_name, identifier, "Skipped (Output Exists)"
    elif output_aln_path.exists():
         logger.warning(f"Output alignment {output_aln_path} exists but is empty. Will overwrite for {base_name}.")

    command = [mafft_exe] + mafft_args + [str(fasta_file_path)]
    try:
        with open(output_aln_path, 'w') as f_out, open(output_log_path, 'w') as f_err:
            process = subprocess.run(command, stdout=f_out, stderr=f_err, check=True, text=True, encoding='utf-8')
        if not output_aln_path.stat().st_size > 0:
             error_info = "Unknown MAFFT issue (empty output)"
             try: 
                  with open(output_log_path, 'r', encoding='utf-8') as f_log_read:
                       log_content = f_log_read.read(500).strip()
                       if log_content: error_info = f"MAFFT Error (see log): {log_content}..."
             except Exception: pass 
             logger.warning(f"MAFFT created an empty alignment file for {base_name} (ID: {identifier}). Check log: {output_log_path}")
             return base_name, identifier, error_info
        return base_name, identifier, "Success"
    except FileNotFoundError:
        logger.error(f"MAFFT executable '{mafft_exe}' not found. Failed for {base_name} (ID: {identifier}).")
        return base_name, identifier, f"Error: MAFFT not found ('{mafft_exe}')"
    except subprocess.CalledProcessError as e:
        logger.error(f"MAFFT failed for {base_name} (ID: {identifier}) with exit code {e.returncode}. Check log: {output_log_path}")
        return base_name, identifier, f"Error: MAFFT failed (Code {e.returncode})"
    except Exception as e:
        logger.exception(f"An unexpected error occurred while running MAFFT for {base_name} (ID: {identifier}): {e}")
        return base_name, identifier, f"Error: Unexpected ({type(e).__name__})"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MAFFT alignment in parallel for initial OG alignments (ultra-filtered large OGs track).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input_dir", type=Path, required=True,
                        help="Input directory containing raw FASTA files for alignment.")
    parser.add_argument("-o", "--output_dir", type=Path, required=True,
                        help="Output directory for MAFFT aligned files.")
    parser.add_argument("-l", "--log_dir", type=Path, required=True,
                        help="Output directory for MAFFT stderr logs.")
    parser.add_argument("--input_suffix", default=".fasta",
                        help="Suffix of input FASTA files to process.")
    parser.add_argument("--output_suffix", default=".aln",
                        help="Suffix to use for output alignment files.")
    parser.add_argument("--mafft_exe", default="mafft",
                        help="Path to the MAFFT executable.")
    parser.add_argument("--mafft_args", default="--auto --thread 1", 
                        help="Arguments to pass to MAFFT (as a single string).")
    
    default_cores = max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)
    parser.add_argument("-n", "--num_cores", type=int, default=default_cores,
                        help="Number of parallel MAFFT jobs to run.")

    return parser.parse_args()

def main():
    args = parse_arguments()
    start_time = time.time()
    logger.info(f"Script arguments: {args}")

    mafft_exe_path = shutil.which(args.mafft_exe)
    if not mafft_exe_path:
        logger.error(f"MAFFT executable '{args.mafft_exe}' not found in PATH or is not executable. Please install MAFFT or provide the correct path."); sys.exit(1)
    logger.info(f"Using MAFFT executable: {mafft_exe_path}")

    mafft_args_list = args.mafft_args.split()
    logger.info(f"Using MAFFT arguments: {mafft_args_list}")

    # Ensure directories exist (input dir should already exist from previous script)
    args.input_dir.mkdir(parents=True, exist_ok=True) 
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input FASTA files from: {args.input_dir}")
    logger.info(f"Output alignments will be saved in: {args.output_dir}")
    logger.info(f"Log files will be saved in: {args.log_dir}")

    fasta_files = sorted(list(args.input_dir.glob(f"*{args.input_suffix}")))
    if not fasta_files:
        logger.error(f"No files matching '*{args.input_suffix}' found in {args.input_dir}. Stopping."); sys.exit(1)

    logger.info(f"Found {len(fasta_files)} FASTA files to align for the ultra-filtered large OGs set.")
    logger.info(f"Running up to {args.num_cores} MAFFT jobs in parallel...")

    success_count, skipped_count, error_count = 0, 0, 0
    
    with ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        futures = {
            executor.submit(
                run_mafft_on_file,
                f_path, args.output_dir, args.log_dir,
                mafft_exe_path, mafft_args_list,
                args.input_suffix, args.output_suffix
            ): f_path
            for f_path in fasta_files
        }

        for future in tqdm(as_completed(futures), total=len(fasta_files), desc="Aligning Ultra-Filtered Large OGs", unit="OG"):
            input_filepath = futures[future]
            input_fname = input_filepath.name
            try:
                _, identifier, status = future.result() 
                if status == "Success":
                    logger.info(f"Completed: {identifier} ({input_fname}) - Success")
                    success_count += 1
                elif status == "Skipped (Output Exists)":
                    logger.info(f"Completed: {identifier} ({input_fname}) - Skipped (Output Exists)")
                    skipped_count += 1
                else: 
                    logger.error(f"Completed: {identifier} ({input_fname}) - {status}")
                    error_count += 1
            except Exception as exc:
                logger.error(f"Job for {input_fname} (ID: {input_filepath.stem}) generated an exception in the executor: {exc}", exc_info=True)
                error_count += 1
            
    end_time = time.time()
    logger.info("--- Initial MAFFT Alignment Summary (Ultra-Filtered Large OGs) ---")
    logger.info(f"Successfully aligned files: {success_count}")
    logger.info(f"Skipped (output existed): {skipped_count}")
    logger.info(f"Errors during alignment: {error_count}")
    logger.info(f"Total files processed/attempted: {len(fasta_files)}")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info("Initial MAFFT alignment process for ultra-filtered large OGs completed.")
    logger.info(f"Next step: Run 'filter_mafft_alignments_by_length.py' (adapted for these new paths) on alignments in '{args.output_dir}'.")

    if error_count > 0:
        logger.error(f"{error_count} jobs failed. Check logs in {args.log_dir} for details.")
    
if __name__ == '__main__':
    main()
