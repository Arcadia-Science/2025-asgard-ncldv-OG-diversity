# scripts/refine_alignments.py
#
# Description:
# V3 Workflow: Step 5
# This script implements the alignment refinement pipeline. For each
# length-filtered alignment, it:
#   1. Unaligns the sequences (removes all gaps).
#   2. Re-aligns the cleaned sequence set with MAFFT.
#   3. Trims the new alignment using TrimAl to remove poorly aligned columns.
# v2: Updated MAFFT call to be more robust by passing the input file as a
#     direct argument instead of using stdin.

import argparse
import logging
from pathlib import Path
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def unalign_fasta(input_path: Path, output_path: Path):
    """Reads a FASTA alignment and writes a new FASTA with all gap characters removed."""
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('>'):
                f_out.write(line)
            else:
                f_out.write(line.replace('-', ''))

def run_command(command: list, log_path: Path):
    """Runs a generic command and logs stdout/stderr."""
    with open(log_path, 'w') as f_log:
        try:
            subprocess.run(command, check=True, stdout=f_log, stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError:
            logging.error(f"Command '{' '.join(command)}' failed. See log: {log_path}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred running command '{' '.join(command)}': {e}")
            return False

def process_og_refinement(og_id: str, args: argparse.Namespace):
    """Runs the full unalign -> realign -> trim pipeline for a single OG."""
    len_filtered_path = args.input_dir / f"{og_id}{args.input_suffix}"
    
    # Define intermediate and final paths
    temp_unaligned_path = args.temp_dir / f"{og_id}_unaligned.fasta"
    realigned_path = args.temp_dir / f"{og_id}_realigned.fasta"
    final_trimmed_path = args.output_dir / f"{og_id}{args.output_suffix}"
    
    mafft_log_path = args.log_dir / f"{og_id}_mafft_realign.log"
    trimal_log_path = args.log_dir / f"{og_id}_trimal.log"

    try:
        # Step 1: Unalign
        unalign_fasta(len_filtered_path, temp_unaligned_path)

        # Step 2: Re-align with MAFFT
        # **FIXED LOGIC**: Pass input file directly as an argument instead of using stdin.
        mafft_command = [args.mafft_exe, *args.mafft_opts.split(), str(temp_unaligned_path)]
        with open(realigned_path, 'w') as f_out, open(mafft_log_path, 'w') as f_log:
             subprocess.run(mafft_command, check=True, stdout=f_out, stderr=f_log)

        # Step 3: Trim with TrimAl
        trimal_command = [args.trimal_exe, "-in", str(realigned_path), "-out", str(final_trimmed_path), *args.trimal_opts.split()]
        run_command(trimal_command, trimal_log_path)
        
        return og_id, "Success"
    except Exception as e:
        logging.error(f"Failed processing {og_id}: {e}")
        return og_id, "Failure"

def main():
    parser = argparse.ArgumentParser(description="Refine alignments: unalign, realign with MAFFT, and trim with TrimAl.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of length-filtered alignments.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for final, refined MSAs.")
    parser.add_argument("--input-suffix", type=str, default=".fasta")
    parser.add_argument("--output-suffix", type=str, default="_final.fasta")
    parser.add_argument("--temp-dir", type=Path, help="Temp directory. Defaults to output_dir/temp.")
    parser.add_argument("--log-dir", type=Path, help="Log directory. Defaults to output_dir/logs.")
    parser.add_argument("--mafft-exe", type=str, default="mafft")
    parser.add_argument("--trimal-exe", type=str, default="trimal")
    parser.add_argument("--mafft-opts", type=str, default="--auto --thread 1")
    parser.add_argument("--trimal-opts", type=str, default="-gappyout")
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    args.temp_dir = args.temp_dir or args.output_dir / "temp"
    args.log_dir = args.log_dir or args.output_dir / "logs"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(exist_ok=True)
    args.log_dir.mkdir(exist_ok=True)
    
    files_to_process = list(args.input_dir.glob(f"*{args.input_suffix}"))
    tasks = [f.stem.replace(args.input_suffix, '') for f in files_to_process]
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_og_refinement, og_id, args) for og_id in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Refining Alignments"):
            future.result()

    logging.info("Alignment refinement process complete.")

if __name__ == "__main__":
    main()
