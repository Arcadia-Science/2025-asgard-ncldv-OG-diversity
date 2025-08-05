import pandas as pd
import numpy as np
from Bio import AlignIO, SeqIO
import os
import glob
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
import logging
from pathlib import Path
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
        if handler.level > logging.INFO:
             handler.setLevel(logging.INFO)
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    if logger.level > logging.INFO : logger.setLevel(logging.INFO)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter MAFFT alignments based on sequence length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing MAFFT alignment files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for length-filtered alignment files."
    )
    parser.add_argument(
        "--summary-log",
        type=Path,
        default=None,
        help="Path to save the summary log CSV file. Defaults to 'length_filtering_log.csv' in the output directory."
    )
    parser.add_argument(
        "--min-length-perc",
        type=float,
        default=70.0,
        help="Minimum length percentage of the median non-gap length to keep a sequence."
    )
    parser.add_argument(
        "--min-abs-length",
        type=int,
        default=0,
        help="Minimum absolute non-gap length to keep a sequence. Overrides percentage if higher."
    )
    parser.add_argument(
        "--input-suffix",
        type=str,
        default=".aln",
        help="Suffix of input alignment files."
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_len_filtered.aln",
        help="Suffix for output filtered alignment files."
    )
    return parser.parse_args()

def calculate_non_gap_length(seq_str: str) -> int:
    """Calculates the number of non-gap characters in a sequence string."""
    return len(seq_str.replace('-', ''))

def main():
    """Main execution function."""
    args = parse_arguments()

    # If summary log is not specified, create it in the output directory
    summary_log_file = args.summary_log
    if summary_log_file is None:
        summary_log_file = args.output_dir / "length_filtering_log.csv"

    logger.info("Starting Length-Based Filtering of MAFFT Aligned Sequences...")
    logger.info(f"Input MAFFT directory: {args.input_dir}")
    logger.info(f"Output directory for length-filtered alignments: {args.output_dir}")
    logger.info(f"Keeping sequences >= {args.min_length_perc}% of median non-gap length.")
    if args.min_abs_length > 0:
        logger.info(f"Additionally, keeping sequences with absolute non-gap length >= {args.min_abs_length} AAs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    alignment_files = sorted(list(args.input_dir.glob(f"*{args.input_suffix}")))
    
    if not alignment_files:
        logger.warning(f"No alignment files found in '{args.input_dir}' with pattern '*{args.input_suffix}'. Exiting.")
        return

    logger.info(f"Found {len(alignment_files)} alignment files to process.")
    filtering_summary_log = []

    for aln_filepath_obj in tqdm(alignment_files, desc="Filtering Alignments by Length"):
        base_aln_filename = aln_filepath_obj.name
        
        # Extract OG ID
        identifier_parts = base_aln_filename.split('.')
        if len(identifier_parts) > 1 and base_aln_filename.endswith(args.input_suffix):
            og_id_for_log = base_aln_filename[:-len(args.input_suffix)]
        else:
            og_id_for_log = aln_filepath_obj.stem

        logger.debug(f"Processing: {base_aln_filename} for OG: {og_id_for_log}")

        try:
            alignment_records = list(AlignIO.read(aln_filepath_obj, "fasta"))
            if not alignment_records: 
                logger.warning(f"Alignment file '{base_aln_filename}' is empty or could not be read. Skipping.")
                filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": 0, "Median_NonGap_Length": np.nan,
                                          "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": 0, "Status": "Empty/Unreadable"})
                continue

            initial_seq_count = len(alignment_records)
            
            seq_data_for_filtering = [{'record': r, 'non_gap_length': calculate_non_gap_length(str(r.seq))} for r in alignment_records]
            
            df_seq_lengths = pd.DataFrame(seq_data_for_filtering)
            
            if df_seq_lengths.empty or df_seq_lengths['non_gap_length'].isnull().all():
                 logger.warning(f"Could not calculate non-gap lengths for sequences in '{base_aln_filename}'. Skipping OG.")
                 filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": initial_seq_count, "Median_NonGap_Length": np.nan,
                                           "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": 0, "Status": "Non-gap length calculation failed"})
                 continue

            median_non_gap_length = df_seq_lengths['non_gap_length'].median()
            length_threshold_percent_based = (args.min_length_perc / 100.0) * median_non_gap_length
            
            final_length_threshold = max(length_threshold_percent_based, args.min_abs_length)
            
            logger.debug(f"OG {og_id_for_log}: Median non-gap len: {median_non_gap_length:.0f}, "
                         f"Threshold due to %: {length_threshold_percent_based:.0f}, Final Threshold: {final_length_threshold:.0f}")

            filtered_records = [
                item['record'] for item in seq_data_for_filtering if item['non_gap_length'] >= final_length_threshold
            ]
            
            num_after_filter = len(filtered_records)
            logger.info(f"OG {og_id_for_log}: Initial: {initial_seq_count}, Kept after length filter: {num_after_filter}")
            
            filtering_summary_log.append({
                "Orthogroup": og_id_for_log, 
                "Initial_Seqs": initial_seq_count, 
                "Median_NonGap_Length": round(median_non_gap_length, 1) if pd.notna(median_non_gap_length) else np.nan,
                "Threshold_Length_Used": round(final_length_threshold, 1) if pd.notna(final_length_threshold) else np.nan,
                "Seqs_After_Length_Filter": num_after_filter,
                "Status": "Processed"
            })

            if num_after_filter >= 2: 
                output_filtered_filename = args.output_dir / f"{og_id_for_log}{args.output_suffix}"
                try:
                    SeqIO.write(filtered_records, output_filtered_filename, "fasta")
                except Exception as e:
                    logger.error(f"Error writing length-filtered FASTA for OG {og_id_for_log}: {e}")
            elif num_after_filter > 0:
                logger.warning(f"OG {og_id_for_log}: Only {num_after_filter} sequence(s) remained after length filtering. Not writing file as < 2 sequences.")
            else:
                logger.warning(f"No sequences remained for OG {og_id_for_log} after length filtering. No file written.")

        except Exception as e:
            logger.error(f"Failed to process alignment file '{base_aln_filename}': {e}", exc_info=True)
            filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": "Error", "Median_NonGap_Length": np.nan,
                                      "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": "Error", "Status": "Processing Error"})
    
    if filtering_summary_log:
        df_summary = pd.DataFrame(filtering_summary_log)
        try:
            df_summary.to_csv(summary_log_file, index=False)
            logger.info(f"Length filtering summary log saved to: {summary_log_file}")
        except Exception as e:
            logger.error(f"Could not save length filtering summary log: {e}")

    logger.info(f"Length-based filtering finished. Filtered alignments are in: {args.output_dir}")
    logger.info(f"Next step: Run 'refine_alignments.py' on alignments in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
