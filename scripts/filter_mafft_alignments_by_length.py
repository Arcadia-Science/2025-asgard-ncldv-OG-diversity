import pandas as pd
import numpy as np
import os
import glob
from Bio import AlignIO, SeqIO # Ensure SeqIO is imported
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
import logging
from pathlib import Path

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers: # Use .handlers instead of .hasHandlers() for Python 3.7+
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.propagate = False # Avoid duplicate logs if root logger is also configured
else:
    # If logger already has handlers, ensure its level is set
    # This might not be necessary if the above block always runs once per module import
    for handler in logger.handlers: # Ensure all handlers are at least INFO
        if handler.level > logging.INFO:
             handler.setLevel(logging.INFO)
    if logger.level > logging.INFO : logger.setLevel(logging.INFO)


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent # Assumes script is in 'scripts' directory
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"


# Input directory for MAFFT alignments (output from the initial MAFFT script)
INPUT_ALN_DIR = DATA_DIR / "initial_mafft_alignments"

# Output directory for length-filtered FASTA alignment files
OUTPUT_LEN_FILTERED_ALN_DIR = DATA_DIR / "mafft_len_filtered_output"

# Input file suffix (from initial MAFFT script)
INPUT_ALN_SUFFIX = ".aln"
# Output file suffix for this script
OUTPUT_FILTERED_SUFFIX = "_len_filtered.aln"


# Filtering Parameters
MIN_LENGTH_PERCENTAGE_OF_MEDIAN = 70.0 
MIN_ABSOLUTE_NON_GAP_LENGTH = 0 

SUMMARY_LOG_FILE = OUTPUT_LEN_FILTERED_ALN_DIR / "length_filtering_log.csv"
# --- End Configuration ---

def calculate_non_gap_length(seq_str: str) -> int:
    """Calculates the number of non-gap characters in a sequence string."""
    return len(seq_str.replace('-', ''))

# --- Main Script ---
def main():
    logger.info(f"Starting Length-Based Filtering of MAFFT Aligned Sequences...")
    logger.info(f"Input MAFFT directory: {INPUT_ALN_DIR}")
    logger.info(f"Output directory for length-filtered alignments: {OUTPUT_LEN_FILTERED_ALN_DIR}")
    logger.info(f"Keeping sequences >= {MIN_LENGTH_PERCENTAGE_OF_MEDIAN}% of median non-gap length.")
    if MIN_ABSOLUTE_NON_GAP_LENGTH and MIN_ABSOLUTE_NON_GAP_LENGTH > 0:
        logger.info(f"Additionally, keeping sequences with absolute non-gap length >= {MIN_ABSOLUTE_NON_GAP_LENGTH} AAs.")

    OUTPUT_LEN_FILTERED_ALN_DIR.mkdir(parents=True, exist_ok=True)

    # Use pathlib glob and ensure input suffix matches
    alignment_files = sorted(list(INPUT_ALN_DIR.glob(f"*{INPUT_ALN_SUFFIX}")))
    
    if not alignment_files:
        logger.warning(f"No alignment files found in '{INPUT_ALN_DIR}' with pattern '*{INPUT_ALN_SUFFIX}'. Exiting.")
        return

    logger.info(f"Found {len(alignment_files)} alignment files to process.")
    filtering_summary_log = []

    for aln_filepath_obj in tqdm(alignment_files, desc="Filtering Alignments by Length"):
        aln_filepath = str(aln_filepath_obj) # Convert Path object to string for older BioPython or os.path functions if needed
        base_aln_filename = aln_filepath_obj.name
        
        # Extract OG ID: assumes filename like OGXXXX.ASG.aln or OGXXXX.GV.aln
        # Takes the part before the first dot if multiple dots, or before .aln
        identifier_parts = base_aln_filename.split('.')
        if len(identifier_parts) > 1 and identifier_parts[-1] == INPUT_ALN_SUFFIX.lstrip('.'): # e.g. .aln
            og_id_for_log = '.'.join(identifier_parts[:-1]) # Joins back if OG_ID had dots, e.g. OGXXXX.ASG
        else:
            og_id_for_log = aln_filepath_obj.stem # Fallback: OGXXXX if filename is OGXXXX.aln

        logger.debug(f"Processing: {base_aln_filename} for OG: {og_id_for_log}")

        try:
            alignment_records = list(AlignIO.read(aln_filepath, "fasta")) 
            if not alignment_records: 
                logger.warning(f"Alignment file '{aln_filepath}' is empty or could not be read. Skipping.")
                filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": 0, "Median_NonGap_Length": np.nan,
                                          "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": 0, "Status": "Empty/Unreadable"})
                continue

            initial_seq_count = len(alignment_records)
            
            seq_data_for_filtering = []
            for record in alignment_records:
                non_gap_len = calculate_non_gap_length(str(record.seq))
                seq_data_for_filtering.append({'record': record, 'non_gap_length': non_gap_len})
            
            df_seq_lengths = pd.DataFrame(seq_data_for_filtering)
            
            if df_seq_lengths.empty or df_seq_lengths['non_gap_length'].isnull().all():
                 logger.warning(f"Could not calculate non-gap lengths for sequences in '{aln_filepath}'. Skipping OG.")
                 filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": initial_seq_count, "Median_NonGap_Length": np.nan,
                                           "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": 0, "Status": "Non-gap length calculation failed"})
                 continue

            median_non_gap_length = df_seq_lengths['non_gap_length'].median()
            length_threshold_percent_based = (MIN_LENGTH_PERCENTAGE_OF_MEDIAN / 100.0) * median_non_gap_length
            
            final_length_threshold = length_threshold_percent_based
            if MIN_ABSOLUTE_NON_GAP_LENGTH and MIN_ABSOLUTE_NON_GAP_LENGTH > 0:
                final_length_threshold = max(length_threshold_percent_based, MIN_ABSOLUTE_NON_GAP_LENGTH)
            
            logger.debug(f"OG {og_id_for_log}: Median non-gap len: {median_non_gap_length:.0f}, "+
                         f"Threshold due to %: {length_threshold_percent_based:.0f}, Final Threshold: {final_length_threshold:.0f}")

            filtered_records = [
                item['record'] for item in seq_data_for_filtering if item['non_gap_length'] >= final_length_threshold
            ]
            
            num_after_filter = len(filtered_records)
            logger.info(f"OG {og_id_for_log}: Initial: {initial_seq_count}, Kept after length filter: {num_after_filter}")
            
            filtering_summary_log.append({
                "Orthogroup": og_id_for_log, 
                "Initial_Seqs": initial_seq_count, 
                "Median_NonGap_Length": round(median_non_gap_length,1) if pd.notna(median_non_gap_length) else np.nan,
                "Threshold_Length_Used": round(final_length_threshold,1) if pd.notna(final_length_threshold) else np.nan,
                "Seqs_After_Length_Filter": num_after_filter,
                "Status": "Processed"
            })

            if num_after_filter >= 2: 
                # Use the new output suffix
                output_filtered_filename = OUTPUT_LEN_FILTERED_ALN_DIR / f"{og_id_for_log}{OUTPUT_FILTERED_SUFFIX}"
                try:
                    SeqIO.write(filtered_records, output_filtered_filename, "fasta")
                except Exception as e:
                    logger.error(f"Error writing length-filtered FASTA for OG {og_id_for_log}: {e}")
            elif num_after_filter > 0:
                logger.warning(f"OG {og_id_for_log}: Only {num_after_filter} sequence(s) remained after length filtering. Not writing file as < 2 sequences.")
            else:
                logger.warning(f"No sequences remained for OG {og_id_for_log} after length filtering. No file written.")

        except Exception as e:
            logger.error(f"Failed to process alignment file '{aln_filepath}': {e}", exc_info=True)
            filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": "Error", "Median_NonGap_Length": np.nan,
                                      "Threshold_Length_Used": np.nan, "Seqs_After_Length_Filter": "Error", "Status": "Processing Error"})
    
    if filtering_summary_log:
        df_summary = pd.DataFrame(filtering_summary_log)
        try:
            df_summary.to_csv(SUMMARY_LOG_FILE, index=False)
            logger.info(f"Length filtering summary log saved to: {SUMMARY_LOG_FILE}")
        except Exception as e:
            logger.error(f"Could not save length filtering summary log: {e}")

    logger.info(f"Length-based filtering finished. Filtered alignments are in: {OUTPUT_LEN_FILTERED_ALN_DIR}")
    logger.info(f"Next step: Run 'run_realign_trimal.sh' on alignments in '{OUTPUT_LEN_FILTERED_ALN_DIR}'.")

if __name__ == "__main__":
    main()
