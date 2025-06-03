import pandas as pd
import numpy as np
import os
import glob
from Bio import AlignIO, SeqIO 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.auto import tqdm
import logging
from pathlib import Path
import argparse 
import sys # Added for console_handler

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers: 
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
else:
    for handler in logger.handlers: 
        if handler.level > logging.INFO: handler.setLevel(logging.INFO)
    if logger.level > logging.INFO : logger.setLevel(logging.INFO)

# --- Default Configuration (can be overridden by args) ---
# Assumes this script is in a 'scripts' subdirectory of your main project BASE_DIR
SCRIPT_DIR_DEFAULT = Path(__file__).resolve().parent
BASE_DIR_DEFAULT = SCRIPT_DIR_DEFAULT.parent 

# These defaults are set up for an "ESP OGs" track.
# Adjust NEW_PIPELINE_DATA_DIR_BASE_NAME and ANALYSIS_SUBDIR_NAME if your ESP track uses different top-level folder names.
NEW_PIPELINE_DATA_DIR_BASE_NAME = "data_esp_ogs" 
ANALYSIS_SUBDIR_NAME = "analysis_esp_ogs"

NEW_PIPELINE_DATA_DIR_DEFAULT = BASE_DIR_DEFAULT / "data" / NEW_PIPELINE_DATA_DIR_BASE_NAME
ANALYSIS_OUTPUTS_DIR_DEFAULT = BASE_DIR_DEFAULT / "analysis_outputs" / ANALYSIS_SUBDIR_NAME

INPUT_ALN_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "initial_esp_mafft_alignments" # MAFFT outputs for ESPs
OUTPUT_LEN_FILTERED_ALN_DIR_DEFAULT = NEW_PIPELINE_DATA_DIR_DEFAULT / "mafft_len_filtered_esp_output" # Output of this script
CURATED_OG_ANCHOR_FILE_DEFAULT = ANALYSIS_OUTPUTS_DIR_DEFAULT / "curated_esp_og_anchor_list_v1.csv" # From Cell 2_ESP

INPUT_ALN_SUFFIX_DEFAULT = ".aln" 
OUTPUT_FILTERED_SUFFIX_DEFAULT = "_len_filtered.aln" # Suffix for files produced by this script

MIN_LENGTH_PERCENTAGE_OF_MEDIAN_DEFAULT = 70.0 
MIN_ABSOLUTE_NON_GAP_LENGTH_DEFAULT = 0 

# Default summary log file name, placed within the output directory
DEFAULT_LOG_FILENAME = "length_filtering_log_esp_track.csv"
SUMMARY_LOG_FILE_DEFAULT = OUTPUT_LEN_FILTERED_ALN_DIR_DEFAULT / DEFAULT_LOG_FILENAME
# --- End Default Configuration ---

def calculate_non_gap_length(seq_str: str) -> int:
    return len(seq_str.replace('-', ''))

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter MAFFT alignments by sequence length (always keeping anchors). Configured for ESP track.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_aln_dir", type=Path, default=INPUT_ALN_DIR_DEFAULT,
                        help="Input directory for initial MAFFT alignments.")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_LEN_FILTERED_ALN_DIR_DEFAULT,
                        help="Output directory for length-filtered alignments.")
    parser.add_argument("--anchor_list_csv", type=Path, default=CURATED_OG_ANCHOR_FILE_DEFAULT,
                        help="CSV file mapping OG_ID to Anchor_ProteinID.")
    parser.add_argument("--input_suffix", default=INPUT_ALN_SUFFIX_DEFAULT,
                        help="Suffix of input alignment files (e.g., .aln, .fasta).")
    parser.add_argument("--output_suffix", default=OUTPUT_FILTERED_SUFFIX_DEFAULT,
                        help="Suffix for output filtered alignment files.")
    parser.add_argument("--min_percent_median", type=float, default=MIN_LENGTH_PERCENTAGE_OF_MEDIAN_DEFAULT,
                        help="Minimum non-gap length as percentage of median non-gap length of sequences in the alignment.")
    parser.add_argument("--min_abs_len", type=int, default=MIN_ABSOLUTE_NON_GAP_LENGTH_DEFAULT,
                        help="Minimum absolute non-gap length. If >0, this is an additional floor to the percentage-based threshold.")
    parser.add_argument("--summary_log_file", type=Path, default=None, 
                        help=f"Path for the summary log CSV file. Default is '{DEFAULT_LOG_FILENAME}' in the output_dir.")
    return parser.parse_args()

def main(args: argparse.Namespace):
    logger.info(f"Starting Length-Based Filtering (v3.1 - Always Keep Anchor, ESP Track)...")
    logger.info(f"Input MAFFT directory: {args.input_aln_dir}")
    logger.info(f"Output directory for filtered alignments: {args.output_dir}")
    logger.info(f"Anchor list CSV: {args.anchor_list_csv}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine summary log path: use argument if provided, else use default based on output_dir
    summary_log_path = args.summary_log_file if args.summary_log_file else args.output_dir / DEFAULT_LOG_FILENAME
    logger.info(f"Summary log will be saved to: {summary_log_path}")

    if not args.anchor_list_csv.exists():
        logger.error(f"Anchor list file '{args.anchor_list_csv}' not found. Exiting."); return
    try:
        df_anchors = pd.read_csv(args.anchor_list_csv)
        if not all(col in df_anchors.columns for col in ["OG_ID", "Anchor_ProteinID"]):
            logger.error(f"Anchor list CSV '{args.anchor_list_csv}' missing 'OG_ID' or 'Anchor_ProteinID'. Exiting."); return
        # Create a dictionary for quick lookup: OG_ID -> Anchor_ProteinID
        anchor_map = pd.Series(df_anchors["Anchor_ProteinID"].astype(str).values, index=df_anchors["OG_ID"].astype(str)).to_dict()
        logger.info(f"Loaded {len(anchor_map)} OG-to-Anchor mappings.")
    except Exception as e: 
        logger.error(f"Error reading anchor list CSV '{args.anchor_list_csv}': {e}", exc_info=True); return

    # Ensure input suffix starts with a dot if it's meant to be an extension
    input_suffix_glob = args.input_suffix if args.input_suffix.startswith('.') else '.' + args.input_suffix
    alignment_files = sorted(list(args.input_aln_dir.glob(f"*{input_suffix_glob}")))
    
    if not alignment_files:
        logger.warning(f"No alignment files found in '{args.input_aln_dir}' with pattern '*{input_suffix_glob}'. Exiting."); return

    logger.info(f"Found {len(alignment_files)} alignment files to process.")
    filtering_summary_log = []

    for aln_filepath_obj in tqdm(alignment_files, desc="Filtering ESP Alignments"):
        base_aln_filename = aln_filepath_obj.name
        # Robustly extract OG_ID from filename, assuming format like OGXXXX.ASG.aln or OGXXXX.aln
        og_id_for_log = base_aln_filename.removesuffix(input_suffix_glob) # Python 3.9+
        # Fallback for older Python or more complex names:
        # if base_aln_filename.endswith(input_suffix_glob):
        #     og_id_for_log = base_aln_filename[:-len(input_suffix_glob)]
        # else:
        #     og_id_for_log = aln_filepath_obj.stem

        anchor_protein_id = anchor_map.get(str(og_id_for_log)) # Ensure OG_ID is string for lookup
        
        if not anchor_protein_id:
            logger.warning(f"No anchor in map for OG: {og_id_for_log} (File: {base_aln_filename}). Skipping this file."); 
            filtering_summary_log.append({"Orthogroup": og_id_for_log, "Initial_Seqs": "Skipped", "Status": "No anchor in map"}); continue

        logger.debug(f"Processing: {base_aln_filename} for OG: {og_id_for_log}, Anchor: {anchor_protein_id}")
        try:
            alignment_records = list(AlignIO.read(str(aln_filepath_obj), "fasta")) 
            if not alignment_records: 
                logger.warning(f"Alignment file '{aln_filepath_obj}' is empty. Skipping."); 
                filtering_summary_log.append({"Orthogroup": og_id_for_log, "Anchor_ProteinID": anchor_protein_id, "Initial_Seqs": 0, "Status": "Empty file"}); continue

            initial_seq_count = len(alignment_records)
            seq_data_for_filtering, anchor_record_in_alignment = [], None
            
            for record in alignment_records:
                non_gap_len = calculate_non_gap_length(str(record.seq))
                is_anchor_flag = str(record.id) == str(anchor_protein_id) 
                seq_data_for_filtering.append({'record': record, 'non_gap_length': non_gap_len, 'is_anchor': is_anchor_flag})
                if is_anchor_flag: anchor_record_in_alignment = record
            
            if anchor_record_in_alignment is None:
                logger.warning(f"Designated anchor {anchor_protein_id} NOT FOUND in initial MAFFT alignment {base_aln_filename} for OG {og_id_for_log}. This OG's anchor might have been filtered out before this step, or ID mismatch. Will proceed without forcing this specific anchor, but this is unusual.")
            
            df_seq_lengths = pd.DataFrame(seq_data_for_filtering)
            if df_seq_lengths.empty: 
                logger.warning(f"No sequence data extracted for {base_aln_filename}. Skipping."); 
                filtering_summary_log.append({"Orthogroup": og_id_for_log, "Anchor_ProteinID": anchor_protein_id, "Initial_Seqs": initial_seq_count, "Status": "No seq data extracted"}); continue
            
            median_non_gap_length = df_seq_lengths['non_gap_length'].median()
            final_length_threshold = 0.0
            if pd.notna(median_non_gap_length) and median_non_gap_length > 0 : # Ensure median is valid
                length_threshold_percent_based = (args.min_percent_median / 100.0) * median_non_gap_length
                final_length_threshold = length_threshold_percent_based
                if args.min_abs_len > 0: final_length_threshold = max(length_threshold_percent_based, float(args.min_abs_len))
            else: 
                logger.warning(f"Median non-gap length is NaN or zero for OG {og_id_for_log}. Using threshold 0 for non-anchors if min_abs_len is also 0.")
                if args.min_abs_len > 0: final_length_threshold = float(args.min_abs_len)


            median_display = f"{median_non_gap_length:.0f}" if pd.notna(median_non_gap_length) else 'NaN'
            threshold_display = f"{final_length_threshold:.0f}" if pd.notna(final_length_threshold) else 'NaN'
            logger.debug(f"OG {og_id_for_log}: Median non-gap len: {median_display}, Final Threshold for non-anchors: {threshold_display}")

            filtered_records = []
            for item in seq_data_for_filtering:
                if item['is_anchor'] and anchor_record_in_alignment is not None: # Only force keep if anchor was actually found
                    filtered_records.append(item['record'])
                    logger.debug(f"  OG {og_id_for_log}: Kept anchor {item['record'].id} (Non-gap len: {item['non_gap_length']})")
                elif item['non_gap_length'] >= final_length_threshold:
                    filtered_records.append(item['record'])
            
            num_after_filter = len(filtered_records)
            logger.info(f"OG {og_id_for_log}: Initial: {initial_seq_count}, Kept after filter (anchor forced if found): {num_after_filter}")
            
            anchor_non_gap_len_val = np.nan
            if anchor_record_in_alignment: # If the designated anchor was found in the input alignment
                anchor_data = df_seq_lengths[df_seq_lengths['is_anchor']] # Should find it if anchor_record_in_alignment is not None
                if not anchor_data.empty: anchor_non_gap_len_val = anchor_data['non_gap_length'].iloc[0]

            filtering_summary_log.append({
                "Orthogroup": og_id_for_log, "Anchor_ProteinID": anchor_protein_id,
                "Initial_Seqs": initial_seq_count, 
                "Median_NonGap_Length": round(median_non_gap_length,1) if pd.notna(median_non_gap_length) else np.nan,
                "Threshold_Length_Used_For_NonAnchors": round(final_length_threshold,1) if pd.notna(final_length_threshold) else np.nan,
                "Seqs_After_Length_Filter": num_after_filter,
                "Anchor_Present_In_Initial_MSA": anchor_record_in_alignment is not None,
                "Anchor_NonGap_Length": anchor_non_gap_len_val if pd.notna(anchor_non_gap_len_val) else np.nan, 
                "Status": "Processed"
            })

            if num_after_filter >= 2: 
                output_filtered_filename = args.output_dir / f"{og_id_for_log}{args.output_suffix}"
                SeqIO.write(filtered_records, output_filtered_filename, "fasta")
            else: 
                logger.warning(f"OG {og_id_for_log}: Only {num_after_filter} sequence(s) remained. Not writing file as < 2 sequences are needed for most downstream steps.")
        except Exception as e:
            logger.error(f"Failed to process alignment file '{aln_filepath_obj}': {e}", exc_info=True)
            filtering_summary_log.append({"Orthogroup": og_id_for_log, "Anchor_ProteinID": anchor_protein_id, "Status": f"Processing Error: {type(e).__name__}"})
    
    if filtering_summary_log:
        pd.DataFrame(filtering_summary_log).to_csv(summary_log_path, index=False, float_format='%.1f')
        logger.info(f"Length filtering summary log saved to: {summary_log_path}")

    logger.info(f"Length-based filtering finished. Filtered alignments are in: {args.output_dir}")
    logger.info(f"Next step: Run 'run_realign_trimal.sh' (adapted for these new paths) on alignments in '{args.output_dir}'.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
