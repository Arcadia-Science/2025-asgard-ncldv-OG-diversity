#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for Step 0.1 & 0.2 of Phase 1 Alignment Pipeline:
Extracts member sequences for selected Orthogroups (OGs) and saves them
into individual FASTA files.
(v3 - Refactored for command-line arguments and improved robustness)

Input:
- A CSV file containing a list of OG IDs to process.
- The main proteome database CSV containing sequences and metadata.

Output:
- FASTA files, one per OG, in the specified output directory.
  Headers will be like: >ProteinID UniProtKB_AC=P12345 OG_ID=OG000XXXX
"""

import pandas as pd
from pathlib import Path
import logging
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm.auto import tqdm
import argparse
import sys

def setup_logging():
    """Sets up a logger that prints to the console."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
        logger.propagate = False
    return logger

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract member sequences for specified Orthogroups into individual FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--og_list_csv",
        type=Path,
        required=True,
        help="Path to the input CSV file containing the list of OG_IDs to process."
    )
    parser.add_argument(
        "--proteome_db",
        type=Path,
        required=True,
        help="Path to the main proteome database CSV file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output directory where FASTA files will be saved."
    )
    parser.add_argument(
        "--og_id_col",
        default="OG_ID",
        help="Column name for Orthogroup IDs in the input CSVs."
    )
    parser.add_argument(
        "--protein_id_col",
        default="ProteinID",
        help="Column name for Protein IDs in the proteome database."
    )
    parser.add_argument(
        "--seq_col",
        default="Sequence",
        help="Column name for protein sequences in the proteome database."
    )
    parser.add_argument(
        "--uniprot_col",
        default="UniProtKB_AC",
        help="Column name for UniProt Accession numbers in the proteome database."
    )
    return parser.parse_args()

def main(args: argparse.Namespace, logger: logging.Logger):
    """Main function to orchestrate sequence extraction."""
    logger.info("Starting extraction of OG member sequences to FASTA files...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output FASTA files will be saved to: {args.output_dir}")

    try:
        df_curated_ogs = pd.read_csv(args.og_list_csv)
        if args.og_id_col not in df_curated_ogs.columns:
            logger.error(f"Column '{args.og_id_col}' not found in '{args.og_list_csv}'. Exiting.")
            return
        ogs_to_process = df_curated_ogs[args.og_id_col].unique()
        logger.info(f"Found {len(ogs_to_process)} unique OGs to process from '{args.og_list_csv}'.")
    except FileNotFoundError:
        logger.error(f"Curated OG list file '{args.og_list_csv}' not found. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error reading curated OG list '{args.og_list_csv}': {e}", exc_info=True)
        return

    if not ogs_to_process.size: 
        logger.warning("No OGs found in the curated list. Exiting.")
        return

    try:
        logger.info("Loading proteome database... This might take a moment.")
        cols_to_load = [args.protein_id_col, args.og_id_col, args.seq_col, args.uniprot_col]
        df_proteome = pd.read_csv(args.proteome_db, usecols=lambda c: c in cols_to_load, low_memory=False)
        logger.info(f"Read {len(df_proteome)} total protein entries from '{args.proteome_db}'.")
    except FileNotFoundError:
        logger.error(f"Proteome database file '{args.proteome_db}' not found. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error reading proteome database '{args.proteome_db}': {e}", exc_info=True)
        return

    ogs_processed_count = 0
    for og_id in tqdm(ogs_to_process, desc="Processing OGs for FASTA export"):
        if pd.isna(og_id):
            logger.warning("Encountered a NaN OG_ID in the curated list. Skipping.")
            continue

        og_members_df = df_proteome[df_proteome[args.og_id_col] == og_id]

        if og_members_df.empty:
            logger.warning(f"No members found in proteome database for OG: {og_id}. Skipping.")
            continue

        fasta_records = []
        for _, row in og_members_df.iterrows():
            protein_id = str(row[args.protein_id_col])
            sequence_str = row.get(args.seq_col)
            uniprot_ac = row.get(args.uniprot_col)

            if pd.isna(sequence_str) or not isinstance(sequence_str, str) or not sequence_str.strip():
                logger.warning(f"Protein {protein_id} in OG {og_id} has an invalid or empty sequence. Skipping.")
                continue
            
            description_parts = []
            if pd.notna(uniprot_ac) and str(uniprot_ac).strip():
                description_parts.append(f"UniProtKB_AC={str(uniprot_ac).strip()}")
            description_parts.append(f"OG_ID={og_id}")
            description_str = " ".join(description_parts)

            record = SeqRecord(Seq(str(sequence_str)), id=protein_id, description=description_str)
            fasta_records.append(record)

        if fasta_records:
            safe_og_id_filename = str(og_id).replace('/', '_')
            output_fasta_file = args.output_dir / f"{safe_og_id_filename}.fasta"
            SeqIO.write(fasta_records, output_fasta_file, "fasta")
            ogs_processed_count += 1

    logger.info(f"--- OG Member Sequence Extraction Finished ---")
    logger.info(f"Successfully wrote FASTA files for {ogs_processed_count} OGs.")

if __name__ == "__main__":
    logger = setup_logging()
    args = parse_arguments()
    main(args, logger)
