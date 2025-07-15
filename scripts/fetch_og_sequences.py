# scripts/fetch_og_sequences.py
#
# Description:
# V3 Workflow: Step 2
# Extracts member sequences for a specified list of Orthogroups (OGs) and
# saves them into individual FASTA files.

import pandas as pd
import argparse
import logging
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Extract member sequences for a list of OGs into separate FASTA files.")
    parser.add_argument("--og-list-csv", type=Path, required=True, help="Path to the CSV file containing the list of target OG_IDs.")
    parser.add_argument("--proteome-db", type=Path, required=True, help="Path to the master proteome database CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the output FASTA files.")
    args = parser.parse_args()

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output FASTA files will be saved to: {args.output_dir}")

    # --- Load Data ---
    try:
        logging.info(f"Loading list of target OGs from: {args.og_list_csv}")
        df_ogs = pd.read_csv(args.og_list_csv)
        ogs_to_process = df_ogs['OG_ID'].unique()
        logging.info(f"Found {len(ogs_to_process)} unique OGs to process.")

        logging.info(f"Loading proteome database from: {args.proteome_db}")
        cols_to_load = ['ProteinID', 'OG_ID', 'Sequence', 'Virus_Family', 'Virus_Name']
        df_proteome = pd.read_csv(args.proteome_db, usecols=cols_to_load, low_memory=False)
    except Exception as e:
        logging.error(f"Failed to load input files: {e}")
        return

    # --- Process OGs ---
    logging.info("Extracting sequences for each OG...")
    for og_id in tqdm(ogs_to_process, desc="Writing OG FASTA files"):
        og_members_df = df_proteome[df_proteome['OG_ID'] == og_id]

        if og_members_df.empty:
            logging.warning(f"No members found in proteome database for OG: {og_id}. Skipping.")
            continue

        fasta_records = []
        for _, row in og_members_df.iterrows():
            # Basic validation
            if pd.isna(row['ProteinID']) or pd.isna(row['Sequence']):
                continue
            
            protein_id = str(row['ProteinID'])
            virus_family = row['Virus_Family']
            
            # **FIXED**: Construct the ID string correctly
            # Combine ProteinID and Phylum into a single ID string for the header
            # Example: >NWF94827.1|Thorarchaeota
            if pd.notna(virus_family):
                fasta_id = f"{protein_id}|{virus_family}"
            else:
                fasta_id = protein_id

            # **FIXED**: Correctly call the SeqRecord constructor
            record = SeqRecord(
                Seq(str(row['Sequence'])),
                id=fasta_id,
                description="" # Keep description clean
            )
            fasta_records.append(record)

        if fasta_records:
            output_fasta_file = args.output_dir / f"{og_id}.fasta"
            try:
                SeqIO.write(fasta_records, output_fasta_file, "fasta")
            except Exception as e:
                logging.error(f"Error writing FASTA file for OG {og_id}: {e}")
        else:
            logging.warning(f"No valid sequences found to write for OG: {og_id}.")
            
    logging.info("Sequence extraction complete.")

if __name__ == "__main__":
    main()
