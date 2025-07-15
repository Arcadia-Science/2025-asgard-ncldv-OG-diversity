# scripts/prepare_og_list.py
#
# Description:
# V3 Workflow: Step 1
# This script prepares the initial list of Orthologous Groups (OGs) for the
# V3 analysis. It filters the main proteome database to identify all OGs
# from the 'Asgard' phylum that contain more than 20 members, creating the
# target list for the downstream all-vs-all structural comparison.

import pandas as pd
import argparse
import logging
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Prepare the list of Asgard OGs with > 20 members for V3 analysis.")
    parser.add_argument("--proteome-db", type=Path, required=True, help="Path to the master proteome database CSV.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to save the output CSV of target OG IDs.")
    parser.add_argument("--min-members", type=int, default=20, help="Minimum number of members for an OG to be included.")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading proteome data from: {args.proteome_db}")
    try:
        # Load only necessary columns to be memory efficient
        df_proteome = pd.read_csv(args.proteome_db, usecols=['OG_ID', 'Asgard_Phylum'], low_memory=False)
    except Exception as e:
        logging.error(f"Failed to load proteome database: {e}")
        return

    # Filter for Asgard OGs
    df_asgard = df_proteome[df_proteome['Asgard_Phylum'].notna()].copy()
    logging.info(f"Found {len(df_asgard)} protein entries belonging to Asgard phyla.")

    # Count members per OG
    og_counts = df_asgard['OG_ID'].value_counts()

    # Filter for OGs with more than the minimum number of members
    ogs_to_keep = og_counts[og_counts > args.min_members].index.tolist()
    logging.info(f"Found {len(ogs_to_keep)} Asgard OGs with > {args.min_members} members.")

    # Save the list to a CSV
    output_df = pd.DataFrame(ogs_to_keep, columns=['OG_ID'])
    try:
        output_df.to_csv(args.output_csv, index=False)
        logging.info(f"Successfully saved list of target OGs to: {args.output_csv}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")

if __name__ == "__main__":
    main()
