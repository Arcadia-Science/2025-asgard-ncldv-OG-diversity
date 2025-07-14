# scripts/extract_hq_structures_for_og.py
#
# Description:
# For a single, specified Orthologous Group (OG), this script identifies all
# member proteins that have a high-quality structure (pLDDT above a given
# threshold) and copies their corresponding PDB files from a local database
# into a new, dedicated directory.

import pandas as pd
import argparse
import logging
from pathlib import Path
import shutil
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Extract high-quality structures for a specific OG.")
    parser.add_argument("--og-id", type=str, required=True, help="The specific OG_ID to process.")
    parser.add_argument("--proteome-db", type=Path, required=True, help="Path to the master proteome database CSV.")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Source directory containing all AlphaFold PDB files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to copy the target PDB files into.")
    parser.add_argument("--plddt-threshold", type=float, default=70.0, help="Minimum average pLDDT score to include a structure.")
    args = parser.parse_args()

    # --- Setup ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory for structures: {args.output_dir}")

    # --- 1. Load Proteome Data ---
    logging.info(f"Loading proteome data from: {args.proteome_db}")
    try:
        cols_to_load = ['OG_ID', 'ProteinID', 'UniProtKB_AC', 'Avg_pLDDT']
        df_proteome = pd.read_csv(args.proteome_db, usecols=cols_to_load)
    except Exception as e:
        logging.error(f"Failed to load proteome database: {e}")
        return

    # --- 2. Filter for Target OG and High-Quality Structures ---
    logging.info(f"Finding high-quality structures for OG: {args.og_id}...")
    
    df_og = df_proteome[df_proteome['OG_ID'] == args.og_id].copy()
    
    if df_og.empty:
        logging.warning(f"No members found for OG {args.og_id} in the proteome database.")
        return
        
    df_og['Avg_pLDDT'] = pd.to_numeric(df_og['Avg_pLDDT'], errors='coerce')
    df_og_hq = df_og[df_og['Avg_pLDDT'] >= args.plddt_threshold].copy()
    df_og_hq.dropna(subset=['UniProtKB_AC'], inplace=True)
    
    if df_og_hq.empty:
        logging.warning(f"No members for OG {args.og_id} met the pLDDT threshold of {args.plddt_threshold}.")
        return
        
    logging.info(f"Found {len(df_og_hq)} high-quality structures to copy.")

    # --- 3. Copy PDB Files ---
    copied_count = 0
    for _, row in tqdm(df_og_hq.iterrows(), total=len(df_og_hq), desc=f"Copying PDBs for {args.og_id}"):
        uniprot_ac = row['UniProtKB_AC']
        pdb_filename = f"AF-{uniprot_ac}-F1-model_v4.pdb"
        source_path = args.pdb_dir / pdb_filename
        dest_path = args.output_dir / pdb_filename
        
        if source_path.exists():
            shutil.copy(source_path, dest_path)
            copied_count += 1
        else:
            logging.warning(f"PDB file not found for {row['ProteinID']} ({uniprot_ac}) at {source_path}")

    logging.info("--- Process Complete ---")
    logging.info(f"Successfully copied {copied_count} PDB files to: {args.output_dir}")

if __name__ == "__main__":
    main()
