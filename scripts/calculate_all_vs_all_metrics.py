# scripts/calculate_all_vs_all_metrics.py
#
# Description:
# V3 Workflow: Step 8
# This script performs the core, all-vs-all structural comparison for each OG.
# For each target OG, it identifies all member proteins with high-quality
# structures (pLDDT > 70) and runs TM-align on every unique pair. The process
# is highly parallelized to handle the large number of comparisons.
# v2: Includes more detailed progress logging.

import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import subprocess
import re
import itertools
import concurrent.futures

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def parse_tmalign_output(output: str) -> dict:
    """Parses the text output of TMalign to extract key metrics."""
    results = {"TMscore_1": np.nan, "TMscore_2": np.nan, "RMSD": np.nan}
    try:
        tm_score_1_match = re.search(r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_1", output)
        if tm_score_1_match:
            results["TMscore_1"] = float(tm_score_1_match.group(1))
        
        tm_score_2_match = re.search(r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_2", output)
        if tm_score_2_match:
            results["TMscore_2"] = float(tm_score_2_match.group(1))
        
        aligned_length_rmsd_match = re.search(r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+)", output)
        if aligned_length_rmsd_match:
            results["RMSD"] = float(aligned_length_rmsd_match.group(2))
        
        if pd.isna(results["TMscore_1"]):
            raise ValueError("Could not parse TM-score from output.")
            
    except Exception as e:
        logging.warning(f"Failed to parse TM-align output. Error: {e}")
    return results

def run_tmalign_pair(protein1_path: Path, protein2_path: Path, tmalign_exe: str = "TMalign") -> dict:
    """Runs TM-align on a single pair of PDB files."""
    if not protein1_path.exists() or not protein2_path.exists():
        return {"Error": "PDB_Not_Found"}
        
    command = [tmalign_exe, str(protein1_path), str(protein2_path)]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        return parse_tmalign_output(process.stdout)
    except subprocess.CalledProcessError as e:
        logging.debug(f"TM-align failed for {protein1_path.name} vs {protein2_path.name}. Stderr: {e.stderr.strip()}")
        return {"Error": "TMalign_Execution_Error"}
    except Exception as e:
        logging.warning(f"An unexpected error occurred running TMalign: {e}")
        return {"Error": "Python_Exception"}

def main():
    parser = argparse.ArgumentParser(description="Calculate all-vs-all structural metrics for a list of OGs.")
    parser.add_argument("--og-list-csv", type=Path, required=True, help="Path to the list of target OG IDs.")
    parser.add_argument("--proteome-db", type=Path, required=True, help="Path to the master proteome database.")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory containing AlphaFold PDB files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to save the output CSV of all-vs-all results.")
    parser.add_argument("--plddt-threshold", type=float, default=70.0, help="Minimum pLDDT score for a structure to be included.")
    parser.add_argument("--max-workers", type=int, default=12, help="Number of parallel processes to use.")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    logging.info("--- V3 All-vs-All Structural Metrics Calculation ---")
    logging.info("Step 1: Loading required data...")
    try:
        target_ogs = pd.read_csv(args.og_list_csv)['OG_ID'].tolist()
        df_proteome = pd.read_csv(args.proteome_db, usecols=['ProteinID', 'OG_ID', 'UniProtKB_AC', 'Avg_pLDDT'], low_memory=False)
    except Exception as e:
        logging.error(f"Failed to load input files: {e}")
        return

    logging.info("Step 2: Filtering for high-quality structures...")
    df_proteome['Avg_pLDDT'] = pd.to_numeric(df_proteome['Avg_pLDDT'], errors='coerce')
    hq_proteome = df_proteome[df_proteome['Avg_pLDDT'] >= args.plddt_threshold].copy()
    hq_proteome.dropna(subset=['UniProtKB_AC'], inplace=True)
    protein_to_uniprot = pd.Series(hq_proteome.UniProtKB_AC.values, index=hq_proteome.ProteinID).to_dict()
    logging.info(f"Found {len(hq_proteome)} total high-quality proteins to consider.")

    with open(args.output_csv, 'w') as f:
        f.write("OG_ID,Protein1,Protein2,TMscore_1,TMscore_2,RMSD,Error\n")

    logging.info("Step 3: Beginning per-OG processing...")
    for og_id in tqdm(target_ogs, desc="Processing OGs"):
        logging.info(f"--- Starting OG: {og_id} ---")
        og_members = hq_proteome[hq_proteome['OG_ID'] == og_id]['ProteinID'].tolist()
        
        if len(og_members) < 2:
            logging.info(f"Skipping OG {og_id}: Fewer than 2 high-quality structures available.")
            continue
        logging.info(f"Found {len(og_members)} high-quality members for {og_id}.")

        pairs = list(itertools.combinations(og_members, 2))
        logging.info(f"Preparing {len(pairs)} pairwise comparisons for {og_id}.")
        
        tasks = []
        for p1, p2 in pairs:
            uniprot1 = protein_to_uniprot.get(p1)
            uniprot2 = protein_to_uniprot.get(p2)
            if uniprot1 and uniprot2:
                path1 = args.pdb_dir / f"AF-{uniprot1}-F1-model_v4.pdb"
                path2 = args.pdb_dir / f"AF-{uniprot2}-F1-model_v4.pdb"
                if path1.exists() and path2.exists():
                    tasks.append({'protein1_path': path1, 'protein2_path': path2})

        if not tasks:
            logging.warning(f"No valid pairs with existing PDBs found for OG {og_id}.")
            continue
            
        logging.info(f"Running TM-align in parallel for {len(tasks)} pairs in {og_id}...")
        og_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_pair = {executor.submit(run_tmalign_pair, t['protein1_path'], t['protein2_path']): t for t in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(tasks), desc=f"TM-align {og_id}", leave=False):
                task = future_to_pair[future]
                p1_id = task['protein1_path'].stem.replace('AF-', '').replace('-F1-model_v4', '')
                p2_id = task['protein2_path'].stem.replace('AF-', '').replace('-F1-model_v4', '')
                result = future.result()
                
                og_results.append({
                    "OG_ID": og_id,
                    "Protein1": p1_id,
                    "Protein2": p2_id,
                    **result
                })

        if og_results:
            df_og_results = pd.DataFrame(og_results)
            df_og_results.to_csv(args.output_csv, mode='a', header=False, index=False, float_format='%.4f')
            logging.info(f"Finished processing OG {og_id}. Appended {len(og_results)} results to file.")

    logging.info(f"All-vs-all structural comparison complete. Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
