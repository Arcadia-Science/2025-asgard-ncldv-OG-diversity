# scripts/calculate_sequence_diversity.py
#
# Description:
# V3 Workflow: Step 7
# This script calculates a suite of sequence and phylogenetic diversity metrics
# for a given set of OGs, using their newly generated MSAs and trees. It includes
# per-column MSA entropy, tree diversity (Hill number), and richness-normalized
# tree diversity in a single, parallelized process.

import pandas as pd
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from Bio import Phylo, AlignIO
import numpy as np
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# --- Metric Calculation Functions ---

def calculate_msa_metrics(alignment):
    """Calculates APSI and entropy metrics from a Bio.Align.MultipleSeqAlignment."""
    num_sequences = len(alignment)
    aln_len = alignment.get_alignment_length()
    
    # APSI (Average Pairwise Sequence Identity)
    total_identity_ungapped = 0.0
    num_pairs = 0
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            seq1 = str(alignment[i].seq)
            seq2 = str(alignment[j].seq)
            matches_ungapped, positions_compared = 0, 0
            for k in range(aln_len):
                res1, res2 = seq1[k], seq2[k]
                if res1 != '-' and res2 != '-':
                    positions_compared += 1
                    if res1 == res2:
                        matches_ungapped += 1
            if positions_compared > 0:
                total_identity_ungapped += (matches_ungapped / positions_compared)
            num_pairs += 1
    apsi = (total_identity_ungapped / num_pairs) * 100 if num_pairs > 0 else 0.0

    # Per-column Shannon Entropy
    col_entropies = []
    for i in range(aln_len):
        col = alignment[:, i]
        counts = Counter(c for c in col if c != '-')
        col_entropy = 0.0
        effective_n = sum(counts.values())
        if effective_n > 1:
            for count in counts.values():
                p = count / effective_n
                if p > 0: col_entropy -= p * math.log2(p)
        col_entropies.append(col_entropy)
    
    mean_entropy = np.mean(col_entropies) if col_entropies else 0.0
    
    return {
        "MSA_N_Seqs": num_sequences,
        "MSA_Length": aln_len,
        "APSI_Ungapped": apsi,
        "MSA_Mean_Col_Entropy": mean_entropy
    }

def calculate_tree_metrics(tree):
    """Calculates PD and Hill Diversity (including normalized) from a Bio.Phylo.BaseTree."""
    # Phylogenetic Diversity (Faith's PD)
    total_branch_length = sum(clade.branch_length for clade in tree.find_clades() if clade.branch_length)
    
    # Hill Diversity (q=1)
    terminals = tree.get_terminals()
    branch_lengths = np.array([t.branch_length for t in terminals if t.branch_length is not None and t.branch_length > 1e-9])
    total_terminal_branch_length = np.sum(branch_lengths)
    
    hill_q1 = 0.0
    if total_terminal_branch_length > 1e-9:
        proportions = branch_lengths / total_terminal_branch_length
        proportions = proportions[proportions > 1e-9]
        raw_shannon = -np.sum(proportions * np.log(proportions))
        hill_q1 = np.exp(raw_shannon)
        
    num_tips = tree.count_terminals()
    
    return {
        "Tree_PD_TotalBranchLength": total_branch_length,
        "Tree_N_Tips": num_tips,
        "Tree_Hill_Diversity_q1": hill_q1,
        "Tree_Hill_Diversity_q1_NormByTips": hill_q1 / num_tips if num_tips > 0 else 0.0
    }

def process_single_og(og_id, msa_path, tree_path):
    """Process one OG to calculate all sequence diversity metrics."""
    results = {"OG_ID": og_id, "Error": None}
    
    try:
        # Process MSA
        alignment = AlignIO.read(msa_path, "fasta")
        if not alignment: raise ValueError("MSA file is empty or invalid.")
        results.update(calculate_msa_metrics(alignment))
    except Exception as e:
        results["Error"] = f"MSA_Error: {e}"
        return results

    try:
        # Process Tree
        if tree_path and tree_path.exists():
            tree = Phylo.read(tree_path, "newick")
            if not tree: raise ValueError("Tree file is empty or invalid.")
            results.update(calculate_tree_metrics(tree))
        else:
            results["Error"] = "Tree_File_Not_Found"
    except Exception as e:
        results["Error"] = (results["Error"] or "") + f"; Tree_Error: {e}"
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Calculate sequence and phylogenetic diversity metrics for a set of OGs.")
    parser.add_argument("--msa-dir", type=Path, required=True, help="Directory containing new MSA files.")
    parser.add_argument("--tree-dir", type=Path, required=True, help="Directory containing new tree files.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to save the output CSV with diversity metrics.")
    parser.add_argument("--msa-suffix", type=str, default=".fasta", help="Suffix for MSA files.")
    parser.add_argument("--tree-suffix", type=str, default="_fasttree.nwk", help="Suffix for tree files.")
    parser.add_argument("--max-workers", type=int, default=12, help="Number of parallel processes.")
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    msa_files = list(args.msa_dir.glob(f"*{args.msa_suffix}"))
    if not msa_files:
        logging.error(f"No MSA files found in {args.msa_dir} with suffix {args.msa_suffix}.")
        return

    tasks = []
    for msa_path in msa_files:
        og_id = msa_path.stem.replace(args.msa_suffix, '')
        tree_path = args.tree_dir / f"{og_id}{args.tree_suffix}"
        tasks.append({'og_id': og_id, 'msa_path': msa_path, 'tree_path': tree_path})

    all_results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_og = {executor.submit(process_single_og, **task): task for task in tasks}
        
        for future in tqdm(as_completed(future_to_og), total=len(tasks), desc="Calculating Diversity"):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                og_id = future_to_og[future]['og_id']
                logging.error(f"Task for OG {og_id} generated an exception: {e}")
                all_results.append({"OG_ID": og_id, "Error": "Unhandled_Exception"})

    df_final = pd.DataFrame(all_results)
    df_final.sort_values("OG_ID", inplace=True)
    df_final.to_csv(args.output_csv, index=False, float_format="%.4f")
    logging.info(f"Diversity metrics for {len(df_final)} OGs saved to {args.output_csv}")

if __name__ == "__main__":
    main()
