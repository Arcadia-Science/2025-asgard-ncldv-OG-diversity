#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for Phase 1.1: Orthogroup (OG) and Anchor Protein Identification.
(v7.2 - Corrected column names IPR_Signatures, Euk_Hit_SourceDB handling, verified tqdm)

This script processes a proteome database to:
1. Incorporate a predefined list of pilot OGs and their anchors.
2. Identify new candidate OGs and select the best anchor protein for each based on a
   tiered approach, strongly preferring anchors with locally downloaded structures.
3. Filter OGs for taxonomic group (Asgard/GV) and minimum size.
4. Output a curated list of OGs/anchors and lists of structures to download/log.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from tqdm.auto import tqdm # Standard tqdm import for direct use

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment for maximum verbosity

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent 
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
PILOT_ANCHORS_FILE = DATA_DIR / "chosen_functional_anchors_summary.csv"
STRUCTURE_LOG_FILE = DATA_DIR / "structure_download_log_v2.csv" 

LOCAL_PDB_DIR = DATA_DIR / "downloaded_structures" / "rcsb_pdb"
LOCAL_AFDB_DIR = DATA_DIR / "downloaded_structures" / "alphafold_structures" # Corrected based on user's setup
AFDB_MODEL_VERSION_FOR_FILENAME = "v4" 

OUTPUT_CURATED_LIST_FILE = ANALYSIS_OUTPUTS_DIR / "curated_og_anchor_list_v7.2.csv" 
OUTPUT_PDB_TO_DOWNLOAD_FILE = ANALYSIS_OUTPUTS_DIR / "pdb_anchors_to_download_v7.2.csv" 
OUTPUT_AFDB_MISSING_LOG_FILE = ANALYSIS_OUTPUTS_DIR / "afdb_anchors_to_download_or_log_v7.2.csv"

MIN_OG_MEMBERS = 20
TARGET_GROUPS = ["Asgard", "GV"] 

ANNOTATION_KEYWORDS = [ # User-customized list
    "cofilin", "vps4", "vps2", "vps20", "vps22", "vps25", "vps32", "chmp", "escrt", "actin", "tubulin",
    "profilin", "roadblock", "dynein", "dynamin", "kinesin", "arp2/3", "caspase", "ubiquitin", "sumo", "atpase",
    "helicase", "polymerase", "ligase", "nuclease", "synthetase", "kinase", "phosphatase", "protease",
    "methyltransferase", "acetyltransferase", "transferase", "transporter", "receptor", "channel",
    "capsid", "major capsid protein", "minor capsid protein", "dna polymerase", "rna polymerase",
    "reverse transcriptase", "ribosomal protein", "class_i_samdependent_methyltransferase",
    "4fe4s_binding_protein", "acetyl-coa synthetase", "fes_binding_protein", "fadbinding_oxidoreductase",
    "aminotransferase_class_i_iifold_pyridoxal_phosphatedependent_enzyme", "coabinding_protein",
    "aldo_keto_reductase", "aldehyde_ferredoxin_oxidoreductase", "hydrogenase_ironsulfur_subunit",
    "transcription_initiation_factor_iib", "50s_ribosomebinding_gtpase",
    "abc_transporter_substratebinding_protein", "llm_class_flavindependent_oxidoreductase",
    "aminotransferase_class_vfold_plpdependent_enzyme", "abc_transporter_permease_subunit",
    "pyruvate_ferredoxin_oxidoreductase", "fadbinding_protein", "n6_dna_methylase",
    "acylcoa_dehydrogenase", "dna_repair_and_recombination_protein_rada", "2hydroxyacylcoa_dehydratase",
    "2oxoacid_acceptor_oxidoreductase_subunit_alpha",
    "aminotransferase_class_iiifold_pyridoxal_phosphatedependent_enzyme", "nacetyltransferase",
    "myo-inositol-1-phosphate synthase", "pyridoxalphosphate_dependent_enzyme",
    "adpribosylation_factorlike_protein", "4-coumarate--coa ligase 1",
    "ironcontaining_alcohol_dehydrogenase", "cobcom_heterodisulfide_reductase_subunit_b",
    "dnadirected_rna_polymerase_subunit_a", "thioredoxin", "coenzyme_f4200_lglutamate_ligase",
    "long-chain-fatty-acid--coa ligase", "longchain_fatty_acidcoa_ligase", "poxvirus a32 protein",
    "trnaintron_lyase", "peptidylprolyl_isomerase", "dna-binding_protein", "ADP-ribosylation factor-like protein Arf6",
    "arf1p", "cetz", "Rab family GTPase", "Rab-family small GTPase", "small GTPase Rab7",
    "dynamin_family_GTPase", 
]
INTERPRO_KEYWORDS = [ # User-customized list
    "IPR003593", "IPR001806", "IPR027417", "IPR016024", "IPR006092", "IPR015421", "IPR036395",
    "IPR013785", "IPR002000", "IPR001680", "IPR001250", "IPR011056"
]

PROTEOME_COLS_TO_LOAD = [ 
    'ProteinID', 'UniProtKB_AC', 'Orthogroup', 'Source_Protein_Annotation', 'Group', 'Avg_pLDDT',
    'SeqSearch_PDB_Hit', 'SeqSearch_AFDB_Hit', 
    'IPR_Signatures',             # CORRECTED: Was IPR_Domains_Summary
    'Euk_Hit_Protein_Name', 
    # 'Euk_Hit_SourceDB',         # REMOVED: User confirmed this column does not exist
    'Num_OG_Sequences'
]

def get_expected_afdb_filename(uniprot_ac):
    if pd.notna(uniprot_ac) and isinstance(uniprot_ac, str) and uniprot_ac.strip():
        return f"AF-{uniprot_ac.strip()}-F1-model_{AFDB_MODEL_VERSION_FOR_FILENAME}.pdb"
    return None

def get_expected_pdb_filename(pdb_id_val):
    if pd.notna(pdb_id_val) and isinstance(pdb_id_val, str):
        pdb_id_clean = pdb_id_val.strip().split('_')[0].split(';')[0].upper()
        if len(pdb_id_clean) == 4: 
            return f"{pdb_id_clean}.pdb"
    return None

def load_data_and_initial_paths():
    logger.info(f"Loading proteome database from: {PROTEOME_DB_FILE}")
    # Ensure all columns in PROTEOME_COLS_TO_LOAD are attempted
    # If a column is missing, pandas will raise an error if 'usecols' doesn't find it,
    # unless the lambda c: c in PROTEOME_COLS_TO_LOAD handles it by simply not finding it.
    # It's better to load what's available and then check.
    try:
        df_proteome_full = pd.read_csv(PROTEOME_DB_FILE, low_memory=True)
        cols_to_actually_load = [col for col in PROTEOME_COLS_TO_LOAD if col in df_proteome_full.columns]
        missing_requested_cols = [col for col in PROTEOME_COLS_TO_LOAD if col not in df_proteome_full.columns]
        if missing_requested_cols:
            logger.warning(f"Requested columns not found in {PROTEOME_DB_FILE}: {missing_requested_cols}. These will be ignored.")
        df_proteome = df_proteome_full[cols_to_actually_load]

    except Exception as e:
        logger.error(f"Could not load proteome database {PROTEOME_DB_FILE}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), {} # Return empty DataFrames on critical error
    
    logger.info(f"Loading pilot anchors from: {PILOT_ANCHORS_FILE}")
    df_pilot_anchors = pd.read_csv(PILOT_ANCHORS_FILE)
    
    initial_structure_paths_from_log = {} 
    if STRUCTURE_LOG_FILE.exists():
        logger.info(f"Loading structure log from: {STRUCTURE_LOG_FILE}")
        try:
            df_structure_log = pd.read_csv(STRUCTURE_LOG_FILE)
            if 'ProteinID' in df_structure_log.columns and 'Saved_Filename' in df_structure_log.columns and 'Retrieved_From' in df_structure_log.columns:
                for _, row in df_structure_log.iterrows():
                    protein_id, filename, source = row['ProteinID'], row['Saved_Filename'], str(row['Retrieved_From']).lower()
                    if pd.notna(protein_id) and pd.notna(filename):
                        if "alphafold" in source:
                            initial_structure_paths_from_log[protein_id] = str(LOCAL_AFDB_DIR / filename)
                        elif "rcsb" in source or "pdb" in source:
                            initial_structure_paths_from_log[protein_id] = str(LOCAL_PDB_DIR / filename)
            else:
                logger.warning(f"Structure log {STRUCTURE_LOG_FILE} is missing required columns (ProteinID, Saved_Filename, Retrieved_From).")
        except Exception as e:
            logger.error(f"Error reading structure log {STRUCTURE_LOG_FILE}: {e}")
    else:
        logger.warning(f"Structure log file {STRUCTURE_LOG_FILE} not found. Initial paths from log will be empty.")
        
    return df_proteome, df_pilot_anchors, initial_structure_paths_from_log

def get_anchor_functional_annotation(protein_row_series, pilot_anchor_info_series=None):
    if pilot_anchor_info_series is not None and 'Protein Name/Family' in pilot_anchor_info_series.index and pd.notna(pilot_anchor_info_series['Protein Name/Family']):
        return pilot_anchor_info_series['Protein Name/Family']
    if isinstance(protein_row_series, pd.Series):
        # Order of preference for annotation
        for col in ['Source_Protein_Annotation', 'Euk_Hit_Protein_Name', 'IPR_Signatures']: # CORRECTED to IPR_Signatures
            if col in protein_row_series.index and pd.notna(protein_row_series[col]):
                return protein_row_series[col]
    return "Unknown Function"

def process_pilot_ogs(df_pilot_anchors, df_proteome, initial_structure_paths_from_log, pdb_to_download_list):
    logger.info("Processing pilot OGs...")
    curated_pilot_list = []
    for _, pilot_row in df_pilot_anchors.iterrows():
        og_id, anchor_pid = pilot_row['Orthogroup'], pilot_row['ProteinID']
        anchor_details_series_df = df_proteome[df_proteome['ProteinID'] == anchor_pid]
        if anchor_details_series_df.empty: 
            logger.warning(f"Pilot anchor {anchor_pid} for OG {og_id} not in proteome DB. Skipping."); continue
        anchor_details_series = anchor_details_series_df.iloc[0]

        selection_tier = "Pilot_Study_Anchor"
        anchor_path = initial_structure_paths_from_log.get(anchor_pid) 

        if not anchor_path or not Path(anchor_path).exists(): 
            logger.debug(f"Pilot anchor {anchor_pid} for OG {og_id}: Not found in log or file at log path missing. Checking local dirs directly.")
            pdb_id_val = anchor_details_series.get('SeqSearch_PDB_Hit')
            expected_pdb_fn = get_expected_pdb_filename(pdb_id_val)
            if expected_pdb_fn and (LOCAL_PDB_DIR / expected_pdb_fn).exists():
                anchor_path = str(LOCAL_PDB_DIR / expected_pdb_fn)
                selection_tier = "Pilot_PDB_Local"
            elif 'UniProtKB_AC' in anchor_details_series and pd.notna(anchor_details_series['UniProtKB_AC']):
                expected_af_fn = get_expected_afdb_filename(anchor_details_series['UniProtKB_AC'])
                if expected_af_fn and (LOCAL_AFDB_DIR / expected_af_fn).exists():
                    anchor_path = str(LOCAL_AFDB_DIR / expected_af_fn)
                    selection_tier = "Pilot_AFDB_Local"
            elif expected_pdb_fn: 
                pdb_id_to_download = expected_pdb_fn.split('.')[0]
                anchor_path = f"{pdb_id_to_download}_Needs_Download"
                selection_tier = "Pilot_PDB_Needs_Download"
                pdb_to_download_list.append({'OG_ID': og_id, 'Anchor_ProteinID': anchor_pid, 'PDB_ID': pdb_id_to_download, 'Reason': selection_tier})
            else: 
                anchor_path = "Structure_File_Not_Found_Locally_Or_In_Log"
                logger.warning(f"  Pilot OG {og_id}: Anchor {anchor_pid} structure not found in log or locally.")
        
        func_annot = get_anchor_functional_annotation(anchor_details_series, pilot_row)
        plddt_val = anchor_details_series.get('Avg_pLDDT', pilot_row.get('Avg_pLDDT', np.nan))
        plddt = float(plddt_val) if pd.notna(plddt_val) else np.nan

        og_members = df_proteome[df_proteome['Orthogroup'] == og_id]
        num_asg, num_gv = (og_members[og_members['Group'] == g].shape[0] if 'Group' in og_members.columns else 0 for g in ['Asgard', 'GV'])
        
        curated_pilot_list.append({
            'OG_ID': og_id, 'Anchor_ProteinID': anchor_pid, 'Anchor_Structure_Path': anchor_path,
            'Anchor_Functional_Annotation': func_annot, 'Anchor_Avg_pLDDT': plddt,
            'Anchor_Selection_Tier': selection_tier, 'Num_ASG_Members': num_asg,
            'Num_GV_Members': num_gv, 'Total_OG_Members': len(og_members)})
    logger.info(f"Processed {len(curated_pilot_list)} pilot OGs.")
    return pd.DataFrame(curated_pilot_list)

def select_new_ogs_and_anchors(df_proteome, pilot_og_ids, pdb_to_download_list, afdb_to_download_or_log_list):
    logger.info("Selecting new OGs and anchors, prioritizing local files...")
    df_new_ogs_domain = df_proteome[~df_proteome['Orthogroup'].isin(pilot_og_ids)].copy()
    if df_new_ogs_domain.empty: logger.info("No proteins for new OG selection after removing pilot OGs."); return pd.DataFrame()
    
    num_og_seq_col = 'Num_OG_Sequences_Calculated'
    if 'Orthogroup' not in df_new_ogs_domain.columns: logger.error("Orthogroup column missing, cannot calculate OG sizes."); return pd.DataFrame()
    df_new_ogs_domain.loc[:, num_og_seq_col] = df_new_ogs_domain.groupby('Orthogroup')['ProteinID'].transform('count')
    
    group_col = 'Group'
    if group_col not in df_new_ogs_domain.columns: logger.warning(f"'{group_col}' column missing. Cannot filter by target groups."); 
    df_pre_filtered_ogs = df_new_ogs_domain[
        (df_new_ogs_domain[num_og_seq_col] >= MIN_OG_MEMBERS) & 
        (df_new_ogs_domain[group_col].isin(TARGET_GROUPS) if group_col in df_new_ogs_domain.columns else True) 
    ].copy()
    logger.info(f"Proteins after MIN_OG_MEMBERS & TARGET_GROUPS filter: {len(df_pre_filtered_ogs)}")
    if df_pre_filtered_ogs.empty: logger.info("No proteins left after size/group filtering for new OGs."); return pd.DataFrame()

    potential_anchors_list = []
    unique_ogs_to_process = df_pre_filtered_ogs['Orthogroup'].unique()
    logger.info(f"Processing {len(unique_ogs_to_process)} potential new OGs.")

    plddt_col = 'Avg_pLDDT'
    if plddt_col in df_pre_filtered_ogs.columns: df_pre_filtered_ogs.loc[:, plddt_col] = pd.to_numeric(df_pre_filtered_ogs[plddt_col], errors='coerce')
    else: logger.warning(f"'{plddt_col}' column missing. Creating with NaNs."); df_pre_filtered_ogs.loc[:, plddt_col] = np.nan

    for og_id in tqdm(unique_ogs_to_process, desc="Selecting New OG Anchors"): # tqdm used here
        og_data = df_pre_filtered_ogs[df_pre_filtered_ogs['Orthogroup'] == og_id]
        selected_anchor_series, selection_tier, anchor_path = None, "None", "Not_Applicable"

        # Tier 1: Local PDB Hit
        if 'SeqSearch_PDB_Hit' in og_data.columns:
            pdb_candidates = []
            for _, row in og_data[og_data['SeqSearch_PDB_Hit'].notna()].iterrows():
                expected_fn = get_expected_pdb_filename(row['SeqSearch_PDB_Hit'])
                if expected_fn and (LOCAL_PDB_DIR / expected_fn).exists():
                    candidate_data = row.to_dict()
                    candidate_data['local_path'] = str(LOCAL_PDB_DIR / expected_fn)
                    pdb_candidates.append(candidate_data)
            if pdb_candidates:
                selected_anchor_series = pd.DataFrame(pdb_candidates).sort_values(by=plddt_col, ascending=False, na_position='last').iloc[0]
                selection_tier, anchor_path = "Tier1_PDB_Local", selected_anchor_series['local_path']
                logger.debug(f"  OG {og_id}: Selected Tier1_PDB_Local: {selected_anchor_series['ProteinID']}")

        # Tier 1b: PDB Hit Needs Download
        if selected_anchor_series is None and 'SeqSearch_PDB_Hit' in og_data.columns:
            pdb_needs_download_candidates = []
            for _, row in og_data[og_data['SeqSearch_PDB_Hit'].notna()].iterrows():
                pdb_id_for_file = get_expected_pdb_filename(row['SeqSearch_PDB_Hit']) 
                if pdb_id_for_file: 
                    candidate_data = row.to_dict()
                    candidate_data['_pdb_id_to_download_'] = pdb_id_for_file.split('.')[0] 
                    pdb_needs_download_candidates.append(candidate_data)
            if pdb_needs_download_candidates:
                candidate_series = pd.DataFrame(pdb_needs_download_candidates).sort_values(by=plddt_col, ascending=False, na_position='last').iloc[0]
                pdb_id_to_flag = candidate_series['_pdb_id_to_download_']
                selected_anchor_series, anchor_path, selection_tier = candidate_series, f"{pdb_id_to_flag}_Needs_Download", "Tier1_PDB_Needs_Download"
                pdb_to_download_list.append({'OG_ID': og_id, 'Anchor_ProteinID': selected_anchor_series['ProteinID'], 'PDB_ID': pdb_id_to_flag, 'Reason': selection_tier})
                logger.info(f"  OG {og_id}: Selected Tier1_PDB_Needs_Download: {selected_anchor_series['ProteinID']} (PDB_ID: {pdb_id_to_flag})")
        
        # AFDB Tiers
        if selected_anchor_series is None and 'SeqSearch_AFDB_Hit' in og_data.columns and 'UniProtKB_AC' in og_data.columns:
            afdb_potential_all = og_data[og_data['SeqSearch_AFDB_Hit'].notna() & og_data['UniProtKB_AC'].notna()]
            
            for _, protein_row in afdb_potential_all.iterrows():
                if pd.notna(protein_row.get(plddt_col)): 
                    expected_af_fn = get_expected_afdb_filename(protein_row['UniProtKB_AC'])
                    if expected_af_fn and not (LOCAL_AFDB_DIR / expected_af_fn).exists():
                        afdb_to_download_or_log_list.append({
                            'OG_ID': og_id, 'ProteinID': protein_row['ProteinID'], 
                            'UniProtKB_AC': protein_row['UniProtKB_AC'], 'Avg_pLDDT': protein_row.get(plddt_col),
                            'Expected_Filename': expected_af_fn, 'Reason': 'Indicated_AFDB_Not_Local'})
            
            afdb_local_candidates_annot = []
            for _, row in afdb_potential_all.iterrows():
                expected_af_fn = get_expected_afdb_filename(row['UniProtKB_AC'])
                if expected_af_fn and (LOCAL_AFDB_DIR / expected_af_fn).exists():
                    candidate_data = row.to_dict()
                    candidate_data['local_path'] = str(LOCAL_AFDB_DIR / expected_af_fn)
                    afdb_local_candidates_annot.append(candidate_data)

            if afdb_local_candidates_annot:
                df_afdb_local = pd.DataFrame(afdb_local_candidates_annot)
                best_afdb_anchor_series, best_afdb_tier, best_afdb_plddt = None, "None", -1.0

                # Tier definitions: (TierName, AnnotationColumn, Keywords, ExtraConditionColumn, ExtraConditionValue)
                # For Tier 3 (Euk Homolog), we remove the Euk_Hit_SourceDB check as per user.
                tier_definitions = [
                    ("Tier2_AFDB_Local_Annotation", 'Source_Protein_Annotation', ANNOTATION_KEYWORDS, None, None),
                    ("Tier3_AFDB_Local_EukHomolog", 'Euk_Hit_Protein_Name', ANNOTATION_KEYWORDS, None, None), # Euk_Hit_SourceDB condition removed
                    ("Tier4_AFDB_Local_InterPro", 'IPR_Signatures', INTERPRO_KEYWORDS, None, None) # CORRECTED to IPR_Signatures
                ]

                for tier_name, annot_col, keywords_list, extra_cond_col, extra_cond_val in tier_definitions:
                    if annot_col not in df_afdb_local.columns: 
                        logger.debug(f"  OG {og_id}: Annot col '{annot_col}' missing for tier {tier_name}."); continue
                    if extra_cond_col and extra_cond_col not in df_afdb_local.columns: 
                        logger.debug(f"  OG {og_id}: Extra cond col '{extra_cond_col}' missing for tier {tier_name}."); continue

                    tier_matches = df_afdb_local[df_afdb_local[annot_col].astype(str).str.contains('|'.join(keywords_list), case=False, na=False)]
                    if extra_cond_col: 
                        tier_matches = tier_matches[tier_matches[extra_cond_col] == extra_cond_val] # This line is now conditional
                    
                    if not tier_matches.empty:
                        current_tier_best = tier_matches.sort_values(by=plddt_col, ascending=False, na_position='last').iloc[0]
                        current_plddt_val = current_tier_best.get(plddt_col, -1.0) 
                        current_plddt = float(current_plddt_val) if pd.notna(current_plddt_val) else -1.0

                        if current_plddt > best_afdb_plddt: 
                            best_afdb_anchor_series, best_afdb_tier, best_afdb_plddt = current_tier_best, tier_name, current_plddt
                
                if best_afdb_anchor_series is not None:
                    selected_anchor_series, selection_tier, anchor_path = best_afdb_anchor_series, best_afdb_tier, best_afdb_anchor_series['local_path']

        if selected_anchor_series is not None:
            anchor_pid = selected_anchor_series['ProteinID']
            func_annot = get_anchor_functional_annotation(selected_anchor_series)
            plddt_val = selected_anchor_series.get(plddt_col, np.nan)
            plddt_disp = "N/A"
            if pd.notna(plddt_val):
                try: plddt_disp = f"{float(plddt_val):.2f}"
                except: plddt_disp = str(plddt_val)
            
            num_asg, num_gv = (og_data[og_data['Group'] == g].shape[0] if 'Group' in og_data.columns else 0 for g in ['Asgard', 'GV'])
            potential_anchors_list.append({
                'OG_ID': og_id, 'Anchor_ProteinID': anchor_pid, 'Anchor_Structure_Path': anchor_path,
                'Anchor_Functional_Annotation': func_annot, 'Anchor_Avg_pLDDT': float(plddt_val) if pd.notna(plddt_val) else np.nan,
                'Anchor_Selection_Tier': selection_tier, 'Num_ASG_Members': num_asg,
                'Num_GV_Members': num_gv, 'Total_OG_Members': len(og_data)})
            logger.info(f"  OG {og_id}: Selected anchor {anchor_pid} via {selection_tier} (pLDDT: {plddt_disp}). Path: {anchor_path}")
        else: logger.debug(f"  OG {og_id}: No suitable anchor found prioritizing local files.")
            
    logger.info(f"Identified {len(potential_anchors_list)} new OGs with anchors.")
    return pd.DataFrame(potential_anchors_list)

def main():
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PDB_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_AFDB_DIR.mkdir(parents=True, exist_ok=True)

    df_proteome, df_pilot_anchors, initial_structure_paths_from_log = load_data_and_initial_paths()
    if df_proteome.empty: logger.error("Proteome database empty or failed to load critical columns. Exiting."); return
    
    # Check for essential columns after loading and fill if totally missing, to prevent key errors later
    # This is a fallback; ideally, PROTEOME_COLS_TO_LOAD ensures they are attempted.
    essential_cols_for_logic = ['Group', 'Avg_pLDDT', 'UniProtKB_AC', 'SeqSearch_PDB_Hit', 
                                'SeqSearch_AFDB_Hit', 'Source_Protein_Annotation', 
                                'Euk_Hit_Protein_Name', 'IPR_Signatures'] # Euk_Hit_SourceDB removed
    for col in essential_cols_for_logic:
        if col not in df_proteome.columns:
            logger.warning(f"'{col}' column was not found in the loaded proteome data. Creating with NaNs/None. Tier selection relying on this column will be affected.")
            if col == 'Avg_pLDDT': df_proteome.loc[:, col] = np.nan
            else: df_proteome.loc[:, col] = None 

    pdb_to_download_list = []
    afdb_to_download_or_log_list = [] 

    df_curated_pilot = process_pilot_ogs(df_pilot_anchors, df_proteome, initial_structure_paths_from_log, pdb_to_download_list)
    pilot_og_ids = set(df_curated_pilot['OG_ID'].unique()) if not df_curated_pilot.empty else set()
    
    df_curated_new = select_new_ogs_and_anchors(df_proteome, pilot_og_ids, pdb_to_download_list, afdb_to_download_or_log_list)
    
    to_concat = [df for df in [df_curated_pilot, df_curated_new] if df is not None and not df.empty] 
    df_final_curated_list = pd.concat(to_concat, ignore_index=True) if to_concat else pd.DataFrame()

    if not df_final_curated_list.empty:
        df_final_curated_list.drop_duplicates(subset=['OG_ID'], keep='first', inplace=True)
        df_final_curated_list.sort_values(by="OG_ID", inplace=True)
        logger.info(f"Saving final curated list of {len(df_final_curated_list)} OGs to: {OUTPUT_CURATED_LIST_FILE}")
        df_final_curated_list.to_csv(OUTPUT_CURATED_LIST_FILE, index=False, float_format='%.2f')
    else: logger.info("Final curated list empty. No file saved.")

    if pdb_to_download_list:
        pd.DataFrame(pdb_to_download_list).drop_duplicates().to_csv(OUTPUT_PDB_TO_DOWNLOAD_FILE, index=False)
        logger.info(f"Saved {len(pdb_to_download_list)} PDB anchors to download to: {OUTPUT_PDB_TO_DOWNLOAD_FILE}")
    else: logger.info("No PDB anchors flagged for download.")

    if afdb_to_download_or_log_list:
        pd.DataFrame(afdb_to_download_or_log_list).drop_duplicates().to_csv(OUTPUT_AFDB_MISSING_LOG_FILE, index=False, float_format='%.2f')
        logger.info(f"Saved {len(afdb_to_download_or_log_list)} AFDB entries (indicated in DB, not local) to: {OUTPUT_AFDB_MISSING_LOG_FILE}")
    else: logger.info("No AFDB structures found to be missing locally that were indicated in DB.")

    logger.info(f"--- OG and Anchor Selection Script ({Path(__file__).name}) Finished ---")

if __name__ == "__main__":
    main()
