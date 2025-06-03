#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to filter the output of extract_og_annotations_script.py.

This script reads the 'og_annotation_term_frequencies.csv' file and
filters out common, less informative annotation terms based on specified criteria.

Input file (expected in ../analysis_outputs/ relative to script location):
- og_annotation_term_frequencies.csv

Output file (in ../analysis_outputs/ relative to script location):
- og_annotation_term_frequencies_filtered.csv
"""

import pandas as pd
from pathlib import Path
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

INPUT_ANNOTATION_COUNTS_FILE = ANALYSIS_OUTPUTS_DIR / "og_annotation_term_frequencies_filtered_v3.csv"
OUTPUT_FILTERED_ANNOTATION_COUNTS_FILE = ANALYSIS_OUTPUTS_DIR / "og_annotation_term_frequencies_filtered_v4.csv"

# --- Filtering Criteria ---
# Terms to remove if the annotation *starts with* (case-insensitive)
PREFIX_FILTERS = [
    "MAG "
]

# Terms to remove if the annotation is an *exact match* (case-insensitive)
EXACT_MATCH_FILTERS = [
    "domain_containing_protein",
    "hypothetical protein",
    "uncharacterized protein",
    "unnamed protein product",
    "unnamed protein product, partial",
    "domaincontaining",
    "putative",
    "MAG",
    "domain",
    "tetratricopeptide_repeat_protein",
    "radical_SAM_protein",
    "ABC_transporter_permease",
    "ABC transporter protein",
    "ankyrin_repeat_protein",
    "predicted protein",
    "glycosyltransferase",
    "MFS_transporter",
    "GTPbinding_protein",
    "alpha_beta_fold_hydrolase",
    "HAD_family_hydrolase",
    "righthanded_parallel_betahelix_repeatcontaining_protein",
    "transposase",
    "CPBP_family_intramembrane_metalloprotease",
    "sitespecific_DNAmethyltransferase",
    "ArsR_family_transcriptional_regulator",
    "DMT_family_transporter",
    "AMPbinding_protein",
    "response_regulator",
    "SDR_family_oxidoreductase",
    "ATPbinding_protein",
    "helicase",
    "sulfotransferase",
    "leucinerich_repeat_protein",
    "DEAD_DEAH_box_helicase",
    "alpha_beta_hydrolase",
    "protein of unknown function",
    "Gfo_Idh_MocA_family_oxidoreductase",
    "methyltransferase",
    "acyltransferase",
    "amidohydrolase",
    "phosphoribosyltransferase",
    "RNAbinding_protein",
    "serine_threonine_protein_kinase",
    "GTPase",
    "histidine kinase",
    "protein" # Filter out if the annotation is just "protein"
]

# Terms to remove if the annotation *contains* (case-insensitive)
# Useful for broader categories of generic terms
CONTAINS_FILTERS = [
    "hypothetical", # Catches variations like "conserved hypothetical protein"
    "unknown function",
    "MAG",
    "ATPbinding_protein"
    "cationtranslocating_Ptype_ATPase",
    "MBL_fold_metallohydrolase",
    "putative",
    "TATAboxbinding_protein",
    "ABC transporter protein",
    "type_II_methionyl_aminopeptidase",
    "sulfataselike_hydrolase_transferase",
    "domain",
    "MFS_transporter",
    "ABC_transporter_ATPbinding_protein",
    "metallophosphoesterase",
    "probable D-lactate dehydrogenase, mitochondrial",
    "non-specific lipid-transfer protein",
    "ankyrin_repeatcontaining_protein",
    "transcriptional_regulator",
    "DUF",
    "alpha_beta_hydrolase",
    "NAD_P_binding_protein",
    "cationtransporting_Ptype_ATPase",
    "metallophosphoesterase",
    "_family_",
    "FADdependent_oxidoreductase",
    "Sbox",
    "putative",
    "ATP-binding ABC transporter",
    "ABC transporter, putative",
    "ATPdependent_DNA_ligase",
    "helixturnhelix",
    "NAD_P_dependent_oxidoreductase",
    "predicted protein, partial",
    "domaincontaining",
    "unnamed",
    "uncharacterized" # Catches "uncharacterized conserved protein", etc.
]

# Terms to remove if the annotation *ends with* (case-insensitive)
SUFFIX_FILTERS = [
    "_family_protein", # e.g., "ABC_family_protein"
    " family protein",  # e.g., "ABC family protein"
    " family", # e.g., "kinase family" - might be too broad, review carefully
    " domain" # e.g., "ABC domain" - might be too broad, review carefully
]


def load_annotation_counts():
    """Loads the annotation term frequencies CSV."""
    if not INPUT_ANNOTATION_COUNTS_FILE.exists():
        logger.error(f"Input file not found: {INPUT_ANNOTATION_COUNTS_FILE}")
        return None
    logger.info(f"Loading annotation term frequencies from: {INPUT_ANNOTATION_COUNTS_FILE}")
    try:
        df = pd.read_csv(INPUT_ANNOTATION_COUNTS_FILE)
        logger.info(f"Successfully loaded {len(df)} annotation terms.")
        return df
    except Exception as e:
        logger.error(f"Error loading {INPUT_ANNOTATION_COUNTS_FILE}: {e}")
        return None

def filter_annotations(df_counts):
    """Applies filtering rules to the annotation terms."""
    if df_counts is None or df_counts.empty:
        logger.warning("Input DataFrame is empty or None. No filtering applied.")
        return pd.DataFrame()

    # Ensure 'Annotation_Term' column exists
    if 'Annotation_Term' not in df_counts.columns:
        logger.error("'Annotation_Term' column not found in the input DataFrame. Cannot filter.")
        return df_counts # Return original if column is missing

    # Create a boolean Series to mark rows for removal, initialize to False
    rows_to_remove = pd.Series([False] * len(df_counts), index=df_counts.index)
    
    # Convert Annotation_Term to lowercase for case-insensitive matching
    annotation_terms_lower = df_counts['Annotation_Term'].astype(str).str.lower().str.strip()

    # Apply prefix filters
    for prefix in PREFIX_FILTERS:
        prefix_lower = prefix.lower()
        condition = annotation_terms_lower.str.startswith(prefix_lower)
        count = condition.sum()
        if count > 0:
            logger.info(f"Marking {count} rows for removal due to prefix: '{prefix}'")
        rows_to_remove |= condition

    # Apply exact match filters
    for term in EXACT_MATCH_FILTERS:
        term_lower = term.lower()
        condition = (annotation_terms_lower == term_lower)
        count = condition.sum()
        if count > 0:
            logger.info(f"Marking {count} rows for removal due to exact match: '{term}'")
        rows_to_remove |= condition
        
    # Apply contains filters
    for term_fragment in CONTAINS_FILTERS:
        term_fragment_lower = term_fragment.lower()
        # Avoid removing terms that are *just* the fragment but could be part of a more specific term
        # This 'contains' is broad, so ensure it's not too aggressive.
        # For example, if "protein" is in CONTAINS_FILTERS, it would remove "kinase protein".
        # It's generally better to use exact match for "protein" if that's the only word.
        condition = annotation_terms_lower.str.contains(term_fragment_lower, regex=False)
        count = condition.sum()
        if count > 0:
            logger.info(f"Marking {count} rows for removal due to containing: '{term_fragment}'")
        rows_to_remove |= condition

    # Apply suffix filters
    for suffix in SUFFIX_FILTERS:
        suffix_lower = suffix.lower()
        condition = annotation_terms_lower.str.endswith(suffix_lower)
        count = condition.sum()
        if count > 0:
            logger.info(f"Marking {count} rows for removal due to suffix: '{suffix}'")
        rows_to_remove |= condition

    # Filter out the marked rows
    df_filtered = df_counts[~rows_to_remove].copy() # Use .copy() to avoid SettingWithCopyWarning

    logger.info(f"Original number of terms: {len(df_counts)}")
    logger.info(f"Number of terms after filtering: {len(df_filtered)}")
    logger.info(f"Number of terms removed: {len(df_counts) - len(df_filtered)}")
    
    return df_filtered

def main():
    """Main function to orchestrate annotation filtering."""
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_annotation_counts = load_annotation_counts()
    
    if df_annotation_counts is None:
        return

    df_filtered_counts = filter_annotations(df_annotation_counts)
    
    if not df_filtered_counts.empty:
        # Sort by frequency again after filtering
        df_filtered_counts.sort_values(by='Frequency', ascending=False, inplace=True)
        logger.info(f"Saving filtered annotation term frequencies to: {OUTPUT_FILTERED_ANNOTATION_COUNTS_FILE}")
        df_filtered_counts.to_csv(OUTPUT_FILTERED_ANNOTATION_COUNTS_FILE, index=False)
        logger.info(f"Filtered list contains {len(df_filtered_counts)} unique annotation terms.")
        logger.info("Review this filtered file to further refine your keyword selection.")
    elif df_annotation_counts is not None and not df_annotation_counts.empty : 
        logger.info("All annotation terms were filtered out. Output file will be empty or not created if it was empty to begin with.")
        # Create an empty file with headers if all were filtered
        pd.DataFrame(columns=df_annotation_counts.columns).to_csv(OUTPUT_FILTERED_ANNOTATION_COUNTS_FILE, index=False)
    else: 
        logger.info("Initial annotation term list was empty. No filtering performed.")


    logger.info("--- Annotation Filtering Script Finished ---")

if __name__ == "__main__":
    main()
