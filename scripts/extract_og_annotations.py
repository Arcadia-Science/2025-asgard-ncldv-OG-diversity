#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to extract and count unique annotations from specified OGs
to aid in curating keyword lists for anchor selection.

This script:
1. Loads the main proteome database.
2. Filters for OGs with a minimum number of members (e.g., >30).
3. Collects unique annotations from 'Source_Protein_Annotation',
   'IPR_Domains_Summary', and 'Euk_Hit_Protein_Name' for proteins in these OGs.
4. Counts the frequency of each unique annotation string.
5. Outputs the annotations and their frequencies to a CSV file.
"""

import pandas as pd
from pathlib import Path
import logging
from collections import Counter

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
DATA_DIR = BASE_DIR / "data"
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"

PROTEOME_DB_FILE = DATA_DIR / "proteome_database_v3.2.csv"
OUTPUT_ANNOTATION_COUNTS_FILE = ANALYSIS_OUTPUTS_DIR / "og_annotation_term_frequencies.csv"

MIN_OG_MEMBERS_FOR_ANNOT_EXTRACTION = 30

# Columns from proteome_database_v3.2.csv to extract annotations from
ANNOTATION_SOURCE_COLUMNS = [
    'Source_Protein_Annotation',
    'IPR_Domains_Summary',
    'Euk_Hit_Protein_Name'
]
# Columns needed for filtering and processing
PROTEOME_COLS_TO_LOAD_FOR_ANNOT = ['ProteinID', 'Orthogroup', 'Num_OG_Sequences'] + ANNOTATION_SOURCE_COLUMNS

def load_proteome_data():
    """Loads the proteome database, focusing on necessary columns."""
    logger.info(f"Loading proteome database from: {PROTEOME_DB_FILE}")
    try:
        # Attempt to load only specified columns
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, usecols=lambda c: c in PROTEOME_COLS_TO_LOAD_FOR_ANNOT, low_memory=False)
    except ValueError:
        logger.warning(f"Could not load all specified columns from {PROTEOME_DB_FILE} using usecols. Attempting to load all columns and then select.")
        df_proteome = pd.read_csv(PROTEOME_DB_FILE, low_memory=False)
        # Ensure essential columns are present after loading all
        missing_essential = [col for col in PROTEOME_COLS_TO_LOAD_FOR_ANNOT if col not in df_proteome.columns and col != 'Num_OG_Sequences'] # Num_OG_Sequences can be calculated
        if missing_essential:
            logger.error(f"Essential annotation source columns missing from proteome DB even after loading all: {missing_essential}")
            raise ValueError("Essential annotation source columns missing in proteome DB.")
        df_proteome = df_proteome[[col for col in PROTEOME_COLS_TO_LOAD_FOR_ANNOT if col in df_proteome.columns]]


    # Calculate Num_OG_Sequences if not reliable or missing
    if 'Num_OG_Sequences' not in df_proteome.columns or df_proteome['Num_OG_Sequences'].isnull().any():
        logger.info("Recalculating Num_OG_Sequences as it's missing or has NaNs.")
        if 'Orthogroup' in df_proteome.columns and 'ProteinID' in df_proteome.columns:
            og_counts = df_proteome.groupby('Orthogroup')['ProteinID'].transform('count')
            df_proteome['Num_OG_Sequences'] = og_counts
        else:
            logger.error("'Orthogroup' or 'ProteinID' column missing, cannot calculate Num_OG_Sequences.")
            # As a fallback, allow proceeding but filtering by OG size might not work
            df_proteome['Num_OG_Sequences'] = 0 # Placeholder to avoid errors later

    return df_proteome

def extract_and_count_annotations(df_proteome):
    """
    Filters OGs by size, then extracts and counts annotations from specified columns.
    """
    if 'Num_OG_Sequences' not in df_proteome.columns:
        logger.error("Cannot filter OGs by size as 'Num_OG_Sequences' is not available.")
        df_relevant_proteins = df_proteome # Process all proteins if size filtering fails
    else:
        df_large_ogs = df_proteome[df_proteome['Num_OG_Sequences'] > MIN_OG_MEMBERS_FOR_ANNOT_EXTRACTION]
        if df_large_ogs.empty:
            logger.warning(f"No OGs found with > {MIN_OG_MEMBERS_FOR_ANNOT_EXTRACTION} members. Processing all OGs for annotations.")
            df_relevant_proteins = df_proteome
        else:
            logger.info(f"Found {df_large_ogs['Orthogroup'].nunique()} OGs with > {MIN_OG_MEMBERS_FOR_ANNOT_EXTRACTION} members.")
            df_relevant_proteins = df_large_ogs
    
    all_annotations = []
    for col_name in ANNOTATION_SOURCE_COLUMNS:
        if col_name in df_relevant_proteins.columns:
            # Drop NaNs and convert to string to handle mixed types before collecting
            annotations_from_col = df_relevant_proteins[col_name].dropna().astype(str).tolist()
            all_annotations.extend([(term, col_name) for term in annotations_from_col if term.strip()]) # Store term and its source column
            logger.info(f"Collected {len(annotations_from_col)} non-empty annotations from column: {col_name}")
        else:
            logger.warning(f"Annotation source column '{col_name}' not found in DataFrame.")

    if not all_annotations:
        logger.warning("No annotations collected. Output will be empty.")
        return pd.DataFrame(columns=['Annotation_Term', 'Source_Column', 'Frequency'])

    # Count frequencies of each annotation term
    annotation_counts = Counter(all_annotations)
    
    df_counts = pd.DataFrame([
        {'Annotation_Term': term_source_tuple[0], 'Source_Column': term_source_tuple[1], 'Frequency': count}
        for term_source_tuple, count in annotation_counts.items()
    ])
    
    df_counts.sort_values(by='Frequency', ascending=False, inplace=True)
    
    return df_counts

def main():
    """Main function to orchestrate annotation extraction and counting."""
    ANALYSIS_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df_proteome = load_proteome_data()

    if df_proteome.empty:
        logger.error("Proteome database is empty or failed to load. Exiting.")
        return

    df_annotation_counts = extract_and_count_annotations(df_proteome)
    
    if not df_annotation_counts.empty:
        logger.info(f"Saving annotation term frequencies to: {OUTPUT_ANNOTATION_COUNTS_FILE}")
        df_annotation_counts.to_csv(OUTPUT_ANNOTATION_COUNTS_FILE, index=False)
        logger.info(f"Found {len(df_annotation_counts)} unique annotation terms.")
        logger.info("Review this file to identify common and informative keywords for your anchor selection script.")
    else:
        logger.info("No annotation terms were processed or counted. Output file will not be created or will be empty.")

    logger.info("--- Annotation Extraction Script Finished ---")

if __name__ == "__main__":
    main()
