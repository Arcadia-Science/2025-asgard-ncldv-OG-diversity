#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Maps UniProt ACs to PDB IDs (RCSB with EBI fallback for mapping) and downloads
PDB structures (RCSB with EBI fallback for 404 on download).
Logs to a dedicated PDB download log.
(v6 - Clean rewrite, simplified PDB ID collection loop)

Input: uniprot_ids_for_pdb_candidates.csv
Output: Downloads PDB files to downloaded_structures/rcsb_pdb/
        Appends to or creates structure_download_log_pdb.csv (new dedicated log)
"""

import pandas as pd
import requests
import os
import time
import logging
from pathlib import Path
from tqdm.auto import tqdm
import json

# --- Setup Logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s.%(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
else:
    logger.setLevel(logging.INFO)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ANALYSIS_OUTPUTS_DIR = BASE_DIR / "analysis_outputs"
DATA_DIR = BASE_DIR / "data"

INPUT_UNIPROT_LIST_CSV = ANALYSIS_OUTPUTS_DIR / "uniprot_ids_for_pdb_candidates.csv"

BASE_STRUCTURE_OUTPUT_DIR = DATA_DIR / "downloaded_structures"
PDB_STRUCTURE_OUTPUT_DIR = BASE_STRUCTURE_OUTPUT_DIR / "rcsb_pdb" # All PDBs go here

# Dedicated log file for this PDB fetching script
PDB_MASTER_LOG_CSV_FILE = DATA_DIR / "structure_download_log_pdb.csv" 

PROTEIN_ID_COL = "ProteinID"
UNIPROT_AC_COL = "UniProtKB_AC"

RCSB_SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
EBI_PDBe_API_UNIPROT_MAPPING_URL = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/summary/{uniprot_ac}"
RCSB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"
EBI_PDB_DOWNLOAD_URL_TEMPLATE = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}.pdb"

REQUEST_DELAY = 0.5 
API_TIMEOUT = 45    # Slightly increased API timeout
DOWNLOAD_TIMEOUT = 90 # Slightly increased download timeout
MAX_PDB_IDS_PER_UNIPROT = 1 # Process this many PDB IDs if multiple are found

# --- End Configuration ---

def query_api_for_pdb_ids(uniprot_ac: str, api_source: str) -> list:
    """Queries either RCSB or EBI PDBe API to find PDB IDs for a given UniProt AC."""
    pdb_ids = []
    if not uniprot_ac or pd.isna(uniprot_ac): return pdb_ids
    
    uniprot_ac_clean = uniprot_ac.strip()
    query_url = ""
    query_json = None
    headers = {'Accept': 'application/json'} # Common header

    if api_source == "RCSB":
        query_url = RCSB_SEARCH_API_URL
        query_json = {
            "query": {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", "operator": "exact_match", "value": uniprot_ac_clean}},
            "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": MAX_PDB_IDS_PER_UNIPROT + 5}, "sort": [{"sort_by": "score", "direction": "desc"}], "results_content_type": ["experimental"]}}
        headers['Content-Type'] = 'application/json'
        http_method = requests.post
    elif api_source == "EBI":
        query_url = EBI_PDBe_API_UNIPROT_MAPPING_URL.format(uniprot_ac=uniprot_ac_clean)
        http_method = requests.get
    else:
        logger.error(f"Unknown API source: {api_source}"); return pdb_ids

    response = None
    try:
        logger.debug(f"[{api_source} Query] UniProt: {uniprot_ac_clean}, URL: {query_url}")
        if query_json: # POST request for RCSB
            response = http_method(query_url, headers=headers, json=query_json, timeout=API_TIMEOUT)
        else: # GET request for EBI
            response = http_method(query_url, headers=headers, timeout=API_TIMEOUT)
        response.raise_for_status()

        if not response.text or not response.text.strip().startswith(("{", "[")):
            logger.warning(f"[{api_source} Query] Response for {uniprot_ac_clean} not JSON/empty. Status: {response.status_code}. Response: {response.text[:500]}"); return pdb_ids
        
        results = response.json()

        if api_source == "RCSB" and "result_set" in results and results["result_set"]:
            for item in results["result_set"]:
                pdb_ids.append(item["identifier"])
                if len(pdb_ids) >= MAX_PDB_IDS_PER_UNIPROT: break
        elif api_source == "EBI" and uniprot_ac_clean in results and results[uniprot_ac_clean]:
            for entry in results[uniprot_ac_clean]:
                pdb_ids.append(entry["pdb_id"])
                if len(pdb_ids) >= MAX_PDB_IDS_PER_UNIPROT: break
        logger.debug(f"[{api_source} Query] Found PDB IDs for {uniprot_ac_clean}: {pdb_ids}")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404 and api_source == "EBI": logger.info(f"[{api_source} Query] UniProt AC {uniprot_ac_clean} not found (404).")
        else: logger.warning(f"[{api_source} Query] HTTP Error for {uniprot_ac_clean}: {e.response.status_code}. Response: {e.response.text[:500]}")
    except json.JSONDecodeError as e:
        response_text = response.text[:500] if response and response.text else "N/A"; logger.error(f"[{api_source} Query] JSONDecodeError for {uniprot_ac_clean}: {e}. Response: {response_text}")
    except requests.exceptions.RequestException as e: logger.error(f"[{api_source} Query] RequestException for {uniprot_ac_clean}: {e}")
    except Exception as e: logger.error(f"[{api_source} Query] Unexpected error for {uniprot_ac_clean}: {e}", exc_info=True)
    
    time.sleep(REQUEST_DELAY); return pdb_ids


def attempt_pdb_download(pdb_id: str, output_dir: Path, protein_id_log: str) -> tuple:
    """Attempts to download a PDB file, first from RCSB, then from EBI PDBe if RCSB returns 404."""
    pdb_id_upper = pdb_id.upper()
    common_filename = f"{pdb_id_upper}.pdb"
    output_path = output_dir / common_filename
    download_sources = [
        ("RCSB_PDB", RCSB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id_upper)),
        ("EBI_PDBe_Download", EBI_PDB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id_upper))
    ]
    
    for source_name, url in download_sources:
        logger.info(f"[{protein_id_log} | PDB: {pdb_id_upper}] Attempting {source_name}: {url}")
        try:
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            logger.info(f"[{protein_id_log} | PDB: {pdb_id_upper}] Success ({source_name}): {output_path}")
            return True, source_name, common_filename, f"Success ({source_name}: {pdb_id_upper})"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"[{protein_id_log} | PDB: {pdb_id_upper}] {source_name} download 404. Trying next source if available.")
                if source_name == "RCSB_PDB": continue # Try EBI next
                else: return False, None, None, f"Failed (Both RCSB & EBI 404 for {pdb_id_upper})" # EBI was last try
            else: # Other HTTP error
                logger.error(f"[{protein_id_log} | PDB: {pdb_id_upper}] {source_name} HTTP Error {e.response.status_code}. Response: {e.response.text[:200]}")
                return False, None, None, f"Failed ({source_name} HTTP Error {e.response.status_code})"
        except requests.exceptions.RequestException as e_req:
            logger.error(f"[{protein_id_log} | PDB: {pdb_id_upper}] {source_name} RequestException: {e_req}")
            return False, None, None, f"Failed ({source_name} RequestException)" # Stop on other request errors
        except Exception as e_generic:
            logger.error(f"[{protein_id_log} | PDB: {pdb_id_upper}] {source_name} unexpected error: {e_generic}", exc_info=True)
            return False, None, None, f"Failed ({source_name} Unexpected Error)"
        time.sleep(REQUEST_DELAY) # Delay between attempts if the first one failed and we loop
    
    return False, None, None, f"Failed (PDB: {pdb_id_upper} - All download sources attempted)"


def load_existing_log(log_file_path: Path) -> pd.DataFrame:
    if log_file_path.exists():
        try: 
            df = pd.read_csv(log_file_path)
            logger.info(f"Loaded {len(df)} entries from log: {log_file_path}")
            return df
        except Exception as e: 
            logger.error(f"Could not read log {log_file_path}: {e}. Treating as empty.")
    return pd.DataFrame()

def get_successfully_logged_pdb_downloads(existing_log_df: pd.DataFrame) -> set:
    logged_items = set() # Store (ProteinID, PDB_ID_Found)
    if not existing_log_df.empty and all(c in existing_log_df.columns for c in [PROTEIN_ID_COL, 'PDB_ID_Found', 'Download_Status', 'Retrieved_From']):
        successful_logs = existing_log_df[
            existing_log_df['Retrieved_From'].astype(str).str.contains("RCSB_PDB|EBI_PDBe_Download", case=False, na=False) &
            existing_log_df['Download_Status'].astype(str).str.startswith("Success")
        ]
        for _, row in successful_logs.iterrows():
            pid, pdb_found = row[PROTEIN_ID_COL], row['PDB_ID_Found']
            if pd.notna(pid) and pd.notna(pdb_found):
                logged_items.add((pid, str(pdb_found).upper()))
    logger.info(f"Found {len(logged_items)} (ProteinID, PDB_ID_Found) pairs successfully logged from RCSB/EBI."); return logged_items

def main():
    logger.info("Starting UniProt AC to PDB ID mapping and download process (v6 Clean Rewrite)...")
    PDB_STRUCTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_log_entries_df = load_existing_log(PDB_MASTER_LOG_CSV_FILE)
    successfully_logged_pdb_downloads = get_successfully_logged_pdb_downloads(existing_log_entries_df)
    new_log_records = []

    if not INPUT_UNIPROT_LIST_CSV.exists(): logger.critical(f"Input '{INPUT_UNIPROT_LIST_CSV}' not found. Exiting."); return
    try: df_input = pd.read_csv(INPUT_UNIPROT_LIST_CSV); logger.info(f"Read {len(df_input)} from: '{INPUT_UNIPROT_LIST_CSV}'.")
    except Exception as e: logger.critical(f"Error reading '{INPUT_UNIPROT_LIST_CSV}': {e}"); return
    if not all(c in df_input.columns for c in [PROTEIN_ID_COL, UNIPROT_AC_COL]):
        logger.critical(f"Input CSV needs '{PROTEIN_ID_COL}' and '{UNIPROT_AC_COL}'. Exiting."); return

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Processing UniProt ACs"):
        protein_id, uniprot_ac, og_id = row[PROTEIN_ID_COL], row.get(UNIPROT_AC_COL), row.get("Orthogroup", None)
        if pd.isna(uniprot_ac) or not str(uniprot_ac).strip(): logger.warning(f"[{protein_id}] Skipping: missing/empty UniProt AC."); continue

        retrieved_pdb_ids, api_source_for_ids = query_api_for_pdb_ids(str(uniprot_ac), "RCSB"), "RCSB"
        if not retrieved_pdb_ids:
            logger.info(f"[{protein_id} | UniProt: {uniprot_ac}] No PDB IDs from RCSB. Trying EBI PDBe API...")
            retrieved_pdb_ids, api_source_for_ids = query_api_for_pdb_ids(str(uniprot_ac), "EBI"), "EBI"
        
        if not retrieved_pdb_ids:
            logger.info(f"[{protein_id} | UniProt: {uniprot_ac}] No PDB IDs found via any API.")
            new_log_records.append({PROTEIN_ID_COL: protein_id, UNIPROT_AC_COL: uniprot_ac, "PDB_ID_Found": None, "Retrieved_From": None, "Saved_Filename": None, "Download_Status": f"No PDB IDs found via {api_source_for_ids}", "OG_ID": og_id, "API_Source_For_PDB_ID": api_source_for_ids})
            continue

        pdb_ids_to_process = [pid.upper() for pid in retrieved_pdb_ids[:MAX_PDB_IDS_PER_UNIPROT]]
        logger.info(f"[{protein_id} | UniProt: {uniprot_ac}] Will process PDB IDs from {api_source_for_ids}: {pdb_ids_to_process}")

        for pdb_id_to_fetch in pdb_ids_to_process:
            if (protein_id, pdb_id_to_fetch) in successfully_logged_pdb_downloads:
                logger.info(f"[{protein_id} | PDB: {pdb_id_to_fetch}] Pair already successfully logged. Skipping."); continue
            
            common_filename = f"{pdb_id_to_fetch}.pdb"
            output_path = PDB_STRUCTURE_OUTPUT_DIR / common_filename

            if not output_path.exists():
                success, retrieved_from, saved_filename, status_message = attempt_pdb_download(pdb_id_to_fetch, PDB_STRUCTURE_OUTPUT_DIR, protein_id)
            else:
                logger.info(f"[{protein_id} | PDB: {pdb_id_to_fetch}] File exists: {output_path}.")
                success, retrieved_from, saved_filename, status_message = True, "RCSB_PDB (LocalFileExists)", common_filename, f"Success (RCSB_PDB - LocalFileExists: {pdb_id_to_fetch})" # Assume RCSB if local
            
            new_log_records.append({
                PROTEIN_ID_COL: protein_id, UNIPROT_AC_COL: uniprot_ac, "PDB_ID_Found": pdb_id_to_fetch,
                "Retrieved_From": retrieved_from, "Saved_Filename": saved_filename,
                "Download_Status": status_message, "OG_ID": og_id, "API_Source_For_PDB_ID": api_source_for_ids
            })

    if new_log_records:
        df_new_logs = pd.DataFrame(new_log_records)
        df_combined_log = pd.concat([existing_log_entries_df, df_new_logs], ignore_index=True)
        # Deduplication based on ProteinID and the specific PDB_ID_Found
        subset_cols = [PROTEIN_ID_COL, 'PDB_ID_Found', 'Retrieved_From', 'Saved_Filename']
        actual_subset_cols = [col for col in subset_cols if col in df_combined_log.columns]

        if actual_subset_cols:
             df_combined_log.sort_values(by=[PROTEIN_ID_COL, 'Download_Status'], ascending=[True, True], inplace=True)
             if 'PDB_ID_Found' in df_combined_log.columns: df_combined_log['PDB_ID_Found'] = df_combined_log['PDB_ID_Found'].astype(str).str.upper().replace("NAN", "", case=False)
             df_combined_log.drop_duplicates(subset=actual_subset_cols, keep='last', inplace=True)
        else: logger.warning("Could not perform robust deduplication due to missing key columns.")
        try:
            df_combined_log.sort_values(by=[PROTEIN_ID_COL, UNIPROT_AC_COL, 'PDB_ID_Found'], na_position='first', inplace=True)
            df_combined_log.to_csv(PDB_MASTER_LOG_CSV_FILE, index=False) # Use dedicated log file
            logger.info(f"PDB download log updated: {PDB_MASTER_LOG_CSV_FILE} ({len(df_combined_log)} entries).")
        except Exception as e: logger.error(f"Error writing PDB log CSV file: {e}", exc_info=True)
    else: logger.info("No new PDB download attempts or log entries generated.")
    logger.info("--- UniProt to PDB Mapping and Download Process Finished (v6 Clean Rewrite) ---")

if __name__ == "__main__":
    main()
