# topic_model_pretrainer_cloud/wikipedia_parser.py
import os
import glob
import subprocess
from loguru import logger
from config_pretrainer import (
    WIKIPEDIA_DUMP_DIR,
    WIKIPEDIA_EXTRACTED_DIR, 
    LANGUAGES_TO_PROCESS,
    WIKIEXTRACTOR_PROCESSES,
    WIKIEXTRACTOR_BYTES_PER_CHUNK,
    WIKIEXTRACTOR_TEMPLATES_DIR, 
    BASE_PROJECT_DIR 
)

# Construct the path to the cloned WikiExtractor-V2 script
WIKIEXTRACTOR_V2_FORK_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "Wikiextractor-V2")
WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE = os.path.join(WIKIEXTRACTOR_V2_FORK_ROOT_DIR, "wikiextractor", "WikiExtractor.py")

def extract_text_from_dump(lang_code: str, dump_file_path: str, output_base_dir_for_lang_extraction: str) -> bool:
    lang_specific_output_dir = os.path.join(output_base_dir_for_lang_extraction, lang_code)
    
    os.makedirs(WIKIEXTRACTOR_TEMPLATES_DIR, exist_ok=True) 
    templates_cache_file = os.path.join(WIKIEXTRACTOR_TEMPLATES_DIR, f"{lang_code}_wikiextractor_v2_templates.dat")

    logger.info(f"Starting extraction for '{lang_code}' from '{dump_file_path}' to '{lang_specific_output_dir}'")
    logger.info(f"Using WikiExtractorV2 script: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
    logger.info(f"Templates cache will be stored/used at: {templates_cache_file}")
    os.makedirs(lang_specific_output_dir, exist_ok=True)

    if os.path.exists(lang_specific_output_dir) and os.listdir(lang_specific_output_dir):
        logger.info(f"Output directory '{lang_specific_output_dir}' for '{lang_code}' already contains files/subdirectories. "
                      "Skipping WikiExtractorV2 run. Delete content of this directory to re-extract.")
        return True

    if not os.path.exists(WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE):
        logger.error(f"WikiExtractorV2 script not found at specified path: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
        logger.error(f"Ensure 'Wikiextractor-V2' is cloned into: {BASE_PROJECT_DIR} (which is {os.path.abspath(BASE_PROJECT_DIR)}) "
                     f"and that the Wikiextractor-V2 fork has the file 'wikiextractor/WikiExtractor.py' within it.")
        return False

    command = [
        "python3", WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE,
        dump_file_path,
        "--output", lang_specific_output_dir,    
        "--bytes", WIKIEXTRACTOR_BYTES_PER_CHUNK, 
        "--templates", templates_cache_file,      
        "--json",                                 
        "--discard_sections",                     
        "--discard_templates",                    
        "--ignore_templates",                     
        "--processes", str(WIKIEXTRACTOR_PROCESSES)
        # Add --debug here IF WikiExtractorV2 supports it and you want its internal debug logs
        # "--debug" 
    ]

    logger.info(f"Executing WikiExtractorV2 command for '{lang_code}': {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True, 
            check=False, 
            cwd=WIKIEXTRACTOR_V2_FORK_ROOT_DIR 
        )

        if process.returncode == 0:
            logger.success(f"WikiExtractorV2 successfully completed extraction for '{lang_code}'. Output in '{lang_specific_output_dir}'.")
            # Log snippets of output even on success for confirmation
            if process.stdout: logger.debug(f"WikiExtractorV2 STDOUT (last 2k chars on success):\n{process.stdout[-2000:]}")
            if process.stderr: logger.debug(f"WikiExtractorV2 STDERR (last 2k chars on success, should be minimal):\n{process.stderr[-2000:]}")
            return True
        else:
            logger.error(f"WikiExtractorV2 subprocess failed for '{lang_code}' with return code {process.returncode}.")
            # Log FULL stdout and stderr from the failed subprocess for detailed debugging
            logger.error("--- WikiExtractorV2 STDOUT (Full on Error) ---")
            if process.stdout:
                logger.error(f"\n{process.stdout}\n")
            else:
                logger.error("(No STDOUT captured)")
            logger.error("--- WikiExtractorV2 STDERR (Full on Error) ---")
            if process.stderr: # This will contain the Python traceback from WikiExtractorV2
                logger.error(f"\n{process.stderr}\n")
            else:
                logger.error("(No STDERR captured)")
            logger.error("--- End of WikiExtractorV2 Output ---")
            return False
    except FileNotFoundError: 
        logger.error(f"Python3 or WikiExtractorV2 script not found. Path tried for script: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during WikiExtractorV2 execution for '{lang_code}': {e}", exc_info=True)
        return False

def parse_all_downloaded_dumps():
    logger.info("Starting Wikipedia text extraction process using WikiExtractorV2...")
    os.makedirs(WIKIPEDIA_EXTRACTED_DIR, exist_ok=True)
    all_successful = True
    dump_patterns = [
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-pages-articles.xml.bz2"),
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-articles.xml.bz2")
    ]
    all_found_dump_files = []
    for pattern in dump_patterns:
        all_found_dump_files.extend(glob.glob(pattern))
    
    if not all_found_dump_files:
        logger.error(f"No Wikipedia dump files (.xml.bz2 matching patterns) found in '{WIKIPEDIA_DUMP_DIR}'. "
                       "Please ensure dumps are downloaded by wikipedia_downloader.py first.")
        return False

    logger.info(f"Found potential dump files: {all_found_dump_files}")

    for lang_code in LANGUAGES_TO_PROCESS: 
        lang_specific_dump_file = None
        for df_path in all_found_dump_files:
            df_name = os.path.basename(df_path)
            if df_name.startswith(f"{lang_code}wiki"):
                lang_specific_dump_file = df_path
                break
        
        if lang_specific_dump_file and os.path.exists(lang_specific_dump_file):
            logger.info(f"Processing dump file for language '{lang_code}': {lang_specific_dump_file}")
            if not extract_text_from_dump(lang_code, lang_specific_dump_file, WIKIPEDIA_EXTRACTED_DIR):
                all_successful = False
                logger.error(f"Extraction failed for language: {lang_code} using WikiExtractorV2.")
        else:
            logger.warning(f"No dump file found or accessible for language '{lang_code}' with expected name patterns in '{WIKIPEDIA_DUMP_DIR}'. "
                           "Skipping its extraction.")
            if lang_code in LANGUAGES_TO_PROCESS: # If a configured language is missing, consider it a failure point.
                 all_successful = False 

    if all_successful:
        logger.success("Wikipedia text extraction completed (or skipped if already done) for all specified languages using WikiExtractorV2.")
    else:
        logger.warning("One or more Wikipedia text extractions failed or were skipped due to missing dumps or errors during parsing.")
    return all_successful

if __name__ == "__main__":
    logger.info("Testing wikipedia_parser.py standalone...")
    # For standalone test, ensure config_pretrainer.py is correctly set up
    # and the WikiExtractor-V2 fork is cloned in BASE_PROJECT_DIR.
    # Also, ensure dumps are present in WIKIPEDIA_DUMP_DIR.
    if parse_all_downloaded_dumps():
        logger.info("Wikipedia parser module test finished successfully.")
    else:
        logger.error("Wikipedia parser module test encountered errors or incomplete processing.")