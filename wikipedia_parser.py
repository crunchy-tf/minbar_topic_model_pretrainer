# topic_model_pretrainer_cloud/wikipedia_parser.py
import os
import glob
import subprocess
from loguru import logger
from config_pretrainer import (
    WIKIPEDIA_DUMP_DIR,
    WIKIPEDIA_EXTRACTED_DIR, # This is the base for lang_output_dir, e.g., .../wikipedia_extracted_raw_v2
    LANGUAGES_TO_PROCESS,
    WIKIEXTRACTOR_PROCESSES,
    WIKIEXTRACTOR_BYTES_PER_CHUNK,
    WIKIEXTRACTOR_TEMPLATES_DIR, # For V2's template cache
    BASE_PROJECT_DIR # Root of your topic_model_pretrainer_cloud project on the VM
)

# Construct the path to the cloned WikiExtractor-V2 script
# Assumes 'Wikiextractor-V2' is cloned inside your BASE_PROJECT_DIR
WIKIEXTRACTOR_V2_FORK_ROOT_DIR = os.path.join(BASE_PROJECT_DIR, "Wikiextractor-V2")
WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE = os.path.join(WIKIEXTRACTOR_V2_FORK_ROOT_DIR, "wikiextractor", "WikiExtractor.py")

def extract_text_from_dump(lang_code: str, dump_file_path: str, output_base_dir_for_lang_extraction: str) -> bool:
    """
    Extracts text from a single Wikipedia dump file using the cloned WikiExtractor V2 fork.
    output_base_dir_for_lang_extraction is typically config_pretrainer.WIKIPEDIA_EXTRACTED_DIR
    """
    # Specific output directory for this language's extracted files, e.g., .../wikipedia_extracted_raw_v2/en
    lang_specific_output_dir = os.path.join(output_base_dir_for_lang_extraction, lang_code)
    
    # Path for the template cache file for this language, specific to WikiExtractor V2
    # WIKIEXTRACTOR_TEMPLATES_DIR is the base for these cache files
    os.makedirs(WIKIEXTRACTOR_TEMPLATES_DIR, exist_ok=True) 
    templates_cache_file = os.path.join(WIKIEXTRACTOR_TEMPLATES_DIR, f"{lang_code}_wikiextractor_v2_templates.dat")

    logger.info(f"Starting extraction for '{lang_code}' from '{dump_file_path}' to '{lang_specific_output_dir}'")
    logger.info(f"Using WikiExtractorV2 script: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
    logger.info(f"Templates cache will be stored/used at: {templates_cache_file}")
    os.makedirs(lang_specific_output_dir, exist_ok=True)

    # Check if the language-specific output directory already has content (e.g., subdirectories AA, AB created by wikiextractor)
    if os.path.exists(lang_specific_output_dir) and os.listdir(lang_specific_output_dir):
        # A more robust check would be for a specific marker file or expected number of subdirs/files.
        # For now, if not empty, assume it was processed.
        logger.info(f"Output directory '{lang_specific_output_dir}' for '{lang_code}' already contains files/subdirectories. "
                      "Skipping WikiExtractorV2 run. Delete content of this directory to re-extract.")
        return True

    if not os.path.exists(WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE):
        logger.error(f"WikiExtractorV2 script not found at specified path: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
        logger.error("Please ensure the 'Wikiextractor-V2' fork is cloned into your project root directory "
                     f"({BASE_PROJECT_DIR}) and the path is correct.")
        return False

    # Construct command based on WikiExtractor V2 README recommendations
    command = [
        "python3", WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE,
        dump_file_path,
        "--output", lang_specific_output_dir,    # Output for this language
        "--bytes", WIKIEXTRACTOR_BYTES_PER_CHUNK, # Max size per output file chunk
        "--templates", templates_cache_file,      # Path to use/create template cache
        "--json",                                 # Output in JSONL format (one JSON object per article per line)
        "--discard_sections",                     # Recommended V2 option
        "--discard_templates",                    # Recommended V2 option
        "--ignore_templates",                     # Recommended V2 option
        "--processes", str(WIKIEXTRACTOR_PROCESSES) # Number of CPU cores to use
    ]

    logger.info(f"Executing WikiExtractorV2 command for '{lang_code}': {' '.join(command)}")
    try:
        # Run the WikiExtractorV2 script from its own root directory
        # This helps it find its 'config/' folder for discard_sections.txt etc.
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # We will check returncode manually
            cwd=WIKIEXTRACTOR_V2_FORK_ROOT_DIR # Set Current Working Directory to the fork's root
        )

        if process.returncode == 0:
            logger.success(f"WikiExtractorV2 successfully initiated extraction for '{lang_code}'. Output in '{lang_specific_output_dir}'.")
            # WikiExtractorV2 can be verbose; log snippets for debugging.
            if process.stdout: logger.debug(f"WikiExtractorV2 STDOUT (last 1k chars):\n{process.stdout[-1000:]}")
            if process.stderr: logger.debug(f"WikiExtractorV2 STDERR (last 1k chars):\n{process.stderr[-1000:]}")
            return True
        else:
            logger.error(f"WikiExtractorV2 subprocess failed for '{lang_code}' with return code {process.returncode}.")
            logger.error(f"WikiExtractorV2 STDOUT:\n{process.stdout}") # Log full output on error
            logger.error(f"WikiExtractorV2 STDERR:\n{process.stderr}") # Log full output on error
            return False
    except FileNotFoundError: # Should not happen if python3 exists and script path is correct
        logger.error(f"Python3 or WikiExtractorV2 script not found. Path tried: {WIKIEXTRACTOR_V2_SCRIPT_EXECUTABLE}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during WikiExtractorV2 execution for '{lang_code}': {e}", exc_info=True)
        return False

def parse_all_downloaded_dumps():
    logger.info("Starting Wikipedia text extraction process using WikiExtractorV2...")
    # WIKIPEDIA_EXTRACTED_DIR is the base where lang subdirs (e.g., 'en', 'fr') will be created
    os.makedirs(WIKIPEDIA_EXTRACTED_DIR, exist_ok=True)
    all_successful = True

    # Find all potential dump files
    dump_patterns = [
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-pages-articles.xml.bz2"),
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-articles.xml.bz2") # Another common pattern
    ]
    all_found_dump_files = []
    for pattern in dump_patterns:
        all_found_dump_files.extend(glob.glob(pattern))
    
    if not all_found_dump_files:
        logger.error(f"No Wikipedia dump files (.xml.bz2 matching patterns) found in '{WIKIPEDIA_DUMP_DIR}'. "
                       "Please ensure dumps are downloaded by wikipedia_downloader.py first.")
        return False

    logger.info(f"Found potential dump files: {all_found_dump_files}")

    for lang_code in LANGUAGES_TO_PROCESS: # From config_pretrainer, e.g., ["en"]
        lang_specific_dump_file = None
        # Find the correct dump file for the current language code
        for df_path in all_found_dump_files:
            df_name = os.path.basename(df_path)
            if df_name.startswith(f"{lang_code}wiki"): # e.g., "enwiki-..."
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
            # If a configured language dump is missing, consider it a failure for that lang
            all_successful = False 

    if all_successful:
        logger.success("Wikipedia text extraction completed (or skipped if already done) for all specified languages using WikiExtractorV2.")
    else:
        logger.warning("One or more Wikipedia text extractions failed or were skipped due to missing dumps or errors during parsing.")
    return all_successful

if __name__ == "__main__":
    # For standalone testing of this parser module
    # Ensure config_pretrainer.py is correctly set up.
    # Ensure Wikipedia dumps are in WIKIPEDIA_DUMP_DIR.
    # Ensure WikiExtractor-V2 fork is cloned in BASE_PROJECT_DIR.
    logger.info("Testing wikipedia_parser.py standalone...")
    if parse_all_downloaded_dumps():
        logger.info("Wikipedia parser module test finished successfully.")
    else:
        logger.error("Wikipedia parser module test encountered errors.")