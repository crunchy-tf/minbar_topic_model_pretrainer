# topic_model_pretrainer_cloud/wikipedia_parser.py
import os
import glob
import subprocess
from loguru import logger
from config_pretrainer import (
    WIKIPEDIA_DUMP_DIR,
    WIKIPEDIA_EXTRACTED_DIR, # This is the base for lang_output_dir
    LANGUAGES_TO_PROCESS,
    WIKIEXTRACTOR_PROCESSES,
    WIKIEXTRACTOR_BYTES_PER_CHUNK,
    WIKIEXTRACTOR_TEMPLATES_DIR # New config for templates cache
)

def extract_text_from_dump(lang_code: str, dump_file_path: str, output_base_dir: str) -> bool:
    """Extracts text from a single Wikipedia dump file using WikiExtractor V2."""
    lang_output_dir = os.path.join(output_base_dir, lang_code) # e.g., .../wikipedia_extracted_raw_v2/en
    
    # Path for the template cache file for this language
    # WikiExtractor V2 recommends a path for --templates
    os.makedirs(WIKIEXTRACTOR_TEMPLATES_DIR, exist_ok=True) # Ensure templates cache base dir exists
    templates_cache_file = os.path.join(WIKIEXTRACTOR_TEMPLATES_DIR, f"{lang_code}_templates_v2.dat")

    logger.info(f"Starting extraction for '{lang_code}' from '{dump_file_path}' to '{lang_output_dir}' using WikiExtractorV2")
    logger.info(f"Templates cache will be stored/used at: {templates_cache_file}")
    os.makedirs(lang_output_dir, exist_ok=True)

    # Check if output directory already has *some* expected output (e.g., .txt or .json files)
    # This is a simple check; more robust would be a success marker file.
    # WikiExtractorV2 might create subdirectories like AA, AB.
    # So, check if lang_output_dir has any subdirectories or files.
    if os.path.exists(lang_output_dir) and os.listdir(lang_output_dir):
        logger.info(f"Output directory '{lang_output_dir}' for '{lang_code}' already contains files/subdirectories. "
                      "Skipping WikiExtractorV2 run. Delete content of this directory to re-extract.")
        return True

    # Construct command based on WikiExtractor V2 README recommendations
    command = [
        "python3", "-m", "wikiextractor.WikiExtractor", # Assuming the fork installs as 'wikiextractor' module
        dump_file_path,
        "--output", lang_output_dir,
        "--bytes", WIKIEXTRACTOR_BYTES_PER_CHUNK,
        "--templates", templates_cache_file, # Use or create template cache
        "--json",                             # Output JSON for easier category parsing later
        # The V2 README recommended options:
        "--discard_sections",
        "--discard_templates",
        "--ignore_templates",
        "--processes", str(WIKIEXTRACTOR_PROCESSES)
    ]
    # Note: --no-templates is not used with V2; --templates handles template processing.
    # Note: --min_text_length is not in V2's example; assuming default or later filtering.

    logger.info(f"Executing WikiExtractorV2 command for '{lang_code}': {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.success(f"Successfully initiated extraction for '{lang_code}' using WikiExtractorV2. Output in '{lang_output_dir}'.")
            if process.stdout: logger.debug(f"WikiExtractorV2 STDOUT (last 1k chars):\n{process.stdout[-1000:]}")
            if process.stderr: logger.debug(f"WikiExtractorV2 STDERR (last 1k chars):\n{process.stderr[-1000:]}")
            # WikiExtractorV2 might take time; this return True means the command was launched.
            # True success is when all files are processed by wikiextractor itself.
            # This function only checks if the subprocess call was okay.
            return True
        else:
            logger.error(f"WikiExtractorV2 subprocess failed for '{lang_code}' with return code {process.returncode}.")
            logger.error(f"WikiExtractorV2 STDOUT:\n{process.stdout}")
            logger.error(f"WikiExtractorV2 STDERR:\n{process.stderr}") # Log full stderr on error
            return False
    except FileNotFoundError:
        logger.error("WikiExtractorV2 command (or python3 or wikiextractor.WikiExtractor module) not found. "
                       "Please ensure the chosen fork is installed correctly and runnable as 'python3 -m wikiextractor.WikiExtractor', "
                       "or adjust the command path.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during WikiExtractorV2 execution for '{lang_code}': {e}", exc_info=True)
        return False

def parse_all_downloaded_dumps():
    logger.info("Starting Wikipedia text extraction process using WikiExtractorV2...")
    os.makedirs(WIKIPEDIA_EXTRACTED_DIR, exist_ok=True) # Base for language-specific extracted dirs
    all_successful = True

    # Construct path to look for general dump files
    # Example: enwiki-latest-pages-articles.xml.bz2
    dump_pattern1 = os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-pages-articles.xml.bz2")
    dump_pattern2 = os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-articles.xml.bz2") # Another common pattern

    # Get all matching dump files
    all_dump_files = glob.glob(dump_pattern1) + glob.glob(dump_pattern2)
    
    if not all_dump_files:
        logger.error(f"No Wikipedia dump files (.xml.bz2 matching patterns) found in '{WIKIPEDIA_DUMP_DIR}'. Please download them first.")
        return False

    logger.info(f"Found potential dump files: {all_dump_files}")

    for lang in LANGUAGES_TO_PROCESS: # Now only "en"
        lang_dump_file = None
        # Find the specific dump file for the current language
        for df_path in all_dump_files:
            df_name = os.path.basename(df_path)
            # Simple check: e.g., "enwiki-..."
            if df_name.startswith(f"{lang}wiki"):
                lang_dump_file = df_path
                break
        
        if lang_dump_file and os.path.exists(lang_dump_file):
            logger.info(f"Found dump file for language '{lang}': {lang_dump_file}")
            if not extract_text_from_dump(lang, lang_dump_file, WIKIPEDIA_EXTRACTED_DIR):
                all_successful = False
                logger.error(f"Extraction failed for language: {lang}")
        else:
            logger.warning(f"No dump file found or accessible for language '{lang}' with expected name patterns in '{WIKIPEDIA_DUMP_DIR}'. Skipping its extraction.")
            # If processing only English and English dump isn't found, this is a critical failure.
            if lang == "en": # If English is mandatory and not found
                all_successful = False


    if all_successful:
        logger.success("Wikipedia text extraction completed (or skipped if already done) for all specified languages.")
    else:
        logger.error("One or more Wikipedia text extractions failed or were skipped due to missing dumps.")
    return all_successful

if __name__ == "__main__":
    # For standalone testing of the parser module
    if parse_all_downloaded_dumps():
        logger.info("Wikipedia parser module test finished successfully.")
    else:
        logger.error("Wikipedia parser module test encountered errors.")