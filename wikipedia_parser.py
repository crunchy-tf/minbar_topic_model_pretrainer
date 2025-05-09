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
    WIKIEXTRACTOR_BYTES_PER_CHUNK
)

def extract_text_from_dump(lang_code: str, dump_file_path: str, output_base_dir: str) -> bool:
    """Extracts text from a single Wikipedia dump file using wikiextractor."""
    lang_output_dir = os.path.join(output_base_dir, lang_code)
    logger.info(f"Starting extraction for '{lang_code}' from '{dump_file_path}' to '{lang_output_dir}'")
    os.makedirs(lang_output_dir, exist_ok=True)

    # Check if output directory already has content (simple check)
    if os.path.exists(lang_output_dir) and any(os.scandir(lang_output_dir)): # Check if directory is not empty
        # A more robust check would be to see if specific expected output files exist
        # or if a success marker from a previous run is present.
        logger.info(f"Output directory '{lang_output_dir}' for '{lang_code}' already contains files. "
                      "Skipping extraction. Delete files to re-extract.")
        return True

    # Command for wikiextractor
    # --json: Output JSON objects, one per article, one per line. Easier to parse.
    # --no-templates: Try to remove templates
    # --processes: Use multiple CPU cores
    # --bytes: Split output into chunks
    # --min_text_length 0 : extract all articles initially, filter length later
    command = [
        "python3", "-m", "wikiextractor.WikiExtractor",
        dump_file_path,
        "--output", lang_output_dir,
        "--json",
        "--no-templates",
        "--processes", str(WIKIEXTRACTOR_PROCESSES),
        "--bytes", WIKIEXTRACTOR_BYTES_PER_CHUNK,
        "--min_text_length", "0" # We will filter length later
    ]
    logger.info(f"Executing wikiextractor command for '{lang_code}': {' '.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode == 0:
            logger.success(f"Successfully extracted text for '{lang_code}'. Output in '{lang_output_dir}'.")
            # Log some output details from wikiextractor if available
            if process.stdout: logger.debug(f"WikiExtractor STDOUT:\n{process.stdout[-1000:]}") # Log last 1000 chars
            if process.stderr: logger.debug(f"WikiExtractor STDERR:\n{process.stderr[-1000:]}")
            return True
        else:
            logger.error(f"wikiextractor failed for '{lang_code}' with return code {process.returncode}.")
            logger.error(f"wikiextractor stdout: {process.stdout}")
            logger.error(f"wikiextractor stderr: {process.stderr}")
            return False
    except FileNotFoundError:
        logger.error("wikiextractor command (or python3) not found. Please ensure it's installed and in PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during wikiextractor execution for '{lang_code}': {e}")
        return False


def parse_all_downloaded_dumps():
    logger.info("Starting Wikipedia text extraction process...")
    os.makedirs(WIKIPEDIA_EXTRACTED_DIR, exist_ok=True)
    all_successful = True

    dump_files = glob.glob(os.path.join(WIKIPEDIA_DUMP_DIR, "*-pages-articles.xml.bz2"))
    if not dump_files: # Try another common pattern
        dump_files.extend(glob.glob(os.path.join(WIKIPEDIA_DUMP_DIR, "*-articles.xml.bz2")))

    if not dump_files:
        logger.error(f"No Wikipedia dump files (.xml.bz2) found in '{WIKIPEDIA_DUMP_DIR}'. Please download them first.")
        return False

    logger.info(f"Found dump files: {dump_files}")

    for lang in LANGUAGES_TO_PROCESS:
        # Find the dump file for the current language
        lang_dump_file = None
        for df_path in dump_files:
            df_name = os.path.basename(df_path)
            if df_name.startswith(f"{lang}wiki"):
                lang_dump_file = df_path
                break
        
        if lang_dump_file:
            if not extract_text_from_dump(lang, lang_dump_file, WIKIPEDIA_EXTRACTED_DIR):
                all_successful = False
                logger.error(f"Extraction failed for language: {lang}")
        else:
            logger.warning(f"No dump file found for language '{lang}' in '{WIKIPEDIA_DUMP_DIR}'. Skipping its extraction.")
            # all_successful = False # Or just skip if a language is optional

    if all_successful:
        logger.success("Wikipedia text extraction completed for all specified languages (or those found).")
    else:
        logger.error("One or more Wikipedia text extractions failed.")
    return all_successful

if __name__ == "__main__":
    if parse_all_downloaded_dumps():
        logger.info("Parser test finished successfully.")
    else:
        logger.error("Parser test encountered errors.")