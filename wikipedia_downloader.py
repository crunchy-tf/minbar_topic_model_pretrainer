# topic_model_pretrainer_cloud/wikipedia_downloader.py
import os
import subprocess
import requests
from loguru import logger
from config_pretrainer import WIKIPEDIA_DUMP_DIR, LANGUAGES_TO_PROCESS

BASE_WIKIMEDIA_URL = "https://dumps.wikimedia.org"

def get_latest_dump_url(lang_code: str) -> str | None:
    """Gets the URL for the latest pages-articles.xml.bz2 dump for a language."""
    status_url = f"{BASE_WIKIMEDIA_URL}/{lang_code}wiki/latest/{lang_code}wiki-latest-dumpstatus.json"
    try:
        response = requests.get(status_url)
        response.raise_for_status()
        status_data = response.json()
        # The structure for pages-articles can vary, often under 'jobs' -> 'articlesmultistreamdump'
        # This is a common path, but might need adjustment if Wikimedia changes JSON structure
        if "jobs" in status_data and "articlesmultistreamdump" in status_data["jobs"] and \
           "files" in status_data["jobs"]["articlesmultistreamdump"] and \
           status_data["jobs"]["articlesmultistreamdump"]["status"] == "done":
            
            files = status_data["jobs"]["articlesmultistreamdump"]["files"]
            # Find the .xml.bz2 file, which is usually the largest or has a specific name pattern
            for filename, file_info in files.items():
                if filename.endswith("-pages-articles.xml.bz2") or filename.endswith("-articles.xml.bz2"): # Common patterns
                    return f"{BASE_WIKIMEDIA_URL}{file_info['url']}"
            logger.warning(f"Could not find pages-articles.xml.bz2 in dumpstatus.json for {lang_code}")
            # Fallback to a common direct link pattern if parsing status fails
            return f"{BASE_WIKIMEDIA_URL}/{lang_code}wiki/latest/{lang_code}wiki-latest-pages-articles.xml.bz2"
        else: # Fallback if structure is different
            logger.warning(f"Could not parse dumpstatus.json for {lang_code} as expected. Falling back to direct URL pattern.")
            return f"{BASE_WIKIMEDIA_URL}/{lang_code}wiki/latest/{lang_code}wiki-latest-pages-articles.xml.bz2"

    except requests.RequestException as e:
        logger.error(f"Error fetching dump status for {lang_code}: {e}. Falling back to direct URL pattern.")
        return f"{BASE_WIKIMEDIA_URL}/{lang_code}wiki/latest/{lang_code}wiki-latest-pages-articles.xml.bz2"
    except KeyError as e:
        logger.error(f"KeyError parsing dump status JSON for {lang_code}: {e}. Falling back to direct URL pattern.")
        return f"{BASE_WIKIMEDIA_URL}/{lang_code}wiki/latest/{lang_code}wiki-latest-pages-articles.xml.bz2"


def download_wikipedia_dump(lang_code: str, dump_url: str, output_path: str) -> bool:
    """Downloads a Wikipedia dump using wget."""
    logger.info(f"Attempting to download Wikipedia dump for '{lang_code}' from: {dump_url}")
    logger.info(f"Output path: {output_path}")

    # Using wget for robustness (handles retries, etc.)
    # -c continues if interrupted
    # -O specifies output file
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Using subprocess.run for better control and error handling than os.system
        process = subprocess.run(
            ["wget", "-c", "-O", output_path, dump_url],
            capture_output=True, text=True, check=False # check=False to handle errors manually
        )
        if process.returncode == 0:
            logger.success(f"Successfully downloaded Wikipedia dump for '{lang_code}'.")
            return True
        else:
            logger.error(f"wget failed for '{lang_code}' with return code {process.returncode}.")
            logger.error(f"wget stdout: {process.stdout}")
            logger.error(f"wget stderr: {process.stderr}")
            # Attempt to remove partially downloaded file on failure
            if os.path.exists(output_path):
                 try: os.remove(output_path); logger.info(f"Removed partial download: {output_path}")
                 except: pass
            return False
    except FileNotFoundError:
        logger.error("wget command not found. Please ensure wget is installed and in your PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download for '{lang_code}': {e}")
        return False

def download_all_dumps():
    logger.info(f"Ensuring Wikipedia dump directory exists: {WIKIPEDIA_DUMP_DIR}")
    os.makedirs(WIKIPEDIA_DUMP_DIR, exist_ok=True)
    all_successful = True

    for lang in LANGUAGES_TO_PROCESS:
        dump_url = get_latest_dump_url(lang)
        if not dump_url:
            logger.error(f"Could not determine download URL for {lang}. Skipping.")
            all_successful = False
            continue
        
        # Infer filename from URL or construct it
        filename_from_url = dump_url.split('/')[-1]
        output_file_path = os.path.join(WIKIPEDIA_DUMP_DIR, filename_from_url)

        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 1000000: # Basic check if file exists and is >1MB
            logger.info(f"Wikipedia dump for '{lang}' ({filename_from_url}) already seems to exist. Skipping download.")
            continue
        
        if not download_wikipedia_dump(lang, dump_url, output_file_path):
            all_successful = False
            logger.error(f"Failed to download dump for {lang}.")
        else:
            logger.info(f"Download for {lang} complete.")
            
    if all_successful:
        logger.success("All specified Wikipedia dumps downloaded (or already existed).")
    else:
        logger.warning("One or more Wikipedia dump downloads failed or were skipped.")
    return all_successful

if __name__ == "__main__":
    # This part is for testing the downloader independently
    if download_all_dumps():
        logger.info("Downloader test finished successfully.")
    else:
        logger.error("Downloader test encountered errors.")