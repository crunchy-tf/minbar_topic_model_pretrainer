# topic_model_pretrainer_cloud/main_pretrainer.py
from loguru import logger
import os
import sys

# Add project root to sys.path to allow imports from sibling modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config_pretrainer import WORKING_DIR
from wikipedia_downloader import download_all_dumps
from wikipedia_parser import parse_all_downloaded_dumps
from wikipedia_filterer import filter_all_extracted_wikipedia
from bertopic_trainer import run_bertopic_training_pipeline # Renamed main function in bertopic_trainer

def main():
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
    
    logger.info("***** STARTING TOPIC MODEL PRETRAINER PIPELINE *****")
    os.makedirs(WORKING_DIR, exist_ok=True)
    logger.info(f"All intermediate and output files will be stored under: {WORKING_DIR}")

    # # Step 1: Download Wikipedia Dumps
    # logger.info("--- Step 1: Downloading Wikipedia Dumps ---")
    # if not download_all_dumps():
    #     logger.error("Failed to download all Wikipedia dumps. Aborting pipeline.")
    #     return
    # logger.success("Wikipedia dumps downloaded successfully (or already existed).")

    # # Step 2: Parse Dumps with WikiExtractor
    # logger.info("--- Step 2: Parsing Wikipedia Dumps with WikiExtractor ---")
    # if not parse_all_downloaded_dumps():
    #     logger.error("Failed to parse all Wikipedia dumps. Aborting pipeline.")
    #     return
    # logger.success("Wikipedia dumps parsed successfully.")

    # # Step 3: Filter Articles for Health Relevance
    # logger.info("--- Step 3: Filtering Wikipedia Articles for Health Relevance ---")
    # if not filter_all_extracted_wikipedia():
    #     logger.error("Failed to filter Wikipedia articles or no relevant articles found. Aborting pipeline.")
    #     return
    # logger.success("Wikipedia articles filtered successfully.")

    # Step 4: Run BERTopic Training Pipeline (which now includes Stage 1 (optional base) and Stage 2 (final guided))
    logger.info("--- Step 4: Running BERTopic Training Pipeline ---")
    run_bertopic_training_pipeline() # This will save the final guided model

    logger.info("***** TOPIC MODEL PRETRAINER PIPELINE FINISHED *****")

if __name__ == "__main__":
    main()