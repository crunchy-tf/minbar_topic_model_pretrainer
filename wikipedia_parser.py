# topic_model_pretrainer_cloud/wikipedia_parser.py
import os
import glob
from loguru import logger
from gensim.corpora import WikiCorpus # <--- Using gensim
from config_pretrainer import (
    WIKIPEDIA_DUMP_DIR,
    WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, # Output base for this script
    LANGUAGES_TO_PROCESS
)

# Define a clear article separator that's unlikely to appear naturally in text
ARTICLE_SEPARATOR = "\n<ARTICLE_SEPARATOR_MINBAR>\n"

def extract_text_with_gensim(lang_code: str, dump_file_path: str, output_dir_for_lang_extraction: str) -> bool:
    """
    Extracts text from a Wikipedia dump using gensim.corpora.WikiCorpus.
    Saves all articles for a language into a single text file.
    output_dir_for_lang_extraction is config_pretrainer.WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR
    """
    lang_output_basedir = os.path.join(output_dir_for_lang_extraction, lang_code)
    os.makedirs(lang_output_basedir, exist_ok=True)
    output_text_file = os.path.join(lang_output_basedir, f"{lang_code}_gensim_extracted_articles.txt")

    logger.info(f"Starting gensim extraction for '{lang_code}' from '{dump_file_path}'")
    logger.info(f"Output will be saved to: {output_text_file}")

    # Check if output file already exists and is reasonably sized to allow skipping
    if os.path.exists(output_text_file) and os.path.getsize(output_text_file) > 1024 * 1024: # e.g., > 1MB
        logger.info(f"Gensim output file '{output_text_file}' for '{lang_code}' already exists and is non-trivial. "
                      "Skipping extraction. Delete file to re-extract.")
        return True
    
    try:
        # MODIFIED LINE: Removed lemmatize=False as it's no longer supported
        # dictionary={} means don't build a dictionary during this pass.
        # lower=False to preserve casing initially (cleaning/lowering can happen later if needed).
        wiki_corpus = WikiCorpus(dump_file_path, dictionary={}, lower=False) 
        
        article_count = 0
        with open(output_text_file, 'w', encoding='utf-8') as f_out:
            # get_texts() yields a list of tokens (words) for each article.
            for article_tokens in wiki_corpus.get_texts():
                # Reconstruct the article text by joining tokens.
                article_text = " ".join(article_tokens)
                
                # Basic check to avoid writing tiny fragments if gensim produces them
                if len(article_text.strip()) > 50: # Arbitrary small length to ensure some content
                    f_out.write(article_text.strip() + ARTICLE_SEPARATOR)
                    article_count += 1
                
                if article_count > 0 and article_count % 20000 == 0: # Log progress every 20,000 articles
                    logger.info(f"[{lang_code}] Gensim processed and wrote {article_count} articles...")
        
        logger.success(f"[{lang_code}] Gensim extraction finished. Total articles written: {article_count} to {output_text_file}")
        return True

    except Exception as e:
        logger.error(f"Error during gensim parsing for language '{lang_code}' on dump '{dump_file_path}': {e}", exc_info=True)
        return False

def parse_all_downloaded_dumps():
    logger.info("Starting Wikipedia text extraction process using Gensim...")
    os.makedirs(WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, exist_ok=True)
    all_successful = True

    dump_patterns = [
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-pages-articles.xml.bz2"),
        os.path.join(WIKIPEDIA_DUMP_DIR, "*-latest-articles.xml.bz2") # Another common pattern
    ]
    all_found_dump_files = []
    for pattern in dump_patterns:
        all_found_dump_files.extend(glob.glob(pattern))
    
    if not all_found_dump_files:
        logger.error(f"No Wikipedia dump files (.xml.bz2 matching patterns) found in '{WIKIPEDIA_DUMP_DIR}'. "
                       "Ensure dumps are downloaded by wikipedia_downloader.py first.")
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
            if not extract_text_with_gensim(lang_code, lang_specific_dump_file, WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR):
                all_successful = False
                logger.error(f"Gensim extraction failed for language: {lang_code}.")
        else:
            logger.warning(f"No dump file found or accessible for language '{lang_code}' in '{WIKIPEDIA_DUMP_DIR}'. Skipping its extraction.")
            if lang_code in LANGUAGES_TO_PROCESS: # If a configured language dump is missing, consider it a failure for that language processing.
                 all_successful = False 

    if all_successful:
        logger.success("Gensim Wikipedia text extraction completed (or skipped if already done) for all specified languages.")
    else:
        logger.warning("One or more Gensim Wikipedia text extractions failed or were skipped.")
    return all_successful

if __name__ == "__main__":
    # This allows testing this module standalone
    logger.info("Testing wikipedia_parser.py (gensim version) standalone...")
    # For standalone test, ensure:
    # 1. config_pretrainer.py is correctly set up (esp. paths and LANGUAGES_TO_PROCESS).
    # 2. Wikipedia dumps are present in WIKIPEDIA_DUMP_DIR.
    if parse_all_downloaded_dumps():
        logger.info("Gensim Wikipedia parser module test finished successfully.")
    else:
        logger.error("Gensim Wikipedia parser module test encountered errors or incomplete processing.")