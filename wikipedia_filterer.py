# topic_model_pretrainer_cloud/wikipedia_filterer.py
import os
import re
from loguru import logger
from typing import List, Dict, Optional # Keep these for good practice

from config_pretrainer import (
    WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, # Input for this script
    FILTERED_WIKIPEDIA_TEXT_DIR,         # Output for this script
    LANGUAGES_TO_PROCESS,
    HEALTH_CATEGORIES_KEYWORDS_EN,       # Now using keyword lists from config
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH
)
# Assuming wikipedia_parser.py is in the same directory or accessible in PYTHONPATH
from wikipedia_parser import ARTICLE_SEPARATOR # Import the separator used by gensim parser

# Attempt to import FR and AR keyword dictionaries if they might be defined in config
# This makes the script flexible if you later re-add FR/AR to LANGUAGES_TO_PROCESS
# and define their keyword dictionaries in config_pretrainer.py
try: from config_pretrainer import HEALTH_CATEGORIES_KEYWORDS_FR
except ImportError: HEALTH_CATEGORIES_KEYWORDS_FR = {}
try: from config_pretrainer import HEALTH_CATEGORIES_KEYWORDS_AR
except ImportError: HEALTH_CATEGORIES_KEYWORDS_AR = {}


def get_health_keywords_dict_for_lang(lang_code: str) -> Dict[str, List[str]]:
    """Returns the health topic keyword dictionary for the language."""
    if lang_code == "en": return HEALTH_CATEGORIES_KEYWORDS_EN
    elif lang_code == "fr": return HEALTH_CATEGORIES_KEYWORDS_FR
    elif lang_code == "ar": return HEALTH_CATEGORIES_KEYWORDS_AR
    else:
        logger.warning(f"No specific health keyword dictionary defined for lang '{lang_code}' in config_pretrainer.py.")
        return {}

def check_article_relevance_by_keywords(
    article_full_text: str, 
    lang_health_keywords_dict: Dict[str, List[str]],
    general_health_keywords: List[str]
) -> bool:
    """
    Checks relevance by searching for keywords within the article text.
    """
    if not article_full_text or not article_full_text.strip():
        return False
        
    text_lower = article_full_text.lower()

    # Check against specific health category keywords from HEALTH_CATEGORIES_KEYWORDS_XX
    if lang_health_keywords_dict: # Only proceed if there are specific keywords for this language
        for category_key, keywords_for_category in lang_health_keywords_dict.items():
            for keyword in keywords_for_category:
                # Use regex for whole word matching to avoid partial matches within larger words
                # re.escape handles special characters in the keyword string itself.
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                    logger.trace(f"Relevant by specific keyword: '{keyword}' (from category '{category_key}') in article snippet: '{article_full_text[:100]}...'")
                    return True
    
    # Fallback to general health keywords if no specific match or if lang_health_keywords_dict is empty
    for general_keyword in general_health_keywords:
        if re.search(r'\b' + re.escape(general_keyword.lower()) + r'\b', text_lower):
            logger.trace(f"Relevant by general keyword: '{general_keyword}' in article snippet: '{article_full_text[:100]}...'")
            return True
            
    return False

def simple_clean_text_for_filtering(text: str) -> str:
    """Basic cleaning, primarily for length check and reducing noise before keyword search."""
    # Remove any potential HTML tags that might have slipped through gensim
    text = re.sub(r'<[^>]+>', ' ', text) 
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_articles_for_language(lang_code: str):
    # Input is now a single large file from gensim parser
    input_gensim_file_path = os.path.join(WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, lang_code, f"{lang_code}_gensim_extracted_articles.txt")
    output_lang_filtered_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_filtered_dir, exist_ok=True)

    if not os.path.exists(input_gensim_file_path):
        logger.warning(f"Gensim parsed text file not found for language '{lang_code}': {input_gensim_file_path}")
        return 0 # No articles to process

    logger.info(f"Filtering relevant articles for language '{lang_code}' from gensim output: '{input_gensim_file_path}'")
    
    specific_lang_health_keywords_dict = get_health_keywords_dict_for_lang(lang_code)
    if not specific_lang_health_keywords_dict and not GENERAL_HEALTH_KEYWORDS_FILTER:
        logger.error(f"No specific or general health keywords defined for filtering language '{lang_code}'. "
                       "This will result in no articles being filtered as relevant. Aborting filtering for this language.")
        return 0
    elif not specific_lang_health_keywords_dict:
         logger.warning(f"No specific health keyword dictionary found for lang '{lang_code}'. "
                       "Filtering will rely solely on GENERAL_HEALTH_KEYWORDS_FILTER.")


    articles_processed_count = 0
    relevant_articles_saved_count = 0
    current_article_lines_buffer = [] # Buffer to hold lines of the current article

    try:
        with open(input_gensim_file_path, 'r', encoding='utf-8') as f_in:
            for line_number, line_content in enumerate(f_in):
                # Check if the current line is the article separator
                # .strip() handles cases where separator might have leading/trailing whitespace from read
                if line_content.strip() == ARTICLE_SEPARATOR.strip():
                    if current_article_lines_buffer: # Process the accumulated article
                        article_text_raw = "".join(current_article_lines_buffer).strip()
                        articles_processed_count += 1
                        
                        # Heuristic for title: first non-empty line of the raw article text
                        potential_title_lines = [l.strip() for l in article_text_raw.split('\n') if l.strip()]
                        title_heuristic = potential_title_lines[0] if potential_title_lines else "untitled_article"
                        
                        # For relevance check and length, use the raw (but whitespace normalized) text
                        full_text_for_check = simple_clean_text_for_filtering(article_text_raw)

                        if len(full_text_for_check) >= MIN_FILTERED_ARTICLE_LENGTH:
                            if check_article_relevance_by_keywords(full_text_for_check, 
                                                                   specific_lang_health_keywords_dict, 
                                                                   GENERAL_HEALTH_KEYWORDS_FILTER):
                                relevant_articles_saved_count += 1
                                
                                # Prepare content for saving (title + body)
                                # The body is the raw article text after the first line (heuristic title)
                                body_text_parts = article_text_raw.split('\n', 1)
                                body_text_to_save = body_text_parts[1].strip() if len(body_text_parts) > 1 else article_text_raw # Fallback if no clear body
                                
                                safe_title_for_filename = re.sub(r'[^\w\.-]', '_', title_heuristic[:60]) # Sanitize title for filename
                                out_filename = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title_for_filename}.txt")
                                
                                with open(out_filename, 'w', encoding='utf-8') as out_f:
                                    out_f.write(title_heuristic + "\n\n" + body_text_to_save) 
                                
                                if relevant_articles_saved_count > 0 and relevant_articles_saved_count % 200 == 0:
                                    logger.info(f"[{lang_code}] Saved {relevant_articles_saved_count} health-relevant articles.")
                        
                        if articles_processed_count > 0 and articles_processed_count % 20000 == 0:
                            logger.debug(f"[{lang_code}] Filtered {articles_processed_count} articles from gensim output...")
                    
                    current_article_lines_buffer = [] # Reset buffer for the next article
                else:
                    current_article_lines_buffer.append(line_content)
            
            # After the loop, process any remaining content in the buffer (for the last article if no trailing separator)
            if current_article_lines_buffer:
                article_text_raw = "".join(current_article_lines_buffer).strip()
                if article_text_raw: # Ensure there's content
                    articles_processed_count += 1
                    potential_title_lines = [l.strip() for l in article_text_raw.split('\n') if l.strip()]
                    title_heuristic = potential_title_lines[0] if potential_title_lines else "untitled_article_eof"
                    body_text_parts = article_text_raw.split('\n', 1)
                    body_text_to_save = body_text_parts[1].strip() if len(body_text_parts) > 1 else article_text_raw
                    
                    full_text_for_check = simple_clean_text_for_filtering(title_heuristic + " " + body_text_to_save)
                    if len(full_text_for_check) >= MIN_FILTERED_ARTICLE_LENGTH:
                        if check_article_relevance_by_keywords(full_text_for_check, 
                                                               specific_lang_health_keywords_dict, 
                                                               GENERAL_HEALTH_KEYWORDS_FILTER):
                            relevant_articles_saved_count += 1
                            safe_title_for_filename = re.sub(r'[^\w\.-]', '_', title_heuristic[:60])
                            out_filename = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title_for_filename}.txt")
                            with open(out_filename, 'w', encoding='utf-8') as out_f:
                                out_f.write(title_heuristic + "\n\n" + body_text_to_save)
                            logger.info(f"[{lang_code}] Saved last relevant article (total: {relevant_articles_saved_count}).")

    except FileNotFoundError: # Should be caught by the os.path.exists check earlier
        logger.error(f"Input file not found during iteration: {input_gensim_file_path}")
    except Exception as e_file:
        logger.error(f"Error processing gensim output file {input_gensim_file_path}: {e_file}", exc_info=True)
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant articles saved: {relevant_articles_saved_count} "
                f"from {articles_processed_count} processed articles.")
    return relevant_articles_saved_count

def filter_all_extracted_wikipedia():
    logger.info("Starting filtering of all (gensim) extracted Wikipedia articles for health relevance...")
    os.makedirs(FILTERED_WIKIPEDIA_TEXT_DIR, exist_ok=True)
    total_relevant_across_all_langs = 0

    for lang_code in LANGUAGES_TO_PROCESS:
        total_relevant_across_all_langs += filter_articles_for_language(lang_code)
    
    if total_relevant_across_all_langs > 0:
        logger.success(f"Filtering complete. Total health-relevant articles saved: {total_relevant_across_all_langs}")
    else:
        logger.warning("Filtering complete, but no health-relevant articles were found/saved. "
                       "Check HEALTH_CATEGORIES_KEYWORDS_XX in config and gensim output quality.")
    return total_relevant_across_all_langs > 0

if __name__ == "__main__":
    logger.info("Testing wikipedia_filterer.py (gensim version) standalone...")
    if filter_all_extracted_wikipedia():
        logger.info("Gensim Wikipedia filterer module test finished successfully.")
    else:
        logger.warning("Gensim Wikipedia filterer module test completed, but no relevant articles were found/saved.")