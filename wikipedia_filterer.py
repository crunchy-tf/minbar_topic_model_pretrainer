# topic_model_pretrainer_cloud/wikipedia_filterer.py
import os
import re
from loguru import logger
from typing import List, Dict, Optional

from config_pretrainer import (
    WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, # Input for this script
    FILTERED_WIKIPEDIA_TEXT_DIR,         # Output for this script
    LANGUAGES_TO_PROCESS,
    # These will be dictionaries of {'topic_key': ['kw1', 'kw2']}
    HEALTH_CATEGORIES_KEYWORDS_EN,
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH
)
from wikipedia_parser import ARTICLE_SEPARATOR # Import the separator used by gensim parser

# Attempt to import FR and AR keyword dictionaries if they might be defined in config
try: from config_pretrainer import HEALTH_CATEGORIES_KEYWORDS_FR
except ImportError: HEALTH_CATEGORIES_KEYWORDS_FR = {}
try: from config_pretrainer import HEALTH_CATEGORIES_KEYWORDS_AR
except ImportError: HEALTH_CATEGORIES_KEYWORDS_AR = {}


def get_health_keywords_dict_for_lang(lang_code: str) -> Dict[str, List[str]]:
    """Returns the health topic keyword dictionary for the language."""
    if lang_code == "en": return HEALTH_CATEGORIES_KEYWORDS_EN
    if lang_code == "fr": return HEALTH_CATEGORIES_KEYWORDS_FR
    if lang_code == "ar": return HEALTH_CATEGORIES_KEYWORDS_AR
    logger.warning(f"No specific health keyword dictionary defined for lang '{lang_code}' in config_pretrainer.py.")
    return {}

def check_article_relevance_by_keywords(
    article_full_text: str, # Gensim provides full text, title might be first line
    lang_health_keywords_dict: Dict[str, List[str]],
    general_health_keywords: List[str]
) -> bool:
    """
    Checks relevance by searching for keywords within the article text.
    """
    if not article_full_text.strip():
        return False
        
    text_lower = article_full_text.lower()

    # Check against specific health category keywords from HEALTH_CATEGORIES_KEYWORDS_XX
    if lang_health_keywords_dict:
        for category_key, keywords_for_category in lang_health_keywords_dict.items():
            for keyword in keywords_for_category:
                # Use regex for whole word matching to avoid partial matches within larger words
                # \b ensures word boundaries. re.escape handles special characters in keyword.
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                    logger.trace(f"Relevant by specific keyword: '{keyword}' (from category '{category_key}')")
                    return True
    
    # Fallback to general health keywords
    for general_keyword in general_health_keywords:
        if re.search(r'\b' + re.escape(general_keyword.lower()) + r'\b', text_lower):
            logger.trace(f"Relevant by general keyword: '{general_keyword}'")
            return True
            
    return False

def simple_clean_text_for_filtering(text: str) -> str:
    """Basic cleaning, primarily for length check and reducing noise before keyword search."""
    text = re.sub(r'<[^>]+>', ' ', text) # Remove any lingering HTML-like tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_articles_for_language(lang_code: str):
    input_gensim_file = os.path.join(WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, lang_code, f"{lang_code}_gensim_extracted_articles.txt")
    output_lang_filtered_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_filtered_dir, exist_ok=True)

    if not os.path.exists(input_gensim_file):
        logger.warning(f"Gensim parsed text file not found for language '{lang_code}': {input_gensim_file}")
        return 0

    logger.info(f"Filtering relevant articles for language '{lang_code}' from gensim output: '{input_gensim_file}'")
    
    specific_lang_health_keywords_dict = get_health_keywords_dict_for_lang(lang_code)
    if not specific_lang_health_keywords_dict:
        logger.warning(f"No specific health keyword dictionary found for lang '{lang_code}'. "
                       "Filtering will rely solely on GENERAL_HEALTH_KEYWORDS_FILTER.")

    articles_processed_count = 0
    relevant_articles_saved_count = 0

    try:
        with open(input_gensim_file, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
            articles_data = content.split(ARTICLE_SEPARATOR)
            
            logger.info(f"[{lang_code}] Processing approximately {len(articles_data)} articles from gensim output file.")

            for i, article_text_raw in enumerate(articles_data):
                articles_processed_count = i + 1
                if not article_text_raw.strip():
                    continue

                # Heuristic: assume first non-empty line is title, rest is body
                lines = article_text_raw.strip().split('\n', 1)
                title_heuristic = lines[0].strip()
                body_text = lines[1].strip() if len(lines) > 1 else ""
                if not body_text: # If only title was found, use title as body for filtering
                    body_text = title_heuristic
                
                # Use the full article text (title + body) for relevance check and length check
                full_text_for_check = title_heuristic + " " + body_text
                cleaned_for_length_check = simple_clean_text_for_filtering(full_text_for_check)

                if len(cleaned_for_length_check) >= MIN_FILTERED_ARTICLE_LENGTH:
                    if check_article_relevance_by_keywords(full_text_for_check, 
                                                           specific_lang_health_keywords_dict, 
                                                           GENERAL_HEALTH_KEYWORDS_FILTER):
                        relevant_articles_saved_count += 1
                        
                        # Save the relevant article (title heuristic + body)
                        out_filename = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}.txt")
                        with open(out_filename, 'w', encoding='utf-8') as out_f:
                            out_f.write(title_heuristic + "\n\n" + body_text) 
                        
                        if relevant_articles_saved_count % 200 == 0:
                            logger.info(f"[{lang_code}] Saved {relevant_articles_saved_count} health-relevant articles.")
                
                if articles_processed_count % 10000 == 0: # Log progress every 10k articles processed
                    logger.debug(f"[{lang_code}] Filtered {articles_processed_count} articles from gensim output...")

    except Exception as e_file:
        logger.error(f"Error processing gensim output file {input_gensim_file}: {e_file}", exc_info=True)
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant articles saved: {relevant_articles_saved_count} "
                f"from {articles_processed_count} processed articles.")
    return relevant_articles_saved_count

def filter_all_extracted_wikipedia():
    logger.info("Starting filtering of all (gensim) extracted Wikipedia articles for health relevance...")
    os.makedirs(FILTERED_WIKIPEDIA_TEXT_DIR, exist_ok=True)
    total_relevant_across_all_langs = 0

    for lang_code in LANGUAGES_TO_PROCESS: # This will be ["en"] from current config
        total_relevant_across_all_langs += filter_articles_for_language(lang_code)
    
    if total_relevant_across_all_langs > 0:
        logger.success(f"Filtering complete. Total health-relevant articles saved: {total_relevant_across_all_langs}")
    else:
        logger.warning("Filtering complete, but no health-relevant articles found/saved. "
                       "Check HEALTH_CATEGORIES_KEYWORDS_XX in config and gensim output quality.")
    return total_relevant_across_all_langs > 0

if __name__ == "__main__":
    logger.info("Testing wikipedia_filterer.py (gensim version) standalone...")
    if filter_all_extracted_wikipedia():
        logger.info("Gensim Wikipedia filterer module test finished successfully.")
    else:
        logger.warning("Gensim Wikipedia filterer module test completed, but no relevant articles found/saved.")