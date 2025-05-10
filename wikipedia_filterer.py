# topic_model_pretrainer_cloud/wikipedia_filterer.py
import os
import glob
import json
import re
from loguru import logger
from typing import List, Dict, Optional 
from config_pretrainer import (
    WIKIPEDIA_EXTRACTED_DIR,
    FILTERED_WIKIPEDIA_TEXT_DIR,
    LANGUAGES_TO_PROCESS,
    HEALTH_CATEGORIES_FILTER_EN,
    HEALTH_CATEGORIES_FILTER_FR,
    HEALTH_CATEGORIES_FILTER_AR,
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH
)

def get_health_categories_for_lang(lang_code: str) -> list:
    if lang_code == "en": return HEALTH_CATEGORIES_FILTER_EN
    if lang_code == "fr": return HEALTH_CATEGORIES_FILTER_FR
    if lang_code == "ar": return HEALTH_CATEGORIES_FILTER_AR
    return []

def check_article_relevance(title: str, text_content: str, categories_from_json: List[str], 
                              lang_health_categories: List[str]) -> bool:
    """
    Checks if an article is relevant based on explicit categories or keywords in title/text.
    """
    # 1. Check explicit categories first (most reliable if wikiextractor provides them well)
    if categories_from_json:
        # Normalize categories from JSON (lowercase, strip)
        normalized_json_cats = {cat.lower().strip() for cat in categories_from_json}
        for health_cat_phrase in lang_health_categories:
            # Check if any part of the multi-word health category phrase is in the article's categories
            # This is a simple check; more sophisticated category tree traversal could be better.
            if health_cat_phrase.lower() in normalized_json_cats: # Exact match of a category phrase
                 logger.trace(f"Relevant by category (exact): '{health_cat_phrase}' in article categories for title '{title}'")
                 return True
            # Check if any keyword from health_cat_phrase is present as a substring in any of the article's categories
            # This is broader and might catch sub-categories.
            for health_keyword in health_cat_phrase.split():
                if len(health_keyword) < 3: continue # Skip very short keywords
                for article_cat in normalized_json_cats:
                    if health_keyword in article_cat:
                        logger.trace(f"Relevant by category (substring): '{health_keyword}' in article category '{article_cat}' for title '{title}'")
                        return True
    
    # 2. Fallback: Check title and first ~500 chars of text for general health keywords
    # This helps if category information from wikiextractor is sparse or poorly parsed.
    search_corpus = (title + " " + text_content[:500]).lower()
    for keyword in GENERAL_HEALTH_KEYWORDS_FILTER: # Assuming GENERAL_HEALTH_KEYWORDS_FILTER contains lowercase
        if keyword in search_corpus:
            logger.trace(f"Relevant by keyword: '{keyword}' in title/text for title '{title}'")
            return True
            
    return False


def clean_wiki_text(text: str) -> str:
    """Basic cleaning of extracted Wikipedia text."""
    # Remove residual MediaWiki section headings (e.g., == Section ==, === Sub-section ===)
    text = re.sub(r'^=+\s*(.*?)\s*=+\s*$', '', text, flags=re.MULTILINE)
    # Remove newlines that are not paragraph breaks (heuristics)
    # This is tricky. For now, just normalize multiple newlines to one.
    text = re.sub(r'\n{2,}', '\n', text) # Keep paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # Replace single newlines (likely mid-sentence) with space
    text = re.sub(r'\s{2,}', ' ', text).strip() # Normalize multiple spaces
    return text


def filter_articles_for_language(lang_code: str):
    input_lang_dir = os.path.join(WIKIPEDIA_EXTRACTED_DIR, lang_code)
    output_lang_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_dir, exist_ok=True)

    if not os.path.isdir(input_lang_dir):
        logger.warning(f"Input directory for extracted text not found for language '{lang_code}': {input_lang_dir}")
        return 0

    logger.info(f"Filtering relevant articles for language '{lang_code}' from '{input_lang_dir}'")
    
    health_categories_for_lang = get_health_categories_for_lang(lang_code)
    if not health_categories_for_lang:
        logger.warning(f"No health category filters defined for language '{lang_code}'. Relying on general keywords only.")

    total_files_processed = 0
    relevant_articles_found = 0

    # wikiextractor creates subdirs like AA, AB, etc.
    for subdir_name in os.listdir(input_lang_dir):
        subdir_path = os.path.join(input_lang_dir, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in glob.glob(os.path.join(subdir_path, "wiki_*")): # Process files from wikiextractor
                total_files_processed += 1
                if total_files_processed % 100 == 0:
                    logger.debug(f"[{lang_code}] Processed {total_files_processed} input files...")
                
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        for line_number, line in enumerate(f):
                            try:
                                data = json.loads(line)
                                article_id = data.get("id", f"unknown_{relevant_articles_found}")
                                title = data.get("title", "")
                                text_content = data.get("text", "")
                                # Assuming 'categories' is a list of strings in the JSON if wikiextractor provided it.
                                # This is an OPTIMISTIC assumption. You may need to parse categories from raw wikitext if not.
                                categories_from_json = data.get("categories", []) 
                                if not isinstance(categories_from_json, list): categories_from_json = []


                                if text_content and len(text_content) >= MIN_FILTERED_ARTICLE_LENGTH:
                                    if check_article_relevance(title, text_content, categories_from_json, health_categories_for_lang):
                                        relevant_articles_found += 1
                                        cleaned_text = clean_wiki_text(text_content)
                                        
                                        out_filename = os.path.join(output_lang_dir, f"{lang_code}_{article_id}.txt")
                                        with open(out_filename, 'w', encoding='utf-8') as out_f:
                                            out_f.write(title + "\n\n" + cleaned_text) # Save title and cleaned text
                                        
                                        if relevant_articles_found % 500 == 0:
                                            logger.info(f"[{lang_code}] Found {relevant_articles_found} health-relevant articles so far.")
                            except json.JSONDecodeError:
                                logger.trace(f"Skipping non-JSON line in {filename}:{line_number+1}")
                                continue
                            except Exception as e_line:
                                logger.warning(f"Error processing line in {filename}:{line_number+1} - {e_line}")
                                continue
                except Exception as e_file:
                    logger.error(f"Could not process file {filename}: {e_file}")
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant articles found: {relevant_articles_found}")
    return relevant_articles_found

def filter_all_extracted_wikipedia():
    logger.info("Starting filtering of all extracted Wikipedia articles for health relevance...")
    os.makedirs(FILTERED_WIKIPEDIA_TEXT_DIR, exist_ok=True)
    total_relevant = 0
    for lang in LANGUAGES_TO_PROCESS:
        total_relevant += filter_articles_for_language(lang)
    
    if total_relevant > 0:
        logger.success(f"Filtering complete. Total health-relevant articles saved: {total_relevant}")
    else:
        logger.warning("Filtering complete, but no health-relevant articles were found or saved. Check configurations and input data.")
    return total_relevant > 0


if __name__ == "__main__":
    if filter_all_extracted_wikipedia():
        logger.info("Filterer test finished successfully.")
    else:
        logger.error("Filterer test encountered errors or found no relevant articles.")