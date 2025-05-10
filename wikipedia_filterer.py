# topic_model_pretrainer_cloud/wikipedia_filterer.py
import os
import glob
import json
import re
from loguru import logger
from typing import List, Dict, Optional # Keep these for good practice

# Import only what's guaranteed to be in config_pretrainer.py
# when processing only English, or what's generally needed.
from config_pretrainer import (
    WIKIPEDIA_EXTRACTED_DIR,
    FILTERED_WIKIPEDIA_TEXT_DIR,
    LANGUAGES_TO_PROCESS,       # This will determine which categories are looked up
    HEALTH_CATEGORIES_FILTER_EN,# Required if "en" is in LANGUAGES_TO_PROCESS
    # If you decide to process FR or AR later, you'd un-comment their
    # category filter definitions in config_pretrainer.py and then this script
    # would pick them up via get_health_categories_for_lang.
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH
)
# Attempt to import FR and AR category filters, but don't fail if they're not there
# This allows config_pretrainer.py to only define what's actively being processed.
try:
    from config_pretrainer import HEALTH_CATEGORIES_FILTER_FR
except ImportError:
    HEALTH_CATEGORIES_FILTER_FR = [] # Default to empty if not defined
try:
    from config_pretrainer import HEALTH_CATEGORIES_FILTER_AR
except ImportError:
    HEALTH_CATEGORIES_FILTER_AR = [] # Default to empty if not defined


def get_health_categories_for_lang(lang_code: str) -> list:
    """
    Returns the appropriate health category filter list for the given language code.
    Relies on constants being defined in config_pretrainer.py if that language is active.
    """
    if lang_code == "en":
        # HEALTH_CATEGORIES_FILTER_EN is imported directly and must exist if 'en' is processed
        return HEALTH_CATEGORIES_FILTER_EN
    elif lang_code == "fr":
        # HEALTH_CATEGORIES_FILTER_FR is imported via try-except
        return HEALTH_CATEGORIES_FILTER_FR
    elif lang_code == "ar":
        # HEALTH_CATEGORIES_FILTER_AR is imported via try-except
        return HEALTH_CATEGORIES_FILTER_AR
    else:
        logger.warning(f"No specific health category filter list configured for language code: '{lang_code}'. "
                       "Relevance check will rely more on general keywords for this language if processed.")
        return []


def check_article_relevance(title: str, text_content: str, categories_from_json: List[str],
                              lang_health_categories: List[str]) -> bool:
    """
    Checks if an article is relevant based on explicit categories or keywords in title/text.
    'lang_health_categories' is the specific list for the current language being processed.
    """
    # 1. Check explicit categories first (most reliable if wikiextractor provides them well)
    if categories_from_json and lang_health_categories: # Only check if we have categories for this lang
        normalized_json_cats = {cat.lower().strip() for cat in categories_from_json}
        for health_cat_phrase in lang_health_categories:
            # health_cat_phrase is a string from your config list, e.g., "public health"
            if health_cat_phrase.lower() in normalized_json_cats: # Exact match
                logger.trace(f"Relevant by category (exact match): '{health_cat_phrase}' in article categories for title '{title}'")
                return True
            # Optional: Broader check for keywords within category phrases
            # for health_keyword in health_cat_phrase.split():
            #     if len(health_keyword) > 2: # Avoid very short "keywords"
            #         for article_cat_tokenized in normalized_json_cats:
            #             if health_keyword in article_cat_tokenized.split(): # Check against tokenized article cats
            #                 logger.trace(f"Relevant by category (keyword substring): '{health_keyword}' in article category '{article_cat_tokenized}' for title '{title}'")
            #                 return True
    
    # 2. Fallback: Check title and first ~500 chars of text for general health keywords
    # This helps if category information from wikiextractor is sparse or poorly parsed,
    # or if no specific lang_health_categories were provided.
    search_corpus = (title + " " + text_content[:500]).lower()
    for keyword in GENERAL_HEALTH_KEYWORDS_FILTER:
        # GENERAL_HEALTH_KEYWORDS_FILTER can contain keywords from all languages
        if keyword.lower() in search_corpus:
            logger.trace(f"Relevant by general keyword: '{keyword}' in title/text for title '{title}'")
            return True
            
    return False


def clean_wiki_text(text: str) -> str:
    """Basic cleaning of extracted Wikipedia text."""
    # Remove residual MediaWiki section headings (e.g., == Section ==, === Sub-section ===)
    text = re.sub(r'^=+\s*(.*?)\s*=+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\[Category:.*?\]\]', '', text) # Remove category tags if still present
    # Remove newlines that are not paragraph breaks (heuristics)
    text = re.sub(r'\n{2,}', '\n', text) # Keep paragraph breaks (becomes single \n)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # Replace single newlines (likely mid-sentence) with space
    text = re.sub(r'\s{2,}', ' ', text).strip() # Normalize multiple spaces
    return text


def filter_articles_for_language(lang_code: str):
    input_lang_dir = os.path.join(WIKIPEDIA_EXTRACTED_DIR, lang_code)
    output_lang_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_dir, exist_ok=True)

    if not os.path.isdir(input_lang_dir):
        logger.warning(f"Input directory for extracted text not found for language '{lang_code}': {input_lang_dir}")
        return 0 # No articles to process

    logger.info(f"Filtering relevant articles for language '{lang_code}' from '{input_lang_dir}'")
    
    # Get the specific category list for the current language
    health_categories_for_this_lang = get_health_categories_for_lang(lang_code)
    if not health_categories_for_this_lang:
        logger.warning(f"No specific health category filters found for language '{lang_code}' in config. "
                       "Filtering will rely solely on GENERAL_HEALTH_KEYWORDS_FILTER for this language.")

    total_files_processed = 0
    relevant_articles_found = 0

    # WikiExtractorV2 output is often nested (e.g., AA/wiki_00, AB/wiki_01)
    for root, dirs, files in os.walk(input_lang_dir):
        for filename in files:
            # Process only files that look like WikiExtractor output (e.g., "wiki_XX")
            # This check might need adjustment based on actual WikiExtractorV2 output filenames
            if not filename.startswith("wiki_"): 
                continue

            filepath = os.path.join(root, filename)
            total_files_processed += 1
            if total_files_processed % 500 == 0: # Log progress less frequently for file iteration
                logger.debug(f"[{lang_code}] Scanned {total_files_processed} input files from WikiExtractor output...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f):
                        try:
                            data = json.loads(line) # Expecting JSONL format from WikiExtractorV2 with --json
                            article_id = data.get("id", f"unknownid_{relevant_articles_found}")
                            title = data.get("title", "")
                            text_content = data.get("text", "")
                            
                            # How WikiExtractorV2 --json outputs categories needs to be confirmed.
                            # The README mentions "doc_id,title,url,languages,text" for --generator.
                            # For non-generator --json, it might be a "categories": [...] field.
                            # Assuming it's a list in a "categories" field for now.
                            # If not, this part will need adjustment after inspecting V2's JSON output.
                            categories_from_json = data.get("categories", []) 
                            if not isinstance(categories_from_json, list): 
                                categories_from_json = []

                            if text_content and len(text_content) >= MIN_FILTERED_ARTICLE_LENGTH:
                                if check_article_relevance(title, text_content, categories_from_json, health_categories_for_this_lang):
                                    relevant_articles_found += 1
                                    cleaned_text = clean_wiki_text(text_content)
                                    
                                    # Ensure article_id is filesystem-safe
                                    safe_article_id = re.sub(r'[^\w\.-]', '_', str(article_id))
                                    out_filename = os.path.join(output_lang_dir, f"{lang_code}_{safe_article_id}.txt")
                                    
                                    with open(out_filename, 'w', encoding='utf-8') as out_f:
                                        out_f.write(title + "\n\n" + cleaned_text)
                                    
                                    if relevant_articles_found % 200 == 0: # Log progress for found articles
                                        logger.info(f"[{lang_code}] Found and saved {relevant_articles_found} health-relevant articles so far.")
                        except json.JSONDecodeError:
                            logger.trace(f"Skipping non-JSON line in {filepath}:{line_number+1}")
                            continue
                        except Exception as e_line:
                            logger.warning(f"Error processing line in {filepath}:{line_number+1} - {e_line}")
                            continue
            except Exception as e_file:
                logger.error(f"Could not process file {filepath}: {e_file}")
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant articles found and saved: {relevant_articles_found} "
                f"from {total_files_processed} WikiExtractor output files scanned.")
    return relevant_articles_found

def filter_all_extracted_wikipedia():
    logger.info("Starting filtering of all extracted Wikipedia articles for health relevance...")
    os.makedirs(FILTERED_WIKIPEDIA_TEXT_DIR, exist_ok=True)
    total_relevant_across_all_langs = 0

    # LANGUAGES_TO_PROCESS will now correctly reflect what's in config (e.g., just ["en"])
    for lang_code in LANGUAGES_TO_PROCESS:
        total_relevant_across_all_langs += filter_articles_for_language(lang_code)
    
    if total_relevant_across_all_langs > 0:
        logger.success(f"Filtering complete. Total health-relevant articles saved across all processed languages: {total_relevant_across_all_langs}")
    else:
        logger.warning("Filtering complete, but no health-relevant articles were found or saved for the processed languages. "
                       "Check configurations, WikiExtractor output, and category filter lists.")
    return total_relevant_across_all_langs > 0


if __name__ == "__main__":
    # For standalone testing of this filterer module
    # Ensure config_pretrainer.py is set up, esp. LANGUAGES_TO_PROCESS and category lists.
    # Ensure WikiExtractorV2 has run and populated WIKIPEDIA_EXTRACTED_DIR/[lang_code]
    logger.info("Testing wikipedia_filterer.py standalone...")
    if filter_all_extracted_wikipedia():
        logger.info("Wikipedia filterer module test finished successfully (found relevant articles).")
    else:
        logger.warning("Wikipedia filterer module test completed, but no relevant articles were found/saved.")