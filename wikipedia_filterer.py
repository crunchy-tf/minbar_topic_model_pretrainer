# topic_model_pretrainer_cloud/wikipedia_filterer.py
import os
import re
from loguru import logger
from typing import List, Dict, Optional, Tuple
import multiprocessing
from functools import partial

from config_pretrainer import (
    WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR,
    FILTERED_WIKIPEDIA_TEXT_DIR,
    LANGUAGES_TO_PROCESS,
    # Import the keyword dictionaries directly from config, which now imports them from separate files
    HEALTH_CATEGORIES_KEYWORDS_EN,
    HEALTH_CATEGORIES_KEYWORDS_FR, # These will be populated if defined & imported in config
    HEALTH_CATEGORIES_KEYWORDS_AR, # These will be populated if defined & imported in config
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH,
    FILTERING_PROCESSES,
    FILTERING_CHUNK_SIZE
)
from wikipedia_parser import ARTICLE_SEPARATOR

# This function now directly uses the imported dictionaries
def get_health_keywords_dict_for_lang(lang_code: str) -> Dict[str, List[str]]:
    if lang_code == "en": return HEALTH_CATEGORIES_KEYWORDS_EN
    elif lang_code == "fr": return HEALTH_CATEGORIES_KEYWORDS_FR
    elif lang_code == "ar": return HEALTH_CATEGORIES_KEYWORDS_AR
    else:
        logger.warning(f"No specific health keyword dictionary configured for lang '{lang_code}'.")
        return {}

# _check_relevance_worker function remains the same as the multiprocessing version
# It correctly receives lang_health_keywords_dict as an argument
def _check_relevance_worker(
    article_text_raw_for_worker: str,
    specific_lang_health_keywords_dict_for_worker: Dict[str, List[str]],
    general_health_keywords_for_worker: List[str],
    min_article_length_for_worker: int
    ) -> Optional[Tuple[str, str]]: 

    if not article_text_raw_for_worker or not article_text_raw_for_worker.strip():
        return None

    potential_title_lines = [l.strip() for l in article_text_raw_for_worker.split('\n') if l.strip()]
    title_heuristic = potential_title_lines[0] if potential_title_lines else "untitled_article"
    
    body_text_parts = article_text_raw_for_worker.split('\n', 1)
    body_text_to_save = body_text_parts[1].strip() if len(body_text_parts) > 1 else article_text_raw_for_worker
    
    full_text_for_check = simple_clean_text_for_filtering(title_heuristic + " " + body_text_to_save)

    if len(full_text_for_check) < min_article_length_for_worker:
        return None

    text_lower = full_text_for_check.lower()
    is_relevant = False
    if specific_lang_health_keywords_dict_for_worker: # Check if the passed dict is not empty
        for keywords_for_category in specific_lang_health_keywords_dict_for_worker.values():
            for keyword in keywords_for_category:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                    is_relevant = True; break
            if is_relevant: break
    
    if not is_relevant:
        for general_keyword in general_health_keywords_for_worker:
            if re.search(r'\b' + re.escape(general_keyword.lower()) + r'\b', text_lower):
                is_relevant = True; break
    
    if is_relevant:
        return title_heuristic, body_text_to_save
    return None

# simple_clean_text_for_filtering function remains the same
def simple_clean_text_for_filtering(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# filter_articles_for_language function remains the same as the multiprocessing version
def filter_articles_for_language(lang_code: str):
    input_gensim_file_path = os.path.join(WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, lang_code, f"{lang_code}_gensim_extracted_articles.txt")
    output_lang_filtered_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_filtered_dir, exist_ok=True)

    if not os.path.exists(input_gensim_file_path):
        logger.warning(f"Gensim parsed text file not found for '{lang_code}': {input_gensim_file_path}")
        return 0

    logger.info(f"Filtering relevant articles for '{lang_code}' from '{input_gensim_file_path}' using {FILTERING_PROCESSES} processes.")
    
    specific_lang_health_keywords_dict = get_health_keywords_dict_for_lang(lang_code)
    has_specific_keywords = bool(specific_lang_health_keywords_dict and any(specific_lang_health_keywords_dict.values()))

    if not has_specific_keywords and not GENERAL_HEALTH_KEYWORDS_FILTER:
        logger.error(f"No keywords for filtering '{lang_code}'. Aborting.")
        return 0
    elif not has_specific_keywords:
         logger.warning(f"No specific keywords for '{lang_code}'. Relying on GENERAL_HEALTH_KEYWORDS_FILTER.")

    articles_processed_count = 0
    relevant_articles_saved_count = 0
    
    process_func = partial(_check_relevance_worker,
                           specific_lang_health_keywords_dict_for_worker=specific_lang_health_keywords_dict,
                           general_health_keywords_for_worker=GENERAL_HEALTH_KEYWORDS_FILTER,
                           min_article_length_for_worker=MIN_FILTERED_ARTICLE_LENGTH)

    article_batch_for_pool = [] 

    try:
        with open(input_gensim_file_path, 'r', encoding='utf-8') as f_in, \
             multiprocessing.Pool(processes=FILTERING_PROCESSES) as pool:
            
            current_article_lines_buffer = []
            for line_content in f_in:
                if line_content.strip() == ARTICLE_SEPARATOR.strip():
                    if current_article_lines_buffer:
                        article_text_raw = "".join(current_article_lines_buffer).strip()
                        if article_text_raw: 
                            article_batch_for_pool.append(article_text_raw)
                            articles_processed_count += 1
                        
                        if len(article_batch_for_pool) >= FILTERING_CHUNK_SIZE:
                            results = pool.map(process_func, article_batch_for_pool)
                            for result_tuple in results:
                                if result_tuple:
                                    title_heuristic, body_text_to_save = result_tuple
                                    relevant_articles_saved_count += 1
                                    safe_title = re.sub(r'[^\w\.-]', '_', title_heuristic[:60])
                                    out_fn = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title}.txt")
                                    with open(out_fn, 'w', encoding='utf-8') as out_f:
                                        out_f.write(title_heuristic + "\n\n" + body_text_to_save)
                                    if relevant_articles_saved_count % 1000 == 0:
                                        logger.info(f"[{lang_code}] Saved {relevant_articles_saved_count} relevant articles (processed approx. {articles_processed_count}).")
                            article_batch_for_pool = [] 
                    current_article_lines_buffer = [] 
                else:
                    current_article_lines_buffer.append(line_content)
            
            if current_article_lines_buffer:
                article_text_raw = "".join(current_article_lines_buffer).strip()
                if article_text_raw: article_batch_for_pool.append(article_text_raw)
                articles_processed_count +=1
            
            if article_batch_for_pool:
                results = pool.map(process_func, article_batch_for_pool)
                for result_tuple in results:
                    if result_tuple:
                        title_heuristic, body_text_to_save = result_tuple
                        relevant_articles_saved_count += 1
                        safe_title = re.sub(r'[^\w\.-]', '_', title_heuristic[:60])
                        out_fn = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title}.txt")
                        with open(out_fn, 'w', encoding='utf-8') as out_f:
                            out_f.write(title_heuristic + "\n\n" + body_text_to_save)
                logger.info(f"[{lang_code}] Processed final batch. Total relevant saved for this lang: {relevant_articles_saved_count}.")

    except Exception as e_file:
        logger.error(f"Error processing file {input_gensim_file_path} with multiprocessing: {e_file}", exc_info=True)
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant saved: {relevant_articles_saved_count} from {articles_processed_count} articles read.")
    return relevant_articles_saved_count

# filter_all_extracted_wikipedia and if __name__ == "__main__": remain the same
def filter_all_extracted_wikipedia():
    logger.info("Starting filtering of all (gensim) extracted Wikipedia articles for health relevance...")
    os.makedirs(FILTERED_WIKIPEDIA_TEXT_DIR, exist_ok=True)
    total_relevant_across_all_langs = 0
    for lang_code in LANGUAGES_TO_PROCESS:
        total_relevant_across_all_langs += filter_articles_for_language(lang_code)
    
    if total_relevant_across_all_langs > 0:
        logger.success(f"Filtering complete. Total health-relevant articles saved across all processed languages: {total_relevant_across_all_langs}")
    else:
        logger.warning("Filtering complete, but no health-relevant articles were found or saved. Check keyword lists and gensim output quality.")
    return total_relevant_across_all_langs > 0

if __name__ == "__main__":
    logger.info("Testing wikipedia_filterer.py (gensim + multiprocessing version) standalone...")
    if filter_all_extracted_wikipedia():
        logger.info("Gensim Wikipedia filterer module test finished successfully.")
    else:
        logger.warning("Gensim Wikipedia filterer module test completed, but no relevant articles were found/saved.")