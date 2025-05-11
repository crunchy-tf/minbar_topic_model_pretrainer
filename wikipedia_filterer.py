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
    HEALTH_CATEGORIES_KEYWORDS_EN,
    HEALTH_CATEGORIES_KEYWORDS_FR,
    HEALTH_CATEGORIES_KEYWORDS_AR,
    GENERAL_HEALTH_KEYWORDS_FILTER,
    MIN_FILTERED_ARTICLE_LENGTH,
    FILTERING_PROCESSES,
    FILTERING_CHUNK_SIZE,
    MAX_ARTICLES_TO_READ_FROM_GENSIM_PER_LANG # New import
)
from wikipedia_parser import ARTICLE_SEPARATOR

def get_health_keywords_dict_for_lang(lang_code: str) -> Dict[str, List[str]]:
    if lang_code == "en": return HEALTH_CATEGORIES_KEYWORDS_EN
    elif lang_code == "fr": return HEALTH_CATEGORIES_KEYWORDS_FR
    elif lang_code == "ar": return HEALTH_CATEGORIES_KEYWORDS_AR
    else:
        logger.warning(f"No specific health keyword dictionary for lang '{lang_code}'.")
        return {}

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
    if specific_lang_health_keywords_dict_for_worker:
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

def simple_clean_text_for_filtering(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_articles_for_language(lang_code: str):
    input_gensim_file_path = os.path.join(WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR, lang_code, f"{lang_code}_gensim_extracted_articles.txt")
    output_lang_filtered_dir = os.path.join(FILTERED_WIKIPEDIA_TEXT_DIR, lang_code)
    os.makedirs(output_lang_filtered_dir, exist_ok=True)

    if not os.path.exists(input_gensim_file_path):
        logger.warning(f"Gensim parsed text file not found for '{lang_code}': {input_gensim_file_path}")
        return 0

    # Get the limit for how many articles to read from the gensim output for this language
    max_articles_to_read = MAX_ARTICLES_TO_READ_FROM_GENSIM_PER_LANG.get(lang_code, float('inf')) 

    logger.info(f"Filtering relevant articles for '{lang_code}' from '{input_gensim_file_path}' using {FILTERING_PROCESSES} processes. "
                f"Will read up to {max_articles_to_read if max_articles_to_read != float('inf') else 'all'} articles from gensim output.")
    
    specific_lang_health_keywords_dict = get_health_keywords_dict_for_lang(lang_code)
    has_specific_keywords = bool(specific_lang_health_keywords_dict and any(specific_lang_health_keywords_dict.values()))

    if not has_specific_keywords and not GENERAL_HEALTH_KEYWORDS_FILTER:
        logger.error(f"No keywords for filtering '{lang_code}'. Aborting filtering for this language.")
        return 0
    elif not has_specific_keywords:
         logger.warning(f"No specific keywords for '{lang_code}'. Relying solely on GENERAL_HEALTH_KEYWORDS_FILTER.")

    articles_submitted_to_pool_count = 0
    relevant_articles_saved_count = 0
    articles_read_from_file_count = 0 # Count articles read from the input gensim file
    
    process_func = partial(_check_relevance_worker,
                           specific_lang_health_keywords_dict_for_worker=specific_lang_health_keywords_dict,
                           general_health_keywords_for_worker=GENERAL_HEALTH_KEYWORDS_FILTER,
                           min_article_length_for_worker=MIN_FILTERED_ARTICLE_LENGTH)

    article_batch_for_pool = [] 

    try:
        with open(input_gensim_file_path, 'r', encoding='utf-8') as f_in, \
             multiprocessing.Pool(processes=FILTERING_PROCESSES) as pool:
            
            current_article_lines_buffer = []
            for line_content in f_in: # Read line by line from the large gensim output
                if articles_read_from_file_count >= max_articles_to_read:
                    logger.info(f"[{lang_code}] Reached limit of {max_articles_to_read} articles read from gensim file. Stopping reading for this language.")
                    # Process any remaining articles in the current_article_lines_buffer before breaking
                    if current_article_lines_buffer:
                        article_text_raw_from_buffer = "".join(current_article_lines_buffer).strip()
                        if article_text_raw_from_buffer:
                             article_batch_for_pool.append(article_text_raw_from_buffer)
                             # This article was read before hitting the limit but not processed yet
                        current_article_lines_buffer = []
                    break # Exit the file reading loop

                if line_content.strip() == ARTICLE_SEPARATOR.strip():
                    if current_article_lines_buffer:
                        article_text_raw = "".join(current_article_lines_buffer).strip()
                        articles_read_from_file_count += 1 # Increment when a full article is identified
                        
                        if article_text_raw: 
                            article_batch_for_pool.append(article_text_raw)
                        
                        if len(article_batch_for_pool) >= FILTERING_CHUNK_SIZE:
                            logger.debug(f"[{lang_code}] Submitting chunk of {len(article_batch_for_pool)} articles to filter pool (Total articles read from file: {articles_read_from_file_count}).")
                            results = pool.map(process_func, article_batch_for_pool)
                            articles_submitted_to_pool_count += len(article_batch_for_pool)
                            for result_tuple in results:
                                if result_tuple:
                                    title_heuristic, body_text_to_save = result_tuple
                                    relevant_articles_saved_count += 1
                                    safe_title = re.sub(r'[^\w\.-]', '_', title_heuristic[:60])
                                    out_fn = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title}.txt")
                                    with open(out_fn, 'w', encoding='utf-8') as out_f:
                                        out_f.write(title_heuristic + "\n\n" + body_text_to_save)
                                    if relevant_articles_saved_count % 1000 == 0:
                                        logger.info(f"[{lang_code}] Saved {relevant_articles_saved_count} relevant articles (submitted to pool approx. {articles_submitted_to_pool_count}).")
                            article_batch_for_pool = [] # Reset batch
                    current_article_lines_buffer = [] 
                else:
                    current_article_lines_buffer.append(line_content)
            
            # Process any remaining content in current_article_lines_buffer (if limit not hit, or last article before limit)
            if current_article_lines_buffer and (articles_read_from_file_count < max_articles_to_read or max_articles_to_read == float('inf')):
                article_text_raw = "".join(current_article_lines_buffer).strip()
                if article_text_raw: 
                    article_batch_for_pool.append(article_text_raw)
                    # articles_read_from_file_count +=1 # Counted when separator is found or loop ends
            
            # Process final accumulated batch for the pool
            if article_batch_for_pool:
                logger.debug(f"[{lang_code}] Submitting final chunk of {len(article_batch_for_pool)} articles to filter pool (Total articles read from file: {articles_read_from_file_count}).")
                results = pool.map(process_func, article_batch_for_pool)
                articles_submitted_to_pool_count += len(article_batch_for_pool)
                for result_tuple in results:
                    if result_tuple:
                        title_heuristic, body_text_to_save = result_tuple
                        relevant_articles_saved_count += 1
                        safe_title = re.sub(r'[^\w\.-]', '_', title_heuristic[:60])
                        out_fn = os.path.join(output_lang_filtered_dir, f"{lang_code}_article_{relevant_articles_saved_count}_{safe_title}.txt")
                        with open(out_fn, 'w', encoding='utf-8') as out_f:
                            out_f.write(title_heuristic + "\n\n" + body_text_to_save)
                logger.info(f"[{lang_code}] Processed final batch. Total relevant saved: {relevant_articles_saved_count}.")

    except Exception as e_file:
        logger.error(f"Error processing file {input_gensim_file_path} with multiprocessing: {e_file}", exc_info=True)
    
    logger.info(f"Finished filtering for '{lang_code}'. Total relevant articles saved: {relevant_articles_saved_count}. "
                f"Total articles read from gensim file: {articles_read_from_file_count}. "
                f"Total articles submitted to filter pool: {articles_submitted_to_pool_count}.")
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
        logger.warning("Filtering complete, but no health-relevant articles were found/saved. Check keyword lists and gensim output.")
    return total_relevant_across_all_langs > 0

if __name__ == "__main__":
    logger.info("Testing wikipedia_filterer.py (gensim + multiprocessing version) standalone...")
    if filter_all_extracted_wikipedia():
        logger.info("Gensim Wikipedia filterer module test finished successfully.")
    else:
        logger.warning("Gensim Wikipedia filterer module test completed, but no relevant articles were found/saved.")