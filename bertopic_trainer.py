# topic_model_pretrainer_cloud/bertopic_trainer.py
import os
import glob
import random
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from loguru import logger
from typing import List, Dict

from config_pretrainer import (
    FILTERED_WIKIPEDIA_TEXT_DIR, SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH,
    LANGUAGES_TO_PROCESS, WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT,
    FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_FULL_PATH,
    BERTOPIC_MIN_TOPIC_SIZE, BERTOPIC_NR_TOPICS,
    SEED_KEYWORD_MIN_LEN, SEED_MAX_KEYWORDS_PER_TOPIC
)
from health_topics_data import HEALTH_TOPICS_KEYWORDS 

# --- Helper: SBERT Model Loading ---
def load_or_download_sbert(model_name: str, save_path: str) -> SentenceTransformer | None:
    try:
        if os.path.exists(save_path) and os.listdir(save_path):
            logger.info(f"Loading SBERT model from local path: {save_path}")
            model = SentenceTransformer(save_path)
        else:
            logger.info(f"SBERT model not found at {save_path} or directory empty. Downloading '{model_name}'...")
            model = SentenceTransformer(model_name)
            os.makedirs(save_path, exist_ok=True)
            model.save(save_path)
            logger.info(f"SBERT model '{model_name}' downloaded and saved to {save_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading/downloading SBERT model '{model_name}': {e}", exc_info=True)
        return None

# --- Helper: Load Filtered Wikipedia Documents ---
def load_filtered_wikipedia_docs(base_filtered_dir: str, langs_to_process: List[str], 
                                 limit_per_lang_dict: Dict[str, int]) -> List[str]:
    documents = []
    total_docs_loaded = 0
    logger.info(f"Loading filtered Wikipedia documents (post-gensim & filter) from: {base_filtered_dir}")

    for lang_code in langs_to_process:
        current_lang_limit = limit_per_lang_dict.get(lang_code, 10000)
        lang_dir = os.path.join(base_filtered_dir, lang_code)
        
        if not os.path.isdir(lang_dir):
            logger.warning(f"Filtered Wikipedia directory not found for language: {lang_dir}. Skipping this language.")
            continue

        all_files_in_lang = glob.glob(os.path.join(lang_dir, "*.txt"))
        if not all_files_in_lang:
            logger.warning(f"No .txt files found in filtered directory for language {lang_code}: {lang_dir}")
            continue

        logger.info(f"Found {len(all_files_in_lang)} filtered articles for language '{lang_code}'. Applying limit of {current_lang_limit}.")
        
        if len(all_files_in_lang) > current_lang_limit:
            selected_files = random.sample(all_files_in_lang, current_lang_limit)
            logger.info(f"[{lang_code}] Sampled {len(selected_files)} filtered articles for BERTopic training.")
        else:
            selected_files = all_files_in_lang
            logger.info(f"[{lang_code}] Using all {len(selected_files)} filtered articles for BERTopic training.")

        lang_docs_count = 0
        for filepath in selected_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    title = f.readline().strip() 
                    f.readline() 
                    content_body = f.read().strip()
                    if content_body: 
                        documents.append(content_body)
                        lang_docs_count += 1
                    else:
                        logger.warning(f"Empty content body in file {filepath}. Skipping.")
            except Exception as e:
                logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
        
        logger.info(f"[{lang_code}] Loaded {lang_docs_count} documents.")
        total_docs_loaded += lang_docs_count
    
    if any(doc is None for doc in documents):
        logger.error("CRITICAL: Found None values in the final documents list before returning!")
        documents = [doc for doc in documents if doc is not None] 
    if any(not doc.strip() for doc in documents if isinstance(doc, str)):
        logger.error("CRITICAL: Found empty strings in the final documents list before returning!")
        documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()]

    logger.info(f"Loaded a total of {total_docs_loaded} documents from all processed languages for BERTopic training. After final check, document count is {len(documents)}.")
    return documents

# --- Helper: Generate Seed Keywords (Still called but its output might be overridden for this test) ---
def generate_seed_keywords(health_topics_dict: Dict[str, str],
                           min_keyword_len: int,
                           max_keywords_per_topic: int) -> List[List[str]]:
    import re 
    seed_topic_list = []
    logger.info("Generating seed keywords for Guided BERTopic from health_topics_data.py...")
    for topic_key, description_keywords_english in health_topics_dict.items():
        if not isinstance(description_keywords_english, str):
            logger.warning(f"Description for '{topic_key}' is not a string: {description_keywords_english}. Skipping this seed.")
            continue
        cleaned_description = re.sub(r'\s+', ' ', description_keywords_english.lower()).strip()
        keywords_initial = cleaned_description.split(' ')
        keywords_filtered_len = [kw for kw in keywords_initial if len(kw) >= min_keyword_len]
        keywords_final = [kw.strip() for kw in keywords_filtered_len if kw.strip()]

        if keywords_final:
            unique_keywords = list(dict.fromkeys(keywords_final)) 
            seed_topic_list.append(unique_keywords[:max_keywords_per_topic])
            logger.trace(f"Generated seed for '{topic_key}': {unique_keywords[:max_keywords_per_topic]}")
        else:
            logger.warning(f"No suitable seed keywords extracted for HEALTH_TOPICS_KEYWORDS key: '{topic_key}' (description: '{description_keywords_english}') - Will not be used for guidance.")
    logger.info(f"Generated {len(seed_topic_list)} non-empty sets of seed keywords from health_topics_data.py.")
    return seed_topic_list

# --- Train Final Guided Model (on combined multilingual Wikipedia data) -
# Name reverted to original, but behavior changed via run_bertopic_training_pipeline for this test
def train_final_guided_model(sbert_model: SentenceTransformer, 
                             all_language_documents: List[str], 
                             seed_keywords: List[List[str]] | None) -> BERTopic | None: # Allow None for seeds
    logger.info("--- Starting BERTopic Trainer: Guided Model (SIMPLE SEED TEST or FULL) ---")
    
    if not all_language_documents or len(all_language_documents) < BERTOPIC_MIN_TOPIC_SIZE:
        logger.error(f"Insufficient documents ({len(all_language_documents)}) for final BERTopic training. Minimum required: {BERTOPIC_MIN_TOPIC_SIZE}")
        return None
    
    logger.info(f"Number of documents for BERTopic: {len(all_language_documents)}")
    if all_language_documents:
        none_docs = sum(1 for doc in all_language_documents if doc is None)
        empty_docs = sum(1 for doc in all_language_documents if isinstance(doc, str) and not doc.strip())
        if none_docs > 0:
            logger.critical(f"CRITICAL PRE-FIT CHECK: Found {none_docs} None documents in input list to BERTopic!")
        if empty_docs > 0:
            logger.critical(f"CRITICAL PRE-FIT CHECK: Found {empty_docs} empty string documents in input list to BERTopic!")
        if all_language_documents: 
             logger.info(f"First document sample for BERTopic (first 100 chars): {all_language_documents[0][:100]}")
        else:
            logger.error("All documents were None or empty after checks! Cannot proceed with BERTopic.")
            return None

    logger.info(f"Number of seed keyword sets being used for BERTopic: {len(seed_keywords) if seed_keywords else 0}")
    if seed_keywords:
        empty_seed_sets_count = 0
        for i, seed_set in enumerate(seed_keywords):
            if not seed_set: 
                logger.error(f"CRITICAL PRE-FIT CHECK: Empty seed keyword set found at index {i} in seed_topic_list!")
                empty_seed_sets_count +=1
        if empty_seed_sets_count > 0:
            logger.error(f"Found a total of {empty_seed_sets_count} empty seed sets. This can cause issues.")
        logger.info(f"First seed keyword set sample (if any): {seed_keywords[0] if seed_keywords else 'None'}")
    
    logger.info(f"Initializing BERTopic model for guided training with {len(all_language_documents)} documents.")
    
    final_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual", 
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=BERTOPIC_NR_TOPICS,
        seed_topic_list=seed_keywords, # This will be the simple list or None
        verbose=True,
        calculate_probabilities=True
    )

    try:
        guidance_type = "HEALTH_TOPICS seeds" if seed_keywords else "NO seeds (unguided)"
        if seed_keywords and len(seed_keywords) < 10: # Heuristic for simple test
             guidance_type = "SIMPLE TEST seeds"
        logger.info(f"Fitting final BERTopic model on combined multilingual Wikipedia data with {guidance_type}...")
        
        final_model.fit_transform(all_language_documents) 
        
        num_topics = len(final_model.get_topic_info()) -1 
        logger.success(f"BERTopic model training completed (with {guidance_type}). Found {num_topics} topics.")
        
        if num_topics > 0: 
            logger.info(f"Sample of topics (with {guidance_type}):\n{final_model.get_topic_info().head(20)}")
        
        os.makedirs(FINAL_BERTOPIC_OUTPUT_DIR, exist_ok=True)

        try:
            model_save_path = FINAL_BERTOPIC_FULL_PATH
            if seed_keywords and len(seed_keywords) < 10: # Simple test save path
                model_save_path = FINAL_BERTOPIC_FULL_PATH.replace(".joblib", "_guided_simple_test.joblib")
            elif not seed_keywords: # Unguided save path
                 model_save_path = FINAL_BERTOPIC_FULL_PATH.replace(".joblib", "_unguided_test.joblib")


            logger.info(f"Attempting to save BERTopic model (with {guidance_type}) to: {model_save_path}")
            final_model.save(model_save_path, save_embedding_model=False)
            
            MIN_EXPECTED_FILE_SIZE_KB = 100 
            if os.path.exists(model_save_path) and os.path.getsize(model_save_path) > 1024 * MIN_EXPECTED_FILE_SIZE_KB: 
                 logger.success(f"BERTopic model (with {guidance_type}) SUCCESSFULLY saved to: {model_save_path}")
                 return final_model
            else:
                 file_size = os.path.getsize(model_save_path) if os.path.exists(model_save_path) else -1
                 error_msg = f"BERTopic model (with {guidance_type}) save call finished, but file {model_save_path} was not found or too small ({file_size} bytes). Expected > {MIN_EXPECTED_FILE_SIZE_KB} KB."
                 logger.error(error_msg)
                 return None

        except Exception as e_save: 
            logger.error(f"ERROR occurred during saving the BERTopic model (with {guidance_type}) to {model_save_path}: {e_save}", exc_info=True)
            return None

    except ValueError as ve: 
        logger.error(f"ValueError during BERTopic fit_transform (with {guidance_type} - potentially inhomogeneous shape): {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"ERROR during BERTopic training (with {guidance_type} - fit_transform or other): {e}", exc_info=True)
        return None


def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline (Gensim Path, Multilingual) Initiated (SIMPLE GUIDED TEST MODE) ===")
    
    sbert = load_or_download_sbert(SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH)
    if not sbert:
        logger.error("SBERT model could not be loaded/downloaded. Aborting pipeline.")
        return

    wiki_docs_multilingual = load_filtered_wikipedia_docs(
        FILTERED_WIKIPEDIA_TEXT_DIR,
        LANGUAGES_TO_PROCESS,
        WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT
    )
    if not wiki_docs_multilingual: 
        logger.error("No documents loaded for BERTopic training after filtering. Aborting.")
        return

    # Generate full seeds to see logs, but override for test
    _ = generate_seed_keywords( 
        HEALTH_TOPICS_KEYWORDS,
        SEED_KEYWORD_MIN_LEN,
        SEED_MAX_KEYWORDS_PER_TOPIC
    )
    
    # +++ HARDCODED SIMPLE SEED LIST FOR THIS DEBUGGING RUN +++
    logger.info("DEBUG: Using a small, hardcoded simple seed list for this Guided BERTopic test run.")
    simple_test_seeds = [
        ["disease", "symptom", "illness", "health", "medical"], 
        ["fever", "cough", "headache", "fatigue", "pain"],
        ["virus", "bacteria", "infection", "outbreak", "pandemic"] # Added another simple one
    ] 
    # Ensure all keywords in simple_test_seeds meet SEED_KEYWORD_MIN_LEN (which is 3) - they do.
    effective_seeds_for_bertopic = simple_test_seeds
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # The `train_final_guided_model` function is now used for all cases,
    # its behavior regarding seed_keywords depends on what's passed to it.
    final_model = train_final_guided_model(sbert, wiki_docs_multilingual, effective_seeds_for_bertopic) 

    if final_model:
        logger.success("=== BERTopic Training Pipeline (SIMPLE GUIDED TEST MODE) Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTopic Training Pipeline (SIMPLE GUIDED TEST MODE) FAILED. Check logs. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()