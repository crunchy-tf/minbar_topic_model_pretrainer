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
    LANGUAGES_TO_PROCESS, WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT, # This is now a dict
    FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_FULL_PATH,
    BERTOPIC_MIN_TOPIC_SIZE, BERTOPIC_NR_TOPICS,
    SEED_KEYWORD_MIN_LEN, SEED_MAX_KEYWORDS_PER_TOPIC
)
# HEALTH_TOPICS_KEYWORDS will be used for seed generation.
# These seeds are primarily English-based from your health_topics_data.py,
# and SBERT's multilingual capabilities will help map them across languages.
from health_topics_data import HEALTH_TOPICS_KEYWORDS 

# --- Helper: SBERT Model Loading ---
def load_or_download_sbert(model_name: str, save_path: str) -> SentenceTransformer | None:
    try:
        if os.path.exists(save_path) and os.listdir(save_path): # Check if dir not empty
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

    for lang_code in langs_to_process: # Will iterate through "en", "fr", "ar"
        current_lang_limit = limit_per_lang_dict.get(lang_code, 10000) # Default to 10k if lang not in dict (should be)
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
                    if content_body: # Ensure non-empty content
                        documents.append(content_body)
                        lang_docs_count += 1
                    else:
                        logger.warning(f"Empty content body in file {filepath}. Skipping.")
            except Exception as e:
                logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
        
        logger.info(f"[{lang_code}] Loaded {lang_docs_count} documents.")
        total_docs_loaded += lang_docs_count
    
    # Final check for None or empty strings in the entire documents list
    if any(doc is None for doc in documents):
        logger.error("CRITICAL: Found None values in the final documents list before returning!")
        documents = [doc for doc in documents if doc is not None] # Filter out None
    if any(not doc.strip() for doc in documents if isinstance(doc, str)):
        logger.error("CRITICAL: Found empty strings in the final documents list before returning!")
        documents = [doc for doc in documents if isinstance(doc, str) and doc.strip()] # Filter out empty strings

    logger.info(f"Loaded a total of {total_docs_loaded} documents from all processed languages for BERTopic training. After final check, document count is {len(documents)}.")
    return documents

# --- Helper: Generate Seed Keywords ---
def generate_seed_keywords(health_topics_dict: Dict[str, str], # This is HEALTH_TOPICS_KEYWORDS from health_topics_data.py
                           min_keyword_len: int,
                           max_keywords_per_topic: int) -> List[List[str]]:
    import re 
    seed_topic_list = []
    logger.info("Generating seed keywords for Guided BERTopic from health_topics_data.py...")
    for topic_key, description_keywords_english in health_topics_dict.items():
        cleaned_description = re.sub(r'\s+', ' ', description_keywords_english.lower()).strip()
        keywords = [kw for kw in cleaned_description.split(' ') if len(kw) >= min_keyword_len]
        if keywords:
            unique_keywords = list(dict.fromkeys(keywords)) 
            seed_topic_list.append(unique_keywords[:max_keywords_per_topic])
            logger.trace(f"Generated seed for '{topic_key}': {unique_keywords[:max_keywords_per_topic]}")
        else:
            logger.warning(f"No suitable seed keywords extracted for HEALTH_TOPICS_KEYWORDS key: '{topic_key}' (description: '{description_keywords_english}') - Will not be used for guidance.")
            # IMPORTANT: Do NOT append an empty list here if no keywords are found for a topic.
            # BERTopic might handle it, but it's safer to just omit that seed set.
    logger.info(f"Generated {len(seed_topic_list)} non-empty sets of seed keywords for guidance.")
    return seed_topic_list

# --- Train Final Guided Model (on combined multilingual Wikipedia data) ---
def train_final_guided_model(sbert_model: SentenceTransformer, 
                             all_language_documents: List[str], 
                             seed_keywords: List[List[str]]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: Final Guided Model (on multilingual filtered Wikipedia data) ---")
    
    if not all_language_documents or len(all_language_documents) < BERTOPIC_MIN_TOPIC_SIZE:
        logger.error(f"Insufficient documents ({len(all_language_documents)}) for final BERTopic training. Minimum required: {BERTOPIC_MIN_TOPIC_SIZE}")
        return None
    
    # +++ ADDED DEBUGGING LOGS HERE +++
    logger.info(f"Number of documents for BERTopic: {len(all_language_documents)}")
    if all_language_documents:
        none_docs = sum(1 for doc in all_language_documents if doc is None)
        empty_docs = sum(1 for doc in all_language_documents if isinstance(doc, str) and not doc.strip())
        if none_docs > 0:
            logger.critical(f"CRITICAL PRE-FIT CHECK: Found {none_docs} None documents in input list to BERTopic!")
            # Optionally, you could filter them out here again, but it's better if load_filtered_wikipedia_docs ensures this.
            # all_language_documents = [doc for doc in all_language_documents if doc is not None]
            # logger.info(f"Document count after removing None: {len(all_language_documents)}")
        if empty_docs > 0:
            logger.critical(f"CRITICAL PRE-FIT CHECK: Found {empty_docs} empty string documents in input list to BERTopic!")
            # all_language_documents = [doc for doc in all_language_documents if isinstance(doc, str) and doc.strip()]
            # logger.info(f"Document count after removing empty strings: {len(all_language_documents)}")
        if all_language_documents: # Check again if list became empty after filtering
             logger.info(f"First document sample for BERTopic (first 100 chars): {all_language_documents[0][:100]}")
        else:
            logger.error("All documents were None or empty after checks! Cannot proceed with BERTopic.")
            return None


    effective_seed_keywords = seed_keywords if seed_keywords else None # Ensure None if empty, not []
    logger.info(f"Number of seed keyword sets for BERTopic: {len(effective_seed_keywords) if effective_seed_keywords else 0}")
    
    if effective_seed_keywords:
        empty_seed_sets_count = 0
        for i, seed_set in enumerate(effective_seed_keywords):
            if not seed_set: # Checks for empty list []
                logger.error(f"CRITICAL PRE-FIT CHECK: Empty seed keyword set found at index {i} in seed_topic_list!")
                empty_seed_sets_count +=1
        if empty_seed_sets_count > 0:
            logger.error(f"Found a total of {empty_seed_sets_count} empty seed sets. This can cause issues.")
            # Decide how to handle: either proceed with caution, filter them out, or error out.
            # For now, BERTopic might ignore them or error.
        # logger.info(f"First seed keyword set sample: {effective_seed_keywords[0] if effective_seed_keywords else 'None'}")
    # +++ END ADDED DEBUGGING LOGS +++
    
    logger.info(f"Initializing BERTopic model for final guided training with {len(all_language_documents)} documents (from all languages).")
    
    final_guided_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual", 
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=BERTOPIC_NR_TOPICS,
        seed_topic_list=effective_seed_keywords, # Use the potentially filtered list or None
        verbose=True,
        calculate_probabilities=True
    )

    try:
        logger.info("Fitting final guided BERTopic model on combined multilingual Wikipedia data with HEALTH_TOPICS seeds...")
        # This step will take time, proportional to len(all_language_documents)
        final_guided_model.fit_transform(all_language_documents) 
        
        num_topics = len(final_guided_model.get_topic_info()) -1 # Exclude outlier topic
        logger.success(f"Final guided BERTopic model training completed. Found {num_topics} topics.")
        
        if num_topics > 0: 
            logger.info(f"Sample of final guided topics (keywords might be mixed language initially):\n{final_guided_model.get_topic_info().head(20)}")
        
        os.makedirs(FINAL_BERTOPIC_OUTPUT_DIR, exist_ok=True)

        try:
            logger.info(f"Attempting to save FINAL Guided BERTopic model to: {FINAL_BERTOPIC_FULL_PATH}")
            final_guided_model.save(FINAL_BERTOPIC_FULL_PATH, save_embedding_model=False)
            
            MIN_EXPECTED_FILE_SIZE_KB = 100 
            if os.path.exists(FINAL_BERTOPIC_FULL_PATH) and os.path.getsize(FINAL_BERTOPIC_FULL_PATH) > 1024 * MIN_EXPECTED_FILE_SIZE_KB: 
                 logger.success(f"FINAL Guided BERTopic model SUCCESSFULLY saved to: {FINAL_BERTOPIC_FULL_PATH}")
                 return final_guided_model
            else:
                 file_size = os.path.getsize(FINAL_BERTOPIC_FULL_PATH) if os.path.exists(FINAL_BERTOPIC_FULL_PATH) else -1
                 error_msg = f"FINAL Guided BERTopic model save call finished, but file {FINAL_BERTOPIC_FULL_PATH} was not found or too small ({file_size} bytes) after save attempt. Expected > {MIN_EXPECTED_FILE_SIZE_KB} KB."
                 logger.error(error_msg)
                 return None

        except Exception as e_save: 
            logger.error(f"ERROR occurred during saving the final BERTopic model to {FINAL_BERTOPIC_FULL_PATH}: {e_save}", exc_info=True)
            return None

    except ValueError as ve: # Catch ValueError specifically for more targeted logging
        logger.error(f"ValueError during BERTopic fit_transform (potentially inhomogeneous shape): {ve}", exc_info=True)
        # Log shapes for debugging if possible (this is tricky as embeddings are internal to fit_transform)
        # For example, if embeddings were accessible:
        # if 'embeddings' in locals() or 'embeddings' in globals():
        #     logger.error(f"Shape of embeddings (if available): {embeddings.shape if hasattr(embeddings, 'shape') else 'Not an array'}")
        return None
    except Exception as e:
        logger.error(f"ERROR during final guided BERTopic training (fit_transform or other): {e}", exc_info=True)
        return None


def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline (Gensim Path, Multilingual) Initiated ===")
    
    sbert = load_or_download_sbert(SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH)
    if not sbert:
        logger.error("SBERT model could not be loaded/downloaded. Aborting pipeline.")
        return

    wiki_docs_multilingual = load_filtered_wikipedia_docs(
        FILTERED_WIKIPEDIA_TEXT_DIR,
        LANGUAGES_TO_PROCESS,
        WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT
    )
    if not wiki_docs_multilingual: # Check if list is empty after loading and potential filtering
        logger.error("No documents loaded for BERTopic training after filtering. Aborting.")
        return

    health_topic_seeds = generate_seed_keywords(
        HEALTH_TOPICS_KEYWORDS,
        SEED_KEYWORD_MIN_LEN,
        SEED_MAX_KEYWORDS_PER_TOPIC
    )
    # Ensure health_topic_seeds is None if empty, not an empty list, for BERTopic
    if not health_topic_seeds: # If the list itself is empty (no valid seed sets generated)
        logger.warning("No valid seed keywords were generated. Proceeding with unguided BERTopic.")
        effective_seeds_for_bertopic = None
    else:
        # Filter out any inner empty lists from seed_keywords, though generate_seed_keywords should prevent this.
        # This is a defensive measure.
        effective_seeds_for_bertopic = [s for s in health_topic_seeds if s]
        if not effective_seeds_for_bertopic: # If all seed sets ended up empty after filtering
            logger.warning("All generated seed sets were empty after filtering. Proceeding with unguided BERTopic.")
            effective_seeds_for_bertopic = None
        elif len(effective_seeds_for_bertopic) < len(health_topic_seeds):
            logger.warning(f"Some seed sets were empty and have been removed. Using {len(effective_seeds_for_bertopic)} non-empty seed sets.")


    final_model = train_final_guided_model(sbert, wiki_docs_multilingual, effective_seeds_for_bertopic)

    if final_model:
        logger.success("=== BERTopic Training Pipeline Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTOPIC TRAINING PIPELINE FAILED. Check logs for specific errors during training or saving. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()