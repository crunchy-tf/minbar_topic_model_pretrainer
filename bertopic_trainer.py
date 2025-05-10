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
                    if content_body:
                        documents.append(content_body)
                        lang_docs_count += 1
            except Exception as e:
                logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
        
        logger.info(f"[{lang_code}] Loaded {lang_docs_count} documents.")
        total_docs_loaded += lang_docs_count
    
    logger.info(f"Loaded a total of {total_docs_loaded} documents from all processed languages for BERTopic training.")
    return documents

# --- Helper: Generate Seed Keywords ---
def generate_seed_keywords(health_topics_dict: Dict[str, str], # This is HEALTH_TOPICS_KEYWORDS from health_topics_data.py
                           min_keyword_len: int,
                           max_keywords_per_topic: int) -> List[List[str]]:
    import re 
    seed_topic_list = []
    logger.info("Generating seed keywords for Guided BERTopic from health_topics_data.py...")
    for topic_key, description_keywords_english in health_topics_dict.items():
        # Assuming descriptions in HEALTH_TOPICS_KEYWORDS are English or good pivot keywords
        cleaned_description = re.sub(r'\s+', ' ', description_keywords_english.lower()).strip()
        keywords = [kw for kw in cleaned_description.split(' ') if len(kw) >= min_keyword_len]
        if keywords:
            unique_keywords = list(dict.fromkeys(keywords)) # Remove duplicates while preserving order
            seed_topic_list.append(unique_keywords[:max_keywords_per_topic])
            logger.trace(f"Generated seed for '{topic_key}': {unique_keywords[:max_keywords_per_topic]}")
        else:
            logger.warning(f"No suitable seed keywords extracted for HEALTH_TOPICS_KEYWORDS key: '{topic_key}' (description: '{description_keywords_english}')")
    logger.info(f"Generated {len(seed_topic_list)} sets of seed keywords for guidance.")
    return seed_topic_list

# --- Train Final Guided Model (on combined multilingual Wikipedia data) ---
def train_final_guided_model(sbert_model: SentenceTransformer, 
                             all_language_documents: List[str], 
                             seed_keywords: List[List[str]]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: Final Guided Model (on multilingual filtered Wikipedia data) ---")
    
    if not all_language_documents or len(all_language_documents) < BERTOPIC_MIN_TOPIC_SIZE:
        logger.error(f"Insufficient documents ({len(all_language_documents)}) for final BERTopic training. Minimum required: {BERTOPIC_MIN_TOPIC_SIZE}")
        return None
    
    logger.info(f"Initializing BERTopic model for final guided training with {len(all_language_documents)} documents (from all languages).")
    
    final_guided_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual", # Essential for combined EN, FR, AR docs
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=BERTOPIC_NR_TOPICS,
        seed_topic_list=seed_keywords if seed_keywords else None,
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
        final_guided_model.save(FINAL_BERTOPIC_FULL_PATH, serialization="joblib", save_embedding_model=False)
        logger.success(f"FINAL Guided BERTopic model saved to: {FINAL_BERTOPIC_FULL_PATH}")
        return final_guided_model
    except Exception as e:
        logger.error(f"Error during final guided BERTopic training: {e}", exc_info=True)
        return None

def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline (Gensim Path, Multilingual) Initiated ===")
    
    sbert = load_or_download_sbert(SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH)
    if not sbert:
        logger.error("SBERT model could not be loaded/downloaded. Aborting pipeline.")
        return

    # Load filtered Wikipedia documents from ALL languages specified in config
    wiki_docs_multilingual = load_filtered_wikipedia_docs(
        FILTERED_WIKIPEDIA_TEXT_DIR,
        LANGUAGES_TO_PROCESS, # Now ["en", "fr", "ar"]
        WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT # This is a dict for per-lang limits
    )
    if not wiki_docs_multilingual:
        logger.error("No filtered Wikipedia documents loaded (all languages combined). Aborting BERTopic training.")
        return

    # Seed keywords are generated from the English descriptions in HEALTH_TOPICS_KEYWORDS
    # SBERT's multilingual capabilities will map these to concepts in FR and AR documents.
    health_topic_seeds = generate_seed_keywords(
        HEALTH_TOPICS_KEYWORDS, # This is imported from health_topics_data.py
        SEED_KEYWORD_MIN_LEN,
        SEED_MAX_KEYWORDS_PER_TOPIC
    )

    # Directly train the final guided model on the combined multilingual documents
    final_model = train_final_guided_model(sbert, wiki_docs_multilingual, health_topic_seeds)

    if final_model:
        logger.success("=== BERTopic Training Pipeline Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTopic Training Pipeline Failed. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()