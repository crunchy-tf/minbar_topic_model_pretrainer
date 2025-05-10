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
    # BASE_WIKI_BERTOPIC_MODEL_DIR, BASE_WIKI_BERTOPIC_FULL_PATH, # Stage 1 model is optional
    # STAGE1_BERTOPIC_MIN_TOPIC_SIZE, STAGE1_BERTOPIC_NR_TOPICS,
    FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_FULL_PATH,
    BERTOPIC_MIN_TOPIC_SIZE, BERTOPIC_NR_TOPICS, # Using these directly for the final model
    SEED_KEYWORD_MIN_LEN, SEED_MAX_KEYWORDS_PER_TOPIC
)
from health_topics_data import HEALTH_TOPICS_KEYWORDS

# --- Helper: SBERT Model Loading ---
def load_or_download_sbert(model_name: str, save_path: str) -> SentenceTransformer | None:
    # ... (same as before)
    try:
        if os.path.exists(save_path) and os.listdir(save_path):
            logger.info(f"Loading SBERT model from local path: {save_path}")
            model = SentenceTransformer(save_path)
        else:
            logger.info(f"SBERT model not found at {save_path}. Downloading '{model_name}'...")
            model = SentenceTransformer(model_name)
            os.makedirs(save_path, exist_ok=True)
            model.save(save_path)
            logger.info(f"SBERT model '{model_name}' downloaded and saved to {save_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading/downloading SBERT model '{model_name}': {e}", exc_info=True)
        return None

# --- Helper: Load Filtered Wikipedia Documents ---
def load_filtered_wikipedia_docs(base_filtered_dir: str, langs: List[str], limit_per_lang: int) -> List[str]:
    documents = []
    logger.info(f"Loading filtered Wikipedia documents (post-gensim & filter) from: {base_filtered_dir}")
    for lang_code in langs: # Will be just ["en"] with current config
        lang_dir = os.path.join(base_filtered_dir, lang_code)
        if not os.path.isdir(lang_dir):
            logger.warning(f"Filtered Wikipedia directory not found for language: {lang_dir}")
            continue

        all_files_in_lang = glob.glob(os.path.join(lang_dir, "*.txt"))
        if not all_files_in_lang:
            logger.warning(f"No .txt files found in filtered directory: {lang_dir}")
            continue

        if len(all_files_in_lang) > limit_per_lang:
            selected_files = random.sample(all_files_in_lang, limit_per_lang)
            logger.info(f"[{lang_code}] Sampled {len(selected_files)} filtered articles for BERTopic training.")
        else:
            selected_files = all_files_in_lang
            logger.info(f"[{lang_code}] Using all {len(selected_files)} filtered articles for BERTopic training.")

        for filepath in selected_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Filterer saves: title\n\ncontent_body
                    title = f.readline().strip() 
                    f.readline() # Skip the blank line
                    content_body = f.read().strip()
                    if content_body: # Ensure we have actual content
                        documents.append(content_body)
            except Exception as e:
                logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
    
    logger.info(f"Loaded a total of {len(documents)} documents for BERTopic training.")
    return documents

# --- Helper: Generate Seed Keywords ---
def generate_seed_keywords(health_topics_dict: Dict[str, str],
                           min_keyword_len: int,
                           max_keywords_per_topic: int) -> List[List[str]]:
    import re 
    seed_topic_list = []
    logger.info("Generating seed keywords for Guided BERTopic...")
    for topic_key, description_keywords in health_topics_dict.items(): # Using HEALTH_TOPICS_KEYWORDS from health_topics_data.py
        cleaned_description = re.sub(r'\s+', ' ', description_keywords.lower()).strip()
        keywords = [kw for kw in cleaned_description.split(' ') if len(kw) >= min_keyword_len]
        if keywords:
            unique_keywords = list(dict.fromkeys(keywords))
            seed_topic_list.append(unique_keywords[:max_keywords_per_topic])
        else:
            logger.warning(f"No suitable seed keywords found for HEALTH_TOPICS_KEYWORDS key: {topic_key}")
    logger.info(f"Generated {len(seed_topic_list)} sets of seed keywords for guidance.")
    return seed_topic_list

# --- Train Final Guided Model (Simplified to one main training stage) ---
def train_final_guided_model(sbert_model: SentenceTransformer, 
                             documents: List[str], 
                             seed_keywords: List[List[str]]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: Final Guided Model (on filtered Wikipedia data) ---")
    
    if not documents or len(documents) < BERTOPIC_MIN_TOPIC_SIZE: # Adjusted min check
        logger.error(f"Insufficient documents ({len(documents)}) for final BERTopic training. Minimum required: {BERTOPIC_MIN_TOPIC_SIZE}")
        return None
    
    logger.info(f"Initializing BERTopic model for final guided training with {len(documents)} documents.")
    
    final_guided_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual",
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE, # From config
        nr_topics=BERTOPIC_NR_TOPICS,           # From config
        seed_topic_list=seed_keywords if seed_keywords else None,
        verbose=True,
        calculate_probabilities=True
    )

    try:
        logger.info("Fitting final guided BERTopic model on filtered Wikipedia data with HEALTH_TOPICS seeds...")
        final_guided_model.fit_transform(documents)
        num_topics = len(final_guided_model.get_topic_info()) -1 # Exclude outlier topic
        logger.success(f"Final guided BERTopic model training completed. Found {num_topics} topics.")
        if num_topics > 0: logger.info(f"Sample of final guided topics:\n{final_guided_model.get_topic_info().head(15)}")
        
        os.makedirs(FINAL_BERTOPIC_OUTPUT_DIR, exist_ok=True)
        final_guided_model.save(FINAL_BERTOPIC_FULL_PATH, serialization="joblib", save_embedding_model=False)
        logger.success(f"FINAL Guided BERTopic model saved to: {FINAL_BERTOPIC_FULL_PATH}")
        return final_guided_model
    except Exception as e:
        logger.error(f"Error during final guided BERTopic training: {e}", exc_info=True)
        return None

def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline (Gensim Path) Initiated ===")
    
    sbert = load_or_download_sbert(SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH)
    if not sbert:
        return

    wiki_docs = load_filtered_wikipedia_docs(
        FILTERED_WIKIPEDIA_TEXT_DIR,
        LANGUAGES_TO_PROCESS, # This will be ["en"]
        WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT
    )
    if not wiki_docs:
        return

    # Generate seed keywords from health_topics_data.py
    health_topic_seeds = generate_seed_keywords(
        HEALTH_TOPICS_KEYWORDS, # This is imported from health_topics_data.py
        SEED_KEYWORD_MIN_LEN,
        SEED_MAX_KEYWORDS_PER_TOPIC
    )

    # Directly train the final guided model
    final_model = train_final_guided_model(sbert, wiki_docs, health_topic_seeds)

    if final_model:
        logger.success("=== BERTopic Training Pipeline Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTopic Training Pipeline Failed. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()