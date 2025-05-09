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
    BASE_WIKI_BERTOPIC_MODEL_DIR, BASE_WIKI_BERTOPIC_FULL_PATH,
    STAGE1_BERTOPIC_MIN_TOPIC_SIZE, STAGE1_BERTOPIC_NR_TOPICS,
    FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_FULL_PATH,
    STAGE2_BERTOPIC_MIN_TOPIC_SIZE, STAGE2_BERTOPIC_NR_TOPICS,
    SEED_KEYWORD_MIN_LEN, SEED_MAX_KEYWORDS_PER_TOPIC
)
from health_topics_data import HEALTH_TOPICS_KEYWORDS # Your dictionary of topics and seed keywords

# --- Helper: SBERT Model Loading ---
def load_or_download_sbert(model_name: str, save_path: str) -> SentenceTransformer | None:
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
    logger.info(f"Loading filtered Wikipedia documents from: {base_filtered_dir}")
    for lang_code in langs:
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
            logger.info(f"[{lang_code}] Sampled {len(selected_files)} filtered articles for training.")
        else:
            selected_files = all_files_in_lang
            logger.info(f"[{lang_code}] Using all {len(selected_files)} filtered articles for training.")

        for filepath in selected_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Assuming filterer saved title \n\n text
                    title = f.readline().strip() # Read title
                    f.readline() # Skip blank line
                    content = f.read()
                    documents.append(content) # Use only content for BERTopic training
            except Exception as e:
                logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
    
    logger.info(f"Loaded a total of {len(documents)} filtered Wikipedia documents for training.")
    return documents


# --- Helper: Generate Seed Keywords ---
def generate_seed_keywords(health_topics_dict: Dict[str, str],
                           min_keyword_len: int,
                           max_keywords_per_topic: int) -> List[List[str]]:
    import re # Local import for this helper
    seed_topic_list = []
    logger.info("Generating seed keywords for Guided BERTopic (Stage 2)...")
    for topic_key, description_keywords in health_topics_dict.items():
        cleaned_description = re.sub(r'\s+', ' ', description_keywords.lower()).strip()
        keywords = [kw for kw in cleaned_description.split(' ') if len(kw) >= min_keyword_len]
        if keywords:
            unique_keywords = list(dict.fromkeys(keywords))
            seed_topic_list.append(unique_keywords[:max_keywords_per_topic])
        else:
            logger.warning(f"No suitable seed keywords found for HEALTH_TOPICS key: {topic_key}")
    logger.info(f"Generated {len(seed_topic_list)} sets of seed keywords for Stage 2 guidance.")
    return seed_topic_list


# --- Stage 1: Pre-train Base BERTopic model on Wikipedia ---
def train_stage1_wikipedia_base_model(sbert_model: SentenceTransformer, documents: List[str]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: Stage 1 (Wikipedia Base Model) ---")
    if not documents or len(documents) < STAGE1_BERTOPIC_MIN_TOPIC_SIZE * 2:
        logger.error(f"Insufficient documents ({len(documents)}) for Stage 1 BERTopic training. Minimum recommended: {STAGE1_BERTOPIC_MIN_TOPIC_SIZE * 2}")
        return None

    logger.info(f"Initializing BERTopic for Stage 1 with {len(documents)} Wikipedia documents.")
    # Usually unguided for Stage 1 to capture broad themes from Wikipedia
    base_topic_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual",
        min_topic_size=STAGE1_BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=STAGE1_BERTOPIC_NR_TOPICS,
        verbose=True,
        calculate_probabilities=True
    )
    try:
        logger.info("Fitting Stage 1 BERTopic model on Wikipedia data... (This can take a long time)")
        base_topic_model.fit_transform(documents) # Only need to fit
        num_topics = len(base_topic_model.get_topic_info()) -1
        logger.success(f"Stage 1 BERTopic model training on Wikipedia completed. Found {num_topics} base topics.")
        if num_topics > 0: logger.info(f"Sample of Stage 1 topics:\n{base_topic_model.get_topic_info().head()}")
        
        os.makedirs(BASE_WIKI_BERTOPIC_MODEL_DIR, exist_ok=True)
        base_topic_model.save(BASE_WIKI_BERTOPIC_FULL_PATH, serialization="joblib", save_embedding_model=False)
        logger.success(f"Stage 1 (Base Wikipedia) BERTopic model saved to: {BASE_WIKI_BERTOPIC_FULL_PATH}")
        return base_topic_model
    except Exception as e:
        logger.error(f"Error during Stage 1 BERTopic training: {e}", exc_info=True)
        return None

# --- Stage 2: Guide the Base Model with HEALTH_TOPICS (using Wikipedia data as context) ---
def train_stage2_guided_model(sbert_model: SentenceTransformer, 
                              wikipedia_documents_for_stage2: List[str], # Pass the same Wiki docs, or a subset
                              seed_keywords: List[List[str]]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: Stage 2 (Guiding with HEALTH_TOPICS on Wikipedia Data Context) ---")
    
    if not wikipedia_documents_for_stage2:
        logger.error("No Wikipedia documents provided for Stage 2 context. Cannot proceed.")
        return None
    
    # For this simplified Stage 2 (no new PG data), we re-train a BERTopic model
    # on the Wikipedia documents, but this time WITH guidance from HEALTH_TOPICS.
    # The SBERT model provides the rich embeddings.
    logger.info(f"Initializing new BERTopic model for Stage 2 guidance using {len(wikipedia_documents_for_stage2)} Wikipedia documents.")
    
    guided_topic_model = BERTopic(
        embedding_model=sbert_model, # Same SBERT model
        language="multilingual",
        min_topic_size=STAGE2_BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=STAGE2_BERTOPIC_NR_TOPICS, # Could be len(seed_keywords) to try and force alignment
        seed_topic_list=seed_keywords if seed_keywords else None,
        verbose=True,
        calculate_probabilities=True
    )

    try:
        logger.info("Fitting Stage 2 (Guided) BERTopic model on Wikipedia data with HEALTH_TOPIC seeds...")
        guided_topic_model.fit_transform(wikipedia_documents_for_stage2)
        num_topics = len(guided_topic_model.get_topic_info()) -1
        logger.success(f"Stage 2 (Guided) BERTopic model training completed. Found {num_topics} topics.")
        if num_topics > 0: logger.info(f"Sample of Stage 2 (Guided) topics:\n{guided_topic_model.get_topic_info().head(15)}")
        
        os.makedirs(FINAL_BERTOPIC_OUTPUT_DIR, exist_ok=True)
        guided_topic_model.save(FINAL_BERTOPIC_FULL_PATH, serialization="joblib", save_embedding_model=False)
        logger.success(f"FINAL Guided BERTopic model saved to: {FINAL_BERTOPIC_FULL_PATH}")
        return guided_topic_model
    except Exception as e:
        logger.error(f"Error during Stage 2 (Guided) BERTopic training: {e}", exc_info=True)
        return None

def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline Initiated ===")
    
    # 0. Ensure SBERT model is available
    sbert = load_or_download_sbert(SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH)
    if not sbert:
        logger.error("SBERT model could not be loaded/downloaded. Aborting pipeline.")
        return

    # 1. Load filtered Wikipedia documents (used for both stages in this revised approach)
    wiki_docs = load_filtered_wikipedia_docs(
        FILTERED_WIKIPEDIA_TEXT_DIR,
        LANGUAGES_TO_PROCESS,
        WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT
    )
    if not wiki_docs:
        logger.error("No filtered Wikipedia documents loaded. Aborting pipeline.")
        return

    # STAGE 1: Create a base model (optional step if Stage 2 retrains from scratch with SBERT)
    # For this current simplified approach where Stage 2 retrains with guidance,
    # Stage 1 is primarily to see what an unguided model on Wikipedia data looks like,
    # or to potentially reuse its UMAP/HDBSCAN components if doing more advanced BERTopic updates.
    # If you skip Stage 1, ensure Stage 2 uses enough documents.
    
    # For simplicity, let's make Stage 1 optional or only for exploration in this script version.
    # We will directly proceed to Stage 2 which trains a new guided model on Wikipedia data.
    # The "base model" concept is implicitly handled by using a powerful SBERT.
    # If you truly want to load a model and *update* it, BERTopic's `partial_fit` or merging techniques are needed.
    # The current request is to *guide* a model trained on Wikipedia data with HEALTH_TOPICS.

    # STAGE 2 (Effectively the main training now for the final model):
    # Train a BERTopic model on Wikipedia data, guided by HEALTH_TOPICS seeds
    health_topic_seeds = generate_seed_keywords(
        HEALTH_TOPICS_KEYWORDS,
        SEED_KEYWORD_MIN_LEN,
        SEED_MAX_KEYWORDS_PER_TOPIC
    )

    final_model = train_stage2_guided_model(sbert, wiki_docs, health_topic_seeds)

    if final_model:
        logger.success("=== BERTopic Training Pipeline Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTopic Training Pipeline Failed. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()