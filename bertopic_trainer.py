# topic_model_pretrainer_cloud/bertopic_trainer.py
import os
import glob
import random
import time 
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from loguru import logger # Ensure logger is imported early for immediate use
from typing import List, Dict, Optional

# +++ START: Explicitly Print Key from Environment and Loguru Fix +++
logger.info("BERTOPIC_TRAINER.PY MODULE LOADING")
RAW_GEMINI_API_KEY_FROM_ENV = os.environ.get("GEMINI_API_KEY") # Get it once at module load
if RAW_GEMINI_API_KEY_FROM_ENV:
    logger.info(f"Attempting to use GEMINI_API_KEY from environment.")
    logger.info(f"RAW GEMINI_API_KEY string from os.environ (first 5 chars): '{RAW_GEMINI_API_KEY_FROM_ENV[:5]}...'") # Mask most of it
    logger.info(f"Length of raw key string from os.environ: {len(RAW_GEMINI_API_KEY_FROM_ENV)}")
else:
    logger.warning("GEMINI_API_KEY NOT FOUND in environment variables when module loaded.")
# +++ END: Explicitly Print Key from Environment +++

# --- For Gemini API ---
try:
    import google.generativeai as genai
    from retry import retry # For robust API calls
except ImportError:
    logger.warning("google-generativeai or retry library not found. LLM functionality will be skipped.")
    genai = None
    retry = None
# --- End Gemini API Imports ---

from config_pretrainer import (
    FILTERED_WIKIPEDIA_TEXT_DIR, SBERT_MODEL_NAME, SBERT_MODEL_SAVE_PATH,
    LANGUAGES_TO_PROCESS, WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT,
    FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_FULL_PATH,
    BERTOPIC_MIN_TOPIC_SIZE, BERTOPIC_NR_TOPICS
)

# --- Helper: SBERT Model Loading (no changes) ---
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

# --- Helper: Load Filtered Wikipedia Documents (no changes) ---
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
                    title = f.readline().strip(); f.readline() 
                    content_body = f.read().strip()
                    if content_body: documents.append(content_body); lang_docs_count += 1
                    else: logger.warning(f"Empty content body in file {filepath}. Skipping.")
            except Exception as e: logger.warning(f"Could not read or parse filtered file {filepath}: {e}")
        logger.info(f"[{lang_code}] Loaded {lang_docs_count} documents.")
        total_docs_loaded += lang_docs_count
    if any(doc is None for doc in documents):
        logger.error("CRITICAL: Found None values in the final documents list!"); documents = [d for d in documents if d is not None]
    if any(not doc.strip() for doc in documents if isinstance(doc, str)):
        logger.error("CRITICAL: Found empty strings in the final documents list!"); documents = [d for d in documents if isinstance(d, str) and d.strip()]
    logger.info(f"Loaded a total of {total_docs_loaded} initial documents. After final check, document count for BERTopic is {len(documents)}.")
    return documents

# --- LLM Configuration and Helper Functions ---
# GEMINI_API_KEY is now the RAW_GEMINI_API_KEY_FROM_ENV from the top
GEMINI_MODEL_NAME_FOR_NAMING = "gemini-1.5-flash-latest" 
gemini_llm_model_for_naming = None 

def initialize_gemini_model():
    global gemini_llm_model_for_naming 
    if gemini_llm_model_for_naming:
        return True
    
    # Use the RAW_GEMINI_API_KEY_FROM_ENV captured at module load
    if genai and RAW_GEMINI_API_KEY_FROM_ENV:
        try:
            genai.configure(api_key=RAW_GEMINI_API_KEY_FROM_ENV)
            gemini_llm_model_for_naming = genai.GenerativeModel(GEMINI_MODEL_NAME_FOR_NAMING)
            logger.info(f"Gemini client configured for topic naming with model: {GEMINI_MODEL_NAME_FOR_NAMING} using key from env.")
            return True
        except Exception as e:
            error_message = str(e).replace("{", "{{").replace("}", "}}") # Loguru fix
            logger.error(f"Failed to initialize Gemini model for naming: {error_message}")
            gemini_llm_model_for_naming = None
            return False
    else:
        if not genai: logger.warning("Gemini library (google-generativeai) not installed. LLM functionality will be skipped.")
        if not RAW_GEMINI_API_KEY_FROM_ENV: logger.warning("GEMINI_API_KEY (RAW_GEMINI_API_KEY_FROM_ENV) not available during initialization. LLM functionality will be skipped.")
        return False

def test_gemini_connection() -> bool:
    if not initialize_gemini_model() or not gemini_llm_model_for_naming :
        logger.error("Gemini model could not be initialized for connection test.")
        return False
    try:
        logger.info("Performing a quick test call to Gemini API to verify API key and connectivity...")
        test_prompt_parts = ["Explain AI in one sentence briefly."]
        response = gemini_llm_model_for_naming.generate_content(test_prompt_parts)
        if response.text and not response.text.strip().lower().startswith("error:"):
            logger.success(f"Gemini API test successful. Response snippet: {response.text.strip()[:70]}...")
            return True
        else:
            error_message = response.text.replace("{", "{{").replace("}", "}}") # Loguru fix
            logger.error(f"Gemini API test failed or returned error/empty. Response: {error_message}")
            return False
    except Exception as e:
        error_message = str(e).replace("{", "{{").replace("}", "}}") # Loguru fix
        logger.error(f"Error during Gemini API test call: {error_message}", exc_info=False) 
        # logger.exception("Full traceback for Gemini API test call error:") # Uncomment for full traceback if needed
        return False

if retry:
    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def generate_gemini_content_with_retry(prompt_parts: list):
        if not gemini_llm_model_for_naming: 
            logger.error("Gemini model for naming not initialized during retry call.")
            return "Error: LLM Naming Model Not Initialized"
        response = gemini_llm_model_for_naming.generate_content(prompt_parts)
        return response.text.strip()
else: 
    def generate_gemini_content_with_retry(prompt_parts: list):
        if not gemini_llm_model_for_naming:
            logger.error("Gemini model for naming not initialized during direct call.")
            return "Error: LLM Naming Model Not Initialized"
        try:
            response = gemini_llm_model_for_naming.generate_content(prompt_parts)
            return response.text.strip()
        except Exception as e_gemini_direct:
            error_message = str(e_gemini_direct).replace("{", "{{").replace("}", "}}") # Loguru fix
            logger.error(f"Direct Gemini API call failed: {error_message}")
            return f"Error in LLM call: {error_message}"


def generate_llm_topic_labels(
    topic_model: BERTopic, 
    num_keywords_for_prompt: int = 10,
    num_docs_for_prompt: int = 1
) -> Optional[Dict[int, str]]:
    if not gemini_llm_model_for_naming: 
        logger.warning("Gemini model object not initialized, skipping LLM topic label generation.")
        return None
    llm_labels: Dict[int, str] = {}
    topic_info_df = topic_model.get_topic_info()
    valid_topics_df = topic_info_df[topic_info_df.Topic != -1]
    logger.info(f"Attempting to generate LLM labels for {len(valid_topics_df)} topics.")
    for index, row in valid_topics_df.iterrows():
        topic_id = row["Topic"]
        default_bertopic_name = row.get("Name", f"Topic_{topic_id}") 
        keywords = topic_model.get_topic(topic_id)
        if not keywords:
            logger.warning(f"No keywords found for topic {topic_id} ('{default_bertopic_name}'). Skipping LLM naming.")
            continue
        top_keywords_list = [kw for kw, score in keywords[:num_keywords_for_prompt]]
        representative_docs_text = ""
        try:
            docs = topic_model.get_representative_docs(topic_id)
            if docs: representative_docs_text = " ".join([doc[:150] + "..." for doc in docs[:num_docs_for_prompt]])
        except Exception: pass
        prompt_parts = [
            "You are an expert in summarizing topics from lists of keywords and sample documents related to public health and general news from Tunisia.",
            "Generate a concise and descriptive topic label (3-7 words) that captures the essence of the topic defined by the following information.",
            "Focus on clarity and relevance. The keywords might be in English, French, or Arabic.",
            f"Keywords: {', '.join(top_keywords_list)}",
        ]
        if representative_docs_text: prompt_parts.append(f"Representative Document Snippets: {representative_docs_text}")
        prompt_parts.append("Topic Label:")
        try:
            logger.debug(f"Prompting LLM for topic ID {topic_id} ('{default_bertopic_name}')")
            generated_label = generate_gemini_content_with_retry(prompt_parts)
            generated_label = generated_label.replace("\n", " ").replace("Topic Label:", "").strip()
            if generated_label and not generated_label.lower().startswith("error:"):
                llm_labels[topic_id] = generated_label
                logger.info(f"LLM generated label for topic {topic_id} ('{default_bertopic_name}'): '{generated_label}'")
            else: 
                error_message = generated_label.replace("{", "{{").replace("}", "}}") # Loguru fix
                logger.warning(f"LLM returned empty or error label for topic {topic_id}. Using default. LLM response: '{error_message}'")
            time.sleep(0.5) 
        except Exception as e_llm: 
            error_message = str(e_llm).replace("{", "{{").replace("}", "}}") # Loguru fix
            logger.error(f"Error calling LLM for topic {topic_id} ('{default_bertopic_name}'): {error_message}", exc_info=False)
            # logger.exception(f"Full traceback for LLM call error on topic {topic_id}") # Uncomment for full traceback
            
    return llm_labels if llm_labels else None

# --- Train UNGUIDED Model ---
def train_bertopic_model_unsupervised(sbert_model: SentenceTransformer, 
                                     all_language_documents: List[str]) -> BERTopic | None:
    logger.info("--- Starting BERTopic Trainer: UNGUIDED MODE (on multilingual filtered Wikipedia data) ---")
    
    if not all_language_documents or len(all_language_documents) < BERTOPIC_MIN_TOPIC_SIZE:
        logger.error(f"Insufficient documents ({len(all_language_documents)}) for BERTopic training. Min: {BERTOPIC_MIN_TOPIC_SIZE}")
        return None
    
    logger.info(f"Number of documents for BERTopic: {len(all_language_documents)}")
    if all_language_documents:
        none_docs = sum(1 for doc in all_language_documents if doc is None)
        empty_docs = sum(1 for doc in all_language_documents if isinstance(doc, str) and not doc.strip())
        if none_docs > 0: logger.critical(f"CRITICAL PRE-FIT: {none_docs} None documents!")
        if empty_docs > 0: logger.critical(f"CRITICAL PRE-FIT: {empty_docs} empty string documents!")
        if not all_language_documents: logger.error("All documents empty after checks!"); return None
        logger.info(f"First document sample for BERTopic (first 100 chars): {all_language_documents[0][:100]}")
    
    logger.info(f"Initializing BERTopic model for UNGUIDED training with {len(all_language_documents)} documents.")
    
    topic_model = BERTopic(
        embedding_model=sbert_model,
        language="multilingual", 
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=BERTOPIC_NR_TOPICS, 
        seed_topic_list=None,
        verbose=True,
        calculate_probabilities=True
    )

    try:
        logger.info("Fitting UNGUIDED BERTopic model on combined multilingual Wikipedia data...")
        topic_model.fit_transform(all_language_documents) 
        
        num_topics = len(topic_model.get_topic_info()) - 1 
        logger.success(f"UNGUIDED BERTopic model training completed. Found {num_topics} topics.")
        
        if num_topics > 0: 
            logger.info(f"Sample of topics BEFORE LLM naming (UNGUIDED):\n{topic_model.get_topic_info().head(20)}")
            if gemini_llm_model_for_naming:
                logger.info("Attempting to generate LLM-based topic labels using Gemini...")
                try:
                    custom_llm_labels = generate_llm_topic_labels(topic_model) 
                    if custom_llm_labels:
                        topic_model.set_topic_labels(custom_llm_labels)
                        logger.success("Successfully applied LLM-generated topic labels.")
                        logger.info(f"Sample of topics AFTER LLM naming (UNGUIDED):\n{topic_model.get_topic_info().head(20)}")
                    else: logger.warning("LLM topic label generation returned no/empty labels. Using default BERTopic names.")
                except Exception as e_llm: 
                    error_message = str(e_llm).replace("{", "{{").replace("}", "}}") # Loguru fix
                    logger.error(f"Error during LLM topic label generation process: {error_message}. Using default.", exc_info=False)
            else: logger.warning("Gemini model object not initialized (check API key or library install), skipping LLM topic naming.")
        
        os.makedirs(FINAL_BERTOPIC_OUTPUT_DIR, exist_ok=True)
        model_save_path = FINAL_BERTOPIC_FULL_PATH 
        logger.info(f"Attempting to save UNGUIDED BERTopic model (with LLM naming if applied) to: {model_save_path}")
        
        try:
            topic_model.save(model_save_path, save_embedding_model=False)
            MIN_EXPECTED_FILE_SIZE_KB = 100 
            if os.path.exists(model_save_path) and os.path.getsize(model_save_path) > 1024 * MIN_EXPECTED_FILE_SIZE_KB: 
                 logger.success(f"UNGUIDED BERTopic model SUCCESSFULLY saved to: {model_save_path}")
                 return topic_model
            else:
                 file_size = os.path.getsize(model_save_path) if os.path.exists(model_save_path) else -1
                 error_msg = f"UNGUIDED BERTopic model save call finished, but file {model_save_path} not found or too small ({file_size} bytes)."
                 logger.error(error_msg)
                 return None
        except Exception as e_save: 
            logger.error(f"ERROR occurred during saving the UNGUIDED BERTopic model to {model_save_path}: {e_save}", exc_info=True)
            return None
    except ValueError as ve: 
        logger.error(f"ValueError during UNGUIDED BERTopic fit_transform: {ve}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"ERROR during UNGUIDED BERTopic training: {e}", exc_info=True)
        return None

# --- Main pipeline execution function ---
def run_bertopic_training_pipeline():
    logger.info("=== BERTopic Training Pipeline Initiated (TARGET: UNGUIDED with LLM Naming) ===")
    
    gemini_ready = test_gemini_connection()
    if not gemini_ready:
        logger.warning("Gemini API connection test failed or not configured. LLM topic naming will be SKIPPED for this run.")
        # If you want to ABORT if Gemini isn't working:
        # logger.error("Aborting pipeline due to Gemini API pre-check failure.")
        # return
    else:
        logger.info("Gemini API connection test successful. LLM naming will be attempted.")

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
    
    logger.info("Proceeding with UNGUIDED BERTopic model training.")
    final_model = train_bertopic_model_unsupervised(sbert, wiki_docs_multilingual) 

    if final_model:
        logger.success("=== BERTopic Training Pipeline (UNGUIDED with LLM Naming) Completed Successfully. Final model saved. ===")
    else:
        logger.error("=== BERTopic Training Pipeline (UNGUIDED with LLM Naming) FAILED. Check logs. ===")

if __name__ == "__main__":
    run_bertopic_training_pipeline()