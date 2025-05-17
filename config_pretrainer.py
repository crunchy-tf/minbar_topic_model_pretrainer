# topic_model_pretrainer_cloud/config_pretrainer.py
import os
from filter_keywords.en_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_EN
from filter_keywords.fr_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_FR
from filter_keywords.ar_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_AR
# from health_topics_data import HEALTH_TOPICS_KEYWORDS as MASTER_HEALTH_TOPIC_DESCRIPTIONS_FOR_SEEDING # No longer needed by trainer

# --- General Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace")

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_gensim_extracted")
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts_gensim")

LANGUAGES_TO_PROCESS = ["en", "fr", "ar"]

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic Model Output ---
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output_multilang_gensim")
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_multilang_gensim.joblib" # You might rename this if always unguided
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

# BERTopic training parameters
WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT: dict[str, int] = {
    "en": 10000, # Adjusted from your logs
    "fr": 10000, # Adjusted from your logs
    "ar": 10000  # Adjusted from your logs
}
BERTOPIC_MIN_TOPIC_SIZE = 15
BERTOPIC_NR_TOPICS = None # 'auto' for unguided is typical

# SEED_KEYWORD_MIN_LEN = 3 # No longer used by trainer
# SEED_MAX_KEYWORDS_PER_TOPIC = 7 # No longer used by trainer

# --- Limit articles read by the filterer from Gensim output ---
MAX_ARTICLES_TO_READ_FROM_GENSIM_PER_LANG: dict[str, int] = {
    "en": 250000, 
    "fr": 250000,
    "ar": 250000
}

# --- Filtering settings (Still needed for the filtering step) ---
_all_specific_keywords_for_general_filter = []
# ... (rest of your keyword filter generation logic remains) ...
GENERAL_HEALTH_KEYWORDS_FILTER = list(set(_all_specific_keywords_for_general_filter + [
    # ... your general keywords ...
]))
MIN_FILTERED_ARTICLE_LENGTH = 200
FILTERING_PROCESSES = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
FILTERING_CHUNK_SIZE = 2000