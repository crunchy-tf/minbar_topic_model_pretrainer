# topic_model_pretrainer_cloud/config_pretrainer.py
import os
from filter_keywords.en_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_EN
from filter_keywords.fr_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_FR
from filter_keywords.ar_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_AR
from health_topics_data import HEALTH_TOPICS_KEYWORDS as MASTER_HEALTH_TOPIC_DESCRIPTIONS_FOR_SEEDING

# --- General Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace")

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_gensim_extracted") # Parsed by Gensim
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts_gensim") # Output of filterer

LANGUAGES_TO_PROCESS = ["en", "fr", "ar"]

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic Model Output ---
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output_multilang_gensim")
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_multilang_gensim.joblib"
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

# BERTopic training parameters (articles sampled *after* filtering)
WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT: dict[str, int] = {
    "en": 30000,
    "fr": 15000,
    "ar": 10000
}
BERTOPIC_MIN_TOPIC_SIZE = 15
BERTOPIC_NR_TOPICS = None

SEED_KEYWORD_MIN_LEN = 3
SEED_MAX_KEYWORDS_PER_TOPIC = 7

# --- NEW: Limit articles read by the filterer from Gensim output ---
MAX_ARTICLES_TO_READ_FROM_GENSIM_PER_LANG: dict[str, int] = {
    "en": 750000,   # Filter up to the first 750k English articles from Gensim output
    "fr": 400000,
    "ar": 250000
}

# --- Filtering settings for Wikipedia articles ---
# HEALTH_CATEGORIES_KEYWORDS_EN/FR/AR are imported from filter_keywords/
# Ensure these are comprehensively populated in their respective files.

_all_specific_keywords_for_general_filter = []
for lang_specific_keyword_dict in [HEALTH_CATEGORIES_KEYWORDS_EN, HEALTH_CATEGORIES_KEYWORDS_FR, HEALTH_CATEGORIES_KEYWORDS_AR]:
    if lang_specific_keyword_dict:
        for keyword_list_for_category in lang_specific_keyword_dict.values():
            if keyword_list_for_category:
                _all_specific_keywords_for_general_filter.extend([kw.lower() for kw in keyword_list_for_category])

GENERAL_HEALTH_KEYWORDS_FILTER = list(set(_all_specific_keywords_for_general_filter + [
    "health", "medical", "disease", "symptom", "illness", "clinic", "hospital", "doctor",
    "pharma", "vaccin", "epidemic", "pandemic", "virus", "bacteria", "infection", "outbreak",
    "patient", "treatment", "care", "public health", "cdc", "who", "moh",
    "mortality", "morbidity", "surveillance", "prevention", "control measures",
    "santé", "médical", "maladie", "symptôme", "clinique", "hôpital", "docteur", "médecin",
    "pharmaceutique", "épidémie", "pandémie", "infection", "foyer de maladie",
    "صحة", "طبي", "مرض", "عرض", "مستشفى", "طبيب", "دواء", "صيدلية",
    "لقاح", "وباء", "جائحة", "عدوى", "فيروس", "بكتيريا", "تفشي"
]))
MIN_FILTERED_ARTICLE_LENGTH = 200

# --- Multiprocessing settings for filterer ---
FILTERING_PROCESSES = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
FILTERING_CHUNK_SIZE = 2000 # How many articles to batch to the multiprocessing pool at once