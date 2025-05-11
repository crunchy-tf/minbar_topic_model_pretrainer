# topic_model_pretrainer_cloud/config_pretrainer.py
import os

# Import the keyword dictionaries from the new filter_keywords package
# This assumes your 'filter_keywords' directory is in the same root as 'config_pretrainer.py'
# (i.e., both are directly under 'topic_model_pretrainer_cloud/')
# and 'filter_keywords' has an __init__.py file.
from filter_keywords.en_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_EN
from filter_keywords.fr_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_FR
from filter_keywords.ar_filter_keywords import HEALTH_CATEGORIES_KEYWORDS_AR

# health_topics_data.py should be in the same directory or your PYTHONPATH
# It contains your master HEALTH_TOPICS_KEYWORDS dictionary (with English keyword strings)
# This is used by bertopic_trainer.py for generating seed keyword lists for BERTopic.
from health_topics_data import HEALTH_TOPICS_KEYWORDS as MASTER_HEALTH_TOPIC_DESCRIPTIONS_FOR_SEEDING

# --- General Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__)) # Root of this pretrainer project
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace")

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_gensim_extracted")
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts_gensim")

LANGUAGES_TO_PROCESS = ["en", "fr", "ar"] # Processing all three languages

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic Model Output ---
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output_multilang_gensim")
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_multilang_gensim.joblib"
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

# BERTopic training parameters
WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT: dict[str, int] = {
    "en": 50000,
    "fr": 30000,
    "ar": 20000
}
BERTOPIC_MIN_TOPIC_SIZE = 20 # Min documents to form a topic (can be adjusted)
BERTOPIC_NR_TOPICS = None    # 'auto', or an integer, or len(MASTER_HEALTH_TOPIC_DESCRIPTIONS_FOR_SEEDING)

SEED_KEYWORD_MIN_LEN = 3     # For generating seed lists for BERTopic from descriptions
SEED_MAX_KEYWORDS_PER_TOPIC = 7 # For generating seed lists for BERTopic from descriptions

# --- Filtering settings for Wikipedia articles ---
# The HEALTH_CATEGORIES_KEYWORDS_EN/FR/AR dictionaries are imported from the filter_keywords package.
# They provide specific keywords for each language to filter Wikipedia articles.

# Fallback general keywords (lowercase) - built from the imported specific keyword lists plus extras
_all_specific_keywords_for_general_filter = []
# Iterate over the actual imported dictionary objects
for lang_specific_keyword_dict in [HEALTH_CATEGORIES_KEYWORDS_EN, HEALTH_CATEGORIES_KEYWORDS_FR, HEALTH_CATEGORIES_KEYWORDS_AR]:
    if lang_specific_keyword_dict: # Check if dictionary is not None and populated
        for keyword_list_for_category in lang_specific_keyword_dict.values():
            if keyword_list_for_category: # Ensure the list itself is not empty
                _all_specific_keywords_for_general_filter.extend([kw.lower() for kw in keyword_list_for_category])

GENERAL_HEALTH_KEYWORDS_FILTER = list(set(_all_specific_keywords_for_general_filter + [
    # Adding some very generic terms again just in case, and ensuring they are lowercased
    "health", "medical", "disease", "symptom", "illness", "clinic", "hospital", "doctor",
    "pharma", "vaccin", "epidemic", "pandemic", "virus", "bacteria", "infection", "outbreak",
    "patient", "treatment", "care", "public health", "cdc", "who", "moh",
    "mortality", "morbidity", "surveillance", "prevention", "control measures",
    "santé", "médical", "maladie", "symptôme", "clinique", "hôpital", "docteur", "médecin",
    "pharmaceutique", "épidémie", "pandémie", "infection", "foyer de maladie",
    "صحة", "طبي", "مرض", "عرض", "مستشفى", "طبيب", "دواء", "صيدلية",
    "لقاح", "وباء", "جائحة", "عدوى", "فيروس", "بكتيريا", "تفشي"
]))
MIN_FILTERED_ARTICLE_LENGTH = 200 # Min characters for a filtered article to be kept

# --- Multiprocessing settings for filterer ---
FILTERING_PROCESSES = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
FILTERING_CHUNK_SIZE = 5000 # How many articles to batch to the multiprocessing pool at once