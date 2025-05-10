# topic_model_pretrainer_cloud/config_pretrainer.py
import os

# --- General Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace")

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_EXTRACTED_DIR = os.path.join(WORKING_DIR, "wikipedia_extracted_raw_v2") # Changed name slightly
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts_v2") # Changed name slightly

# List of language codes for Wikipedia processing
LANGUAGES_TO_PROCESS = ["en"] # MODIFIED: Only English for now

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic (Stage 1 - Base on Wikipedia, and Stage 2 - Final Guided) ---
# Stage 1 (Optional Base Model - can be skipped if Stage 2 trains from scratch on filtered Wiki data)
BASE_WIKI_BERTOPIC_MODEL_DIR = os.path.join(WORKING_DIR, "bertopic_model_base_wikipedia_v2")
BASE_WIKI_BERTOPIC_MODEL_FILENAME = "bertopic_model_base_wikipedia_v2.joblib"
BASE_WIKI_BERTOPIC_FULL_PATH = os.path.join(BASE_WIKI_BERTOPIC_MODEL_DIR, BASE_WIKI_BERTOPIC_MODEL_FILENAME)

WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT = 50000 # Max *filtered* articles per language to load for BERTopic training
STAGE1_BERTOPIC_MIN_TOPIC_SIZE = 50
STAGE1_BERTOPIC_NR_TOPICS = "auto"

# Stage 2 / Final Model (Trained on filtered Wikipedia, Guided by HEALTH_TOPICS)
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output_v2")
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_v2.joblib"
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

STAGE2_BERTOPIC_MIN_TOPIC_SIZE = 25
STAGE2_BERTOPIC_NR_TOPICS = None # Or len(HEALTH_TOPICS_KEYWORDS)

SEED_KEYWORD_MIN_LEN = 3
SEED_MAX_KEYWORDS_PER_TOPIC = 7

# --- WikiExtractor V2 settings ---
WIKIEXTRACTOR_PROCESSES = 4
WIKIEXTRACTOR_BYTES_PER_CHUNK = "500M"
# Path for WikiExtractor V2's template cache files.
# It's good to place this within the WIKIPEDIA_EXTRACTED_DIR so it's language-specific if needed.
# The actual filename will be constructed in wikipedia_parser.py
WIKIEXTRACTOR_TEMPLATES_DIR = os.path.join(WIKIPEDIA_EXTRACTED_DIR, "templates_cache")


# --- Filtering settings for Wikipedia articles ---
# Only English needed for now
HEALTH_CATEGORIES_FILTER_EN = [
    "medicine", "public health", "diseases and disorders", "pharmacology", "epidemiology",
    "mental health", "viruses", "bacteria", "pandemics", "vaccines", "symptoms", "nutrition",
    "human anatomy", "physiology", "toxicology", "medical specialties", "health care", "health organizations"
]
# HEALTH_CATEGORIES_FILTER_FR = [ ... ] # Removed
# HEALTH_CATEGORIES_FILTER_AR = [ ... ] # Removed

GENERAL_HEALTH_KEYWORDS_FILTER = [
    "health", "medical", "disease", "symptom", "illness", "clinic", "hospital", "doctor",
    "pharma", "vaccin", "epidemic", "pandemic", "virus", "bacteria", "infection", "outbreak",
    # Add French and Arabic keywords here if you still want to use them for filtering text
    # even if not processing FR/AR dumps for pre-training BERTopic.
    # However, if only processing EN dump, EN keywords here are most relevant.
    "santé", "médical", "maladie", "symptôme", "clinique", "hôpital", "docteur", "médecin",
    "pharmaceutique", "épidémie", "pandémie", "infection", "foyer de maladie",
    "صحة", "طبي", "مرض", "عرض", "مستشفى", "طبيب", "دواء", "صيدلية",
    "لقاح", "وباء", "جائحة", "عدوى", "فيروس", "بكتيريا", "تفشي"
]
MIN_FILTERED_ARTICLE_LENGTH = 250