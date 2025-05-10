# topic_model_pretrainer_cloud/config_pretrainer.py
import os
from health_topics_data import HEALTH_TOPICS_KEYWORDS # Import for convenience

# --- General Paths ---
BASE_PROJECT_DIR = os.path.abspath(os.path.dirname(__file__)) # Root of this pretrainer project
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace")

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_GENSIM_EXTRACTED_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_gensim_extracted")
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts_gensim")

LANGUAGES_TO_PROCESS = ["en"] # Focusing on English

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic Model Output ---
# Final Model (Trained on filtered Wikipedia, Guided by HEALTH_TOPICS from health_topics_data.py)
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output_gensim")
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided_gensim.joblib"
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

# BERTopic training parameters
WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT = 50000 # Max *filtered* articles to load for BERTopic training
BERTOPIC_MIN_TOPIC_SIZE = 20 # Min documents to form a topic (adjust based on filtered data size)
BERTOPIC_NR_TOPICS = None    # 'auto', or an integer, or len(HEALTH_TOPICS_KEYWORDS)

SEED_KEYWORD_MIN_LEN = 3
SEED_MAX_KEYWORDS_PER_TOPIC = 7 # Max seed keywords per HEALTH_TOPIC

# --- Filtering settings for Wikipedia articles (using gensim output) ---
# This dictionary maps keys from your HEALTH_TOPICS_KEYWORDS to lists of
# keywords that will be searched for within the article text.
# YOU MUST POPULATE THIS COMPREHENSIVELY FOR 'en'.
HEALTH_CATEGORIES_KEYWORDS_EN: dict[str, list[str]] = {
    # Example: if HEALTH_TOPICS_KEYWORDS has "symptoms_general": "general systemic ..."
    "symptoms_general": ["general", "systemic", "fatigue", "malaise", "weakness", "weight change", "appetite change", "night sweat"],
    "symptoms_fever_temperature": ["fever", "high temperature", "hypothermia", "chill", "shivering", "temperature"],
    "disease_covid19": ["covid-19", "covid", "omicron", "long covid", "pasc", "coronavirus", "sars-cov-2", "pandemic"],
    "public_health_vaccination_general": ["vaccine", "vaccination", "immunization", "jab", "shot", "vaccinate"],
    "mental_health_symptoms_anxiety": ["anxiety", "worry", "nervousness", "panic attack", "stress"],
    # !!! IMPORTANT !!!
    # Add entries here for ALL keys present in your health_topics_data.HEALTH_TOPICS_KEYWORDS
    # For each key, provide a list of 5-15 strong, relevant English keywords.
    # Example for another key:
    # "disease_influenza_seasonal": ["influenza", "flu", "seasonal flu", "h1n1", "h3n2", "grippe"],
    # "health_system_capacity_hospitals": ["hospital", "bed availability", "icu", "emergency room", "overcrowding", "wait time"],
}

# Fallback general keywords (lowercase)
GENERAL_HEALTH_KEYWORDS_FILTER = [
    "health", "medical", "disease", "symptom", "illness", "clinic", "hospital", "doctor",
    "pharma", "vaccin", "epidemic", "pandemic", "virus", "bacteria", "infection", "outbreak",
    "patient", "treatment", "care", "public health", "cdc", "who", "moh", # Common orgs
    "mortality", "morbidity", "surveillance", "prevention", "control measures"
]
MIN_FILTERED_ARTICLE_LENGTH = 200 # Min characters for a filtered article to be kept