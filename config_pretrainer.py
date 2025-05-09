# topic_model_pretrainer_cloud/config_pretrainer.py
import os

# --- General Paths ---
# Base directory for all operations within this pretrainer project
# When running on a GCE VM, this could be a path on its persistent disk.
BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Project root
WORKING_DIR = os.path.join(BASE_PROJECT_DIR, "pretrainer_workspace") # For all intermediate files

# --- Wikipedia Data Paths ---
WIKIPEDIA_DUMP_DIR = os.path.join(WORKING_DIR, "wikipedia_dumps")
WIKIPEDIA_EXTRACTED_DIR = os.path.join(WORKING_DIR, "wikipedia_extracted_raw") # Raw output from wikiextractor
FILTERED_WIKIPEDIA_TEXT_DIR = os.path.join(WORKING_DIR, "wikipedia_filtered_texts") # Health-relevant texts

# List of language codes for Wikipedia processing
LANGUAGES_TO_PROCESS = ["en", "fr", "ar"] # English, French, Arabic

# --- SBERT Model ---
SBERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
SBERT_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sbert_model")

# --- BERTopic Pre-training (Stage 1 - Wikipedia Base) ---
# This model is an intermediate step, trained on Wikipedia
BASE_WIKI_BERTOPIC_MODEL_DIR = os.path.join(WORKING_DIR, "bertopic_model_base_wikipedia")
BASE_WIKI_BERTOPIC_MODEL_FILENAME = "bertopic_model_base_wikipedia.joblib"
BASE_WIKI_BERTOPIC_FULL_PATH = os.path.join(BASE_WIKI_BERTOPIC_MODEL_DIR, BASE_WIKI_BERTOPIC_MODEL_FILENAME)

WIKIPEDIA_ARTICLES_PER_LANGUAGE_LIMIT = 50000 # Max articles PER LANGUAGE to use for Stage 1 (adjust based on resources)
                                            # Total articles = this * len(LANGUAGES_TO_PROCESS)
                                            # 50k * 3 = 150k articles. Still substantial.
                                            # Start smaller if needed (e.g., 10k-20k per lang)
STAGE1_BERTOPIC_MIN_TOPIC_SIZE = 50
STAGE1_BERTOPIC_NR_TOPICS = "auto" # Or a large number like 200-500

# --- BERTopic Fine-tuning/Guiding (Stage 2 - Using HEALTH_TOPICS) ---
# This is the FINAL model output that nlp_analyzer will use.
# The nlp_analyzer's .env.yaml should point to this filename.
FINAL_BERTOPIC_OUTPUT_DIR = os.path.join(WORKING_DIR, "bertopic_model_final_output") # Final output location
FINAL_BERTOPIC_MODEL_FILENAME = "bertopic_model_final_guided.joblib" # This file will be copied to nlp_analyzer
FINAL_BERTOPIC_FULL_PATH = os.path.join(FINAL_BERTOPIC_OUTPUT_DIR, FINAL_BERTOPIC_MODEL_FILENAME)

# Parameters for Stage 2 (guiding the Wikipedia-trained model with HEALTH_TOPICS seeds)
# Since Stage 2 now only uses HEALTH_TOPICS and the Wikipedia model (no new data from PG):
# The number of documents for fit_transform in Stage 2 will be the Wikipedia documents.
# The min_topic_size might need to be adjusted carefully.
STAGE2_BERTOPIC_MIN_TOPIC_SIZE = 25 # Adjust based on expected support for seed topics within Wikipedia data
STAGE2_BERTOPIC_NR_TOPICS = None # Or len(HEALTH_TOPICS_KEYWORDS) if you want to try and force it

# Keyword generation for seeds
SEED_KEYWORD_MIN_LEN = 3
SEED_MAX_KEYWORDS_PER_TOPIC = 7

# --- wikiextractor settings ---
WIKIEXTRACTOR_PROCESSES = 4 # Number of cores for wikiextractor
WIKIEXTRACTOR_BYTES_PER_CHUNK = "500M" # Split wikiextractor output files

# --- Filtering settings for Wikipedia articles ---
# You MUST define these lists accurately for good filtering.
# These are examples; research actual Wikipedia category names for health.
HEALTH_CATEGORIES_FILTER_EN = [
    "medicine", "public health", "diseases and disorders", "pharmacology", "epidemiology",
    "mental health", "viruses", "bacteria", "pandemics", "vaccines", "symptoms", "nutrition",
    "human anatomy", "physiology", "toxicology", "medical specialties", "health care", "health organizations"
]
HEALTH_CATEGORIES_FILTER_FR = [
    "médecine", "santé publique", "maladies et troubles", "pharmacologie", "épidémiologie",
    "santé mentale", "virus", "bactéries", "pandémies", "vaccins", "symptômes", "nutrition",
    "anatomie humaine", "physiologie", "toxicologie", "spécialités médicales", "soins de santé", "organisations de santé"
]
HEALTH_CATEGORIES_FILTER_AR = [
    "طب", "صحة عامة", "أمراض", "اضطرابات", "علم الأدوية", "علم الأوبئة", "صحة نفسية",
    "فيروسات", "بكتيريا", "جوائح", "أوبئة", "لقاحات", "أعراض", "تغذية",
    "تشريح الإنسان", "علم وظائف الأعضاء", "علم السموم", "تخصصات طبية", "رعاية صحية", "منظمات صحية"
]
GENERAL_HEALTH_KEYWORDS_FILTER = [ # For fallback text search if categories are messy
    "health", "medical", "disease", "symptom", "illness", "clinic", "hospital", "doctor",
    "pharma", "vaccin", "epidemic", "pandemic", "virus", "bacteria", "infection", "outbreak",
    "santé", "médical", "maladie", "symptôme", "clinique", "hôpital", "docteur", "médecin",
    "pharmaceutique", "épidémie", "pandémie", "infection", "foyer de maladie",
    "صحة", "طبي", "مرض", "عرض", "مستشفى", "طبيب", "دواء", "صيدلية",
    "لقاح", "وباء", "جائحة", "عدوى", "فيروس", "بكتيريا", "تفشي"
]
MIN_FILTERED_ARTICLE_LENGTH = 250 # Min characters for a filtered article to be kept for training