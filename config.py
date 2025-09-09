import os
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "azerbaijani_legal_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PROCESSED_COLLECTION_NAME = os.getenv("PROCESSED_COLLECTION_NAME")

VESPA_URL = os.getenv("VESPA_URL", "http://vespa-legalens.internal")
VESPA_PORT = os.getenv("VESPA_PORT", "8080")  # Pass as string if you expect int conversion later
STUDY_NAME = os.getenv("STUDY_NAME", "vespa_ranking_study")
STUDY_STORAGE_POSTGRES_URL = os.getenv("STUDY_STORAGE_POSTGRES_URL")
SQLITE_DB_FILE = "queries.db"
SQLITE_URL = F"sqlite:///{SQLITE_DB_FILE}"

MAX_BATCH_COUNT = int(os.getenv("MAX_BATCH_COUNT", 1))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 0))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", 0))  # Seconds
MAX_CONCURRENT_BATCHES = int(os.getenv("MAX_CONCURRENT_BATCHES", 1))
COMPLETION_WINDOW = "24h"

client_id = str(uuid.uuid4())

api_key_tier1 = os.getenv("OPENAI_API_KEY_TIER1")
api_key_tier5 = os.getenv("OPENAI_API_KEY_TIER5")
# OPENAI_KEYS = [api_key_tier1, api_key_tier5]

OPENAI_KEYS = [api_key_tier5]

INPUT_DATA_DIR = os.getenv("INPUT_DATA_DIR", "input_data")