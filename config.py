import os
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

VESPA_URL = os.getenv("VESPA_URL", "http://vespa-legalens.internal")
VESPA_PORT = os.getenv("VESPA_PORT", "8080")  # Pass as string if you expect int conversion later
VESPA_APP_NAME = os.getenv("VESPA_APP_NAME", "chunks")
STUDY_NAME = os.getenv("STUDY_NAME", "vespa_ranking_study")
STUDY_STORAGE_POSTGRES_URL = os.getenv("STUDY_STORAGE_POSTGRES_URL")
SQLITE_DB_FILE = os.getenv("SQLITE_DB_FILE", "queries.db")
SQLITE_URL = F"sqlite:///{SQLITE_DB_FILE}"