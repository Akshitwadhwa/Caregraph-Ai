import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Shared constants ─────────────────────────────────────────────────
DEFAULT_FAISS_DIR = "medical_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("caregraph")


def get_faiss_dir() -> str:
    return os.getenv("FAISS_DIR", DEFAULT_FAISS_DIR)
