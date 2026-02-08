import os
from dotenv import load_dotenv

DEFAULT_FAISS_DIR = "medical_db"


def get_faiss_dir() -> str:
    load_dotenv()
    return os.getenv("FAISS_DIR", DEFAULT_FAISS_DIR)
