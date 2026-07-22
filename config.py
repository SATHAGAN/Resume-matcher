import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'matchmaker.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- NVIDIA NIM ---
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    NVIDIA_CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "meta/llama-3.3-70b-instruct")
    NVIDIA_EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    EMBED_DIM = 1024

    # --- Matching engine tuning ---
    HYBRID_SEARCH_TOP_K = 50          # candidates pulled by hybrid search per JD
    XAI_TOP_N = 10                    # candidates that get a full XAI writeup
    MAX_UPLOAD_MB = 15

    UPLOAD_ALLOWED_EXTENSIONS = {"pdf", "docx"}
