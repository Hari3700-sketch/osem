import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

# Gemini is the fallback: used for chat answers when Groq is rate-limited/unavailable,
# and used for Hindi voice (Groq has no Hindi TTS voice; Gemini's TTS supports Hindi
# clearly). English speech still tries Groq's TTS first, then falls back to Gemini too.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

# Groq's text-to-speech API only supports English and Arabic voices (no Hindi
# voice model exists there), so English replies are spoken with Groq's TTS
# voice below and Hindi replies use Gemini's TTS engine for clear audio.
GROQ_TTS_VOICE_EN = os.getenv("GROQ_TTS_VOICE_EN", "autumn")

SITE_URLS = [
    "https://upicon.in/",
    "https://upicon.in/skill-development",
    "https://upicon.in/contact",
    "https://cmyouthadda.in/",
    "https://upicon.in/csr-training",
    "https://upicon.in/consultancy",
    "https://msmeosem.in/",
    "https://xn--i1bn6adp9emg4dcbcajdeflxp1gua1n7bt10abief.xn--11b7cb3a6a.xn--h2brj9c/",
    "https://msme.up.gov.in/hi/login/error",
    "https://cmyuva.org.in/",
    "https://upid.ac.in/vishwakarma-shram-samman-yojana-vssy/",
    "https://msme.up.gov.in/",
    "https://up.gov.in/",
    "https://msme1connect.up.gov.in",
    "https://msme1connect.up.gov.in/scheme/up-schemem",
    "https://msme1connect.up.gov.in/about",
    "https://invest.up.gov.in",
    "https://startinup.up.gov.in",
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
SIMILARITY_THRESHOLD = 0.55

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(DATA_DIR, "site_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "site_metadata.pkl")
CACHE_PATH = os.path.join(DATA_DIR, "site_chunks.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
TOKEN_LOG_PATH = os.path.join(DATA_DIR, "token_usage.json")
