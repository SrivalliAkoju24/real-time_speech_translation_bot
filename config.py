"""
config.py — Centralised configuration for the speech translation pipeline.

All settings are driven by environment variables loaded from a .env file.
Copy .env.example → .env and fill in your API keys before running.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
AZURE_TTS_KEY: str = os.getenv("AZURE_TTS_KEY", "")
AZURE_REGION: str = os.getenv("AZURE_REGION", "eastus")

# ── Language Settings ────────────────────────────────────────────────────────
# BCP-47 code for Deepgram STT (e.g. "en-US", "es", "fr")
SOURCE_LANGUAGE: str = os.getenv("SOURCE_LANGUAGE", "en-US")
# Short code for translation target + Azure TTS (e.g. "hi", "es", "fr", "de")
TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "hi")

# ── Deepgram Audio Settings ──────────────────────────────────────────────────
SAMPLE_RATE: int = 16000
ENCODING: str = "linear16"
CHANNELS: int = 1
UTTERANCE_END_MS: int = 2000   # Wait this long after speech stops before finalising
ENDPOINTING_MS: int = 1000     # Silence threshold to detect end of utterance

# ── Groq / LLM Settings ─────────────────────────────────────────────────────
GROQ_MODEL: str = "llama3-8b-8192"
GROQ_TEMPERATURE: float = 0.2
GROQ_MAX_TOKENS: int = 1024

# ── Validation ───────────────────────────────────────────────────────────────
_required = {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "AZURE_TTS_KEY": AZURE_TTS_KEY,
    "AZURE_REGION": AZURE_REGION,
}

def validate():
    missing = [k for k, v in _required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Copy .env.example → .env and fill in your credentials."
        )
