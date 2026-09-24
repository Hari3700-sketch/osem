import io
import logging
import re

import requests
from gtts import gTTS

import config
import gemini_client

logger = logging.getLogger(__name__)

GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"

# Groq's speech API ships English and Arabic voices only - there is no Hindi voice
# model on Groq. So:
#   - English replies: try Groq's TTS voice first, fall back to Gemini's TTS if
#     Groq fails or its usage limit is hit.
#   - Hindi replies: use Gemini's TTS engine directly (it supports Hindi with clear,
#     correctly pronounced audio), falling back to gTTS if Gemini is unavailable.
GROQ_EN_VOICE_OPTIONS = [
    {"model": "canopylabs/orpheus-v1-english", "voice": config.GROQ_TTS_VOICE_EN},
    {"model": "playai-tts", "voice": "Fritz-PlayAI"},
]


def clean_text_for_speech(text):
    """Strip HTML tags, links and markdown noise so only speakable words remain."""
    if not text:
        return ""
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[*_#`~|^]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _groq_english(text):
    """Call Groq's TTS API. Raises on failure so the caller can fall back to Gemini."""
    last_error = None
    for option in GROQ_EN_VOICE_OPTIONS:
        try:
            response = requests.post(
                GROQ_SPEECH_URL,
                headers={
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": option["model"],
                    "voice": option["voice"],
                    "input": text,
                    "response_format": "wav",
                },
                timeout=30,
            )
            if response.status_code == 200 and response.content:
                return response.content, "audio/wav"
            last_error = f"{option['model']} returned {response.status_code}: {response.text[:200]}"
            logger.warning("Groq TTS attempt failed: %s", last_error)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Groq TTS attempt errored: %s", last_error)
    raise RuntimeError(f"Groq TTS failed for all voice options: {last_error}")


def _gtts_speech(text, lang):
    """Last-resort speech using gTTS - reliable and never truncates, used if both
    Groq and Gemini fail (or Gemini keeps truncating)."""
    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read(), "audio/mpeg"


def synthesize_english(text):
    try:
        return _groq_english(text)
    except Exception as exc:
        logger.warning("Groq TTS unavailable, falling back to Gemini TTS: %s", exc)
    try:
        return gemini_client.synthesize_speech(text)
    except Exception as exc:
        logger.warning("Gemini TTS unavailable, falling back to gTTS: %s", exc)
    return _gtts_speech(text, "en")


def synthesize_hindi(text):
    try:
        return gemini_client.synthesize_speech(text)
    except Exception as exc:
        logger.warning("Gemini TTS unavailable, falling back to gTTS: %s", exc)
    return _gtts_speech(text, "hi")


def synthesize_speech(text, lang):
    """Return (audio_bytes, mimetype) of clear speech for the given text and language."""
    clean_text = clean_text_for_speech(text)
    if not clean_text:
        raise ValueError("No speakable text provided")
    if lang == "hi":
        return synthesize_hindi(clean_text)
    return synthesize_english(clean_text)
