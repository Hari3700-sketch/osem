import base64
import io
import wave

import requests

import config

GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _pcm_to_wav(pcm_bytes, channels=1, sample_rate=24000, sample_width=2):
    """Gemini's TTS API returns raw 16-bit PCM audio; wrap it in a playable WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer.read()


def generate_answer(system_prompt, question):
    """Fallback chat answer using Gemini, used when the Groq API is rate-limited or unavailable."""
    url = GEMINI_URL_TEMPLATE.format(model=config.GEMINI_TEXT_MODEL)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 600},
    }
    response = requests.post(
        url,
        headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    parts = data["candidates"][0]["content"]["parts"]
    answer = "".join(part.get("text", "") for part in parts).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer")
    return answer


def _request_speech_once(text, voice_name):
    url = GEMINI_URL_TEMPLATE.format(model=config.GEMINI_TTS_MODEL)
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            # Gemini's TTS models are known to silently cut audio short for longer
            # text unless given a generous output budget - set it explicitly high.
            "maxOutputTokens": 8000,
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            },
        },
    }
    response = requests.post(
        url,
        headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    candidate = data["candidates"][0]
    finish_reason = candidate.get("finishReason", "STOP")
    inline_data = candidate["content"]["parts"][0].get("inlineData")
    if not inline_data or not inline_data.get("data"):
        raise RuntimeError(f"Gemini TTS returned no audio (finishReason={finish_reason})")
    pcm_bytes = base64.b64decode(inline_data["data"])

    # Gemini TTS is known to occasionally cut audio short even on a normal 200
    # response with no error. Sanity-check the audio length against the text
    # length (24kHz, 16-bit, mono = 48000 bytes/second) and treat clearly
    # short/incomplete audio as a failure so the caller retries or falls back.
    actual_seconds = len(pcm_bytes) / 48000
    expected_min_seconds = max(0.6, len(text) / 40)
    if finish_reason not in ("STOP", None) or actual_seconds < expected_min_seconds:
        raise RuntimeError(
            f"Gemini TTS audio looks truncated (finishReason={finish_reason}, "
            f"{actual_seconds:.1f}s audio for {len(text)} chars of text)"
        )
    return pcm_bytes


def synthesize_speech(text, voice_name=None):
    """Generate clear speech audio with Gemini's TTS model. Supports Hindi and English
    (and 20+ other languages) with automatic language detection. Returns (wav_bytes, mimetype).
    Retries once, since Gemini's TTS occasionally truncates audio on the first try."""
    voice_name = voice_name or config.GEMINI_TTS_VOICE
    last_error = None
    for attempt in range(2):
        try:
            pcm_bytes = _request_speech_once(text, voice_name)
            return _pcm_to_wav(pcm_bytes), "audio/wav"
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Gemini TTS failed after retries: {last_error}")
