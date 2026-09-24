import logging
import os

from flask import Flask, Response, jsonify, render_template, request, send_file

import rag
import tts
from vectorstore import get_store

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_index_loaded = False


@app.before_request
def _ensure_index_loaded():
    global _index_loaded
    if not _index_loaded:
        try:
            get_store().load()
        except Exception:
            logger.exception("Failed to load the search index")
        _index_loaded = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/logo.jpg")
def logo():
    return send_file(os.path.join(app.root_path, "logo.jpg"), mimetype="image/jpeg", max_age=3600)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()
    if lang not in ("en", "hi"):
        lang = "en"
    if not question:
        return jsonify({"error": "message is required"}), 400
    try:
        answer = rag.answer_query(question, lang=lang)
    except Exception:
        logger.exception("Failed to answer question: %r", question)
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 503
    return jsonify({"answer": answer})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "en").strip().lower()
    if lang not in ("en", "hi"):
        lang = "en"
    if not text:
        return jsonify({"error": "text is required"}), 400
    try:
        audio_bytes, mimetype = tts.synthesize_speech(text, lang)
    except Exception:
        logger.exception("Failed to synthesize speech for lang=%r", lang)
        return jsonify({"error": "Could not generate audio right now. Please try again."}), 503
    return Response(audio_bytes, mimetype=mimetype)


if __name__ == "__main__":
    get_store().load()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
