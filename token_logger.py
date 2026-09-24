import json
import os
import threading
from datetime import datetime, timezone

import config

_lock = threading.Lock()


def _load_log():
    if not os.path.exists(config.TOKEN_LOG_PATH):
        return []
    try:
        with open(config.TOKEN_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def log_question(question, tokens_generated):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with _lock:
        entries = _load_log()
        entries.append(
            {
                "question": question,
                "tokens_generated": tokens_generated,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        with open(config.TOKEN_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
