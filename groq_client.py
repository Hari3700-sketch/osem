import logging
import re

from groq import Groq

import config
import gemini_client

logger = logging.getLogger(__name__)
_client = None

LONG_KEYWORDS = [
    "detail",
    "details",
    "full detail",
    "full details",
    "explain",
    "elaborate",
    "in depth",
    "indepth",
    "long answer",
    "complete information",
    "more information",
    "describe in detail",
    "full information",
    "give me everything",
    "comprehensive",
    "step by step",
    "step-by-step",
    "tell me everything",
    "puri jankari",
    "poori jankari",
    "poora detail",
    "puri detail",
    "sampoorna jankari",
    "vistar se",
    "vistaar se",
    "sab kuch batao",
    "जानकारी",
    "विस्तार से",
    "पूरी जानकारी",
    "सारी जानकारी",
    "पूरी डिटेल",
]


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def detect_length_mode(question):
    question_lower = question.lower()
    for keyword in LONG_KEYWORDS:
        if keyword in question_lower:
            return "long"
    return "short"


def _choose_max_tokens(question, lang="en"):
    question_lower = (question or "").lower()
    length_mode = detect_length_mode(question_lower)
    is_long_question = (
        length_mode == "long"
        or is_services_request(question_lower)
        or len(question_lower.split()) > 18
    )
    if is_long_question:
        return 1800
    if lang == "hi":
        return 1200
    return 1000


def _looks_truncated(answer):
    if not answer or not answer.strip():
        return True
    cleaned = answer.strip()
    if cleaned.endswith((".", "!", "?", ">", ");", "}")):
        return False
    if cleaned.endswith(("</li>", "</ul>", "</p>", "</strong>")):
        return False
    if len(cleaned) < 60:
        return False
    if re.search(r"[.!?][\s>]*$", cleaned):
        return False
    if re.search(r"(?:</li>|</ul>|</p>|</strong>)\s*$", cleaned):
        return False
    return True


def is_services_request(question):
    question_lower = question.lower()
    return any(
        phrase in question_lower
        for phrase in (
            "services", "service", "programs", "programmes", "what do you offer",
            "seva", "sewa", "suvidha", "सेवा", "सेवाएं", "सुविधाएं", "क्या क्या मिलता है",
        )
    )


def _language_instruction(lang):
    if lang == "hi":
        return (
            "Respond ONLY in natural, correct, simple Hindi written in Devanagari script, the way "
            "an Indian government helpdesk would talk to a citizen. Translate everything into Hindi "
            "except official scheme names, organisation names, URLs, phone numbers and email "
            "addresses, which may stay in their original Roman spelling. Do not mix English "
            "sentences into the answer and do not write Hindi in Roman/Hinglish letters - use "
            "Devanagari script only."
        )
    return "Respond only in clear English."


def generate_answer(question, context_chunks, lang="en"):
    lang = "hi" if lang == "hi" else "en"
    client = get_client()
    context_text = "\n\n".join(
        f"Source: {chunk['url']}\n{chunk['text']}" for chunk in context_chunks
    )

    length_mode = detect_length_mode(question)
    if is_services_request(question):
        length_instruction = (
            "Summarize the relevant services and program names found in the WEBSITE CONTENT. "
            "Use short bullet points and complete every bullet. Do not cut a sentence or paragraph."
        )
    else:
        length_instruction = (
            "Give a clean, summarized answer in one complete paragraph or short list. Use the WEBSITE CONTENT as the primary "
            "source and clearly mark details needing official verification. Format simple HTML only "
            "with <strong>, <br>, and <ul><li>. Do not use Markdown, asterisks, pipe characters, "
            "tables, URLs in visible text, or repeated explanations. Answer the exact question directly. "
            "Never stop in the middle of a sentence or paragraph."
        )

    max_tokens = _choose_max_tokens(question, lang)
    if length_mode == "long":
        max_tokens = max(max_tokens, 1600)

    system_prompt = (
        "You are an official virtual assistant for Uttar Pradesh government MSME "
        "and skill development portals including UPICON, MSME OSEM, CM Yuva, "
        "CM Youth Adda, Invest UP, Start In UP and MSME1Connect. Answer the user's "
        "question accurately. CM Youth Adda is the primary website for youth services, "
        "skill development, scheme access, franchise and startup support, MSME helpdesk, "
        "digital enablement, mentorship, workshops, business challenges, and innovation "
        "programs. ODOP always means One District One Product. Keep every answer concise, "
        "use simple clean words, summarize instead of repeating website text, and always finish complete sentences and paragraphs. "
        "Do not claim that a detail is official unless it "
        "is supported by the WEBSITE CONTENT. Mention the relevant website URL when "
        "useful. Never reply with phrases such as 'the provided information does "
        "not include', 'I do not have information', or only 'check the website' "
        "when the question names a supported scheme or portal. In that case, give "
        "the most useful general answer possible and identify which details need "
        "official verification. Return only the requested simple HTML elements; "
        "do not include a code fence.\n\n"
        f"{length_instruction}\n\n"
        f"{_language_instruction(lang)}\n\n"
        f"WEBSITE CONTENT:\n{context_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    completion = None
    answer = ""
    groq_failed = False
    try:
        for attempt in range(3):
            current_max_tokens = max_tokens if attempt == 0 else max(max_tokens, 1800)
            completion = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=0.3 if attempt == 0 else 0.1,
                max_tokens=current_max_tokens,
            )
            answer = (completion.choices[0].message.content or "").strip()
            if answer and not _looks_truncated(answer):
                break
            if not answer:
                messages.append({
                    "role": "user",
                    "content": "Please provide the answer now using the website content. Do not return an empty response.",
                })
            if attempt < 2:
                continue
    except Exception as exc:
        groq_failed = True
        logger.warning("Groq chat completion failed (limit reached?), falling back to Gemini: %s", exc)

    total_tokens = 0
    if groq_failed:
        try:
            answer = gemini_client.generate_answer(system_prompt, question)
        except Exception:
            logger.exception("Gemini fallback also failed")
            answer = _content_fallback(question, context_chunks, lang)
    elif not answer.strip():
        answer = _content_fallback(question, context_chunks, lang)
    else:
        total_tokens = completion.usage.total_tokens if completion.usage else 0
    answer = _clean_answer(answer)
    answer = re.sub(
        r"^\s*(?:<h[1-3]>\s*)?(?:<strong>\s*)?summary(?:\s*</strong>)?(?:\s*</h[1-3]>)?\s*(?:<br\s*/?>)?\s*",
        "",
        answer,
        count=1,
        flags=re.IGNORECASE,
    )
    return answer, total_tokens

def _content_fallback(question, context_chunks, lang="en"):
    """Return useful crawled content when the provider returns no answer."""
    source = next((chunk for chunk in context_chunks if chunk.get("text")), None)
    if not source:
        if lang == "hi":
            return (
                "<strong>क्षमा करें, अभी वेबसाइट की जानकारी नहीं मिल पाई।</strong><br>"
                "कृपया कुछ देर बाद फिर से प्रयास करें।"
            )
        return (
            "<strong>Sorry, I could not retrieve website content right now.</strong><br>"
            "Please try again in a moment."
        )
    text = _clean_crawled_text(source["text"])
    url = source.get("url", "")
    if lang == "hi":
        text = translate_to_hindi(text)
        label = "इस प्रश्न से संबंधित जानकारी:"
        source_label = "स्रोत:"
    else:
        label = "Information related to your question:"
        source_label = "Source:"
    return (
        f"<strong>{label}</strong><br>{text}"
        + (f"<br><br>{source_label} {url}" if url else "")
    )


def translate_to_hindi(html_text):
    """Translate fixed English HTML content into natural Hindi, preserving tags/links."""
    if not html_text or not html_text.strip():
        return html_text
    client = get_client()
    system_prompt = (
        "Translate the following government-scheme text into natural, simple, correct Hindi "
        "written in Devanagari script, the way an Indian government helpdesk would speak to a "
        "citizen. Keep proper nouns, scheme names, organisation names, place names, URLs, phone "
        "numbers, and email addresses unchanged. Keep every HTML tag (such as <a>, <strong>, <b>, "
        "<br>, <ul>, <li>, <ol>) and its attributes exactly as given, translating only the visible "
        "text inside them. Do not add commentary, notes, or code fences. Return only the "
        "translated HTML and nothing else."
    )
    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": html_text},
            ],
            temperature=0.2,
            max_tokens=1200,
        )
        translated = (completion.choices[0].message.content or "").strip()
        translated = _clean_answer(translated)
        return translated or html_text
    except Exception:
        return html_text


def _clean_crawled_text(text):
    """Remove repeated navigation labels from a raw page-text fallback."""
    text = re.sub(
        r"\b(?:home|business|start business|grow business|business listing|finance\s*&\s*support|"
        r"marketplace|experts|youth adda|training|others|services|events|about us|contact|login)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" |:-")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    unique = []
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            unique.append(normalized)
            seen.add(key)
    return " ".join(unique)


def _clean_answer(answer):
    """Remove model clutter without cutting the answer mid-sentence."""
    answer = re.sub(r"```(?:html)?|```", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer
