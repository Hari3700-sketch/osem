import concurrent.futures
import json
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

import config
import crawler
import embedder
import groq_client
import token_logger
from vectorstore import get_store


def _load_intents():
    try:
        with open(config.INTENTS_PATH, encoding="utf-8") as intents_file:
            data = json.load(intents_file)
        return [item for item in data if item.get("patterns") and item.get("responses")]
    except (OSError, json.JSONDecodeError):
        return []


INTENTS = _load_intents()

TOPIC_URL_MAP = {
    "cm yuva udyami": ["https://cmyuva.org.in/"],
    "yuva udyami yojna": ["https://cmyuva.org.in/"],
    "yuva udyami yojana": ["https://cmyuva.org.in/"],
    "yuva udyami": ["https://cmyuva.org.in/"],
    "cm yuva": ["https://cmyuva.org.in/"],
    "cmyuva": ["https://cmyuva.org.in/"],
    "cm youth adda": ["https://cmyouthadda.in/"],
    "youth adda": ["https://cmyouthadda.in/"],
    "cmyouthadda": ["https://cmyouthadda.in/"],
    "msme osem": ["https://msmeosem.in/"],
    "msmeosem": ["https://msmeosem.in/"],
    "osem": ["https://msmeosem.in/"],
    "vssy": ["https://upid.ac.in/vishwakarma-shram-samman-yojana-vssy/"],
    "vishwakarma shram samman": ["https://upid.ac.in/vishwakarma-shram-samman-yojana-vssy/"],
    "ramp yojana": ["https://cmyuva.org.in/"],
    "ramp yojna": ["https://cmyuva.org.in/"],
    "ramp": ["https://cmyuva.org.in/"],
    "skill development": ["https://upicon.in/skill-development"],
    "csr training": ["https://upicon.in/csr-training"],
    "consultancy": ["https://upicon.in/consultancy"],
    "contact": ["https://upicon.in/contact"],
    "upicon": [
        "https://upicon.in/",
        "https://upicon.in/skill-development",
        "https://upicon.in/contact",
        "https://upicon.in/csr-training",
        "https://upicon.in/consultancy",
    ],
    "invest up": ["https://invest.up.gov.in"],
    "invest": ["https://invest.up.gov.in"],
    "start in up": ["https://startinup.up.gov.in"],
    "startinup": ["https://startinup.up.gov.in"],
    "msme1connect": [
        "https://msme1connect.up.gov.in",
        "https://msme1connect.up.gov.in/scheme/up-schemem",
        "https://msme1connect.up.gov.in/about",
    ],
    "msme connect": [
        "https://msme1connect.up.gov.in",
        "https://msme1connect.up.gov.in/scheme/up-schemem",
    ],
    "scheme": ["https://msme1connect.up.gov.in/scheme/up-schemem"],
    "yojna": ["https://msme1connect.up.gov.in/scheme/up-schemem", "https://cmyuva.org.in/"],
    "yojana": ["https://msme1connect.up.gov.in/scheme/up-schemem", "https://cmyuva.org.in/"],
    "loan": ["https://cmyuva.org.in/", "https://msme1connect.up.gov.in/scheme/up-schemem"],
    "subsidy": ["https://cmyuva.org.in/", "https://msme1connect.up.gov.in/scheme/up-schemem"],
    "msme": ["https://msme.up.gov.in/", "https://msmeosem.in/"],
}

SUPPORTED_TOPIC_TERMS = tuple(TOPIC_URL_MAP)
SUPPORTED_GENERAL_TERMS = (
    "msme", "enterprise", "entrepreneur", "business", "startup", "skill",
    "training", "loan", "subsidy", "scheme", "yojana", "yojna", "udyam",
    "odop", "services", "service", "program", "programme", "franchise",
    "helpdesk", "mentorship", "workshop", "innovation", "center", "centre",
    "registration", "register", "eligibility", "apply", "application", "benefit", "contact",
    "market", "marketing", "purchase", "sell", "selling", "product", "products",
    "employment", "job", "jobs", "youth", "education", "finance", "funding",
    "tender", "certificate", "license", "licence", "support", "portal", "website",
)
SUPPORTED_GENERAL_TERMS_HI = (
    "एमएसएमई", "उद्यम", "उद्यमी", "व्यवसाय", "व्यापार", "स्टार्टअप", "कौशल", "प्रशिक्षण",
    "ऋण", "लोन", "सब्सिडी", "योजना", "पंजीकरण", "पात्रता", "आवेदन", "लाभ", "संपर्क",
    "बाज़ार", "बाजार", "विपणन", "उत्पाद", "रोजगार", "नौकरी", "युवा", "शिक्षा", "वित्त",
    "फंडिंग", "टेंडर", "प्रमाणपत्र", "लाइसेंस", "सहायता", "पोर्टल", "वेबसाइट", "फ्रेंचाइज़ी",
    "मार्गदर्शन", "सेवा", "सेवाएं",
)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HINGLISH_WORDS = {
    "kya", "hai", "kaise", "kyu", "kyun", "mera", "meri", "mere", "aap", "apka", "apke",
    "apki", "nahi", "haan", "krna", "karna", "batao", "bata", "chahiye", "kitna", "kitne",
    "paisa", "paise", "yojana", "yojna", "madad", "kripya", "acha", "theek", "mujhe",
    "humein", "hume", "tumhara", "kaha", "kahan", "milega", "milegi", "hoga", "hogi",
    "chal", "raha", "rahi", "abhi", "sakta", "sakti", "kar", "sakte",
}


def _looks_like_hindi_or_hinglish(question):
    if _DEVANAGARI_RE.search(question):
        return True
    words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    return len(words & _HINGLISH_WORDS) >= 2


def _effective_lang(question, requested_lang):
    requested_lang = requested_lang if requested_lang in ("en", "hi") else "en"
    if requested_lang == "hi" or _looks_like_hindi_or_hinglish(question):
        return "hi"
    return "en"


def _is_supported_question(question):
    question_lower = question.lower()
    if any(
        term in question_lower
        for term in SUPPORTED_TOPIC_TERMS + SUPPORTED_GENERAL_TERMS + SUPPORTED_GENERAL_TERMS_HI
    ):
        return True
    return any(
        urlparse(url).netloc.lower().removeprefix("www.") in question_lower
        for url in config.SITE_URLS
    )


def _normalize_question(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _intent_answer(question):
    normalized_question = _normalize_question(question)
    if not normalized_question:
        return None
    best_match = None
    best_score = 0.0
    generic_words = {"what", "is", "are", "the", "a", "an", "about", "tell", "me", "how", "can", "i", "do", "you", "who", "for", "of", "to"}
    question_words = set(normalized_question.split())
    for intent in INTENTS:
        for pattern in intent.get("patterns", []):
            normalized_pattern = _normalize_question(pattern)
            if not normalized_pattern:
                continue
            pattern_words = set(normalized_pattern.split())
            meaningful_overlap = (question_words - generic_words) & (pattern_words - generic_words)
            if not meaningful_overlap and normalized_question != normalized_pattern:
                continue
            overlap = len(question_words & pattern_words) / max(len(pattern_words), 1)
            score = SequenceMatcher(None, normalized_question, normalized_pattern).ratio()
            if normalized_question == normalized_pattern:
                score = 1.0
            elif normalized_pattern in normalized_question or normalized_question in normalized_pattern:
                score = max(score, 0.86)
            score = max(score, overlap * 0.82)
            if score > best_score:
                best_score = score
                best_match = intent
    if best_match is None or best_score < 0.72:
        return None
    return best_match["responses"][0]


def _youth_adda_services_answer(question, effective_lang="en"):
    question_lower = question.lower()
    is_youth_adda = "youth adda" in question_lower or "cmyouthadda" in question_lower
    asks_services = groq_client.is_services_request(question)
    if not (is_youth_adda and asks_services):
        return None
    if effective_lang == "hi":
        services = [
            "कौशल विकास",
            "ऋण एवं योजना सुविधा",
            "फ्रेंचाइज़ी और स्टार्टअप",
            "एमएसएमई विकास एवं हेल्पडेस्क",
            "डिजिटल सक्षमता",
            "मार्गदर्शन एवं नेटवर्किंग",
            "उद्यमिता कार्यशालाएं",
            "वास्तविक व्यवसाय चुनौतियां",
            "आइडिया फेयर एवं नवाचार",
        ]
    else:
        services = [
            "Skill Development",
            "Loan & Scheme Access",
            "Franchise & Startup",
            "MSME Growth & Helpdesk",
            "Digital Enablement",
            "Mentorship & Networking",
            "Entrepreneurship Workshops",
            "Real Business Challenges",
            "Idea Fairs & Innovation",
        ]
    items = "".join(f"<li><strong>{service}</strong></li>" for service in services)
    return f"<ul>{items}</ul>"


def _quick_definition(question, effective_lang="en"):
    normalized = re.sub(r"[^a-z0-9 ]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    is_hindi_ask = bool(re.search(r"(kya hai|क्या है|के बारे में बताओ|के बारे में बताएं)", question.lower()))
    if not re.search(r"^(what is|what are|tell me about|explain)\b", normalized) and not is_hindi_ask:
        return None

    definitions_en = {
        "osem": (
            "MSME OSEM (One State Enterprise Model) is a digital platform for Uttar Pradesh businesses. "
            "It connects entrepreneurs with business registration, listings, finance and support, marketplace, "
            "experts, training, events, and growth resources in one place."
        ),
        "msme": (
            "MSME means Micro, Small and Medium Enterprise. These businesses support jobs, innovation, "
            "manufacturing, services, and local economic growth. UP MSME portals provide information about "
            "registration, schemes, finance, training, markets, and business support."
        ),
        "upicon": (
            "UPICON is a Uttar Pradesh organization that supports skill development, training, consultancy, "
            "CSR programs, and entrepreneurship. Its portal provides information about courses, projects, "
            "business support, and related opportunities for learners and enterprises."
        ),
    }
    definitions_hi = {
        "osem": (
            "एमएसएमई ओसेम (One State Enterprise Model) उत्तर प्रदेश के व्यवसायों के लिए एक डिजिटल प्लेटफ़ॉर्म है। "
            "यह उद्यमियों को व्यवसाय पंजीकरण, लिस्टिंग, वित्त और सहायता, मार्केटप्लेस, विशेषज्ञों, प्रशिक्षण, कार्यक्रमों "
            "और विकास संसाधनों से एक ही स्थान पर जोड़ता है।"
        ),
        "msme": (
            "एमएसएमई का अर्थ है सूक्ष्म, लघु एवं मध्यम उद्यम (Micro, Small and Medium Enterprise)। ये व्यवसाय रोजगार, "
            "नवाचार, विनिर्माण, सेवाओं और स्थानीय आर्थिक विकास में सहायक होते हैं। यूपी एमएसएमई पोर्टल पंजीकरण, योजनाओं, "
            "वित्त, प्रशिक्षण, बाज़ार और व्यवसाय सहायता से जुड़ी जानकारी उपलब्ध कराते हैं।"
        ),
        "upicon": (
            "यूपीआईकॉन (UPICON) उत्तर प्रदेश की एक संस्था है जो कौशल विकास, प्रशिक्षण, परामर्श, सीएसआर कार्यक्रमों और "
            "उद्यमिता को बढ़ावा देती है। इसका पोर्टल शिक्षार्थियों और उद्यमों के लिए कोर्स, परियोजनाओं, व्यवसाय सहायता और "
            "संबंधित अवसरों की जानकारी देता है।"
        ),
    }
    matched = next((name for name in definitions_en if re.search(rf"\b{name}\b", normalized)), None)
    if not matched:
        return None
    return definitions_hi[matched] if effective_lang == "hi" else definitions_en[matched]


def _match_topic_urls(question):
    question_lower = question.lower()
    matched = []
    for keyword, urls in TOPIC_URL_MAP.items():
        if keyword in question_lower:
            for url in urls:
                if url not in matched:
                    matched.append(url)
    for url in config.SITE_URLS:
        hostname = urlparse(url).netloc.lower().removeprefix("www.")
        if hostname and hostname in question_lower and url not in matched:
            matched.append(url)
    if "msme" in question_lower or "osem" in question_lower:
        priority_urls = ["https://msmeosem.in/", "https://msme.up.gov.in/"]
        matched = priority_urls + [url for url in matched if url not in priority_urls]
    return matched


def _score_chunk(chunk_text_value, question_words):
    chunk_words = set(re.findall(r"[a-zA-Z0-9]+", chunk_text_value.lower()))
    return len(chunk_words & question_words)


def _crawl_urls_for_question(urls, question, max_chunks=6):
    question_words = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    scored_chunks = []

    if not urls:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(urls))) as executor:
        future_to_url = {executor.submit(crawler.fetch_url_safe, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                html = future.result()
            except Exception:
                html = ""
            text = crawler.extract_text(html)
            if not text:
                continue
            for chunk in crawler.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                score = _score_chunk(chunk, question_words)
                scored_chunks.append((score, {"url": url, "text": chunk}))

    if not scored_chunks:
        return []
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored_chunks[:max_chunks]]


def _cached_chunks_for_question(question, urls=None, max_chunks=6):
    cached_chunks = crawler.load_chunk_cache(config.CACHE_PATH)
    if urls:
        cached_chunks = [item for item in cached_chunks if item.get("url") in urls]
    if not cached_chunks:
        return []
    question_words = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    scored = [
        (_score_chunk(item.get("text", ""), question_words), item)
        for item in cached_chunks
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:max_chunks]]


def _broad_live_fallback(question):
    return _crawl_urls_for_question(config.SITE_URLS, question)


def answer_query(question, lang="en"):
    effective_lang = _effective_lang(question, lang)

    services_answer = _youth_adda_services_answer(question, effective_lang)
    if services_answer:
        return services_answer

    intent_answer = _intent_answer(question)
    if intent_answer:
        if effective_lang == "hi":
            return groq_client.translate_to_hindi(intent_answer)
        return intent_answer

    if not _is_supported_question(question):
        if effective_lang == "hi":
            return (
                "<strong>क्षमा करें, मैं केवल यूपी एमएसएमई सेवाओं, योजनाओं, प्रशिक्षण, ऋण और "
                "व्यवसाय सहायता से जुड़े सवालों के जवाब दे सकता/सकती हूँ।</strong>"
            )
        return "<strong>Sorry, I can only answer questions about UP MSME services, schemes, training, loans, and business support.</strong>"

    quick_answer = _quick_definition(question, effective_lang)
    if quick_answer:
        return f"<strong>{quick_answer}</strong>"

    store = get_store()
    context_chunks = []
    services_request = groq_client.is_services_request(question)
    topic_urls = _match_topic_urls(question)

    # Prefer cached chunks and the local index. Live crawling is a fallback only.
    context_chunks = _cached_chunks_for_question(question, topic_urls or None)
    if store.index is not None:
        if not context_chunks:
            query_embedding = embedder.embed_query(question)
            result_limit = config.TOP_K if services_request else min(config.TOP_K, 2)
            results = store.search(query_embedding, result_limit)
            strong_results = [item for score, item in results if score >= config.SIMILARITY_THRESHOLD]
            if strong_results:
                context_chunks = strong_results
            elif results:
                context_chunks = [item for _, item in results]

    if not context_chunks and topic_urls:
        context_chunks = _crawl_urls_for_question(topic_urls, question)

    if not context_chunks:
        context_chunks = _broad_live_fallback(question)

    if not context_chunks:
        context_chunks = [{"url": "", "text": "No relevant content was found."}]

    answer, tokens_used = groq_client.generate_answer(question, context_chunks, lang=effective_lang)
    token_logger.log_question(question, tokens_used)
    return answer
