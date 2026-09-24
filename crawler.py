import concurrent.futures
import logging
import os
import pickle
import time

import requests
import urllib3
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

MIN_TEXT_LENGTH = 200

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def _fetch_static(url, timeout=6):
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, verify=True)
        response.raise_for_status()
        return response.text
    except requests.exceptions.SSLError:
        # Only fall back to an unverified request if the failure was specifically
        # a certificate problem (common for some misconfigured .gov.in sites),
        # never as the default behavior.
        logger.warning("SSL verification failed for %s; retrying without verification", url)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return ""
    except requests.exceptions.RequestException:
        return ""


def _fetch_rendered(url, timeout=8):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ""
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            time.sleep(1.0)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return ""


def fetch_url(url, timeout=6):
    html = _fetch_static(url, timeout)
    text_preview = extract_text(html)
    if len(text_preview) < MIN_TEXT_LENGTH:
        rendered_html = _fetch_rendered(url, timeout=8)
        if rendered_html and len(extract_text(rendered_html)) > len(text_preview):
            return rendered_html
    return html


def fetch_url_safe(url, overall_timeout=12):
    future = _executor.submit(fetch_url, url)
    try:
        return future.result(timeout=overall_timeout)
    except concurrent.futures.TimeoutError:
        return ""
    except Exception:
        return ""


def extract_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    cleaned = " ".join(text.split())
    return cleaned


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    if not text:
        return chunks
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks


def build_chunk_cache(urls, cache_path, max_workers=6):
    """Fetch all configured sites, chunk their text, and persist source-aware chunks."""
    cached_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_url_safe, url, 20): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = extract_text(future.result())
            except Exception:
                text = ""
            for chunk in chunk_text(text):
                cached_chunks.append({"url": url, "text": chunk})
    if cached_chunks:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as cache_file:
            pickle.dump(cached_chunks, cache_file)
    return cached_chunks


def load_chunk_cache(cache_path):
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "rb") as cache_file:
            chunks = pickle.load(cache_file)
        return chunks if isinstance(chunks, list) else []
    except (OSError, pickle.PickleError, EOFError):
        return []
