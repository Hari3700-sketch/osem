import config
import crawler
import embedder
from vectorstore import get_store


def build():
    all_metadata = crawler.build_chunk_cache(config.SITE_URLS, config.CACHE_PATH)
    all_chunks = [item["text"] for item in all_metadata]

    if not all_chunks:
        raise RuntimeError("No content could be crawled from the configured sites")

    embeddings = embedder.embed_texts(all_chunks)
    store = get_store()
    store.build(embeddings, all_metadata)
    store.save()
    print(f"Cached and indexed {len(all_chunks)} chunks from {len(config.SITE_URLS)} URLs")


if __name__ == "__main__":
    build()
