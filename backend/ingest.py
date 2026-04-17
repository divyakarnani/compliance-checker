"""Ingest EU regulation texts into ChromaDB for RAG retrieval."""

import os
import re
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

REGULATIONS_DIR = Path(__file__).resolve().parent / "regulations"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "eu_regulations"

client = OpenAI()


MAX_CHUNK_CHARS = 6000  # Keep chunks under ~1500 tokens for good embedding quality


def parse_articles(text: str, regulation_name: str) -> list[dict]:
    """Split regulation text into article-level chunks."""
    # Match "Article N" at the start of a line (avoids Treaty refs in preamble)
    pattern = r"(?m)^(Article\s+\d+[a-z]?)\s*$"
    matches = list(re.finditer(pattern, text))

    articles = []
    seen_numbers = set()

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        article_header = match.group(1).strip()
        article_body = text[match.end():end].strip()

        num_match = re.search(r"(\d+[a-z]?)", article_header)
        article_number = num_match.group(1) if num_match else str(idx + 1)

        # Deduplicate — if we've seen this article number, append a suffix
        base_number = article_number
        counter = 2
        while article_number in seen_numbers:
            article_number = f"{base_number}_{counter}"
            counter += 1
        seen_numbers.add(article_number)

        # Extract title (first non-empty line of body, if short enough)
        lines = [l.strip() for l in article_body.split("\n") if l.strip()]
        title = lines[0] if lines and len(lines[0]) < 120 else ""

        full_text = f"{article_header}\n{article_body}"

        # Skip very short chunks
        if len(full_text) < 50:
            continue

        # If chunk is too large, split into sub-chunks
        if len(full_text) > MAX_CHUNK_CHARS:
            sub_chunks = _split_large_article(full_text, MAX_CHUNK_CHARS)
            for i, chunk in enumerate(sub_chunks):
                sub_num = f"{base_number}_part{i+1}" if len(sub_chunks) > 1 else base_number
                articles.append({
                    "regulation_name": regulation_name,
                    "article_number": sub_num,
                    "article_title": title,
                    "full_text": chunk,
                })
        else:
            articles.append({
                "regulation_name": regulation_name,
                "article_number": article_number,
                "article_title": title,
                "full_text": full_text,
            })

    # If no articles found, chunk the whole text
    if not articles and len(text.strip()) > 100:
        for i, chunk in enumerate(_split_large_article(text.strip(), MAX_CHUNK_CHARS)):
            articles.append({
                "regulation_name": regulation_name,
                "article_number": f"chunk_{i+1}",
                "article_title": regulation_name,
                "full_text": chunk,
            })

    return articles


def _split_large_article(text: str, max_chars: int) -> list[str]:
    """Split a large text block into chunks at paragraph/line boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Try splitting on double newlines first, then single newlines
    for separator in ["\n\n", "\n"]:
        paragraphs = text.split(separator)
        if len(paragraphs) <= 1:
            continue

        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chars and current:
                chunks.append(current.strip())
                current = para
            else:
                current = current + separator + para if current else para

        if current.strip():
            chunks.append(current.strip())

        # Check if all chunks are within limit
        if all(len(c) <= max_chars * 1.2 for c in chunks):
            return chunks

    # Last resort: hard split
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts using OpenAI, batching to stay within API limits."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])
        if i + batch_size < len(texts):
            print(f"  Embedded {i + len(batch)}/{len(texts)}...")
    return all_embeddings


def ensure_ingested():
    """Ingest regulations into ChromaDB if the collection is empty."""
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"ChromaDB already has {collection.count()} chunks. Skipping ingestion.")
        return collection

    print("Ingesting regulations into ChromaDB...")

    regulation_files = {
        "ecgt.txt": "EU Green Claims Directive (ECGT) - Directive 2024/825",
        "ucpd.txt": "Unfair Commercial Practices Directive (UCPD) - Directive 2005/29/EC",
        "espr.txt": "Ecodesign for Sustainable Products Regulation (ESPR) - Regulation 2024/1781",
    }

    all_articles = []
    for filename, reg_name in regulation_files.items():
        filepath = REGULATIONS_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.")
            continue
        text = filepath.read_text(encoding="utf-8")
        articles = parse_articles(text, reg_name)
        print(f"  {reg_name}: {len(articles)} articles parsed")
        all_articles.extend(articles)

    if not all_articles:
        print("No articles found. Check regulation files.")
        return collection

    # Batch embed (OpenAI limit ~2048 per call, we'll be well under)
    texts = [a["full_text"] for a in all_articles]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # Add to ChromaDB
    collection.add(
        ids=[f"{a['regulation_name']}_art_{a['article_number']}" for a in all_articles],
        documents=texts,
        metadatas=[{
            "regulation_name": a["regulation_name"],
            "article_number": a["article_number"],
            "article_title": a["article_title"],
        } for a in all_articles],
        embeddings=embeddings,
    )

    print(f"Ingested {collection.count()} chunks into ChromaDB.")
    return collection


if __name__ == "__main__":
    ensure_ingested()
