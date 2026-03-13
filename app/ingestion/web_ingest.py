"""CLI: scrape web pages → chunk → embed → upsert to Pinecone.

Usage (single pages):
    python -m app.ingestion.web_ingest URL1 [URL2 ...]

Usage (crawl entire site — follows HTML links, good for simple sites):
    python -m app.ingestion.web_ingest --crawl URL1 [URL2 ...]
    python -m app.ingestion.web_ingest --crawl --max-pages 200 URL1

Usage (sitemap mode — reads sitemap.xml to discover all URLs, good for JS-heavy sites):
    python -m app.ingestion.web_ingest --sitemap URL1 [URL2 ...]
    python -m app.ingestion.web_ingest --sitemap --path-filter /recipes/ URL1

Examples:
    # Crawl a blog
    python -m app.ingestion.web_ingest --crawl https://www.theperfectloaf.com/

    # Ingest all recipes from King Arthur (JS-rendered site, sitemap required)
    python -m app.ingestion.web_ingest --sitemap --path-filter /recipes/ https://www.kingarthurbaking.com
"""

import argparse
import os
import re
import sys
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from app.ingestion.chunker import chunk_pages

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BATCH_SIZE = 100
EMBED_BATCH_SIZE = 50

# Paths containing these strings are skipped during crawling
_SKIP_PATH_PATTERNS = [
    "/tag/", "/category/", "/author/", "/wp-", "/feed",
    "/page/", "/cart", "/shop", "/account", "/login", "/register",
    "/search", "/404", "/sitemap",
]
_SKIP_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".xml", ".css", ".js"}


# ---------------------------------------------------------------------------
# HTML → clean text
# ---------------------------------------------------------------------------

def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping nav/footer/script noise."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"class": re.compile(r"post|article|content|entry", re.I)})
        or soup.body
        or soup
    )

    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_links(html: str, base_url: str) -> list[str]:
    """Extract same-domain content links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)

        if parsed.netloc != base_domain:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        # Drop query strings and fragments — they usually point to duplicates
        if parsed.query or parsed.fragment:
            continue

        path = parsed.path.lower()
        if any(p in path for p in _SKIP_PATH_PATTERNS):
            continue
        ext = "." + path.rsplit(".", 1)[-1] if "." in path.split("/")[-1] else ""
        if ext in _SKIP_EXTENSIONS:
            continue
        # Must have a meaningful path (not just the root or a one-char slug)
        if len(path.strip("/")) < 3:
            continue

        links.add(parsed._replace(fragment="", query="").geturl())

    return list(links)


def _friendly_source(url: str) -> str:
    """Turn a URL into a short, readable source name."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    if path:
        path = path.replace("-", " ").replace("_", " ")
        path = re.sub(r"\.(html?|php|aspx?)$", "", path)
    return f"{domain}: {path}" if path else domain


# ---------------------------------------------------------------------------
# Embed + upsert helpers
# ---------------------------------------------------------------------------

def _embed_and_upsert(pages: list[dict], url: str, index, client: OpenAI, model: str):
    """Chunk, embed, and upsert a list of page dicts."""
    chunks = chunk_pages(pages, doc_type="blog")
    print(f"  Split into {len(chunks)} chunks")

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        try:
            resp = client.embeddings.create(input=texts, model=model)
            embeddings = [item.embedding for item in resp.data]
        except Exception as e:
            print(f"  ERROR embedding batch {i}: {e}")
            continue

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            raw_id = f"{chunk['metadata']['source']}::{chunk['metadata']['chunk_index']}"
            vec_id = hashlib.md5(raw_id.encode()).hexdigest()
            metadata = {**chunk["metadata"], "text": chunk["text"], "url": url}
            vectors.append({"id": vec_id, "values": embedding, "metadata": metadata})

        for j in range(0, len(vectors), BATCH_SIZE):
            index.upsert(vectors=vectors[j: j + BATCH_SIZE])

        print(f"  Upserted {len(vectors)} vectors")


# ---------------------------------------------------------------------------
# Single-URL ingest
# ---------------------------------------------------------------------------

def ingest_url(url: str, index, client: OpenAI, model: str):
    """Fetch, chunk, embed, and upsert a single URL."""
    print(f"  Fetching {url} ...")
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "SourdoughGuru/1.0 (knowledge-base ingestion)"})
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"  ERROR: HTTP {e.response.status_code}")
        return
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    text = _extract_text_from_html(resp.text)
    if not text or len(text) < 50:
        print(f"  Skipped (too little text: {len(text)} chars)")
        return

    source = _friendly_source(str(resp.url))
    print(f"  Extracted {len(text)} chars from {source}")
    _embed_and_upsert([{"text": text, "page": 1, "source": source}], url, index, client, model)


# ---------------------------------------------------------------------------
# Site crawler
# ---------------------------------------------------------------------------

def crawl_and_ingest(start_url: str, index, client: OpenAI, model: str, max_pages: int = 150):
    """BFS crawl a site starting from start_url and ingest every content page."""
    visited: set[str] = set()
    queue: list[str] = [start_url]
    ingested = 0

    print(f"  Crawling up to {max_pages} pages from {start_url} ...")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = httpx.get(url, follow_redirects=True, timeout=30,
                             headers={"User-Agent": "SourdoughGuru/1.0 (knowledge-base ingestion)"})
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"  [{len(visited)}] SKIP {url} — HTTP {e.response.status_code}")
            continue
        except Exception as e:
            print(f"  [{len(visited)}] SKIP {url} — {e}")
            continue

        final_url = str(resp.url)
        html = resp.text

        # Queue new same-domain links
        for link in _extract_links(html, final_url):
            if link not in visited:
                queue.append(link)

        # Extract and ingest content
        text = _extract_text_from_html(html)
        if not text or len(text) < 200:
            print(f"  [{len(visited)}] SKIP {url} (only {len(text)} chars)")
            continue

        source = _friendly_source(final_url)
        print(f"  [{len(visited)}] {source} — {len(text)} chars")
        _embed_and_upsert([{"text": text, "page": 1, "source": source}], final_url, index, client, model)
        ingested += 1

    print(f"\n  Crawl complete: visited {len(visited)} pages, ingested {ingested} with content.")


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

_SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def _parse_sitemap_xml(xml_text: str, path_filter: str | None) -> tuple[list[str], list[str]]:
    """Parse a sitemap XML string.

    Returns (content_urls, sub_sitemap_urls).
    content_urls are filtered by path_filter if provided.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  WARNING: Could not parse sitemap XML: {e}")
        return [], []

    ns = _SITEMAP_NS
    content_urls: list[str] = []
    sub_sitemaps: list[str] = []

    for child in root:
        tag = child.tag.replace(f"{{{ns}}}", "")
        loc_elem = child.find(f"{{{ns}}}loc")
        if loc_elem is None or not loc_elem.text:
            continue
        url = loc_elem.text.strip()

        if tag == "sitemap":
            sub_sitemaps.append(url)
        elif tag == "url":
            if path_filter is None or path_filter in url:
                content_urls.append(url)

    return content_urls, sub_sitemaps


def collect_sitemap_urls(
    base_url: str,
    path_filter: str | None = None,
    max_urls: int = 500,
) -> list[str]:
    """Discover all content URLs for a domain by reading its sitemap(s).

    Handles both sitemap index files (which link to sub-sitemaps) and
    regular sitemap files (which list URLs directly).

    Args:
        base_url:    Any URL on the target domain (only the scheme+host is used).
        path_filter: Optional substring — only URLs containing this are returned
                     (e.g. "/recipes/" to get only recipe pages).
        max_urls:    Cap on total URLs returned.
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    # Candidate sitemap locations (tried in order until one responds)
    candidates = [
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
        f"{base}/sitemap-index.xml",
    ]

    visited_sitemaps: set[str] = set()
    sitemap_queue: list[str] = candidates
    all_urls: list[str] = []

    headers = {"User-Agent": "SourdoughGuru/1.0 (knowledge-base ingestion)"}

    while sitemap_queue and len(all_urls) < max_urls:
        sitemap_url = sitemap_queue.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        try:
            resp = httpx.get(sitemap_url, follow_redirects=True, timeout=30, headers=headers)
            if resp.status_code != 200:
                continue
        except Exception as e:
            print(f"  Could not fetch sitemap {sitemap_url}: {e}")
            continue

        content_urls, sub_sitemaps = _parse_sitemap_xml(resp.text, path_filter)
        all_urls.extend(content_urls)
        sitemap_queue.extend(s for s in sub_sitemaps if s not in visited_sitemaps)

        filter_note = f" (filter: {path_filter})" if path_filter else ""
        print(f"  Sitemap {sitemap_url}: {len(content_urls)} URLs{filter_note}, "
              f"{len(sub_sitemaps)} sub-sitemaps")

    return all_urls[:max_urls]


def sitemap_ingest(
    base_url: str,
    index,
    client: OpenAI,
    model: str,
    path_filter: str | None = None,
    max_urls: int = 500,
):
    """Discover URLs via sitemap and ingest each one."""
    print(f"  Discovering URLs from sitemap of {base_url} ...")
    urls = collect_sitemap_urls(base_url, path_filter=path_filter, max_urls=max_urls)

    if not urls:
        print("  No URLs found in sitemap. Try --crawl instead.")
        return

    print(f"  Found {len(urls)} URLs to ingest.\n")
    for i, url in enumerate(urls, 1):
        print(f"\n  [{i}/{len(urls)}] {url}")
        ingest_url(url, index, client, model)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest web pages into the Sourdough Guru knowledge base."
    )
    parser.add_argument("urls", nargs="+", help="One or more URLs to ingest")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--crawl", action="store_true",
        help="Crawl the entire site following HTML links (good for simple/blog sites)"
    )
    mode_group.add_argument(
        "--sitemap", action="store_true",
        help="Discover URLs via sitemap.xml (good for JS-heavy sites like King Arthur)"
    )

    parser.add_argument(
        "--path-filter", type=str, default=None, metavar="PATH",
        help="With --sitemap: only ingest URLs whose path contains this string "
             "(e.g. /recipes/ to get only recipe pages)"
    )
    parser.add_argument(
        "--max-pages", type=int, default=150,
        help="Max pages per URL for --crawl, or max URLs for --sitemap (default: 150)"
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["LLMOD_API_KEY"],
        base_url=os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1"),
    )
    model = os.environ.get("LLMOD_EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "sourdough-knowledge")
    index = pc.Index(index_name)

    if args.sitemap:
        mode = f"sitemap (filter: {args.path_filter or 'none'}, max: {args.max_pages})"
    elif args.crawl:
        mode = f"crawl (max: {args.max_pages})"
    else:
        mode = "single-page"

    print(f"=== Web Ingestion → Pinecone ({index_name}) [{mode}] ===\n")

    for url in args.urls:
        print(f"\n--- {url}")
        if args.sitemap:
            sitemap_ingest(url, index, client, model,
                           path_filter=args.path_filter, max_urls=args.max_pages)
        elif args.crawl:
            crawl_and_ingest(url, index, client, model, max_pages=args.max_pages)
        else:
            ingest_url(url, index, client, model)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
