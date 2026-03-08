"""Split extracted pages into overlapping chunks with metadata."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: list[dict], doc_type: str = "book") -> list[dict]:
    """Split a list of page dicts into smaller chunks.

    Each returned dict has keys: text, metadata.
    metadata contains: source, type, page, chunk_index.
    """
    chunks = []
    chunk_index = 0
    for page in pages:
        splits = _splitter.split_text(page["text"])
        for split_text in splits:
            chunks.append({
                "text": split_text,
                "metadata": {
                    "source": page["source"],
                    "type": doc_type,
                    "page": page["page"],
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1
    return chunks
