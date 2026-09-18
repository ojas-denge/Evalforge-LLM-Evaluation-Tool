from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        chunks.append(
            DocumentChunk(
                chunk_id=f"{document_id}:{chunk_index}",
                document_id=document_id,
                text=chunk,
                metadata={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                },
            )
        )

        if end == len(words):
            break

        start = end - chunk_overlap
        chunk_index += 1

    return chunks
