from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.chunking import DocumentChunk, chunk_text


logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(
    documents_path: str = "data/documents",
) -> list[DocumentChunk]:
    settings = get_settings()

    root = Path(documents_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {root}"
        )

    chunks: list[DocumentChunk] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        document_id = path.relative_to(root).as_posix()

        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            logger.warning(
                "Skipping empty document: %s",
                document_id,
            )
            continue

        document_chunks = chunk_text(
            text=text,
            document_id=document_id,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks.extend(document_chunks)

        logger.info(
            "Loaded %s → %d chunks",
            document_id,
            len(document_chunks),
        )

    logger.info(
        "Ingestion complete: %d chunks from %d documents",
        len(chunks),
        len({chunk.document_id for chunk in chunks}),
    )

    return chunks
