from typing import Any

from fastapi import HTTPException, status

from sqlalchemy import select, insert, update

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

from app.models.chunks import Chunks

from app.schemas.retriever import RetrievedChunks


async def save_document(
    db: AsyncSession,
    filename: str,
    category: str,
) -> Document:
    document = Document(
        filename=filename,
        category=category,
        status="processing",
    )

    db.add(document)

    await db.commit()

    return document

async def save_document_with_chunks(
    db: AsyncSession,
    document_id: int,
    chunks: list[RetrievedChunks],
    embeddings: Any,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch.")

    chunk_records = [
        {
            "document_id": document_id,
            "chunk_index": idx,
            "page_no": chunk.get("page_no"),
            "headings": chunk.get("headings"),
            "text": chunk["text"],
            "embeddings": embedding,
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1)
    ]

    await db.execute(insert(Chunks).values(chunk_records))

    await db.execute(
        update(Document).where(
            Document.id == document_id,
        ).values(status="ready", error_message=None)
    )

    await db.commit()


async def mark_failed(
    db: AsyncSession,
    document_id: int,
    error_message: str,
) -> None:
    await db.execute(
        update(Document).where(
            Document.id == document_id
        ).values(status="failed", error_message=error_message)
    )

    await db.commit()


async def get_document_by_id(db: AsyncSession, document_id: int) -> Document:
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} was not found.",
        )
    return document


async def get_all_documents(db: AsyncSession) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_document_chunks(
    db: AsyncSession, 
    document_id: int
) -> list[Chunks]:
    
    chunks = (
        select(Chunks)
        .where(Chunks.document_id == document_id)
        .order_by(Chunks.chunk_index.asc())
    )

    result = await db.execute(chunks)

    return list(result.scalars().all())


async def delete_document(db: AsyncSession, document_id: int) -> dict[str, Any]:
    document = await get_document_by_id(db, document_id)

    await db.delete(document)

    await db.commit()

    return {
        "message": "Document and associated chunks successfully deleted.",
        "document_id": document_id,
    }