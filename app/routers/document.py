from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, get_db

from fastapi import HTTPException, status, BackgroundTasks, Form, UploadFile, File, Depends, APIRouter

from typing import Annotated

from app.services.pdf import extract_document, save_pdf

import os

from app.services.chunker import chunk_document

from app.services.document import save_document_with_chunks, get_document_chunks, get_document_by_id, get_all_documents, delete_document, save_document, mark_failed

from app.ai.embedding import create_embeddings

from app.schemas.document import DocumentUploadResponse, DocumentResponse, DocumentDetailResponse, DocumentDeleteResponse


router = APIRouter(prefix="/documents", tags=["Document"])


DIR = "uploads"


DBSession = Annotated[AsyncSession, Depends(get_db)]


FileSession = Annotated[UploadFile, File(...)]


FormSession = Annotated[str, Form(...)]



async def process_backgroundtask(
    file_path: str,
    document_id: int
) -> None:
    try:
        document = extract_document(file_path=file_path)

        chunks = chunk_document(document=document)

        if not chunks:
            async with AsyncSessionLocal() as db:
                await mark_failed(db=db, document_id=document_id, error_message="No extractable text found (scanned without OCR)")

            return

        chunks_text = [chunk["text"] for chunk in chunks]

        embeddings = create_embeddings(chunks_text)

        async with AsyncSessionLocal() as db:
            await save_document_with_chunks(
                db=db,
                document_id=document_id,
                embeddings=embeddings,
                chunks=chunks,
            )

    except Exception as exc:
        async with AsyncSessionLocal() as db:
            await mark_failed(
                db=db,
                document_id=document_id,
                error_message=str(exc),
            )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    db: DBSession,
    background_tasks: BackgroundTasks,
    file: FileSession,
    category: FormSession,
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a name."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a pdf."
        )

    file_path = save_pdf(file=file, upload_dir=DIR)

    document = await save_document(
        filename=file.filename,
        category=category,
        db=db,
    )

    background_tasks.add_task(
        process_backgroundtask,
        file_path=file_path,
        document_id=document.id,
    )

    return DocumentUploadResponse(
        message="Document was uploaded and is being processed in background.",
        id=document.id,
        status=document.status,
        filename=file.filename,
        category=category
    )



@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: DBSession) -> list[DocumentResponse]:
    documents = await get_all_documents(db=db)

    return [DocumentResponse.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(db: DBSession, document_id: int) -> DocumentDetailResponse:
    document = await get_document_by_id(db=db, document_id=document_id)
    
    chunks = await get_document_chunks(db=db, document_id=document_id)

    return DocumentDetailResponse(
        id=document.id,
        filename=document.filename,
        category=document.category,
        status=document.status,
        error_message=document.error_message,
        created_at=document.created_at,
        total_chunks=len(chunks),
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def remove_document(db: DBSession, document_id: int) -> DocumentDeleteResponse:
    result = await delete_document(db=db, document_id=document_id)
    
    return DocumentDeleteResponse(**result)