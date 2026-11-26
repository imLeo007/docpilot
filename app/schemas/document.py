from datetime import datetime
from pydantic import BaseModel, ConfigDict


class DocumentUploadResponse(BaseModel):
    id: int
    status: str
    message: str
    filename: str
    category: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    category: str
    status: str
    error_message: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: int