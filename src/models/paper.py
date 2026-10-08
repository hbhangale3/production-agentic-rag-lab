import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from src.db.interfaces.postgresql import Base


class Paper(Base):
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    arxiv_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(String, nullable=False)
    authors = Column(JSON, nullable=False)  # List of author names
    abstract = Column(Text, nullable=False)
    categories = Column(JSON, nullable=False)  # List of category tags
    published_date = Column(DateTime, nullable=False)
    updated_date = Column(DateTime, nullable=True)
    pdf_url = Column(String, nullable=False)
    local_pdf_path = Column(String, nullable=True)
    parser_used = Column(String, nullable=True)
    page_count = Column(Integer, nullable=True)
    raw_text = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
