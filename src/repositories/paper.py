from uuid import UUID

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from src.models.paper import Paper
from src.schemas.paper import PaperCreate, PaperUpsert


class PaperRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, paper: PaperCreate) -> Paper:
        db_paper = Paper(**paper.model_dump())
        self.session.add(db_paper)
        self.session.commit()
        self.session.refresh(db_paper)
        return db_paper

    def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        return self.session.query(Paper).filter(Paper.arxiv_id == arxiv_id).first()

    def get_by_id(self, paper_id: UUID) -> Paper | None:
        return self.session.query(Paper).filter(Paper.id == paper_id).first()

    def get_all(self, limit: int = 100, offset: int = 0) -> list[Paper]:
        return self.session.query(Paper).order_by(Paper.published_date.desc()).limit(limit).offset(offset).all()

    def update(self, paper: Paper) -> Paper:
        self.session.add(paper)
        self.session.commit()
        self.session.refresh(paper)
        return paper

    def upsert(self, paper_data: PaperCreate | PaperUpsert) -> Paper:
        """Atomically insert or update a paper using its unique arXiv ID."""
        values = paper_data.model_dump(exclude_unset=True)
        statement = insert(Paper).values(**values)
        update_values = {
            field: getattr(statement.excluded, field) for field in values if field not in {"id", "arxiv_id", "created_at"}
        }
        update_values["updated_at"] = func.now()
        statement = statement.on_conflict_do_update(
            index_elements=[Paper.arxiv_id],
            set_=update_values,
        ).returning(Paper.id)

        paper_id = self.session.execute(statement).scalar_one()
        self.session.commit()
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise RuntimeError(f"Upserted paper {paper_id} could not be loaded")
        return paper
