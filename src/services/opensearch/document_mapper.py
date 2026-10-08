from typing import Any

from src.models.paper import Paper
from src.schemas.paper import PaperBase, PaperUpsert


def paper_to_document(paper: Paper | PaperBase) -> dict[str, Any]:
    """Map a relational/application paper into the OpenSearch document contract."""
    validated = PaperUpsert.model_validate(paper, from_attributes=True)
    return validated.model_dump(mode="json")
