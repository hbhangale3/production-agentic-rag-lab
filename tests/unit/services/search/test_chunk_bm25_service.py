from unittest.mock import Mock

import pytest
from src.exceptions import ChunkBM25SearchError, ChunkIndexError
from src.services.search.chunk_bm25_service import ChunkBM25SearchService


def test_searches_only_chunk_index_and_normalizes_hits() -> None:
    index = Mock()
    index.search_chunks.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_score": 8.2,
                "_source": {
                    "chunk_id": "p::chunk::000", "arxiv_id": "p", "chunk_index": 0,
                    "chunk_text": "text", "word_count": 1, "embedding": [1.0],
                },
            }],
        }
    }
    service = ChunkBM25SearchService(chunk_index=index)

    result = service.search("query", size=5)

    assert result.hits[0].score == 8.2
    assert "embedding" not in result.hits[0].model_dump()
    body = index.search_chunks.call_args.args[0]
    assert body["size"] == 5
    assert "arxiv-papers" not in str(index.method_calls)
    assert not any(call[0] in {"bulk_index_documents", "delete_paper_chunks"} for call in index.method_calls)


def test_backend_failure_is_wrapped() -> None:
    index = Mock()
    index.ensure_compatible_index.side_effect = ChunkIndexError("missing")
    with pytest.raises(ChunkBM25SearchError, match="missing"):
        ChunkBM25SearchService(chunk_index=index).search("query")
