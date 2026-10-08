"""Central OpenSearch configuration for paper-level retrieval."""

ARXIV_PAPERS_INDEX = "arxiv-papers"

ARXIV_PAPERS_MAPPING = {
    "mappings": {
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "updated_date": {"type": "date"},
            "pdf_url": {"type": "keyword", "index": False},
            "local_pdf_path": {"type": "keyword", "index": False},
            "parser_used": {"type": "keyword"},
            "page_count": {"type": "integer"},
            "raw_text": {"type": "text"},
        }
    }
}


CHUNK_FIELD_TYPES = {
    "chunk_id": "keyword",
    "arxiv_id": "keyword",
    "chunk_index": "integer",
    "section_title": "text",
    "chunk_text": "text",
    "word_count": "integer",
    "has_overlap": "boolean",
    "paper_title": "text",
    "authors": "text",
    "categories": "keyword",
    "published_date": "date",
    "embedding": "knn_vector",
}


def build_arxiv_chunks_mapping(embedding_dimension: int) -> dict:
    """Build the chunk-level BM25 and vector index definition."""
    if embedding_dimension < 1:
        raise ValueError("embedding_dimension must be greater than zero")
    keyword_subfield = {"keyword": {"type": "keyword", "ignore_above": 256}}
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "arxiv_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "section_title": {"type": "text", "fields": keyword_subfield},
                "chunk_text": {"type": "text"},
                "word_count": {"type": "integer"},
                "has_overlap": {"type": "boolean"},
                "paper_title": {"type": "text", "fields": keyword_subfield},
                "authors": {"type": "text", "fields": keyword_subfield},
                "categories": {"type": "keyword"},
                "published_date": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
