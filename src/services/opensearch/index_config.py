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
