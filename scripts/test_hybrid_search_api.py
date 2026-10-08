#!/usr/bin/env python3
"""Run the live Week 4 hybrid retrieval demo (not part of pytest)."""

import argparse
import sys

import httpx

QUERIES = (
    "AI systems helping underserved patients get medical care",
    "bias and fairness in clinical decision making",
    "robots learning representations of the physical world",
    "technology adoption barriers for low income college students",
    "smart buildings using cloud and edge computing",
    "helping disadvantaged patients obtain healthcare",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/hybrid-search"

    with httpx.Client(timeout=args.timeout) as client:
        for query_number, query in enumerate(QUERIES, 1):
            try:
                response = client.post(endpoint, json={"query": query, "size": 5})
                response.raise_for_status()
            except (httpx.HTTPError, ValueError) as exc:
                print(f"Query {query_number} failed: {exc}", file=sys.stderr)
                return 1
            payload = response.json()
            print(f"\n{query_number}. {query}")
            print(f"retrieval_mode={payload['retrieval_mode']} count={payload['count']}")
            for rank, result in enumerate(payload["results"], 1):
                preview = " ".join(result["chunk_text"].split())[:160]
                print(
                    f"  {rank}. rrf={result['rrf_score']:.9f} "
                    f"bm25={result['bm25_rank']} vector={result['vector_rank']} "
                    f"arxiv={result['arxiv_id']} chunk={result['chunk_id']}\n"
                    f"     {result['paper_title']}\n"
                    f"     {preview}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
