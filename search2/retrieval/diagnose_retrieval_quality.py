#!/usr/bin/env python3
"""Validate a corpus/FAISS/vLLM embedding pipeline with reproducible probes.

The default self-retrieval test samples corpus documents, converts each sampled
document into a query, and checks whether its original document id appears in
the top-k results.  This is a smoke/consistency test rather than a benchmark on
human relevance labels; use --probes for a stronger labelled evaluation.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from .index_builder_api import EmbeddingClient, record_to_text
except ImportError:  # direct execution: python retrieval/diagnose_retrieval_quality.py
    from index_builder_api import EmbeddingClient, record_to_text


LOG = logging.getLogger("diagnose_retrieval")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def selected_records(corpus_path: Path, target_ids: Set[int]) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """Read selected non-empty JSONL records and count all corpus records."""
    selected: Dict[int, Dict[str, Any]] = {}
    record_id = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if record_id in target_ids:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON at %s:%d: %s" % (corpus_path, line_number, exc)) from exc
                if not isinstance(record, dict):
                    raise ValueError("Record %d is not a JSON object" % record_id)
                selected[record_id] = record
            record_id += 1
    missing = target_ids.difference(selected)
    if missing:
        raise RuntimeError("Sampled document ids are absent from corpus: %s" % sorted(missing)[:10])
    return selected, record_id


def make_self_query(record: Dict[str, Any], text_field: Optional[str], max_chars: int) -> str:
    # Include a passage excerpt even when a title exists.  Titles alone are
    # frequently duplicated in web corpora, which would make exact-id Hit@K a
    # misleading alignment test.
    text = record_to_text(record, text_field)
    text = " ".join(text.split())
    return text[:max_chars]


def load_probes(path: Path) -> List[Dict[str, Any]]:
    probes: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("query"), str):
                raise ValueError("Probe at line %d needs a string 'query'" % line_number)
            ids = item.get("relevant_ids", item.get("expected_ids", item.get("expected_id")))
            if isinstance(ids, int):
                ids = [ids]
            if not isinstance(ids, list) or not ids or not all(isinstance(value, int) for value in ids):
                raise ValueError(
                    "Probe at line %d needs expected_id or a non-empty relevant_ids list" % line_number
                )
            probes.append({"query": item["query"], "relevant_ids": ids})
    if not probes:
        raise ValueError("Probe file is empty: %s" % path)
    return probes


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def evaluate(
    index: Any,
    client: EmbeddingClient,
    queries: Sequence[str],
    relevant_ids: Sequence[Set[int]],
    query_prefix: str,
    topk: int,
    batch_size: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reciprocal_ranks: List[float] = []
    hits = 0
    latencies_ms: List[float] = []
    details: List[Dict[str, Any]] = []
    score_rows: List[List[float]] = []
    api_norms: List[float] = []

    for start in range(0, len(queries), batch_size):
        query_batch = [query_prefix + query for query in queries[start : start + batch_size]]
        before = time.perf_counter()
        embeddings = client.encode(query_batch)
        encode_end = time.perf_counter()
        norms = np.linalg.norm(embeddings, axis=1)
        api_norms.extend(float(value) for value in norms)
        if np.any(norms <= 0) or not np.isfinite(embeddings).all():
            raise RuntimeError("Query API returned invalid embeddings")
        embeddings = np.ascontiguousarray(embeddings / norms[:, None], dtype=np.float32)
        scores, ids = index.search(embeddings, topk)
        after = time.perf_counter()
        per_query_ms = (after - before) * 1000.0 / len(query_batch)
        latencies_ms.extend([per_query_ms] * len(query_batch))
        for local_idx, (row_ids, row_scores) in enumerate(zip(ids, scores)):
            probe_idx = start + local_idx
            wanted = relevant_ids[probe_idx]
            found_rank = 0
            for rank, doc_id in enumerate(row_ids.tolist(), 1):
                if int(doc_id) in wanted:
                    found_rank = rank
                    break
            if found_rank:
                hits += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
            score_list = [float(value) for value in row_scores]
            score_rows.append(score_list)
            details.append(
                {
                    "query": queries[probe_idx],
                    "relevant_ids": sorted(wanted),
                    "retrieved_ids": [int(value) for value in row_ids],
                    "scores": score_list,
                    "rank": found_rank or None,
                }
            )
        LOG.info(
            "Evaluated %d/%d probes (embedding %.2fs, total %.2fs)",
            min(start + batch_size, len(queries)),
            len(queries),
            encode_end - before,
            after - before,
        )

    metrics = {
        "probe_count": len(queries),
        "topk": topk,
        "hit_at_k": hits / len(queries),
        "mrr_at_k": sum(reciprocal_ranks) / len(queries),
        "latency_ms_per_query_p50": percentile(latencies_ms, 50),
        "latency_ms_per_query_p95": percentile(latencies_ms, 95),
        "api_embedding_norm_min": min(api_norms),
        "api_embedding_norm_max": max(api_norms),
        "top1_score_mean": float(np.mean([row[0] for row in score_rows])) if score_rows else None,
    }
    return metrics, details


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--retrieval-method", default="e5")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--sample-size", type=positive_int, default=100)
    parser.add_argument("--query-max-chars", type=positive_int, default=300)
    parser.add_argument("--topk", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--nprobe", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--probes", type=Path, default=None, help="JSONL with query + expected_id/relevant_ids")
    parser.add_argument("--min-hit-rate", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--details-limit", type=int, default=20)
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Load the entire index into RAM instead of memory-mapping its inverted lists",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if not args.index_path.is_file():
        parser.error("index does not exist: %s" % args.index_path)
    if not args.corpus_path.is_file():
        parser.error("corpus does not exist: %s" % args.corpus_path)
    if args.probes is not None and not args.probes.is_file():
        parser.error("probe file does not exist: %s" % args.probes)
    if not 0.0 <= args.min_hit_rate <= 1.0:
        parser.error("--min-hit-rate must be between 0 and 1")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.details_limit < 0:
        parser.error("--details-limit cannot be negative")
    if args.query_prefix is None:
        args.query_prefix = "query: " if "e5" in args.retrieval_method.lower() else ""
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required to diagnose the index") from exc

    started = time.perf_counter()
    LOG.info("Reading index %s", args.index_path)
    if not args.no_mmap and hasattr(faiss, "IO_FLAG_MMAP"):
        flags = faiss.IO_FLAG_MMAP
        if hasattr(faiss, "IO_FLAG_READ_ONLY"):
            flags |= faiss.IO_FLAG_READ_ONLY
        try:
            index = faiss.read_index(str(args.index_path), flags)
        except RuntimeError as exc:
            LOG.warning("Memory-mapped loading failed (%s); falling back to normal loading", exc)
            index = faiss.read_index(str(args.index_path))
    else:
        index = faiss.read_index(str(args.index_path))
    if not index.is_trained:
        raise RuntimeError("FAISS index is not trained")
    if int(index.ntotal) <= 0:
        raise RuntimeError("FAISS index is empty")
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
        effective_nprobe: Optional[int] = args.nprobe
    except RuntimeError:
        effective_nprobe = None  # Flat indexes do not have nprobe.
        LOG.info("Index has no nprobe parameter (likely a Flat index)")

    if args.probes:
        labelled = load_probes(args.probes)
        queries = [item["query"] for item in labelled]
        relevant_ids = [set(item["relevant_ids"]) for item in labelled]
        invalid = sorted({doc_id for ids in relevant_ids for doc_id in ids if doc_id < 0 or doc_id >= index.ntotal})
        if invalid:
            raise ValueError("Probe relevant ids are outside index range: %s" % invalid[:10])
        _, corpus_count = selected_records(args.corpus_path, set())
        test_kind = "labelled"
    else:
        sample_size = min(args.sample_size, int(index.ntotal))
        rng = np.random.default_rng(args.seed)
        sample_ids = sorted(int(value) for value in rng.choice(int(index.ntotal), size=sample_size, replace=False))
        records, corpus_count = selected_records(args.corpus_path, set(sample_ids))
        queries = [make_self_query(records[doc_id], args.text_field, args.query_max_chars) for doc_id in sample_ids]
        relevant_ids = [{doc_id} for doc_id in sample_ids]
        test_kind = "self_retrieval"

    if corpus_count != int(index.ntotal):
        raise RuntimeError(
            "Alignment failure: corpus has %d non-empty records but index.ntotal=%d" % (corpus_count, index.ntotal)
        )
    client = EmbeddingClient(args.api_url, args.api_model, args.request_timeout, args.retries, args.max_length)
    effective_topk = min(args.topk, int(index.ntotal))
    metrics, details = evaluate(
        index, client, queries, relevant_ids, args.query_prefix, effective_topk, args.batch_size
    )
    # The API/index dimension mismatch is normally surfaced by index.search;
    # retain it explicitly in the report for operational diagnosis.
    report: Dict[str, Any] = {
        "status": "PASS" if metrics["hit_at_k"] >= args.min_hit_rate else "FAIL",
        "test_kind": test_kind,
        "index": {
            "path": str(args.index_path.resolve()),
            "dimension": int(index.d),
            "ntotal": int(index.ntotal),
            "is_trained": bool(index.is_trained),
            "nprobe": effective_nprobe,
        },
        "corpus": {"path": str(args.corpus_path.resolve()), "record_count": corpus_count},
        "thresholds": {"min_hit_rate": args.min_hit_rate},
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "failures": [item for item in details if item["rank"] is None][: args.details_limit],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(rendered + "\n", encoding="utf-8")
        tmp.replace(args.output)
        LOG.info("Wrote report to %s", args.output)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
