#!/usr/bin/env python3
"""Build a FAISS index from a JSONL corpus through a vLLM embedding API.

The expensive API phase is resumable: embeddings are saved as numbered NumPy
chunks before FAISS training starts.  Document ids are exactly the zero-based
order of non-empty JSONL records, which is also the order used by
``datasets.load_dataset('json', ...)`` in the retrieval server.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


LOG = logging.getLogger("index_builder")
MANIFEST_VERSION = 1


def atomic_json_dump(data: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def parse_json_response(url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class EmbeddingClient:
    def __init__(
        self,
        api_url: str,
        model: Optional[str],
        timeout: float,
        retries: int,
        max_length: Optional[int],
    ) -> None:
        self.base_url = api_url.rstrip("/")
        if self.base_url.endswith("/v1/embeddings"):
            self.endpoint = self.base_url
        elif self.base_url.endswith("/v1"):
            self.endpoint = self.base_url + "/embeddings"
        else:
            self.endpoint = self.base_url + "/v1/embeddings"
        self.model = model or self._discover_model()
        self.timeout = timeout
        self.retries = retries
        self.max_length = max_length
        LOG.info("Using embedding model %s at %s", self.model, self.endpoint)

    def _discover_model(self) -> str:
        base = self.base_url
        if base.endswith("/v1/embeddings"):
            models_url = base[: -len("/embeddings")] + "/models"
        elif base.endswith("/v1"):
            models_url = base + "/models"
        else:
            models_url = base + "/v1/models"
        try:
            response = parse_json_response(models_url, None, 30)
            models = response.get("data", [])
            if models and models[0].get("id"):
                return str(models[0]["id"])
        except Exception as exc:  # model discovery is only a convenience
            raise RuntimeError(
                "Could not discover the served model from /v1/models; pass --api-model explicitly"
            ) from exc
        raise RuntimeError("The vLLM /v1/models response contained no model id")

    def _request_once(self, texts: Sequence[str]) -> np.ndarray:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        # Supported by recent vLLM releases.  Omitting it lets the server's
        # model configuration decide how over-long inputs are handled.
        if self.max_length:
            payload["truncate_prompt_tokens"] = self.max_length
        response = parse_json_response(self.endpoint, payload, self.timeout)
        if "data" not in response:
            raise RuntimeError("Embedding response has no 'data' field: %s" % response)
        ordered = sorted(response["data"], key=lambda item: int(item.get("index", 0)))
        vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError(
                "Expected %d embeddings, received shape %s" % (len(texts), tuple(vectors.shape))
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("Embedding API returned NaN or infinite values")
        return vectors

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch, retrying transient errors and splitting large failures."""
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                return self._request_once(texts)
            except (HTTPError, URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = min(30.0, 2.0 ** attempt + random.random())
                    LOG.warning("Embedding request failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)
        if len(texts) > 1:
            # A smaller request often recovers from proxy/body-size limits or a
            # single pathological input without throwing away completed work.
            middle = len(texts) // 2
            LOG.warning("Splitting failed batch of %d inputs", len(texts))
            return np.concatenate((self.encode(texts[:middle]), self.encode(texts[middle:])), axis=0)
        raise RuntimeError("Embedding request failed permanently") from last_error


def record_to_text(record: Dict[str, Any], text_field: Optional[str]) -> str:
    if text_field:
        if text_field not in record:
            raise KeyError("Corpus record has no requested field %r" % text_field)
        value = record[text_field]
    elif record.get("contents") is not None:
        value = record["contents"]
    elif record.get("title") is not None or record.get("text") is not None:
        title = str(record.get("title") or "").strip()
        text = str(record.get("text") or "").strip()
        value = "\n".join(part for part in (title, text) if part)
    elif record.get("text") is not None:
        value = record["text"]
    else:
        raise KeyError("Corpus record needs 'contents', 'text', or title/text fields")
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    value = str(value).strip()
    if not value:
        raise ValueError("Corpus record produced empty text")
    return value


def iter_corpus(path: Path, start_record: int = 0) -> Iterable[Tuple[int, str, Dict[str, Any]]]:
    record_id = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            if record_id < start_record:
                record_id += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_number, exc)) from exc
            if not isinstance(record, dict):
                raise ValueError("Corpus record at %s:%d is not a JSON object" % (path, line_number))
            yield record_id, line_number, record
            record_id += 1


def save_array_atomic(array: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def initial_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    stat = args.corpus_path.stat()
    return {
        "version": MANIFEST_VERSION,
        "corpus_path": str(args.corpus_path.resolve()),
        "corpus_size": stat.st_size,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "retrieval_method": args.retrieval_method,
        "text_field": args.text_field,
        "prefix": args.passage_prefix,
        "storage_dtype": args.embedding_dtype,
        "dimension": None,
        "record_count": 0,
        "embedding_complete": False,
        "chunks": [],
    }


def load_or_create_manifest(args: argparse.Namespace, manifest_path: Path) -> Dict[str, Any]:
    expected = initial_manifest(args)
    if not manifest_path.exists():
        atomic_json_dump(expected, manifest_path)
        return expected
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    keys = ("version", "corpus_path", "corpus_size", "corpus_mtime_ns", "text_field", "prefix", "storage_dtype")
    mismatches = [key for key in keys if manifest.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "Embedding workspace does not match this run (%s). Use another --work-dir or --overwrite."
            % ", ".join(mismatches)
        )
    return manifest


def validate_manifest_chunks(manifest: Dict[str, Any], chunks_dir: Path) -> None:
    expected_start = 0
    expected_dim = manifest.get("dimension")
    for chunk in manifest.get("chunks", []):
        if int(chunk["start_id"]) != expected_start:
            raise RuntimeError("Embedding manifest has a gap before document %d" % expected_start)
        path = chunks_dir / chunk["file"]
        if not path.is_file():
            raise RuntimeError("Missing embedding chunk: %s" % path)
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
        expected_shape = (int(chunk["count"]), int(expected_dim))
        if array.shape != expected_shape:
            raise RuntimeError("Chunk %s has shape %s; expected %s" % (path, array.shape, expected_shape))
        expected_start += int(chunk["count"])
    if expected_start != int(manifest.get("record_count", 0)):
        raise RuntimeError("Manifest record_count does not equal the sum of chunk sizes")


def embed_corpus(args: argparse.Namespace, manifest: Dict[str, Any], manifest_path: Path, chunks_dir: Path) -> None:
    if manifest.get("embedding_complete"):
        LOG.info("Embedding phase already complete (%d records)", manifest["record_count"])
        return
    client = EmbeddingClient(args.api_url, args.api_model, args.request_timeout, args.retries, args.max_length)
    start_id = int(manifest.get("record_count", 0))
    texts: List[str] = []
    vectors: List[np.ndarray] = []
    chunk_start = start_id
    started = time.time()

    def flush_chunk() -> None:
        nonlocal chunk_start, texts, vectors
        if not vectors:
            return
        matrix = np.concatenate(vectors, axis=0)
        dtype = np.float16 if args.embedding_dtype == "float16" else np.float32
        matrix = np.ascontiguousarray(matrix, dtype=dtype)
        chunk_no = len(manifest["chunks"])
        filename = "embeddings-%06d.npy" % chunk_no
        save_array_atomic(matrix, chunks_dir / filename)
        manifest["chunks"].append({"file": filename, "start_id": chunk_start, "count": int(matrix.shape[0])})
        manifest["record_count"] = chunk_start + int(matrix.shape[0])
        atomic_json_dump(manifest, manifest_path)
        elapsed = max(time.time() - started, 1e-6)
        LOG.info(
            "Saved %s: ids [%d, %d), total=%d, %.1f docs/s",
            filename,
            chunk_start,
            manifest["record_count"],
            manifest["record_count"],
            (manifest["record_count"] - start_id) / elapsed,
        )
        chunk_start = int(manifest["record_count"])
        texts = []
        vectors = []

    for record_id, line_number, record in iter_corpus(args.corpus_path, start_id):
        try:
            text = record_to_text(record, args.text_field)
        except Exception as exc:
            raise ValueError("Cannot extract text for document %d (line %d): %s" % (record_id, line_number, exc)) from exc
        texts.append(args.passage_prefix + text)
        if len(texts) >= args.batch_size:
            batch_vectors = client.encode(texts)
            norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise RuntimeError("Embedding API returned a zero vector")
            batch_vectors /= norms
            if manifest["dimension"] is None:
                manifest["dimension"] = int(batch_vectors.shape[1])
            elif int(manifest["dimension"]) != batch_vectors.shape[1]:
                raise RuntimeError("Embedding dimension changed during the run")
            vectors.append(batch_vectors)
            texts = []
            if sum(item.shape[0] for item in vectors) >= args.chunk_size:
                flush_chunk()

    if texts:
        batch_vectors = client.encode(texts)
        norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError("Embedding API returned a zero vector")
        batch_vectors /= norms
        if manifest["dimension"] is None:
            manifest["dimension"] = int(batch_vectors.shape[1])
        elif int(manifest["dimension"]) != batch_vectors.shape[1]:
            raise RuntimeError("Embedding dimension changed during the run")
        vectors.append(batch_vectors)
    flush_chunk()
    manifest["embedding_complete"] = True
    atomic_json_dump(manifest, manifest_path)
    LOG.info("Embedding phase complete: %d documents", manifest["record_count"])


def training_sample(manifest: Dict[str, Any], chunks_dir: Path, size: int, seed: int) -> np.ndarray:
    total = int(manifest["record_count"])
    sample_size = min(size, total)
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(total, size=sample_size, replace=False))
    result = np.empty((sample_size, int(manifest["dimension"])), dtype=np.float32)
    output_pos = 0
    for chunk in manifest["chunks"]:
        start = int(chunk["start_id"])
        end = start + int(chunk["count"])
        left = int(np.searchsorted(ids, start, side="left"))
        right = int(np.searchsorted(ids, end, side="left"))
        if right <= left:
            continue
        array = np.load(str(chunks_dir / chunk["file"]), mmap_mode="r", allow_pickle=False)
        selected = np.asarray(array[ids[left:right] - start], dtype=np.float32)
        result[output_pos : output_pos + len(selected)] = selected
        output_pos += len(selected)
    if output_pos != sample_size:
        raise RuntimeError("Could not assemble the requested FAISS training sample")
    return result


def build_faiss(args: argparse.Namespace, manifest: Dict[str, Any], chunks_dir: Path) -> None:
    if not manifest.get("embedding_complete"):
        raise RuntimeError("Embedding phase is incomplete")
    if int(manifest.get("record_count", 0)) == 0:
        raise RuntimeError("Corpus contains no records")
    if args.index_path.exists() and not args.overwrite:
        metadata_path = args.index_path.with_suffix(args.index_path.suffix + ".meta.json")
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            same_build = (
                int(metadata.get("ntotal", -1)) == int(manifest["record_count"])
                and int(metadata.get("dimension", -1)) == int(manifest["dimension"])
                and metadata.get("faiss_type") == args.faiss_type
                and metadata.get("corpus_path") == str(args.corpus_path.resolve())
            )
            if same_build:
                LOG.info("Matching index already exists, leaving it unchanged: %s", args.index_path)
                return
        raise RuntimeError(
            "Index already exists but cannot be verified as this build: %s. "
            "Move it or pass --overwrite." % args.index_path
        )
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required for the indexing phase") from exc

    dimension = int(manifest["dimension"])
    LOG.info("Creating FAISS %s index (dimension=%d, metric=inner product)", args.faiss_type, dimension)
    index = faiss.index_factory(dimension, args.faiss_type, faiss.METRIC_INNER_PRODUCT)
    if not index.is_trained:
        sample = training_sample(manifest, chunks_dir, args.train_size, args.seed)
        LOG.info("Training FAISS on %d sampled vectors", sample.shape[0])
        index.train(sample)
        del sample
    if not index.is_trained:
        raise RuntimeError("FAISS index did not become trained")

    for chunk_no, chunk in enumerate(manifest["chunks"], 1):
        array = np.load(str(chunks_dir / chunk["file"]), mmap_mode="r", allow_pickle=False)
        for offset in range(0, array.shape[0], args.faiss_add_batch_size):
            batch = np.ascontiguousarray(array[offset : offset + args.faiss_add_batch_size], dtype=np.float32)
            index.add(batch)
        LOG.info("Added chunk %d/%d; ntotal=%d", chunk_no, len(manifest["chunks"]), index.ntotal)

    if int(index.ntotal) != int(manifest["record_count"]):
        raise RuntimeError("FAISS ntotal=%d but corpus has %d records" % (index.ntotal, manifest["record_count"]))
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    building_path = args.index_path.with_suffix(args.index_path.suffix + ".building")
    if building_path.exists():
        building_path.unlink()
    LOG.info("Writing final index to %s", building_path)
    faiss.write_index(index, str(building_path))
    os.replace(str(building_path), str(args.index_path))
    metadata = {
        "index_path": str(args.index_path.resolve()),
        "faiss_type": args.faiss_type,
        "metric": "inner_product",
        "dimension": dimension,
        "ntotal": int(index.ntotal),
        "corpus_path": str(args.corpus_path.resolve()),
        "retrieval_method": args.retrieval_method,
        "passage_prefix": args.passage_prefix,
        "normalized": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json_dump(metadata, args.index_path.with_suffix(args.index_path.suffix + ".meta.json"))
    LOG.info("Index build complete: %s", args.index_path)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None, help="Resumable embedding chunk directory")
    parser.add_argument("--retrieval-method", default="e5")
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--passage-prefix", default=None)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-model", default=None, help="Defaults to the first model returned by /v1/models")
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--chunk-size", type=positive_int, default=100000)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--faiss-type", default="IVF4096,Flat")
    parser.add_argument("--train-size", type=positive_int, default=262144)
    parser.add_argument("--faiss-add-batch-size", type=positive_int, default=65536)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-embeddings", action="store_true", help="Delete chunks only after a successful build")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if not args.corpus_path.is_file():
        parser.error("corpus does not exist: %s" % args.corpus_path)
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.passage_prefix is None:
        args.passage_prefix = "passage: " if "e5" in args.retrieval_method.lower() else ""
    if args.index_path is None:
        args.index_path = args.save_dir / ("%s_%s.index" % (args.retrieval_method, args.faiss_type))
    if args.work_dir is None:
        safe_type = args.faiss_type.replace("/", "_").replace("\\", "_")
        args.work_dir = args.save_dir / (".%s_%s.embedding_chunks" % (args.retrieval_method, safe_type))
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.overwrite and args.work_dir.exists():
        LOG.warning("Removing embedding workspace because --overwrite was supplied: %s", args.work_dir)
        shutil.rmtree(str(args.work_dir))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.work_dir / "manifest.json"
    manifest = load_or_create_manifest(args, manifest_path)
    validate_manifest_chunks(manifest, chunks_dir)
    if not args.skip_embedding:
        embed_corpus(args, manifest, manifest_path, chunks_dir)
    # Reload because the embedding phase updates the persisted manifest.
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest_chunks(manifest, chunks_dir)
    if not args.embedding_only:
        build_faiss(args, manifest, chunks_dir)
        if args.delete_embeddings:
            shutil.rmtree(str(args.work_dir))
            LOG.info("Deleted embedding workspace: %s", args.work_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; completed embedding chunks remain resumable")
        sys.exit(130)
