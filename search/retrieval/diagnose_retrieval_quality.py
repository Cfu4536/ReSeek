import argparse
import re
from typing import List

import datasets
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "und",
    "what",
    "which",
    "who",
}


def add_prefix(items: List[str], retriever_name: str, is_query: bool) -> List[str]:
    name = retriever_name.lower()
    if "e5" in name:
        prefix = "query: " if is_query else "passage: "
        return [prefix + item for item in items]
    if "bge" in name and is_query:
        return [f"Represent this sentence for searching relevant passages: {item}" for item in items]
    if "qwen" in name and is_query:
        return [
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
            f"Query: {item}"
            for item in items
        ]
    return items


def encode(model: SentenceTransformer, texts: List[str], retriever_name: str, is_query: bool) -> np.ndarray:
    texts = add_prefix(texts, retriever_name, is_query)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32, order="C")


def title_from_contents(contents: str) -> str:
    return contents.split("\n", 1)[0].strip().strip('"')


def query_terms(query: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[A-Za-z0-9]+", query.lower())
        if len(tok) > 2 and tok not in STOPWORDS
    }


def lexical_probe(
    corpus,
    index,
    model: SentenceTransformer,
    query: str,
    q_emb: np.ndarray,
    retriever_name: str,
    max_hits: int,
    required_terms: List[str] | None = None,
) -> None:
    terms = [term.lower() for term in (required_terms or sorted(query_terms(query)))]
    if not terms:
        return

    hits = []
    for idx, doc in enumerate(corpus):
        contents = doc.get("contents", "")
        lowered = contents.lower()
        if all(term in lowered for term in terms):
            hits.append((idx, contents))
            if len(hits) >= max_hits:
                break

    print(f"LEXICAL PROBE: first {len(hits)} rows containing all terms {terms}")
    if not hits:
        return

    doc_emb = encode(model, [contents for _, contents in hits], retriever_name, is_query=False)
    scores = (doc_emb @ q_emb[0]).tolist()
    for rank, ((idx, contents), score) in enumerate(zip(hits, scores), start=1):
        index_score, vector_agreement = score_index_vector(index, int(idx), q_emb[0], doc_emb[rank - 1])
        title = title_from_contents(contents)
        print(
            f"  lexical {rank:02d}. idx={idx} recomputed={score:.6f} "
            f"index_vector_score={index_score} vector_agreement={vector_agreement} title={title}"
        )
        print("      " + contents.replace("\n", " ")[:240])


def get_ivf_index(index):
    try:
        candidate = faiss.extract_index_ivf(index)
        if candidate is not None:
            return candidate
    except Exception:
        pass
    if hasattr(index, "invlists") and hasattr(index, "nlist"):
        return index
    return None


def find_ivf_vector_by_id(index, target_id: int):
    ivf = get_ivf_index(index)
    if ivf is None:
        return None, None, None

    code_size = int(ivf.code_size)
    for list_no in range(int(ivf.nlist)):
        list_size = int(ivf.invlists.list_size(list_no))
        if list_size == 0:
            continue

        ids = faiss.rev_swig_ptr(ivf.invlists.get_ids(list_no), list_size)
        matches = np.where(ids == target_id)[0]
        if len(matches) == 0:
            continue

        pos = int(matches[0])
        codes = faiss.rev_swig_ptr(ivf.invlists.get_codes(list_no), list_size * code_size)
        raw = codes[pos * code_size : (pos + 1) * code_size]
        vector = np.frombuffer(raw.tobytes(), dtype=np.float32).copy()
        return vector, list_no, list_size

    return None, None, None


def score_index_vector(index, target_id: int, q_emb: np.ndarray, doc_emb: np.ndarray):
    try:
        stored = np.zeros(index.d, dtype=np.float32)
        index.reconstruct(target_id, stored)
    except Exception as exc:
        stored, _, _ = find_ivf_vector_by_id(index, target_id)
        if stored is None:
            return f"unavailable:{type(exc).__name__}", None

    stored_norm = np.linalg.norm(stored)
    if stored_norm > 0:
        stored = stored / stored_norm
    index_score = float(stored @ q_emb)
    vector_agreement = float(stored @ doc_emb)
    return f"{index_score:.6f}", f"{vector_agreement:.6f}"


def inspect_ivf_id(index, target_id: int, q_emb: np.ndarray, doc_emb: np.ndarray | None = None) -> None:
    ivf = get_ivf_index(index)
    if ivf is None:
        print(f"INSPECT ID {target_id}: index is not an IVF index or IVF internals are unavailable.")
        return

    vector, list_no, list_size = find_ivf_vector_by_id(index, target_id)
    if vector is None:
        print(f"INSPECT ID {target_id}: id not found in IVF inverted lists.")
        return

    norm = np.linalg.norm(vector)
    normalized = vector / norm if norm > 0 else vector
    index_score = float(normalized @ q_emb[0])
    agreement = float(normalized @ doc_emb) if doc_emb is not None else None

    list_rank = None
    try:
        _, coarse_lists = ivf.quantizer.search(q_emb.astype(np.float32, order="C"), int(ivf.nlist))
        matches = np.where(coarse_lists[0] == list_no)[0]
        if len(matches) > 0:
            list_rank = int(matches[0]) + 1
    except Exception as exc:
        list_rank = f"unavailable:{type(exc).__name__}"

    current_nprobe = getattr(ivf, "nprobe", None)
    print(
        f"INSPECT ID {target_id}: list={list_no} list_size={list_size} "
        f"coarse_list_rank={list_rank} current_nprobe={current_nprobe} "
        f"index_vector_score={index_score:.6f} vector_agreement={agreement}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose dense retrieval quality and detect FAISS index/corpus row mismatches. "
            "For each retrieved row, it recomputes query-document similarity from the returned JSONL content."
        )
    )
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "Maximum token length used when recomputing embeddings. "
            "Defaults to 256 to match build_index_vllm_api.sh."
        ),
    )
    parser.add_argument(
        "--truncation_side",
        choices=("left", "right"),
        default="left",
        help=(
            "Tokenizer truncation side used when recomputing embeddings. "
            "Defaults to left to match vLLM truncate_prompt_tokens."
        ),
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=[],
        help="Query to test. Can be passed multiple times.",
    )
    parser.add_argument(
        "--lexical_probe",
        action="store_true",
        help="Also scan the corpus for rows containing all non-stopword query terms and score the first hits.",
    )
    parser.add_argument(
        "--lexical_term",
        action="append",
        dest="lexical_terms",
        default=[],
        help="Term required by lexical probe. Can be passed multiple times; overrides automatic query terms.",
    )
    parser.add_argument("--lexical_max_hits", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=None, help="Override IVF nprobe when supported.")
    parser.add_argument("--ef_search", type=int, default=None, help="Override HNSW efSearch when supported.")
    parser.add_argument(
        "--inspect_id",
        action="append",
        type=int,
        default=[],
        help="Inspect a FAISS id inside IVF inverted lists. Can be passed multiple times.",
    )
    args = parser.parse_args()

    queries = args.queries or [
        "libretto Tristan und Isolde language",
        "Cruyff Football manager",
        "Johan Cruyff football manager",
        "What language is the libretto of Tristan und Isolde written in?",
    ]

    print(f"Loading FAISS index: {args.index_path}")
    index = faiss.read_index(args.index_path)
    print(f"Index repr: {index}")
    print(f"Index vectors: {index.ntotal}, dim: {index.d}")
    ivf = get_ivf_index(index)
    if ivf is not None:
        print(f"IVF params: nlist={ivf.nlist}, nprobe={ivf.nprobe}, code_size={ivf.code_size}")
    if args.nprobe is not None:
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
            print(f"Set FAISS nprobe={args.nprobe}")
            ivf = get_ivf_index(index)
            if ivf is not None:
                print(f"IVF params after override: nlist={ivf.nlist}, nprobe={ivf.nprobe}")
        except Exception as exc:
            print(f"Could not set nprobe: {type(exc).__name__}: {exc}")
    if args.ef_search is not None:
        try:
            faiss.ParameterSpace().set_index_parameter(index, "efSearch", args.ef_search)
            print(f"Set FAISS efSearch={args.ef_search}")
        except Exception as exc:
            print(f"Could not set efSearch: {type(exc).__name__}: {exc}")

    print(f"Loading corpus: {args.corpus_path}")
    corpus = datasets.load_dataset("json", data_files=args.corpus_path, split="train", num_proc=args.num_proc)
    print(f"Corpus rows: {len(corpus)}")
    if len(corpus) != index.ntotal:
        print("WARNING: corpus row count != index vector count. Row-id mapping is very likely broken.")

    print(f"Loading encoder: {args.retriever_model}")
    model = SentenceTransformer(args.retriever_model)
    model.max_seq_length = args.max_length
    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        raise RuntimeError("SentenceTransformer does not expose a tokenizer; cannot configure truncation.")
    model.tokenizer.truncation_side = args.truncation_side
    model.eval()
    print(
        f"Encoder truncation: max_length={model.max_seq_length}, "
        f"side={model.tokenizer.truncation_side}"
    )

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        q_emb = encode(model, [query], args.retriever_name, is_query=True)
        if q_emb.shape[1] != index.d:
            print(f"ERROR: query embedding dim {q_emb.shape[1]} != FAISS index dim {index.d}")
            continue

        scores, idxs = index.search(q_emb, args.topk)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        docs = [corpus[int(idx)] for idx in idxs]
        contents = [doc.get("contents", "") for doc in docs]
        doc_emb = encode(model, contents, args.retriever_name, is_query=False)
        recomputed = (doc_emb @ q_emb[0]).tolist()

        q_terms = query_terms(query)
        mismatch_count = 0
        for rank, (idx, faiss_score, dot_score, doc) in enumerate(zip(idxs, scores, recomputed, docs), start=1):
            contents_text = doc.get("contents", "")
            title = title_from_contents(contents_text)
            overlap = sorted(q_terms.intersection(query_terms(contents_text[:2000])))
            gap = abs(float(faiss_score) - float(dot_score))
            if gap > 0.05:
                mismatch_count += 1
            print(
                f"{rank:02d}. idx={idx} faiss={faiss_score:.6f} recomputed={dot_score:.6f} "
                f"gap={gap:.6f} overlap={overlap} title={title}"
            )
            print("    " + contents_text.replace("\n", " ")[:240])

        if mismatch_count >= max(2, args.topk // 2):
            print(
                "DIAGNOSIS: many FAISS scores do not match recomputed scores for returned documents. "
                "The index and corpus row order/file likely do not match."
            )
        else:
            print(
                "DIAGNOSIS: FAISS scores mostly match recomputed document scores. "
                "Index/corpus mapping looks plausible; poor results are likely from dense-only retrieval/query form."
            )

        if args.lexical_probe:
            lexical_probe(
                corpus=corpus,
                index=index,
                model=model,
                query=query,
                q_emb=q_emb,
                retriever_name=args.retriever_name,
                max_hits=args.lexical_max_hits,
                required_terms=args.lexical_terms or None,
            )

        if args.inspect_id:
            inspect_docs = [corpus[int(inspect_id)].get("contents", "") for inspect_id in args.inspect_id]
            inspect_embs = encode(model, inspect_docs, args.retriever_name, is_query=False)
            for inspect_id, inspect_emb in zip(args.inspect_id, inspect_embs):
                inspect_ivf_id(index, inspect_id, q_emb, inspect_emb)


if __name__ == "__main__":
    main()
