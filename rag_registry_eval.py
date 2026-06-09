from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from pdf_pipeline import run_ocr_pipeline, search_rag_pages


DEFAULT_REGISTRY_PATH = "calculators_registry.json"
DEFAULT_QUERY_CACHE_PATH = os.path.join("processed_files", "rag_query_expansions.json")
DEFAULT_TOP_K = 6
DEFAULT_CANDIDATE_K = 20
DEFAULT_DENSE_WEIGHT = 0.8
DEFAULT_CHUNK_MODES = "token"
DEFAULT_CHUNK_SIZES = "500"
DEFAULT_CHUNK_OVERLAPS = "100"
DEFAULT_QUERY_MODEL = "gemini-2.5-flash"
DEFAULT_RERANKER = "llm"
DEFAULT_RERANK_MODEL = "claude-sonnet-4-20250514"
DEFAULT_RERANK_CONTEXT_CHARS = 500
DEFAULT_RERANK_CHUNKS_PER_PAGE = 3


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, value: Any) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def successful_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if "successful" in str(record.get("code_path", "")).lower()
    ]


def int_pages(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    pages = []
    for page in value:
        try:
            pages.append(int(page))
        except (TypeError, ValueError):
            continue
    return sorted(set(pages))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_chunk_modes(value: str) -> list[str]:
    modes = []
    for item in value.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode in {"page", "pages", "page_level", "page-level"}:
            modes.append("page")
        elif mode in {"token", "tokens", "chunk", "chunks", "token_chunk", "token-chunk"}:
            modes.append("token")
        else:
            raise ValueError(f"Unsupported chunk mode: {mode}")
    return list(dict.fromkeys(modes))


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(cleaned[start:end + 1])
    return value if isinstance(value, dict) else {}


def deterministic_query_expansion(tariff: str) -> list[str]:
    base = [
        tariff,
        f"{tariff} rate formula tariff charges",
        f"How are {tariff} calculated?",
        f"Which vessel parameters are used to calculate {tariff}?",
        f"{tariff} tariff rate basis GT LOA NT DWT days movements",
        f"{tariff} gross tonnage length overall net tonnage deadweight",
    ]
    synonyms = {
        "Light Dues": ["lighthouse dues", "aids to navigation tariff", "light dues"],
        "Port Dues": ["harbour dues", "port charges", "port tariff"],
        "Towage Dues": ["tug dues", "towage charges", "tug assistance tariff"],
        "VTS Dues": ["vessel traffic service dues", "VTS charges"],
        "Pilotage Dues": ["pilotage charges", "pilot dues", "pilot tariff"],
        "Running Lines": ["line handling", "mooring lines", "running lines charges"],
    }
    return list(dict.fromkeys(base + synonyms.get(tariff, [])))


def query_cache_key(record: dict[str, Any], tariff: str, model: str) -> str:
    payload = {
        "document_name": record.get("document_name"),
        "pdf_file": record.get("pdf_file"),
        "target_tariffs": record.get("target_tariffs", []),
        "tariff": tariff,
        "model": model,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def llm_query_expansion(
    record: dict[str, Any],
    tariff: str,
    *,
    model: str,
    cache: dict[str, Any],
) -> tuple[list[str], str]:
    base_queries = deterministic_query_expansion(tariff)
    cache_key = query_cache_key(record, tariff, model)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("queries"), list):
        queries = base_queries + [str(query) for query in cached["queries"]]
        return list(dict.fromkeys(query for query in queries if query.strip())), "cache"

    api_key = os.getenv("Gemini_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return base_queries, "deterministic_no_api_key"

    client = genai.Client(api_key=api_key)
    prompt = {
        "task": (
            "Expand a target tariff name into retrieval queries for a port tariff PDF. "
            "Use the provided document context and neighboring target tariff names only as search context. "
            "Do not include page numbers. Do not solve the tariff calculation."
        ),
        "document_context": {
            "document_name": record.get("document_name"),
            "pdf_file": record.get("pdf_file"),
            "all_target_tariffs_in_record": record.get("target_tariffs", []),
            "target_tariff": tariff,
        },
        "query_requirements": [
            "exact tariff name",
            "domain synonyms and alternative wording",
            "rate, formula, tariff, charges, dues wording",
            "likely calculation parameters: GT, LOA, NT, DWT, days, hours, movements, vessel type, cargo type",
            "questions that would retrieve rules, tables, formulas, exceptions, minimums, maximums, and footnotes",
        ],
        "output_schema": {"queries": ["query", "..."]},
    }

    try:
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(prompt, ensure_ascii=False),
        )
        text = response.text or "{}"
        expanded = parse_json_object(text)
        llm_queries = expanded.get("queries", []) if isinstance(expanded, dict) else []
        cache[cache_key] = {
            "model": model,
            "document_name": record.get("document_name"),
            "pdf_file": record.get("pdf_file"),
            "tariff": tariff,
            "queries": [str(query) for query in llm_queries if str(query).strip()],
        }
        queries = base_queries + cache[cache_key]["queries"]
        return list(dict.fromkeys(query for query in queries if query.strip())), "llm"
    except Exception as exc:
        cache[cache_key] = {
            "model": model,
            "document_name": record.get("document_name"),
            "pdf_file": record.get("pdf_file"),
            "tariff": tariff,
            "queries": [],
            "error": str(exc),
        }
        return base_queries, "deterministic_llm_error"


def evaluate_tariff(
    record: dict[str, Any],
    tariff: str,
    *,
    pages: dict[int, str],
    queries: list[str],
    query_source: str,
    top_k: int,
    candidate_k: int,
    dense_weight: float,
    chunk_mode: str,
    chunk_size_tokens: int | None,
    chunk_overlap_tokens: int | None,
    embedding_model: str | None,
    reranker: str,
    rerank_model: str,
    rerank_context_chars: int,
    rerank_chunks_per_page: int,
) -> dict[str, Any]:
    pdf_file = record["pdf_file"]
    expected_pages = int_pages(record.get("evidence_pages", {}).get(tariff, []))
    query = "\n".join(queries)
    results = search_rag_pages(
        pdf_path=pdf_file,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        embedding_model=embedding_model,
        pages=pages,
        include_text=False,
        dense_weight=dense_weight,
        chunk_mode=chunk_mode,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        reranker=reranker,
        rerank_model=rerank_model,
        rerank_context_chars=rerank_context_chars,
        rerank_chunks_per_page=rerank_chunks_per_page,
    )
    retrieved_pages = [int(item["page_number"]) for item in results]
    expected_set = set(expected_pages)
    retrieved_set = set(retrieved_pages)
    matched_pages = sorted(expected_set & retrieved_set)
    missing_pages = sorted(expected_set - retrieved_set)
    extra_pages = sorted(retrieved_set - expected_set)

    recall = len(matched_pages) / len(expected_set) if expected_set else 0.0
    precision = len(matched_pages) / len(retrieved_set) if retrieved_set else 0.0

    return {
        "code_path": record.get("code_path"),
        "document_name": record.get("document_name"),
        "pdf_file": pdf_file,
        "tariff": tariff,
        "query_source": query_source,
        "query_count": len(queries),
        "expected_pages": expected_pages,
        "retrieved_pages": retrieved_pages,
        "matched_pages": matched_pages,
        "missing_pages": missing_pages,
        "extra_pages": extra_pages,
        "hit": bool(matched_pages),
        "all_expected_found": not missing_pages,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "rag_results": [
            {
                "page_number": int(item["page_number"]),
                "score": round(float(item["score"]), 6),
                "dense_score": round(float(item["dense_score"]), 6),
                "bm25_score": round(float(item["bm25_score"]), 6),
                "metadata": item.get("metadata", {}),
            }
            for item in results
        ],
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    hits = sum(1 for item in results if item["hit"])
    full_hits = sum(1 for item in results if item["all_expected_found"])
    expected_total = sum(len(item["expected_pages"]) for item in results)
    matched_total = sum(len(item["matched_pages"]) for item in results)
    retrieved_total = sum(len(set(item["retrieved_pages"])) for item in results)

    return {
        "total_tariff_evaluations": total,
        "tariffs_with_any_hit": hits,
        "tariffs_with_all_expected_found": full_hits,
        "hit_rate": round(hits / total, 4) if total else 0.0,
        "full_hit_rate": round(full_hits / total, 4) if total else 0.0,
        "page_recall": round(matched_total / expected_total, 4) if expected_total else 0.0,
        "page_precision": round(matched_total / retrieved_total, 4) if retrieved_total else 0.0,
    }


def experiment_sort_key(experiment: dict[str, Any]) -> tuple[float, float, float]:
    summary = experiment["summary"]
    return (
        float(summary["page_recall"]),
        float(summary["full_hit_rate"]),
        float(summary["page_precision"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search RAG retrieval settings against evidence_pages in calculators_registry.json."
    )
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--chunk-modes", default=DEFAULT_CHUNK_MODES)
    parser.add_argument("--chunk-sizes", "--chank-sizes", dest="chunk_sizes", default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--chunk-overlaps", "--overlaps", dest="chunk_overlaps", default=DEFAULT_CHUNK_OVERLAPS)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--query-model", default=DEFAULT_QUERY_MODEL)
    parser.add_argument("--query-cache", default=DEFAULT_QUERY_CACHE_PATH)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-context-chars", type=int, default=DEFAULT_RERANK_CONTEXT_CHARS)
    parser.add_argument("--rerank-chunks-per-page", type=int, default=DEFAULT_RERANK_CHUNKS_PER_PAGE)
    parser.add_argument("--tariff", default=None, help="Evaluate only records containing this target tariff.")
    parser.add_argument("--pdf", default=None, help="Evaluate only records with this pdf_file.")
    parser.add_argument("--output", default=os.path.join("logs", "rag_registry_eval.json"))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    records = successful_records(read_json(args.registry, []))
    if args.tariff:
        records = [
            record
            for record in records
            if args.tariff in [str(tariff) for tariff in record.get("target_tariffs", [])]
        ]
    if args.pdf:
        records = [record for record in records if record.get("pdf_file") == args.pdf]

    chunk_modes = parse_chunk_modes(args.chunk_modes)
    chunk_sizes = parse_int_list(args.chunk_sizes)
    chunk_overlaps = parse_int_list(args.chunk_overlaps)
    parameter_sets = []
    if "page" in chunk_modes:
        parameter_sets.append(("page", None, None))
    if "token" in chunk_modes:
        parameter_sets.extend(
            ("token", chunk_size, chunk_overlap)
            for chunk_size, chunk_overlap in itertools.product(
                chunk_sizes,
                chunk_overlaps,
            )
        )

    query_cache = read_json(args.query_cache, {}) or {}
    query_by_record_tariff: dict[tuple[str, str], tuple[list[str], str]] = {}
    pages_by_pdf: dict[str, dict[int, str]] = {}

    for record in records:
        for tariff in record.get("target_tariffs", []):
            tariff = str(tariff)
            queries, source = llm_query_expansion(
                record,
                tariff,
                model=args.query_model,
                cache=query_cache,
            )
            query_by_record_tariff[(str(record.get("code_path")), tariff)] = (queries, source)
    write_json(args.query_cache, query_cache)

    experiments = []
    for exp_index, (chunk_mode, chunk_size, chunk_overlap) in enumerate(parameter_sets, 1):
        print(
            f"\n[{exp_index}/{len(parameter_sets)}] "
            f"chunk_mode={chunk_mode} chunk_size={chunk_size} overlap={chunk_overlap} "
            f"candidate_k={args.candidate_k} top_k={args.top_k} dense_weight={args.dense_weight} "
            f"reranker={args.reranker}",
            flush=True,
        )
        results = []
        for record_index, record in enumerate(records, 1):
            pdf_file = record.get("pdf_file")
            if not pdf_file:
                continue
            if pdf_file not in pages_by_pdf:
                pages_by_pdf[pdf_file] = run_ocr_pipeline(pdf_file)

            for tariff in record.get("target_tariffs", []):
                tariff = str(tariff)
                queries, query_source = query_by_record_tariff[(str(record.get("code_path")), tariff)]
                print(
                    f"  [{record_index}/{len(records)}] {tariff} source={query_source}",
                    flush=True,
                )
                result = evaluate_tariff(
                    record,
                    tariff,
                    pages=pages_by_pdf[pdf_file],
                    queries=queries,
                    query_source=query_source,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                    dense_weight=args.dense_weight,
                    chunk_mode=chunk_mode,
                    chunk_size_tokens=chunk_size,
                    chunk_overlap_tokens=chunk_overlap,
                    embedding_model=args.embedding_model,
                    reranker=args.reranker,
                    rerank_model=args.rerank_model,
                    rerank_context_chars=args.rerank_context_chars,
                    rerank_chunks_per_page=args.rerank_chunks_per_page,
                )
                results.append(result)
                print(
                    "    "
                    f"expected={result['expected_pages']} retrieved={result['retrieved_pages']} "
                    f"matched={result['matched_pages']} recall={result['recall']} precision={result['precision']}",
                    flush=True,
                )

        experiment = {
            "parameters": {
                "chunk_mode": chunk_mode,
                "chunk_size_tokens": chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
                "candidate_k": args.candidate_k,
                "top_k": args.top_k,
                "dense_weight": args.dense_weight,
                "reranker": args.reranker,
                "rerank_model": args.rerank_model,
                "rerank_context_chars": args.rerank_context_chars,
                "rerank_chunks_per_page": args.rerank_chunks_per_page,
                "embedding_model": args.embedding_model,
                "query_model": args.query_model,
            },
            "summary": summarize_results(results),
            "results": results,
        }
        experiments.append(experiment)
        print(json.dumps(experiment["summary"], ensure_ascii=False, indent=2), flush=True)
        write_json(
            args.output,
            {
                "settings": vars(args),
                "experiment_count": len(parameter_sets),
                "completed_experiment_count": len(experiments),
                "experiments": experiments,
                "best_experiments": sorted(experiments, key=experiment_sort_key, reverse=True)[:10],
            },
        )

    report = {
        "settings": vars(args),
        "experiment_count": len(parameter_sets),
        "completed_experiment_count": len(experiments),
        "experiments": experiments,
        "best_experiments": sorted(experiments, key=experiment_sort_key, reverse=True)[:10],
    }
    write_json(args.output, report)

    print("\n=== Best RAG Experiments ===")
    for index, experiment in enumerate(report["best_experiments"], 1):
        print(f"{index}. {json.dumps(experiment['parameters'], ensure_ascii=False)}")
        print(f"   {json.dumps(experiment['summary'], ensure_ascii=False)}")
    print(f"\nSaved report to: {args.output}")


if __name__ == "__main__":
    main()
