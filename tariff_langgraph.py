from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import time
import traceback
from typing import Any, Literal, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph

from pdf_pipeline import run_ocr_pipeline, search_rag_pages, summarize_pages
from solver import evalute_results_tool, normalize_candidate_code

load_dotenv()

RAW_CLAUDE_MODEL_ID = os.getenv("TARIFF_GRAPH_CLAUDE_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_MODEL_ALIASES = {
    "claude-sonnet 4.6": "claude-sonnet-4-20250514",
    "claude sonnet 4.6": "claude-sonnet-4-20250514",
    "claude-sonnet-4.6": "claude-sonnet-4-20250514",
    "claude-sonnet 4": "claude-sonnet-4-20250514",
    "claude sonnet 4": "claude-sonnet-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-sonnet-4-0": "claude-sonnet-4-0",
}
CLAUDE_MODEL_ID = ANTHROPIC_MODEL_ALIASES.get(
    RAW_CLAUDE_MODEL_ID.strip().lower(),
    RAW_CLAUDE_MODEL_ID.strip(),
)
GEMINI_MODEL_ID = os.getenv("TARIFF_GRAPH_GEMINI_MODEL", "gemini-2.5-flash")
OUTPUT_CODE_PREFIX = "successful_tariff_calculator"
LAST_CODE_PREFIX = "last_tariff_calculator"
CALCULATOR_REGISTRY_PATH = "calculators_registry.json"
MAX_CODE_ATTEMPTS = 2
MAX_RETRIEVAL_ROUNDS = 3
RETRIEVAL_MODE = os.getenv("TARIFF_RETRIEVAL_MODE", "summaries").strip().lower()
RAG_TOP_K = int(os.getenv("TARIFF_RAG_TOP_K", "12"))
RAG_DENSE_WEIGHT = float(os.getenv("TARIFF_RAG_DENSE_WEIGHT", "0.65"))
VERBOSE_LOGS = os.getenv("TARIFF_GRAPH_VERBOSE", "1").lower() not in {"0", "false", "no"}
LOG_PROMPTS = os.getenv("TARIFF_GRAPH_LOG_PROMPTS", "0").lower() in {"1", "true", "yes"}
LOG_FILE: str | None = None
ACTIVE_LLM_PROVIDER: Literal["gemini", "anthropic"] = "gemini"
_LLM_CALL_COUNT = 0
LLM_MAX_RETRIES = 5
LLM_RETRY_BASE_SECONDS = 10
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("Gemini_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def use_rag_retrieval() -> bool:
    return RETRIEVAL_MODE in {"rag", "hybrid_rag", "hybrid-rag", "embeddings"}

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


class GraphState(TypedDict, total=False):
    vessel_data: dict[str, Any]
    enriched_input_data: dict[str, Any]
    page_summaries: dict[int, str]
    pages: dict[int, str]
    pdf_file: str

    target_tariffs: list[str]
    tariff_queries: dict[str, list[str]]

    candidate_pages: dict[str, list[int]]
    pages_to_inspect: list[int]
    inspected_pages: set[int]
    discarded_pages: set[int]

    evidence_pages: dict[str, list[int]]
    extracted_evidence: dict[str, list[dict[str, Any]]]

    calculation_rules: dict[str, dict[str, Any]]
    selected_parameters: dict[str, list[str]]
    selected_parameter_reasons: dict[str, dict[str, str]]

    code: str
    results: dict[str, float]
    submit_feedback: dict[str, str]

    current_tariff: str | None
    current_page: int | None
    errors_or_ambiguities: list[str]

    code_attempts: int
    retrieval_rounds: int
    finished: bool
    existing_calculator: dict[str, Any] | None
    last_code_path: str


def log(message: str) -> None:
    if VERBOSE_LOGS:
        print(message, flush=True)
    if LOG_FILE:
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def short_json(value: Any, max_chars: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def state_summary(state: GraphState) -> str:
    target_tariffs = state.get("target_tariffs", [])
    return (
        f"tariffs={target_tariffs} | "
        f"pages={len(state.get('pages', {}))} | "
        f"summaries={len(state.get('page_summaries', {}))} | "
        f"to_inspect={len(state.get('pages_to_inspect', []))} | "
        f"inspected={len(state.get('inspected_pages', set()))} | "
        f"discarded={len(state.get('discarded_pages', set()))} | "
        f"evidence_tariffs={len(state.get('evidence_pages', {}))} | "
        f"code_attempts={state.get('code_attempts', 0)} | "
        f"retrieval_rounds={state.get('retrieval_rounds', 0)} | "
        f"finished={state.get('finished', False)}"
    )


def trace_node(name: str, fn):
    def wrapped(state: GraphState) -> GraphState:
        global ACTIVE_LLM_PROVIDER
        ACTIVE_LLM_PROVIDER = llm_provider_for_state(state)
        log(f"\n>>> NODE START: {name}")
        log(f"[{name}] llm_provider={ACTIVE_LLM_PROVIDER} model={active_model_id()}")
        log(f"[{name}] input: {state_summary(state)}")
        try:
            next_state = fn(state)
        except Exception:
            log(f"[{name}] ERROR:\n{traceback.format_exc()}")
            raise
        log(f"[{name}] output: {state_summary(next_state)}")
        log(f"<<< NODE END: {name}")
        return next_state

    return wrapped


def llm_provider_for_state(state: GraphState) -> Literal["gemini", "anthropic"]:
    if int(state.get("retrieval_rounds", 0)) == 0 and anthropic_client is not None:
        return "anthropic"
    return "gemini"


def active_model_id() -> str:
    return GEMINI_MODEL_ID if ACTIVE_LLM_PROVIDER == "gemini" else CLAUDE_MODEL_ID


def is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = [
        "503",
        "unavailable",
        "high demand",
        "rate limit",
        "rate_limit",
        "too many requests",
        "timeout",
        "temporarily",
        "overloaded",
        "internal error",
        "500",
        "502",
        "504",
    ]
    return any(marker in text for marker in retry_markers)


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "tariff"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_rules_hash(state: GraphState) -> str:
    evidence_payload = {
        "pdf_sha256": file_sha256(state["pdf_file"]) if state.get("pdf_file") else "",
        "target_tariffs": state.get("target_tariffs", []),
        "evidence_pages": state.get("evidence_pages", {}),
        "page_text": {
            str(page_number): state.get("pages", {}).get(page_number, "")
            for page_numbers in state.get("evidence_pages", {}).values()
            for page_number in page_numbers
        },
    }
    normalized = json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def output_code_path(target_tariffs: list[str], rules_hash: str) -> str:
    suffix = "_".join(safe_filename_part(tariff) for tariff in target_tariffs)
    if len(suffix) > 120:
        suffix = suffix[:120].rstrip("_")
    return f"{OUTPUT_CODE_PREFIX}_{rules_hash[:15]}_{suffix}.py"


def output_last_code_path(target_tariffs: list[str], rules_hash: str) -> str:
    suffix = "_".join(safe_filename_part(tariff) for tariff in target_tariffs)
    if len(suffix) > 120:
        suffix = suffix[:120].rstrip("_")
    return f"{LAST_CODE_PREFIX}_{rules_hash[:15]}_{suffix}.py"


def read_registry() -> list[dict[str, Any]]:
    raw = read_json_file(CALCULATOR_REGISTRY_PATH, [])
    return raw if isinstance(raw, list) else []


def register_calculator(
    state: GraphState,
    code_path: str,
    rules_hash: str,
    *,
    status: str,
    accepted: bool,
) -> None:
    registry = read_registry()
    document_hash = file_sha256(state["pdf_file"]) if state.get("pdf_file") else ""
    record = {
        "code_path": code_path,
        "rules_hash": rules_hash,
        "document_hash": document_hash,
        "document_name": os.path.basename(state.get("pdf_file", "")),
        "pdf_file": state.get("pdf_file", ""),
        "target_tariffs": state.get("target_tariffs", []),
        "evidence_pages": state.get("evidence_pages", {}),
        "status": status,
        "accepted": accepted,
        "results": state.get("results", {}),
        "submit_feedback": state.get("submit_feedback", {}),
    }

    registry = [
        item
        for item in registry
        if not (
            item.get("rules_hash") == rules_hash
            and item.get("target_tariffs") == state.get("target_tariffs", [])
            and item.get("status", "success") == status
        )
    ]
    registry.append(record)
    with open(CALCULATOR_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def register_successful_calculator(state: GraphState, code_path: str, rules_hash: str) -> None:
    register_calculator(
        state,
        code_path,
        rules_hash,
        status="success",
        accepted=True,
    )


def register_last_calculator(state: GraphState, code_path: str, rules_hash: str) -> None:
    register_calculator(
        state,
        code_path,
        rules_hash,
        status="last_attempt",
        accepted=False,
    )


def find_existing_calculator(document_hash: str, target_tariffs: list[str]) -> dict[str, Any] | None:
    requested = set(target_tariffs)
    for record in read_registry():
        if record.get("status", "success") != "success":
            continue
        if record.get("document_hash") != document_hash:
            continue
        if requested <= set(record.get("target_tariffs", [])):
            code_path = record.get("code_path", "")
            if code_path and os.path.exists(code_path):
                return record
    return None


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def llm_text(prompt: str, *, json_mode: bool = False) -> str:
    global _LLM_CALL_COUNT
    _LLM_CALL_COUNT += 1
    provider = ACTIVE_LLM_PROVIDER
    model_id = active_model_id()
    log(
        f"[LLM #{_LLM_CALL_COUNT}] provider={provider} model={model_id} json_mode={json_mode} "
        f"prompt_chars={len(prompt)}"
    )
    if LOG_PROMPTS:
        log(f"[LLM #{_LLM_CALL_COUNT}] prompt:\n{prompt}")

    last_error: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return _llm_text_once(
                prompt=prompt,
                json_mode=json_mode,
                provider=provider,
                model_id=model_id,
                call_number=_LLM_CALL_COUNT,
            )
        except Exception as exc:
            last_error = exc
            if not is_retryable_llm_error(exc) or attempt >= LLM_MAX_RETRIES:
                log(f"[LLM #{_LLM_CALL_COUNT}] failed attempt {attempt}/{LLM_MAX_RETRIES}: {exc}")
                raise
            wait_seconds = LLM_RETRY_BASE_SECONDS * attempt
            log(
                f"[LLM #{_LLM_CALL_COUNT}] retryable error attempt {attempt}/{LLM_MAX_RETRIES}: {exc}. "
                f"Waiting {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"LLM failed after retries: {last_error}")


def _llm_text_once(
    *,
    prompt: str,
    json_mode: bool,
    provider: Literal["gemini", "anthropic"],
    model_id: str,
    call_number: int,
) -> str:

    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini API key is required. Set Gemini_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY.")
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json" if json_mode else None,
        )
        response = gemini_client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=config,
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        log(f"[LLM #{call_number}] response_chars={len(text)}")
        if LOG_PROMPTS:
            log(f"[LLM #{call_number}] response:\n{text}")
        return text

    if anthropic_client is None:
        raise RuntimeError("Anthropic API key is not configured; use Gemini instead.")

    system_prompt = (
        "You are a precise tariff-calculation assistant. "
        "Return only valid JSON, with no Markdown, when the user asks for JSON."
        if json_mode
        else "You are a precise tariff-calculation assistant."
    )
    response = anthropic_client.messages.create(
        model=model_id,
        max_tokens=4000,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    text = "\n".join(text_parts).strip()
    if not text:
        raise RuntimeError("Anthropic returned an empty response")
    log(f"[LLM #{call_number}] response_chars={len(text)}")
    if LOG_PROMPTS:
        log(f"[LLM #{call_number}] response:\n{text}")
    return text


def llm_json(prompt: str) -> Any:
    return extract_json(llm_text(prompt, json_mode=True))


def get_page_by_number(state: GraphState, page_number: int) -> str:
    return state.get("pages", {}).get(page_number, "")


def normalize_useful_map(value: Any, related_tariffs: list[str]) -> dict[str, bool]:
    if isinstance(value, dict):
        return {str(k): bool(v) for k, v in value.items()}
    if isinstance(value, list):
        useful = {}
        for item in value:
            if isinstance(item, str):
                useful[item] = True
            elif isinstance(item, dict):
                tariff = item.get("tariff") or item.get("tariff_name") or item.get("name")
                if tariff:
                    useful[str(tariff)] = bool(item.get("useful", True))
        return useful
    if isinstance(value, bool) and len(related_tariffs) == 1:
        return {related_tariffs[0]: value}
    return {}


def normalize_evidence_map(value: Any, related_tariffs: list[str], page_number: int) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, dict):
        normalized = {}
        for tariff, evidence_items in value.items():
            if isinstance(evidence_items, list):
                normalized[str(tariff)] = [
                    item if isinstance(item, dict) else {"page": page_number, "quote_or_summary": str(item)}
                    for item in evidence_items
                ]
            else:
                normalized[str(tariff)] = [{"page": page_number, "quote_or_summary": str(evidence_items)}]
        return normalized

    if isinstance(value, list):
        if len(related_tariffs) == 1:
            return {
                related_tariffs[0]: [
                    item if isinstance(item, dict) else {"page": page_number, "quote_or_summary": str(item)}
                    for item in value
                ]
            }

        normalized: dict[str, list[dict[str, Any]]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            tariff = item.get("tariff") or item.get("tariff_name") or item.get("name")
            if tariff:
                normalized.setdefault(str(tariff), []).append(item)
        return normalized

    if isinstance(value, str) and len(related_tariffs) == 1:
        return {related_tariffs[0]: [{"page": page_number, "quote_or_summary": value}]}

    return {}


def get_default_pdf() -> str:
    pdf_dir = "pdf_data"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in pdf_data")
    return os.path.join(pdf_dir, pdf_files[0])


def load_inputs(state: GraphState) -> GraphState:
    pdf_file = state.get("pdf_file") or get_default_pdf()
    log(f"[LoadInputs] pdf_file={pdf_file}")
    log(f"[LoadInputs] retrieval_mode={RETRIEVAL_MODE}")
    document_hash = file_sha256(pdf_file)
    pages = run_ocr_pipeline(pdf_file)
    log(f"[LoadInputs] OCR pages loaded: {len(pages)}")
    if use_rag_retrieval():
        page_summaries = {}
        log("[LoadInputs] page summaries skipped for RAG retrieval")
    else:
        summary_result = summarize_pages(pdf_file, pages)
        page_summaries = {
            int(summary["page_number"]): summary.get("plain_summary", "")
            for summary in summary_result.get("summaries", [])
        }
        log(f"[LoadInputs] page summaries loaded: {len(page_summaries)}")

    vessel_data = state.get("vessel_data") or read_json_file("input_param.json", {})
    expected = read_json_file("input.json", {})
    target_tariffs = state.get("target_tariffs") or list(expected.keys())
    log(f"[LoadInputs] target_tariffs={target_tariffs}")
    log(f"[LoadInputs] vessel_data keys={list(vessel_data.keys())}")
    existing_calculator = find_existing_calculator(document_hash, target_tariffs)
    if existing_calculator:
        log(
            "[LoadInputs] existing calculator found: "
            f"{existing_calculator.get('code_path')} "
            f"rules_hash={existing_calculator.get('rules_hash', '')[:15]}"
        )

    return {
        **state,
        "pdf_file": pdf_file,
        "pages": pages,
        "page_summaries": page_summaries,
        "vessel_data": vessel_data,
        "target_tariffs": target_tariffs,
        "candidate_pages": {},
        "pages_to_inspect": [],
        "inspected_pages": set(),
        "discarded_pages": set(),
        "evidence_pages": {},
        "extracted_evidence": {},
        "enriched_input_data": {},
        "calculation_rules": {},
        "selected_parameters": {},
        "selected_parameter_reasons": {},
        "submit_feedback": {},
        "errors_or_ambiguities": [],
        "code_attempts": 0,
        "retrieval_rounds": 0,
        "finished": bool(existing_calculator),
        "existing_calculator": existing_calculator,
    }


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


def expand_tariff_queries(state: GraphState) -> GraphState:
    prompt = {
        "task": (
            "Expand each tariff name into search queries for a port tariff PDF. "
            "Include exact name, synonyms, rate/formula/tariff/charges wording, "
            "questions about parameters, probable parameters GT/LOA/NT/DWT/days/movements, "
            "and domain synonyms."
        ),
        "target_tariffs": state["target_tariffs"],
        "output_schema": {"tariff name": ["query", "..."]},
    }
    try:
        expanded = llm_json(json.dumps(prompt, ensure_ascii=False))
    except Exception as exc:
        expanded = {}
        state.setdefault("errors_or_ambiguities", []).append(f"ExpandTariffQueries fallback: {exc}")

    tariff_queries = {}
    for tariff in state["target_tariffs"]:
        llm_queries = expanded.get(tariff, []) if isinstance(expanded, dict) else []
        queries = deterministic_query_expansion(tariff) + [str(q) for q in llm_queries]
        tariff_queries[tariff] = list(dict.fromkeys(q for q in queries if q.strip()))
        log(f"[ExpandTariffQueries] {tariff}: {len(tariff_queries[tariff])} queries")
        log(short_json(tariff_queries[tariff], max_chars=900))

    return {**state, "tariff_queries": tariff_queries}


def summaries_as_numbered_text(page_summaries: dict[int, str]) -> str:
    return "\n\n".join(
        f"Page {page_number}: {summary}"
        for page_number, summary in sorted(page_summaries.items())
    )


def lexical_candidate_pages(queries: list[str], page_summaries: dict[int, str], limit: int = 12) -> list[int]:
    terms = set()
    for query in queries:
        terms.update(re.findall(r"[A-Za-z0-9]+", query.lower()))
    terms = {t for t in terms if len(t) > 2}
    scored = []
    for page_number, summary in page_summaries.items():
        text = summary.lower()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, page_number))
    return [page for _, page in sorted(scored, reverse=True)[:limit]]


def lexical_page_text_candidate_pages(queries: list[str], pages: dict[int, str], limit: int = 12) -> list[int]:
    terms = set()
    for query in queries:
        terms.update(re.findall(r"[A-Za-z0-9]+", query.lower()))
    terms = {t for t in terms if len(t) > 2}
    scored = []
    for page_number, page_text in pages.items():
        text = page_text.lower()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, page_number))
    return [page for _, page in sorted(scored, reverse=True)[:limit]]


def retrieve_candidate_pages(state: GraphState) -> GraphState:
    prompt = f"""
You select candidate pages from page summaries for tariff calculation.

Return only JSON in this schema:
{{"candidate_pages": {{"Tariff Name": [1, 2, 3]}}}}

Target tariffs and expanded queries:
{json.dumps(state["tariff_queries"], ensure_ascii=False, indent=2)}

Page summaries:
{summaries_as_numbered_text(state["page_summaries"])}
""".strip()
    try:
        raw = llm_json(prompt)
        candidate_pages = raw.get("candidate_pages", raw) if isinstance(raw, dict) else {}
    except Exception as exc:
        candidate_pages = {}
        state.setdefault("errors_or_ambiguities", []).append(f"RetrieveCandidatePages fallback: {exc}")

    normalized: dict[str, list[int]] = {}
    all_pages: set[int] = set()
    for tariff, queries in state["tariff_queries"].items():
        pages = candidate_pages.get(tariff, []) if isinstance(candidate_pages, dict) else []
        pages = [int(p) for p in pages if str(p).isdigit()]
        if not pages:
            pages = lexical_candidate_pages(queries, state["page_summaries"])
        normalized[tariff] = sorted(set(pages))
        all_pages.update(normalized[tariff])
        log(f"[RetrieveCandidatePages] {tariff}: pages={normalized[tariff]}")

    log(f"[RetrieveCandidatePages] pages_to_inspect={sorted(all_pages)}")
    return {
        **state,
        "candidate_pages": normalized,
        "pages_to_inspect": sorted(all_pages),
    }


def retrieve_candidate_pages_rag(state: GraphState) -> GraphState:
    normalized: dict[str, list[int]] = {}
    all_pages: set[int] = set()

    for tariff, queries in state["tariff_queries"].items():
        query = "\n".join(queries)
        try:
            results = search_rag_pages(
                pdf_path=state["pdf_file"],
                query=query,
                top_k=RAG_TOP_K,
                pages=state["pages"],
                include_text=False,
                dense_weight=RAG_DENSE_WEIGHT,
            )
            pages = [int(item["page_number"]) for item in results]
            scores = {
                int(item["page_number"]): {
                    "hybrid": round(float(item["score"]), 4),
                    "dense": round(float(item["dense_score"]), 4),
                    "bm25": round(float(item["bm25_score"]), 4),
                }
                for item in results
            }
            log(f"[RetrieveCandidatePagesRag] {tariff}: pages={pages} scores={scores}")
        except Exception as exc:
            pages = lexical_page_text_candidate_pages(queries, state.get("pages", {}))
            state.setdefault("errors_or_ambiguities", []).append(f"RetrieveCandidatePagesRag fallback: {exc}")
            log(f"[RetrieveCandidatePagesRag] {tariff}: fallback pages={pages} error={exc}")

        normalized[tariff] = sorted(set(pages))
        all_pages.update(normalized[tariff])

    log(f"[RetrieveCandidatePagesRag] pages_to_inspect={sorted(all_pages)}")
    return {
        **state,
        "candidate_pages": normalized,
        "pages_to_inspect": sorted(all_pages),
    }


def inspect_pages(state: GraphState) -> GraphState:
    inspected = set(state.get("inspected_pages", set()))
    discarded = set(state.get("discarded_pages", set()))
    evidence_pages = {k: list(v) for k, v in state.get("evidence_pages", {}).items()}
    extracted_evidence = {k: list(v) for k, v in state.get("extracted_evidence", {}).items()}

    for page_number in state.get("pages_to_inspect", []):
        if page_number in inspected:
            log(f"[InspectPages] skip already inspected page={page_number}")
            continue
        inspected.add(page_number)
        page_text = get_page_by_number(state, page_number)
        log(f"[InspectPages] inspecting page={page_number} chars={len(page_text)}")
        if not page_text.strip():
            log(f"[InspectPages] page={page_number} empty, discarded")
            discarded.add(page_number)
            continue

        related_tariffs = [
            tariff
            for tariff, pages in state.get("candidate_pages", {}).items()
            if page_number in pages
        ] or state["target_tariffs"]
        log(f"[InspectPages] page={page_number} related_tariffs={related_tariffs}")
        prompt = f"""
You inspect one OCR page from a port tariff PDF.

For each tariff, decide whether the page is useful for calculating it.
Useful means it contains rates, formulas, thresholds, definitions, units, rounding rules,
exceptions, table headers, or continuation text needed for calculation.

Return only JSON:
{{
  "useful": {{"Tariff Name": true}},
  "evidence": {{"Tariff Name": [{{"page": {page_number}, "quote_or_summary": "...", "why_useful": "..."}}]}},
  "discard_reason": "..."
}}

Tariffs to check:
{json.dumps(related_tariffs, ensure_ascii=False)}

Expanded queries:
{json.dumps({t: state["tariff_queries"].get(t, []) for t in related_tariffs}, ensure_ascii=False, indent=2)}

Page {page_number} OCR text:
{page_text}
""".strip()
        try:
            decision = llm_json(prompt)
        except Exception as exc:
            discarded.add(page_number)
            state.setdefault("errors_or_ambiguities", []).append(f"InspectPages page {page_number}: {exc}")
            log(f"[InspectPages] page={page_number} failed and discarded: {exc}")
            continue

        raw_useful = decision.get("useful", {}) if isinstance(decision, dict) else {}
        raw_evidence = decision.get("evidence", {}) if isinstance(decision, dict) else {}
        useful = normalize_useful_map(raw_useful, related_tariffs)
        evidence = normalize_evidence_map(raw_evidence, related_tariffs, page_number)
        if not isinstance(raw_evidence, dict):
            log(f"[InspectPages] page={page_number} normalized evidence from {type(raw_evidence).__name__}")
        page_was_useful = False
        for tariff in related_tariffs:
            if useful.get(tariff):
                page_was_useful = True
                evidence_pages.setdefault(tariff, [])
                if page_number not in evidence_pages[tariff]:
                    evidence_pages[tariff].append(page_number)
                extracted_evidence.setdefault(tariff, [])
                extracted_evidence[tariff].extend(evidence.get(tariff, []))
                log(f"[InspectPages] page={page_number} useful for {tariff}")
        if not page_was_useful:
            discarded.add(page_number)
            log(f"[InspectPages] page={page_number} not useful, discarded")

    return {
        **state,
        "inspected_pages": inspected,
        "discarded_pages": discarded,
        "evidence_pages": {k: sorted(set(v)) for k, v in evidence_pages.items()},
        "extracted_evidence": extracted_evidence,
    }


def parse_iso_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def collect_datetime_fields(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.update(collect_datetime_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            found.update(collect_datetime_fields(item, path))
    elif parse_iso_datetime(value):
        found[prefix] = str(value)
    return found


def describe_datetime(value: str, standard_work_start: int = 8, standard_work_end: int = 17) -> dict[str, Any]:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return {"raw": value}
    is_weekend = parsed.weekday() >= 5
    is_standard_working_hours = (
        not is_weekend
        and standard_work_start <= parsed.hour < standard_work_end
    )
    return {
        "raw": value,
        "date": parsed.date().isoformat(),
        "time": parsed.time().replace(microsecond=0).isoformat(),
        "weekday": parsed.strftime("%A"),
        "weekday_number": parsed.isoweekday(),
        "is_weekend": is_weekend,
        "is_standard_workday": not is_weekend,
        "is_standard_working_hours": is_standard_working_hours,
        "standard_working_hours_assumption": f"Monday-Friday {standard_work_start:02d}:00-{standard_work_end:02d}:00 local port time",
    }


def deterministic_input_context(vessel_data: dict[str, Any]) -> dict[str, Any]:
    datetime_fields = collect_datetime_fields(vessel_data)
    return {
        "domain_context": {
            "subject": "vessel",
            "operation": "vessel departing from a port",
            "port": vessel_data.get("port"),
            "default_time_basis": "local port time",
        },
        "date_time_context": {
            path: describe_datetime(value)
            for path, value in sorted(datetime_fields.items())
        },
        "assumptions": [
            "Use standard working time when the tariff text does not define special working hours.",
            "Default standard working time is Monday-Friday 08:00-17:00 in local port time.",
            "Weekend dates are treated as non-standard working days by default.",
        ],
    }


def enrich_input_data(state: GraphState) -> GraphState:
    vessel_data = state.get("vessel_data", {})
    deterministic_context = deterministic_input_context(vessel_data)
    prompt = f"""
Enrich the input data for a port tariff calculation.

The data describes a vessel departing from a port. Add concise descriptive context that may
help decide which tariff parameters matter. Preserve the original values; do not invent
missing vessel measurements or rates.

For any date/time fields, include the weekday and whether the moment is within standard
working time. Unless the tariff document says otherwise, use this default standard working
time: Monday-Friday 08:00-17:00 local port time.

Return only JSON:
{{
  "enriched_vessel_data": {{}},
  "derived_context": {{}},
  "calculation_relevance_notes": [],
  "assumptions": []
}}

Original vessel_data:
{json.dumps(vessel_data, ensure_ascii=False, indent=2)}

Deterministic derived context to include and refine:
{json.dumps(deterministic_context, ensure_ascii=False, indent=2)}
""".strip()
    try:
        raw = llm_json(prompt)
        enriched = raw if isinstance(raw, dict) else {}
    except Exception as exc:
        enriched = {
            "enriched_vessel_data": vessel_data,
            "derived_context": deterministic_context,
            "calculation_relevance_notes": [],
            "assumptions": deterministic_context["assumptions"],
        }
        state.setdefault("errors_or_ambiguities", []).append(f"EnrichInputData fallback: {exc}")
        log(f"[EnrichInputData] failed, using deterministic context: {exc}")

    enriched.setdefault("enriched_vessel_data", vessel_data)
    enriched.setdefault("derived_context", deterministic_context)
    log(f"[EnrichInputData] enriched={short_json(enriched, max_chars=1400)}")
    return {**state, "enriched_input_data": enriched}


def extract_rules(state: GraphState) -> GraphState:
    calculation_rules: dict[str, dict[str, Any]] = {}
    for tariff in state["target_tariffs"]:
        pages = state.get("evidence_pages", {}).get(tariff, [])
        log(f"[ExtractRules] {tariff}: evidence_pages={pages}")
        page_bundle = {
            page: get_page_by_number(state, page)
            for page in pages
        }
        prompt = f"""
Extract calculation rules for this tariff from evidence pages.

Use the selected calculation factors and enriched vessel context to focus on rules that can
affect this specific vessel's result, but do not invent rates or formulas that are absent
from the evidence pages.

Return only JSON:
{{
  "tariff": "{tariff}",
  "rules": [],
  "formula": "",
  "rates": [],
  "parameters_used": [],
  "units": [],
  "rounding": "",
  "minimums_maximums": [],
  "exceptions": [],
  "ambiguities": []
}}

Existing evidence summaries:
{json.dumps(state.get("extracted_evidence", {}).get(tariff, []), ensure_ascii=False, indent=2)}

Selected calculation factors for this tariff:
{json.dumps(state.get("selected_parameters", {}).get(tariff, []), ensure_ascii=False, indent=2)}

Enriched vessel context:
{json.dumps(state.get("enriched_input_data", {}), ensure_ascii=False, indent=2)}

Evidence pages OCR:
{json.dumps(page_bundle, ensure_ascii=False, indent=2)}
""".strip()
        try:
            calculation_rules[tariff] = llm_json(prompt)
        except Exception as exc:
            calculation_rules[tariff] = {
                "tariff": tariff,
                "rules": [],
                "ambiguities": [f"ExtractRules failed: {exc}"],
            }
            state.setdefault("errors_or_ambiguities", []).append(f"ExtractRules {tariff}: {exc}")
            log(f"[ExtractRules] {tariff}: failed: {exc}")
        else:
            rule = calculation_rules[tariff]
            log(
                f"[ExtractRules] {tariff}: rules={len(rule.get('rules', []))} "
                f"rates={len(rule.get('rates', []))} "
                f"params={rule.get('parameters_used', [])}"
            )

    return {**state, "calculation_rules": calculation_rules}


def select_calculation_parameters(state: GraphState) -> GraphState:
    evidence_pages = {}
    for tariff in state["target_tariffs"]:
        evidence_pages[tariff] = {
            page: get_page_by_number(state, page)
            for page in state.get("evidence_pages", {}).get(tariff, [])
        }
    prompt = f"""
Select which vessel/input parameters and derived context may influence each tariff.

This runs before rule extraction. Use the enriched input data plus inspected evidence to
identify the data points that could affect the result, such as tonnage, LOA, cargo quantity,
arrival/departure timestamps, weekday, standard working time/non-working time, number of
operations, port, activity, vessel type, or other context visible in the inputs/evidence.

Return only JSON:
{{
  "selected_parameters": {{"Tariff Name": ["parameter_or_context_path"]}},
  "missing_parameters": {{"Tariff Name": ["parameter_name"]}},
  "why_relevant": {{"Tariff Name": {{"parameter_or_context_path": "short reason"}}}}
}}

Original vessel_data from input_param.json:
{json.dumps(state.get("vessel_data", {}), ensure_ascii=False, indent=2)}

Enriched input data:
{json.dumps(state.get("enriched_input_data", {}), ensure_ascii=False, indent=2)}

Evidence summaries:
{json.dumps(state.get("extracted_evidence", {}), ensure_ascii=False, indent=2)}

Evidence pages OCR:
{json.dumps(evidence_pages, ensure_ascii=False, indent=2)}
""".strip()
    try:
        raw = llm_json(prompt)
        selected = raw.get("selected_parameters", {})
        missing = raw.get("missing_parameters", {})
        reasons = raw.get("why_relevant", {})
    except Exception as exc:
        selected = {}
        missing = {}
        reasons = {}
        state.setdefault("errors_or_ambiguities", []).append(f"SelectCalculationParameters: {exc}")
        log(f"[SelectCalculationParameters] failed: {exc}")

    ambiguities = list(state.get("errors_or_ambiguities", []))
    for tariff, params in missing.items() if isinstance(missing, dict) else []:
        if params:
            ambiguities.append(f"{tariff} missing parameters: {params}")
            log(f"[SelectCalculationParameters] {tariff}: missing={params}")

    log(f"[SelectCalculationParameters] selected={short_json(selected, max_chars=1200)}")
    return {
        **state,
        "selected_parameters": selected,
        "selected_parameter_reasons": reasons,
        "errors_or_ambiguities": ambiguities,
    }


def write_code(state: GraphState) -> GraphState:
    feedback = state.get("submit_feedback", {})
    next_attempt = int(state.get("code_attempts", 0)) + 1
    log(f"[WriteCode] generating code attempt {next_attempt}/{MAX_CODE_ATTEMPTS}")
    if feedback:
        log(f"[WriteCode] previous feedback={short_json(feedback, max_chars=1500)}")
    prompt = f"""
Write a complete Python module that calculates the requested tariff values.

Hard requirements:
- Output only Python code.
- Define calculate(vessel_data: dict) -> dict[str, float].
- The result keys must exactly match: {state["target_tariffs"]}.
- Use only the standard library.
- If a needed parameter is missing, use a clearly named fallback only when the rules justify it.
- Do not infer rates from expected answers, previous calculated outputs, percent differences, or evaluation feedback.
- Do not hardcode final tariff answers.
- Do not read files and do not ask for input.

Vessel data:
{json.dumps(state.get("vessel_data", {}), ensure_ascii=False, indent=2)}

Enriched input context:
{json.dumps(state.get("enriched_input_data", {}), ensure_ascii=False, indent=2)}

Calculation rules:
{json.dumps(state.get("calculation_rules", {}), ensure_ascii=False, indent=2)}

Selected parameters:
{json.dumps(state.get("selected_parameters", {}), ensure_ascii=False, indent=2)}

Why selected parameters may matter:
{json.dumps(state.get("selected_parameter_reasons", {}), ensure_ascii=False, indent=2)}

Previous run feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}
""".strip()
    code = normalize_candidate_code(llm_text(prompt))
    log(f"[WriteCode] generated code chars={len(code)}")
    log("[WriteCode] generated code preview:\n" + code[:1600] + ("\n... <truncated>" if len(code) > 1600 else ""))
    return {
        **state,
        "code": code,
        "code_attempts": next_attempt,
    }


def run_generated_code(code: str, vessel_data: dict[str, Any]) -> dict[str, Any]:
    log(f"[run_generated_code] loading generated module chars={len(code)}")
    with tempfile.TemporaryDirectory() as d:
        file_path = os.path.join(d, "candidate_tariff_calculator.py")
        write_text(file_path, code)
        spec = importlib.util.spec_from_file_location("candidate_tariff_calculator", file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load generated module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calculate = getattr(module, "calculate", None)
        if not callable(calculate):
            raise RuntimeError("Generated code must define calculate(vessel_data: dict) -> dict[str, float]")
        log(f"[run_generated_code] calling calculate with vessel_data keys={list(vessel_data.keys())}")
        result = calculate(vessel_data)
        if not isinstance(result, dict):
            raise RuntimeError(f"calculate returned {type(result).__name__}, expected dict")
        log(f"[run_generated_code] raw result={short_json(result, max_chars=1200)}")
        return result


def sanitize_evaluation_for_agent(evaluation: dict[str, Any]) -> dict[str, Any]:
    sanitized_results: dict[str, Any] = {}
    for key, item in evaluation.get("results", {}).items():
        if not isinstance(item, dict):
            continue
        sanitized_results[key] = {
            "ok": item.get("ok"),
            "key": item.get("key"),
            "result": item.get("result"),
        }

    return {
        "ok": evaluation.get("ok"),
        "results": sanitized_results,
        "errors": evaluation.get("errors", {}),
    }


def evaluation_is_accepted(evaluation: dict[str, Any]) -> bool:
    accepted_labels = {"within_1_percent", "within_5_percent"}
    result_labels = [
        item.get("result")
        for item in evaluation.get("results", {}).values()
        if isinstance(item, dict)
    ]
    return (
        evaluation.get("ok") is True
        and bool(result_labels)
        and all(label in accepted_labels for label in result_labels)
    )


def run_and_submit(state: GraphState) -> GraphState:
    log("[RunAndSubmit] starting generated code execution")
    try:
        raw_results = run_generated_code(state["code"], state.get("vessel_data", {}))
        results = {str(k): float(v) for k, v in raw_results.items()}
        evaluation = evalute_results_tool(results)
        sanitized_evaluation = sanitize_evaluation_for_agent(evaluation)
        feedback = {
            "execution": "ok",
            "evaluation": json.dumps(sanitized_evaluation, ensure_ascii=False),
        }
        log(f"[RunAndSubmit] submit results={short_json(results, max_chars=1200)}")
        log(f"[RunAndSubmit] evaluation={short_json(sanitized_evaluation, max_chars=2000)}")
        if not evaluation_is_accepted(evaluation):
            log("[RunAndSubmit] submitted code was not accepted. Full code:")
            log(state.get("code", ""))
    except Exception:
        results = {}
        feedback = {"execution": "failed", "error": traceback.format_exc()}
        log(f"[RunAndSubmit] execution failed:\n{feedback['error']}")
        log("[RunAndSubmit] failed submitted code:")
        log(state.get("code", ""))

    return {**state, "results": results, "submit_feedback": feedback}


def repair_or_finish(state: GraphState) -> GraphState:
    feedback = state.get("submit_feedback", {})
    evaluation_text = feedback.get("evaluation", "")
    try:
        evaluation = json.loads(evaluation_text) if evaluation_text else {}
    except json.JSONDecodeError:
        evaluation = {}
    result_labels = [
        item.get("result")
        for item in evaluation.get("results", {}).values()
        if isinstance(item, dict)
    ]
    accepted_labels = {"within_1_percent", "within_5_percent"}
    success = feedback.get("execution") == "ok" and evaluation_is_accepted(evaluation)
    all_targets_present = set(state.get("target_tariffs", [])) <= set(state.get("results", {}))
    log(
        "[RepairOrFinish] "
        f"success={success} all_targets_present={all_targets_present} "
        f"accepted_labels={sorted(accepted_labels)} labels={result_labels} "
        f"attempts={state.get('code_attempts', 0)} "
        f"retrieval_rounds={state.get('retrieval_rounds', 0)}"
    )

    if success and all_targets_present:
        rules_hash = source_rules_hash(state)
        path = output_code_path(state.get("target_tariffs", []), rules_hash)
        write_text(path, state["code"])
        register_successful_calculator(state, path, rules_hash)
        log(f"[RepairOrFinish] saved successful code to {path}")
        log(f"[RepairOrFinish] registered calculator rules_hash={rules_hash[:15]}")
        return {**state, "finished": True}

    if int(state.get("code_attempts", 0)) >= MAX_CODE_ATTEMPTS:
        retrieval_rounds = int(state.get("retrieval_rounds", 0)) + 1
        if retrieval_rounds >= MAX_RETRIEVAL_ROUNDS and state.get("code"):
            rules_hash = source_rules_hash(state)
            path = output_last_code_path(state.get("target_tariffs", []), rules_hash)
            write_text(path, state["code"])
            register_last_calculator(state, path, rules_hash)
            log(
                f"[RepairOrFinish] max attempts exhausted without accepted range; "
                f"saved last generated code to {path}"
            )
            log(f"[RepairOrFinish] registered last attempt rules_hash={rules_hash[:15]}")
            return {
                **state,
                "retrieval_rounds": retrieval_rounds,
                "last_code_path": path,
                "finished": False,
            }

        expanded_pages = set(state.get("pages_to_inspect", []))
        for page in list(expanded_pages):
            expanded_pages.update([page - 1, page + 1])
        existing_pages = set(state.get("page_summaries", {}).keys()) or set(state.get("pages", {}).keys())
        expanded_pages = {page for page in expanded_pages if page in existing_pages}
        log(
            f"[RepairOrFinish] max code attempts reached; "
            f"retrieval_rounds={retrieval_rounds}; new pages_to_inspect={sorted(expanded_pages - set(state.get('inspected_pages', set())))}"
        )
        return {
            **state,
            "code_attempts": 0,
            "retrieval_rounds": retrieval_rounds,
            "pages_to_inspect": sorted(expanded_pages - set(state.get("inspected_pages", set()))),
            "finished": False,
        }

    return {**state, "finished": False}


def route_after_repair(state: GraphState) -> Literal["finish", "inspect_pages", "write_code"]:
    if state.get("finished"):
        log("[Route] RepairOrFinish -> END (finished)")
        return "finish"
    if int(state.get("retrieval_rounds", 0)) >= MAX_RETRIEVAL_ROUNDS:
        log("[Route] RepairOrFinish -> END (max retrieval rounds)")
        return "finish"
    if state.get("pages_to_inspect") and int(state.get("code_attempts", 0)) == 0:
        log("[Route] RepairOrFinish -> InspectPages")
        return "inspect_pages"
    log("[Route] RepairOrFinish -> WriteCode")
    return "write_code"


def route_after_load_inputs(state: GraphState) -> Literal["finish", "expand_tariff_queries"]:
    if state.get("existing_calculator"):
        log("[Route] LoadInputs -> END (calculator already exists)")
        return "finish"
    log("[Route] LoadInputs -> ExpandTariffQueries")
    return "expand_tariff_queries"


def route_after_expand_queries(state: GraphState) -> Literal["retrieve_candidate_pages", "retrieve_candidate_pages_rag"]:
    if use_rag_retrieval():
        log("[Route] ExpandTariffQueries -> RetrieveCandidatePagesRag")
        return "retrieve_candidate_pages_rag"
    log("[Route] ExpandTariffQueries -> RetrieveCandidatePages")
    return "retrieve_candidate_pages"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("LoadInputs", trace_node("LoadInputs", load_inputs))
    graph.add_node("ExpandTariffQueries", trace_node("ExpandTariffQueries", expand_tariff_queries))
    graph.add_node("RetrieveCandidatePages", trace_node("RetrieveCandidatePages", retrieve_candidate_pages))
    graph.add_node("RetrieveCandidatePagesRag", trace_node("RetrieveCandidatePagesRag", retrieve_candidate_pages_rag))
    graph.add_node("InspectPages", trace_node("InspectPages", inspect_pages))
    graph.add_node("EnrichInputData", trace_node("EnrichInputData", enrich_input_data))
    graph.add_node("SelectCalculationParameters", trace_node("SelectCalculationParameters", select_calculation_parameters))
    graph.add_node("ExtractRules", trace_node("ExtractRules", extract_rules))
    graph.add_node("WriteCode", trace_node("WriteCode", write_code))
    graph.add_node("RunAndSubmit", trace_node("RunAndSubmit", run_and_submit))
    graph.add_node("RepairOrFinish", trace_node("RepairOrFinish", repair_or_finish))

    graph.add_edge(START, "LoadInputs")
    graph.add_conditional_edges(
        "LoadInputs",
        route_after_load_inputs,
        {
            "finish": END,
            "expand_tariff_queries": "ExpandTariffQueries",
        },
    )
    graph.add_conditional_edges(
        "ExpandTariffQueries",
        route_after_expand_queries,
        {
            "retrieve_candidate_pages": "RetrieveCandidatePages",
            "retrieve_candidate_pages_rag": "RetrieveCandidatePagesRag",
        },
    )
    graph.add_edge("RetrieveCandidatePages", "InspectPages")
    graph.add_edge("RetrieveCandidatePagesRag", "InspectPages")
    graph.add_edge("InspectPages", "EnrichInputData")
    graph.add_edge("EnrichInputData", "SelectCalculationParameters")
    graph.add_edge("SelectCalculationParameters", "ExtractRules")
    graph.add_edge("ExtractRules", "WriteCode")
    graph.add_edge("WriteCode", "RunAndSubmit")
    graph.add_edge("RunAndSubmit", "RepairOrFinish")
    graph.add_conditional_edges(
        "RepairOrFinish",
        route_after_repair,
        {
            "finish": END,
            "inspect_pages": "InspectPages",
            "write_code": "WriteCode",
        },
    )
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tariff calculation LangGraph pipeline.")
    parser.add_argument("--pdf", default=None, help="Path to tariff PDF. Defaults to first PDF in pdf_data.")
    parser.add_argument("--tariff", action="append", help="Target tariff. Can be passed multiple times.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose graph logs.")
    parser.add_argument("--log-prompts", action="store_true", help="Print full LLM prompts and responses.")
    parser.add_argument("--log-file", default=None, help="Write graph logs to this file.")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="Maximum LangGraph node transitions before stopping.",
    )
    return parser.parse_args()


def main() -> None:
    global VERBOSE_LOGS, LOG_PROMPTS, LOG_FILE
    args = parse_args()
    if args.quiet:
        VERBOSE_LOGS = False
    if args.log_prompts:
        LOG_PROMPTS = True
    if args.log_file:
        LOG_FILE = args.log_file
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")

    log("=== Tariff LangGraph run started ===")
    log(f"[main] requested_claude_model={RAW_CLAUDE_MODEL_ID}")
    log(
        "[main] first_round_model="
        + (CLAUDE_MODEL_ID if anthropic_client is not None else f"{GEMINI_MODEL_ID} (Anthropic key not set)")
    )
    log(f"[main] fallback_model={GEMINI_MODEL_ID}")
    log(f"[main] args={short_json(vars(args), max_chars=1000)}")
    log(f"[main] recursion_limit={args.recursion_limit}")

    app = build_graph()
    tariffs_to_run = args.tariff or [None]
    final_states: list[GraphState] = []

    for index, tariff in enumerate(tariffs_to_run, 1):
        log("\n" + "=" * 72)
        log(f"[main] training run {index}/{len(tariffs_to_run)} tariff={tariff or '[all from input.json]'}")
        log("=" * 72)

        initial_state: GraphState = {}
        if args.pdf:
            initial_state["pdf_file"] = args.pdf
        if tariff:
            initial_state["target_tariffs"] = [tariff]

        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": args.recursion_limit},
        )
        final_states.append(final_state)

        print(f"\n=== Final Results: {tariff or 'all tariffs'} ===")
        print(json.dumps(final_state.get("results", {}), ensure_ascii=False, indent=2))
        print("\n=== Feedback ===")
        print(json.dumps(final_state.get("submit_feedback", {}), ensure_ascii=False, indent=2))
        if final_state.get("finished"):
            existing = final_state.get("existing_calculator")
            if existing:
                print(f"\nExisting calculator found: {existing.get('code_path')}")
            else:
                rules_hash = source_rules_hash(final_state)
                print(f"\nSaved successful code to: {output_code_path(final_state.get('target_tariffs', []), rules_hash)}")
        else:
            print("\nGraph stopped without a fully successful submission.")
            if final_state.get("last_code_path"):
                print(f"Saved last generated code to: {final_state['last_code_path']}")

    print("\n=== Training Summary ===")
    for final_state in final_states:
        tariffs = ", ".join(final_state.get("target_tariffs", [])) or "all tariffs"
        status = "finished" if final_state.get("finished") else "not finished"
        existing = final_state.get("existing_calculator")
        code_path = existing.get("code_path") if existing else None
        if not code_path and final_state.get("finished"):
            code_path = output_code_path(
                final_state.get("target_tariffs", []),
                source_rules_hash(final_state),
            )
        if not code_path:
            code_path = final_state.get("last_code_path")
        print(f"{tariffs}: {status}" + (f" | {code_path}" if code_path else ""))


if __name__ == "__main__":
    main()
