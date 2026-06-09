import os
import hashlib
import json
import math
import re
import time
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from anthropic import Anthropic
from dotenv import load_dotenv

# --- КОНФИГУРАЦИЯ ---
MODEL_ID = "gemini-2.5-flash" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "processed_files")
CHUNK_SIZE = 15
CACHE_VERSION = "two_page_spread_v1"
SUMMARY_CACHE_VERSION = "page_summary_v1"
PAGE_EMBEDDING_CACHE_VERSION = "voyage_page_embeddings_v1"
DEFAULT_VOYAGE_EMBEDDING_MODEL = "voyage-3.5-lite"
DEFAULT_HYBRID_DENSE_WEIGHT = 0.8
DEFAULT_VOYAGE_EMBED_BATCH_SIZE = 4
DEFAULT_RAG_CHUNK_SIZE_TOKENS = 500
DEFAULT_RAG_CHUNK_OVERLAP_TOKENS = 100
DEFAULT_RAG_CHUNK_MODE = "token"
DEFAULT_RAG_RERANK_MODEL = "claude-sonnet-4-20250514"
DEFAULT_RAG_RERANK_CONTEXT_CHARS = 500
DEFAULT_RAG_RERANK_CHUNKS_PER_PAGE = 3
DEFAULT_RAG_CANDIDATE_K = 20
DEFAULT_RAG_TOP_K = 6

SUMMARY_SYSTEM_PROMPT = """You are preparing page-level summaries of a tariff PDF for a later calculation agent.

You will receive two consecutive pages:
- previous page
- target page

Your task is to summarize ONLY the target page.

Use the previous page only to understand context, headings, table continuation, units, and references.

Do not solve any final task.
Do not calculate final tariff values.
Do not decide which tariff items are relevant to a later problem.

For the target page, produce a concise but calculation-aware summary.

Include:

1. Page number
2. Main topic / section heading of the target page
3. What happens on this page in plain English
4. Whether the page contains:
   - formulas
   - tariff rates
   - tables
   - definitions
   - examples
   - footnotes or exceptions
5. If there is a formula:
   - explain what it calculates
   - list the input parameters it uses
   - preserve the formula as written if possible
   - preserve units and rounding rules
6. If there is a table:
   - explain what the table lists
   - list the table columns
   - describe what each row category represents
   - preserve important rates, thresholds, units, minimums, and maximums
7. Mention all calculation-relevant numeric values exactly as written.
8. Mention all vessel/cargo/time parameters used on the page, such as:
   - GT / gross tonnage
   - NT / net tonnage
   - DWT / deadweight
   - LOA / length overall
   - draft / draught
   - cargo tonnes
   - days
   - hours
   - movements
   - arrivals/departures
   - vessel type
   - cargo type
9. Say whether understanding the target page requires reading the previous page.
10. If yes, explain exactly why:
   - table header is on previous page
   - section heading is on previous page
   - formula starts on previous page
   - units/rates are defined on previous page
   - footnote or condition refers back
   - page is a continuation of a table/list
11. Say whether the next page is likely needed.
12. List search keywords that would help find this page later.

Output format:

{
  "page_number": "...",
  "section_heading": "...",
  "plain_summary": "...",
  "contains_formula": true/false,
  "contains_tariff_rates": true/false,
  "contains_table": true/false,
  "contains_definitions": true/false,
  "contains_footnotes_or_exceptions": true/false,
  "formulas": [
    {
      "formula_as_written": "...",
      "what_it_calculates": "...",
      "parameters_used": [],
      "units": "...",
      "rounding_or_conditions": "..."
    }
  ],
  "tables": [
    {
      "what_table_lists": "...",
      "columns": [],
      "row_categories": [],
      "important_values": []
    }
  ],
  "calculation_relevant_numbers": [
    {
      "value": "...",
      "unit": "...",
      "context": "..."
    }
  ],
  "parameters_mentioned": [],
  "requires_previous_page": true/false,
  "previous_page_dependency_reason": "...",
  "next_page_likely_needed": true/false,
  "next_page_reason": "...",
  "search_keywords": [],
  "ambiguities_or_risks": []
}

Return only valid JSON. Do not wrap it in Markdown."""

load_dotenv()
client = genai.Client(
    api_key=os.getenv("Gemini_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
)

def get_response_text(response):
    content = response.text
    if content:
        return content

    candidates = getattr(response, "candidates", None) or []
    details = []
    for candidate in candidates:
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason:
            details.append(f"finish_reason={finish_reason}")

        safety_ratings = getattr(candidate, "safety_ratings", None)
        if safety_ratings:
            details.append(f"safety_ratings={safety_ratings}")

    detail_text = "; ".join(details) if details else "empty response"
    raise RuntimeError(f"Gemini did not return text for this chunk: {detail_text}")

def get_pdf_chunk_content_hash(doc, start, end):
    """Генерирует хеш для фрагмента страниц."""
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
    # Используем метод сжатия при записи в байты для консистентности хеша
    chunk_bytes = chunk_doc.write(clean=True, deflate=True)
    chunk_doc.close()
    return hashlib.sha256(chunk_bytes).hexdigest()

def get_file_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_chunk_hash(file_hash, start, end):
    cache_key = f"{file_hash}:{start}:{end}:{MODEL_ID}:{CACHE_VERSION}"
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

def get_pages_hash(pages):
    normalized = json.dumps(
        {str(page_number): pages[page_number] for page_number in sorted(pages)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def write_json_atomic(path, value, ensure_ascii=False, indent=None):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=ensure_ascii, indent=indent)
    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt == 4:
                break
            time.sleep(0.2 * (attempt + 1))

    # Windows can deny os.replace when editors, indexers, or antivirus briefly
    # hold the destination file. For cache files, a plain overwrite is an
    # acceptable fallback and avoids requiring admin privileges.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=ensure_ascii, indent=indent)
    try:
        os.remove(tmp_path)
    except OSError:
        pass

def parse_json_object(text):
    cleaned = (text or "").strip()
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

def get_summary_hash(pdf_path, pages):
    file_hash = get_file_hash(pdf_path)
    pages_hash = get_pages_hash(pages)
    cache_key = f"{file_hash}:{pages_hash}:{MODEL_ID}:{SUMMARY_CACHE_VERSION}:{SUMMARY_SYSTEM_PROMPT}"
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

def get_voyage_embedding_model():
    return os.getenv("VOYAGE_EMBEDDING_MODEL", DEFAULT_VOYAGE_EMBEDDING_MODEL)

def get_page_embedding_hash(
    pdf_path,
    pages,
    embedding_model=None,
    chunk_mode=None,
    chunk_size_tokens=None,
    chunk_overlap_tokens=None,
):
    embedding_model = embedding_model or get_voyage_embedding_model()
    chunk_mode = get_rag_chunk_mode(chunk_mode)
    if chunk_mode == "token":
        chunk_size_tokens = get_rag_chunk_size_tokens(chunk_size_tokens)
        chunk_overlap_tokens = get_rag_chunk_overlap_tokens(chunk_overlap_tokens)
    else:
        chunk_size_tokens = None
        chunk_overlap_tokens = None
    file_hash = get_file_hash(pdf_path)
    pages_hash = get_pages_hash(pages)
    if chunk_mode == "page":
        cache_key = f"{file_hash}:{pages_hash}:{embedding_model}:{PAGE_EMBEDDING_CACHE_VERSION}"
        return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

    cache_key = (
        f"{file_hash}:{pages_hash}:{embedding_model}:"
        f"chunk_mode={chunk_mode}:chunk_size={chunk_size_tokens}:overlap={chunk_overlap_tokens}:"
        f"{PAGE_EMBEDDING_CACHE_VERSION}"
    )
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)

def page_text_for_embedding(page_number, page_content):
    content = page_content.strip() or "[empty page]"
    return f"Page {page_number}\n\n{content}"

def get_voyage_embed_batch_size():
    return max(1, int(os.getenv("VOYAGE_EMBED_BATCH_SIZE", DEFAULT_VOYAGE_EMBED_BATCH_SIZE)))

def get_rag_chunk_mode(chunk_mode=None):
    value = chunk_mode if chunk_mode is not None else os.getenv("TARIFF_RAG_CHUNK_MODE", DEFAULT_RAG_CHUNK_MODE)
    value = str(value).strip().lower()
    if value in {"page", "pages", "page_level", "page-level"}:
        return "page"
    if value in {"token", "tokens", "chunk", "chunks", "token_chunk", "token-chunk"}:
        return "token"
    raise ValueError(f"Unsupported RAG chunk mode: {value}")

def get_rag_chunk_size_tokens(chunk_size_tokens=None):
    value = chunk_size_tokens if chunk_size_tokens is not None else os.getenv("TARIFF_RAG_CHUNK_SIZE_TOKENS")
    return max(1, int(value or DEFAULT_RAG_CHUNK_SIZE_TOKENS))

def get_rag_chunk_overlap_tokens(chunk_overlap_tokens=None):
    value = chunk_overlap_tokens if chunk_overlap_tokens is not None else os.getenv("TARIFF_RAG_CHUNK_OVERLAP_TOKENS")
    return max(0, int(value or DEFAULT_RAG_CHUNK_OVERLAP_TOKENS))

def chunk_text_by_tokens(page_number, page_content, chunk_size_tokens=None, chunk_overlap_tokens=None):
    chunk_size_tokens = get_rag_chunk_size_tokens(chunk_size_tokens)
    chunk_overlap_tokens = get_rag_chunk_overlap_tokens(chunk_overlap_tokens)
    if chunk_overlap_tokens >= chunk_size_tokens:
        chunk_overlap_tokens = max(0, chunk_size_tokens - 1)

    content = page_content.strip() or "[empty page]"
    token_matches = list(re.finditer(r"\S+", content))
    if not token_matches:
        token_matches = [re.match(r".*", content)]

    chunks = []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    for chunk_index, token_start in enumerate(range(0, len(token_matches), step)):
        token_end = min(len(token_matches), token_start + chunk_size_tokens)
        char_start = token_matches[token_start].start()
        char_end = token_matches[token_end - 1].end()
        chunk_text = content[char_start:char_end]
        chunks.append(
            {
                "text": f"Page {page_number}, chunk {chunk_index + 1}\n\n{chunk_text}",
                "metadata": {
                    "page_number": int(page_number),
                    "chunk_index": int(chunk_index),
                    "token_start": int(token_start),
                    "token_end": int(token_end),
                    "char_start": int(char_start),
                    "char_end": int(char_end),
                    "chunk_size_tokens": int(chunk_size_tokens),
                    "chunk_overlap_tokens": int(chunk_overlap_tokens),
                },
            }
        )
        if token_end == len(token_matches):
            break
    return chunks

def page_as_embedding_chunk(page_number, page_content):
    content = page_content.strip() or "[empty page]"
    token_count = len(list(re.finditer(r"\S+", content)))
    return {
        "text": page_text_for_embedding(page_number, page_content),
        "metadata": {
            "page_number": int(page_number),
            "chunk_index": 0,
            "token_start": 0,
            "token_end": int(token_count),
            "char_start": 0,
            "char_end": len(content),
            "chunk_mode": "page",
            "chunk_size_tokens": None,
            "chunk_overlap_tokens": None,
        },
    }

def tokenize_for_search(text):
    return re.findall(r"[A-Za-zА-Яа-я0-9]+", text.lower())

def min_max_normalize(scores_by_page):
    if not scores_by_page:
        return {}
    values = list(scores_by_page.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {page_number: 1.0 if score else 0.0 for page_number, score in scores_by_page.items()}
    return {
        page_number: (score - min_score) / (max_score - min_score)
        for page_number, score in scores_by_page.items()
    }

def bm25_scores(query, pages, k1=1.5, b=0.75):
    query_terms = tokenize_for_search(query)
    if not query_terms:
        return {page_number: 0.0 for page_number in pages}

    page_tokens = {
        page_number: tokenize_for_search(page_text)
        for page_number, page_text in pages.items()
    }
    doc_count = len(page_tokens)
    avg_doc_len = (
        sum(len(tokens) for tokens in page_tokens.values()) / doc_count
        if doc_count
        else 0.0
    )
    doc_freq = {}
    for tokens in page_tokens.values():
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    scores = {}
    for page_number, tokens in page_tokens.items():
        if not tokens or not avg_doc_len:
            scores[page_number] = 0.0
            continue
        term_freq = {}
        for term in tokens:
            term_freq[term] = term_freq.get(term, 0) + 1

        score = 0.0
        doc_len = len(tokens)
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (doc_count - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5))
            denominator = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * (freq * (k1 + 1)) / denominator
        scores[page_number] = score
    return scores

def get_rag_reranker(reranker=None):
    value = reranker if reranker is not None else os.getenv("TARIFF_RAG_RERANKER", "none")
    value = str(value).strip().lower()
    if value in {"", "none", "off", "false", "0"}:
        return "none"
    if value in {"llm", "gemini"}:
        return "llm"
    raise ValueError(f"Unsupported RAG reranker: {value}")

def chunk_context_for_rerank(page_text, metadata, context_chars):
    context_chars = max(0, int(context_chars))
    char_start = metadata.get("char_start")
    char_end = metadata.get("char_end")
    if char_start is None or char_end is None:
        snippet = page_text[: max(context_chars * 2, 1200)]
        return "", snippet, ""

    char_start = max(0, int(char_start))
    char_end = min(len(page_text), int(char_end))
    before = page_text[max(0, char_start - context_chars):char_start]
    matched = page_text[char_start:char_end]
    after = page_text[char_end:min(len(page_text), char_end + context_chars)]
    return before, matched, after

def truncate_text(text, max_chars):
    text = text or ""
    max_chars = int(max_chars)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"

def llm_rerank_pages(
    query,
    candidate_chunks,
    *,
    top_k,
    model=None,
    context_chars=None,
):
    model = model or os.getenv("TARIFF_RAG_RERANK_MODEL", DEFAULT_RAG_RERANK_MODEL)
    context_chars = int(
        context_chars
        if context_chars is not None
        else os.getenv("TARIFF_RAG_RERANK_CONTEXT_CHARS", DEFAULT_RAG_RERANK_CONTEXT_CHARS)
    )
    candidates = []
    for index, candidate in enumerate(candidate_chunks, 1):
        before, matched, after = chunk_context_for_rerank(
            candidate["page_text"],
            candidate["metadata"],
            context_chars,
        )
        candidates.append(
            {
                "candidate_id": index,
                "page_number": int(candidate["page_number"]),
                "chunk_index": candidate["metadata"].get("chunk_index"),
                "initial_chunk_score": round(float(candidate["chunk_score"]), 6),
                "page_score_before_rerank": round(float(candidate["page_score"]), 6),
                "context_before": truncate_text(before.strip(), context_chars),
                "matched_chunk": truncate_text(matched.strip(), max(context_chars * 3, 1200)),
                "context_after": truncate_text(after.strip(), context_chars),
            }
        )

    prompt = {
        "task": (
            "Rerank candidate PDF chunks for a port tariff retrieval task. "
            "Choose the most relevant pages for answering the query. "
            "A page can have multiple candidate chunks; use all chunks as evidence for that page. "
            "Prefer pages containing tariff rules, rates, formulas, tables, exceptions, minimums, maximums, "
            "or calculation parameters. Penalize generic table-of-contents or unrelated pages."
        ),
        "query": query,
        "top_k_pages": int(top_k),
        "candidates": candidates,
        "output_schema": {
            "ranked_pages": [
                {
                    "page_number": 1,
                    "reason": "brief reason",
                    "supporting_candidate_ids": [1, 2],
                }
            ]
        },
    }
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured for LLM reranker.")
    anthropic_client = Anthropic(api_key=api_key)
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=0,
        system="You are a precise reranker. Return only valid JSON matching the requested schema.",
        messages=[
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False),
            }
        ],
    )
    text_parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    data = parse_json_object("\n".join(text_parts))
    ranked_pages = data.get("ranked_pages", []) if isinstance(data, dict) else []
    page_order = []
    reasons = {}
    for item in ranked_pages:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("page_number"))
        except (TypeError, ValueError):
            continue
        if page_number in page_order:
            continue
        page_order.append(page_number)
        reasons[page_number] = {
            "reason": str(item.get("reason", "")),
            "supporting_candidate_ids": item.get("supporting_candidate_ids", []),
        }
    return page_order, reasons

def get_rag_rerank_chunks_per_page(value=None):
    value = value if value is not None else os.getenv("TARIFF_RAG_RERANK_CHUNKS_PER_PAGE", DEFAULT_RAG_RERANK_CHUNKS_PER_PAGE)
    return max(1, int(value))

def build_ocr_prompt(start, end):
    first_page = start + 1
    last_page = end + 1
    first_logical_page = start * 2 + 1
    last_logical_page = (end + 1) * 2
    return (
        "Extract all text and tables as Markdown. No summaries.\n"
        f"The uploaded PDF chunk contains physical PDF pages {first_page}-{last_page}.\n"
        "Each physical PDF page is a two-page spread: it contains two logical document pages side by side.\n"
        "For every physical PDF page, split the content into the left logical page and the right logical page.\n"
        "Read each side independently, preserving text and tables as Markdown.\n"
        "Do not merge the left and right sides.\n"
        f"Output logical document pages {first_logical_page}-{last_logical_page} in reading order.\n"
        f"For physical PDF page {first_page}, the left side is logical page {first_logical_page} "
        f"and the right side is logical page {first_logical_page + 1}. Continue this pattern for each physical page.\n"
        "Before each page, write a marker line exactly like this:\n"
        "--- PAGE N ---\n"
        "Replace N with the logical 1-based document page number. "
        "Include a marker for every logical page, even if the page has no text."
    )

def get_logical_page_range(start, end):
    return start * 2 + 1, (end + 1) * 2

def split_chunk_by_pages(content, start, end):
    first_logical_page, last_logical_page = get_logical_page_range(start, end)
    marker_pattern = re.compile(r"(?m)^--- PAGE (\d+) ---\s*$")
    matches = list(marker_pattern.finditer(content))

    if not matches:
        return [(first_logical_page, content.strip())]

    pages = {}
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        page_start = match.end()
        page_end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        pages[page_number] = content[page_start:page_end].strip()

    return [(page_number, pages.get(page_number, "")) for page_number in range(first_logical_page, last_logical_page + 1)]

def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])

def build_summary_prompt(previous_page_number, previous_page_content, target_page_number, target_page_content):
    return (
        f"{SUMMARY_SYSTEM_PROMPT}\n\n"
        f"PREVIOUS PAGE NUMBER: {previous_page_number}\n"
        "PREVIOUS PAGE OCR MARKDOWN:\n"
        f"{previous_page_content or '[empty page]'}\n\n"
        f"TARGET PAGE NUMBER: {target_page_number}\n"
        "TARGET PAGE OCR MARKDOWN:\n"
        f"{target_page_content or '[empty page]'}"
    )

def summarize_page_pair(previous_page_number, previous_page_content, target_page_number, target_page_content):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            build_summary_prompt(
                previous_page_number,
                previous_page_content,
                target_page_number,
                target_page_content,
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    summary = extract_json_object(get_response_text(response))
    summary["page_number"] = str(summary.get("page_number") or target_page_number)
    return summary

def summarize_pages(pdf_path, pages):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    h = get_summary_hash(pdf_path, pages)
    cache_path = os.path.join(CACHE_DIR, f"{h}.summaries.json")
    page_cache_dir = os.path.join(CACHE_DIR, f"{h}.summaries")

    page_numbers = sorted(pages)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_result = json.load(f)
        if len(cached_result.get("summaries", [])) == len(page_numbers):
            print(f"[summaries] Загружено из кэша (hash: {h[:8]}...)")
            return cached_result
        print(f"[summaries] Найден частичный кэш, продолжаю (hash: {h[:8]}...)")

    if not os.path.exists(page_cache_dir):
        os.makedirs(page_cache_dir)

    print(f"[summaries] Обработка страниц через Gemini API (hash: {h[:8]}...)")
    summaries = []
    result = {
        "pdf_path": pdf_path,
        "pdf_hash": get_file_hash(pdf_path),
        "pages_hash": get_pages_hash(pages),
        "model": MODEL_ID,
        "cache_version": SUMMARY_CACHE_VERSION,
        "cache_path": cache_path,
        "page_cache_dir": page_cache_dir,
        "summaries": summaries,
    }

    for target_page_number in page_numbers:
        previous_page_number = target_page_number - 1
        previous_page_content = pages.get(previous_page_number, "")
        if target_page_number == page_numbers[0]:
            previous_page_number = -1
            previous_page_content = ""

        page_cache_path = os.path.join(page_cache_dir, f"page_{target_page_number}.json")
        if os.path.exists(page_cache_path):
            print(f"[summaries] Страница {target_page_number}: загружено из page-кэша")
            with open(page_cache_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            print(f"[summaries] Страница {target_page_number}: previous={previous_page_number}")
            summary = summarize_page_pair(
                previous_page_number,
                previous_page_content,
                target_page_number,
                pages.get(target_page_number, ""),
            )
            with open(page_cache_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        summaries.append(summary)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[summaries] Сохранено: {cache_path}")
    return result

def embed_page_documents(
    pdf_path,
    pages,
    embedding_model=None,
    chunk_mode=None,
    chunk_size_tokens=None,
    chunk_overlap_tokens=None,
):
    """Build or load Voyage embeddings for token chunks of OCR page texts."""
    import voyageai

    def chunk_id(metadata):
        return (
            f"{metadata['page_number']}:{metadata['chunk_index']}:"
            f"{metadata['token_start']}:{metadata['token_end']}"
        )

    def set_embedding_rows(target_result, row_chunks, existing_embeddings):
        target_result["chunk_ids"] = [chunk["id"] for chunk in row_chunks]
        target_result["page_numbers"] = [int(chunk["metadata"]["page_number"]) for chunk in row_chunks]
        target_result["chunk_texts"] = [chunk["text"] for chunk in row_chunks]
        target_result["embeddings"] = [existing_embeddings[chunk["id"]] for chunk in row_chunks]
        target_result["metadatas"] = [chunk["metadata"] for chunk in row_chunks]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    embedding_model = embedding_model or get_voyage_embedding_model()
    chunk_mode = get_rag_chunk_mode(chunk_mode)
    if chunk_mode == "token":
        chunk_size_tokens = get_rag_chunk_size_tokens(chunk_size_tokens)
        chunk_overlap_tokens = get_rag_chunk_overlap_tokens(chunk_overlap_tokens)
        if chunk_overlap_tokens >= chunk_size_tokens:
            chunk_overlap_tokens = max(0, chunk_size_tokens - 1)
    else:
        chunk_size_tokens = None
        chunk_overlap_tokens = None

    h = get_page_embedding_hash(
        pdf_path,
        pages,
        embedding_model,
        chunk_mode=chunk_mode,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    cache_path = os.path.join(CACHE_DIR, f"{h}.voyage_page_embeddings.json")

    chunks = []
    for page_number in sorted(pages):
        page_content = pages.get(page_number, "")
        page_chunks = (
            [page_as_embedding_chunk(page_number, page_content)]
            if chunk_mode == "page"
            else chunk_text_by_tokens(
                page_number,
                page_content,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )
        for chunk in page_chunks:
            metadata = chunk["metadata"]
            metadata["chunk_mode"] = chunk_mode
            chunks.append({"id": chunk_id(metadata), "text": chunk["text"], "metadata": metadata})

    result = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_result = json.load(f)
        is_legacy_page_cache = (
            chunk_mode == "page"
            and cached_result.get("model") == embedding_model
            and cached_result.get("chunk_mode") is None
            and cached_result.get("embeddings")
            and len(cached_result.get("embeddings", [])) == len(chunks)
        )
        if (
            is_legacy_page_cache
            or (
                cached_result.get("model") == embedding_model
                and cached_result.get("chunk_mode") == chunk_mode
                and cached_result.get("chunk_size_tokens") == chunk_size_tokens
                and cached_result.get("chunk_overlap_tokens") == chunk_overlap_tokens
                and cached_result.get("embeddings")
                and len(cached_result.get("embeddings", [])) == len(chunks)
            )
        ):
            if is_legacy_page_cache:
                existing = {
                    chunk["id"]: embedding
                    for chunk, embedding in zip(chunks, cached_result.get("embeddings", []))
                }
                set_embedding_rows(cached_result, chunks, existing)
                cached_result["chunk_mode"] = chunk_mode
                cached_result["chunk_size_tokens"] = chunk_size_tokens
                cached_result["chunk_overlap_tokens"] = chunk_overlap_tokens
                cached_result["complete"] = True
                write_json_atomic(cache_path, cached_result, ensure_ascii=False)
                print(f"[embeddings] Migrated legacy page cache (hash: {h[:8]}...)")
            print(f"[embeddings] Loaded from cache (hash: {h[:8]}...)")
            return cached_result
        if cached_result.get("model") == embedding_model:
            result = cached_result
            print(f"[embeddings] Resuming partial cache (hash: {h[:8]}...)")

    if result is None:
        result = {
            "pdf_path": pdf_path,
            "pdf_hash": get_file_hash(pdf_path),
            "pages_hash": get_pages_hash(pages),
            "model": embedding_model,
            "cache_version": PAGE_EMBEDDING_CACHE_VERSION,
            "cache_path": cache_path,
            "chunk_mode": chunk_mode,
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "chunk_ids": [],
            "page_numbers": [],
            "chunk_texts": [],
            "embeddings": [],
            "metadatas": [],
        }

    existing = {
        str(chunk_id_value): embedding
        for chunk_id_value, embedding in zip(result.get("chunk_ids", []), result.get("embeddings", []))
    }
    missing_chunks = [chunk for chunk in chunks if chunk["id"] not in existing]
    if not missing_chunks:
        set_embedding_rows(result, chunks, existing)
        result["complete"] = True
        return result

    completed_chunks = [chunk for chunk in chunks if chunk["id"] in existing]
    set_embedding_rows(result, completed_chunks, existing)
    result["complete"] = False
    write_json_atomic(cache_path, result, ensure_ascii=False)

    batch_size = get_voyage_embed_batch_size()
    print(
        f"[embeddings] Building Voyage embeddings: missing_chunks={len(missing_chunks)} "
        f"chunk_mode={chunk_mode} chunk_size={chunk_size_tokens} overlap={chunk_overlap_tokens} "
        f"batch_size={batch_size} model={embedding_model}"
    )

    vo = voyageai.Client()
    for start in range(0, len(missing_chunks), batch_size):
        batch_chunks = missing_chunks[start:start + batch_size]
        response = vo.embed(
            [chunk["text"] for chunk in batch_chunks],
            model=embedding_model,
            input_type="document",
        )
        for chunk, embedding in zip(batch_chunks, response.embeddings):
            existing[chunk["id"]] = embedding

        completed_chunks = [chunk for chunk in chunks if chunk["id"] in existing]
        set_embedding_rows(result, completed_chunks, existing)
        result["complete"] = len(completed_chunks) == len(chunks)
        write_json_atomic(cache_path, result, ensure_ascii=False)
        print(f"[embeddings] Saved progress: {len(completed_chunks)}/{len(chunks)} chunks")

    set_embedding_rows(result, chunks, existing)
    result["complete"] = True
    write_json_atomic(cache_path, result, ensure_ascii=False)

    print(f"[embeddings] Saved: {cache_path}")
    return result

def search_rag_pages(
    pdf_path,
    query,
    top_k=DEFAULT_RAG_TOP_K,
    candidate_k=None,
    embedding_model=None,
    pages=None,
    include_text=True,
    dense_weight=DEFAULT_HYBRID_DENSE_WEIGHT,
    chunk_mode=None,
    chunk_size_tokens=None,
    chunk_overlap_tokens=None,
    reranker="llm",
    rerank_model=None,
    rerank_context_chars=None,
    rerank_chunks_per_page=None,
):
    """
    Run OCR if needed, then use hybrid Voyage embeddings + BM25 to rank pages.

    Returns a list of:
    {"page_number": int, "score": float, "metadata": dict, "text": str}
    """
    import voyageai

    if pages is None:
        pages = run_ocr_pipeline(pdf_path)

    embedding_model = embedding_model or get_voyage_embedding_model()
    index = embed_page_documents(
        pdf_path,
        pages,
        embedding_model=embedding_model,
        chunk_mode=chunk_mode,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )

    vo = voyageai.Client()
    query_vector = vo.embed(
        [query],
        model=embedding_model,
        input_type="query",
    ).embeddings[0]

    dense_scores = {}
    dense_chunk_counts = {}
    chunk_scores = []
    metadatas = index.get("metadatas", [])
    page_numbers = index.get("page_numbers", [])
    chunk_texts = index.get("chunk_texts", [])
    for idx, page_vector in enumerate(index["embeddings"]):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        page_number = metadata.get("page_number")
        if page_number is None and idx < len(page_numbers):
            page_number = page_numbers[idx]
        if page_number is None:
            continue

        page_number = int(page_number)
        chunk_dense_score = cosine_similarity(query_vector, page_vector)
        dense_scores[page_number] = dense_scores.get(page_number, 0.0) + chunk_dense_score
        dense_chunk_counts[page_number] = dense_chunk_counts.get(page_number, 0) + 1
        chunk_scores.append(
            {
                "chunk_index": idx,
                "page_number": page_number,
                "dense_score": chunk_dense_score,
                "metadata": metadata,
                "chunk_text": chunk_texts[idx] if idx < len(chunk_texts) else "",
            }
        )

    bm25_by_page = bm25_scores(query, pages)
    chunk_dense_by_index = {item["chunk_index"]: item["dense_score"] for item in chunk_scores}
    normalized_chunk_dense = min_max_normalize(chunk_dense_by_index)
    normalized_dense = min_max_normalize(dense_scores)
    normalized_bm25 = min_max_normalize(bm25_by_page)
    dense_weight = (
        float(os.getenv("TARIFF_RAG_DENSE_WEIGHT", DEFAULT_HYBRID_DENSE_WEIGHT))
        if dense_weight is None
        else dense_weight
    )
    dense_weight = max(0.0, min(1.0, dense_weight))
    bm25_weight = 1.0 - dense_weight

    scored_pages = []
    for page_number in sorted(pages):
        dense_score = normalized_dense.get(page_number, 0.0)
        bm25_score = normalized_bm25.get(page_number, 0.0)
        hybrid_score = dense_score * dense_weight + bm25_score * bm25_weight
        scored_pages.append(
            {
                "page_number": int(page_number),
                "score": hybrid_score,
                "dense_score": dense_scores.get(page_number, 0.0),
                "bm25_score": bm25_by_page.get(page_number, 0.0),
                "metadata": {
                    "page_number": int(page_number),
                    "chunk_count": dense_chunk_counts.get(page_number, 0),
                    "chunk_mode": index.get("chunk_mode"),
                    "chunk_size_tokens": index.get("chunk_size_tokens"),
                    "chunk_overlap_tokens": index.get("chunk_overlap_tokens"),
                },
                "text": pages.get(page_number, "") if include_text else "",
            }
        )

    scored_pages.sort(key=lambda item: item["score"], reverse=True)
    reranker = get_rag_reranker(reranker)
    if reranker == "none":
        return scored_pages[:top_k]

    candidate_k = int(candidate_k or max(top_k, DEFAULT_RAG_CANDIDATE_K))
    candidate_page_numbers = {item["page_number"] for item in scored_pages[:candidate_k]}
    page_score_by_page = {item["page_number"]: item["score"] for item in scored_pages}
    page_result_by_page = {item["page_number"]: item for item in scored_pages}
    chunks_by_page = {}
    for item in chunk_scores:
        page_number = item["page_number"]
        if page_number not in candidate_page_numbers:
            continue
        chunk_score = (
            normalized_chunk_dense.get(item["chunk_index"], 0.0) * dense_weight
            + normalized_bm25.get(page_number, 0.0) * bm25_weight
        )
        chunks_by_page.setdefault(page_number, []).append(
            {
                **item,
                "chunk_score": chunk_score,
                "page_score": page_score_by_page.get(page_number, 0.0),
                "page_text": pages.get(page_number, ""),
            }
        )
    candidate_chunks = []
    chunks_per_page = get_rag_rerank_chunks_per_page(rerank_chunks_per_page)
    for page_number in sorted(candidate_page_numbers, key=lambda page: page_score_by_page.get(page, 0.0), reverse=True):
        page_chunks = sorted(chunks_by_page.get(page_number, []), key=lambda item: item["chunk_score"], reverse=True)
        candidate_chunks.extend(page_chunks[:chunks_per_page])

    try:
        page_order, rerank_reasons = llm_rerank_pages(
            query,
            candidate_chunks,
            top_k=top_k,
            model=rerank_model,
            context_chars=rerank_context_chars,
        )
    except Exception as exc:
        print(f"[reranker] LLM rerank failed, using initial RAG ranking: {exc}")
        return scored_pages[:top_k]

    reranked = []
    for rank, page_number in enumerate(page_order, 1):
        if page_number not in page_result_by_page:
            continue
        result = dict(page_result_by_page[page_number])
        result["metadata"] = dict(result.get("metadata", {}))
        result["metadata"]["reranker"] = "llm"
        result["metadata"]["rerank_rank"] = rank
        result["metadata"]["rerank_reason"] = rerank_reasons.get(page_number, {}).get("reason", "")
        result["metadata"]["rerank_supporting_candidate_ids"] = rerank_reasons.get(page_number, {}).get(
            "supporting_candidate_ids",
            [],
        )
        reranked.append(result)
        if len(reranked) >= top_k:
            break

    if len(reranked) < top_k:
        seen = {item["page_number"] for item in reranked}
        for result in scored_pages:
            if result["page_number"] in seen:
                continue
            fallback = dict(result)
            fallback["metadata"] = dict(fallback.get("metadata", {}))
            fallback["metadata"]["reranker"] = "llm_fallback_initial_rank"
            reranked.append(fallback)
            if len(reranked) >= top_k:
                break

    return reranked

def get_relevant_page_numbers(pdf_path, query, top_k=5, embedding_model=None, pages=None):
    """Convenience wrapper for RAG retrieval when only page numbers are needed."""
    results = search_rag_pages(
        pdf_path=pdf_path,
        query=query,
        top_k=top_k,
        embedding_model=embedding_model,
        pages=pages,
        include_text=False,
    )
    return [item["page_number"] for item in results]

def run_ocr_pipeline(pdf_path):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    file_hash = get_file_hash(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"--- Запуск pdf-pipeline для: {pdf_path} ({total_pages} стр.) ---")

    final_output = {}

    for i in range(0, total_pages, CHUNK_SIZE):
        start = i
        end = min(i + CHUNK_SIZE - 1, total_pages - 1)
        
        # Считаем хеш
        h = get_chunk_hash(file_hash, start, end)
        cache_path = os.path.join(CACHE_DIR, f"{h}.txt")

        if os.path.exists(cache_path):
            print(f"[{i+1}-{end+1}] Загружено из кэша (hash: {h[:8]}...)")
            with open(cache_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            print(f"[{i+1}-{end+1}] Обработка через Gemini API...")
            
            # Создаем временный файл для отправки в API
            tmp_path = f"tmp_{h}.pdf"
            tmp_doc = fitz.open()
            tmp_doc.insert_pdf(doc, from_page=start, to_page=end)
            tmp_doc.save(tmp_path)
            tmp_doc.close()

            uploaded = None
            try:
                # Загрузка и генерация
                uploaded = client.files.upload(file=tmp_path)
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=[uploaded, build_ocr_prompt(start, end)]
                )
                content = get_response_text(response)
                
                # Сохраняем в кэш
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Удаляем временные файлы
            finally:
                if uploaded is not None:
                    client.files.delete(name=uploaded.name)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        for page_number, page_content in split_chunk_by_pages(content, start, end):
            final_output[page_number] = page_content

    doc.close()
    print("--- Обработка завершена успешно ---")
    return final_output

