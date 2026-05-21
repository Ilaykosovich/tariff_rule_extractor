import os
import hashlib
import json
import math
import re
import fitz  # PyMuPDF
from google import genai
from google.genai import types
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
DEFAULT_HYBRID_DENSE_WEIGHT = 0.65
DEFAULT_VOYAGE_EMBED_BATCH_SIZE = 4

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

def get_summary_hash(pdf_path, pages):
    file_hash = get_file_hash(pdf_path)
    pages_hash = get_pages_hash(pages)
    cache_key = f"{file_hash}:{pages_hash}:{MODEL_ID}:{SUMMARY_CACHE_VERSION}:{SUMMARY_SYSTEM_PROMPT}"
    return hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

def get_voyage_embedding_model():
    return os.getenv("VOYAGE_EMBEDDING_MODEL", DEFAULT_VOYAGE_EMBEDDING_MODEL)

def get_page_embedding_hash(pdf_path, pages, embedding_model=None):
    embedding_model = embedding_model or get_voyage_embedding_model()
    file_hash = get_file_hash(pdf_path)
    pages_hash = get_pages_hash(pages)
    cache_key = f"{file_hash}:{pages_hash}:{embedding_model}:{PAGE_EMBEDDING_CACHE_VERSION}"
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

def embed_page_documents(pdf_path, pages, embedding_model=None):
    """Build or load Voyage embeddings for OCR page texts."""
    import voyageai

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    embedding_model = embedding_model or get_voyage_embedding_model()
    h = get_page_embedding_hash(pdf_path, pages, embedding_model)
    cache_path = os.path.join(CACHE_DIR, f"{h}.voyage_page_embeddings.json")
    page_numbers = sorted(pages)

    result = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_result = json.load(f)
        if (
            cached_result.get("model") == embedding_model
            and cached_result.get("embeddings")
            and len(cached_result.get("embeddings", [])) == len(page_numbers)
        ):
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
            "page_numbers": [],
            "embeddings": [],
        }

    existing = {
        int(page_number): embedding
        for page_number, embedding in zip(result.get("page_numbers", []), result.get("embeddings", []))
    }
    missing_page_numbers = [page_number for page_number in page_numbers if page_number not in existing]
    if not missing_page_numbers:
        result["page_numbers"] = page_numbers
        result["embeddings"] = [existing[page_number] for page_number in page_numbers]
        result["complete"] = True
        return result

    completed_page_numbers = [page_number for page_number in page_numbers if page_number in existing]
    result["page_numbers"] = completed_page_numbers
    result["embeddings"] = [existing[page_number] for page_number in completed_page_numbers]
    result["complete"] = False
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    batch_size = get_voyage_embed_batch_size()
    print(
        f"[embeddings] Building Voyage embeddings: missing={len(missing_page_numbers)} "
        f"batch_size={batch_size} model={embedding_model}"
    )

    vo = voyageai.Client()
    for start in range(0, len(missing_page_numbers), batch_size):
        batch_page_numbers = missing_page_numbers[start:start + batch_size]
        docs = [
            page_text_for_embedding(page_number, pages.get(page_number, ""))
            for page_number in batch_page_numbers
        ]
        response = vo.embed(
            docs,
            model=embedding_model,
            input_type="document",
        )
        for page_number, embedding in zip(batch_page_numbers, response.embeddings):
            existing[page_number] = embedding

        completed_page_numbers = [page_number for page_number in page_numbers if page_number in existing]
        result["page_numbers"] = completed_page_numbers
        result["embeddings"] = [existing[page_number] for page_number in completed_page_numbers]
        result["complete"] = len(completed_page_numbers) == len(page_numbers)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"[embeddings] Saved progress: {len(completed_page_numbers)}/{len(page_numbers)} pages")

    result["page_numbers"] = page_numbers
    result["embeddings"] = [existing[page_number] for page_number in page_numbers]
    result["complete"] = True
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"[embeddings] Saved: {cache_path}")
    return result

def search_rag_pages(
    pdf_path,
    query,
    top_k=5,
    embedding_model=None,
    pages=None,
    include_text=True,
    dense_weight=None,
):
    """
    Run OCR if needed, then use hybrid Voyage embeddings + BM25 to rank pages.

    Returns a list of:
    {"page_number": int, "score": float, "text": str}
    """
    import voyageai

    if pages is None:
        pages = run_ocr_pipeline(pdf_path)

    embedding_model = embedding_model or get_voyage_embedding_model()
    index = embed_page_documents(pdf_path, pages, embedding_model=embedding_model)

    vo = voyageai.Client()
    query_vector = vo.embed(
        [query],
        model=embedding_model,
        input_type="query",
    ).embeddings[0]

    dense_scores = {}
    for page_number, page_vector in zip(index["page_numbers"], index["embeddings"]):
        dense_scores[int(page_number)] = cosine_similarity(query_vector, page_vector)

    bm25_by_page = bm25_scores(query, pages)
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
                "text": pages.get(page_number, "") if include_text else "",
            }
        )

    scored_pages.sort(key=lambda item: item["score"], reverse=True)
    return scored_pages[:top_k]

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

