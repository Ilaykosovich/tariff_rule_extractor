from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pdf_pipeline import run_ocr_pipeline, summarize_pages

DOCUMENT_STORE_DIR = "document_store"
DOCUMENT_REGISTRY_PATH = "documents_registry.json"
CALCULATOR_REGISTRY_PATH = "calculators_registry.json"

app = FastAPI(title="Tariff Calculator Inference API")


class InferenceRequest(BaseModel):
    vessel_data: dict[str, Any] = Field(default_factory=dict)
    target_tariffs: list[str]
    document_hash: str | None = None
    document_name: str | None = None


def read_json_file(path: str, default: Any) -> Any:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "document.pdf"


def load_calculator(code_path: str):
    if not os.path.exists(code_path):
        raise HTTPException(status_code=404, detail=f"Calculator file not found: {code_path}")

    spec = importlib.util.spec_from_file_location("tariff_calculator", code_path)
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=500, detail=f"Cannot load calculator: {code_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calculate = getattr(module, "calculate", None)
    if not callable(calculate):
        raise HTTPException(status_code=500, detail="Calculator must define calculate(vessel_data: dict)")
    return calculate


def calculator_matches(record: dict[str, Any], request: InferenceRequest) -> bool:
    record_tariffs = set(record.get("target_tariffs", []))
    requested_tariffs = set(request.target_tariffs)
    if not requested_tariffs <= record_tariffs:
        return False
    if request.document_hash and record.get("document_hash") != request.document_hash:
        return False
    if request.document_name and record.get("document_name") != request.document_name:
        return False
    return True


def find_calculator(request: InferenceRequest) -> dict[str, Any]:
    registry = read_json_file(CALCULATOR_REGISTRY_PATH, [])
    if not isinstance(registry, list):
        registry = []

    matches = [record for record in registry if calculator_matches(record, request)]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                "No trained calculator found. Run tariff_langgraph.py first for this "
                "document/tariff set, then call inference again."
            ),
        )

    matches.sort(key=lambda item: os.path.getmtime(item["code_path"]) if os.path.exists(item["code_path"]) else 0)
    return matches[-1]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def inference_page() -> str:
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Tariff Inference</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 40px auto; line-height: 1.45; }
    label { display: block; margin-top: 16px; font-weight: 700; }
    input, button { width: 100%; box-sizing: border-box; padding: 10px; margin-top: 6px; }
    button { cursor: pointer; margin-top: 22px; }
    pre { background: #f4f4f4; padding: 14px; overflow: auto; }
  </style>
</head>
<body>
  <h1>Tariff Inference</h1>
  <form action="/ui/infer" method="post" enctype="multipart/form-data">
    <label>Port Tariff.pdf</label>
    <input name="pdf_file" type="file" accept=".pdf,application/pdf" required>

    <label>PDF filename</label>
    <input name="document_name" value="Port Tariff.pdf" required>

    <label>input_param.json</label>
    <input name="input_json" type="file" accept=".json,application/json" required>

    <label>Tariff name</label>
    <input name="target_tariff" value="Light Dues" required>

    <button type="submit">Calculate</button>
  </form>
</body>
</html>
""".strip()


@app.post("/ui/infer", response_class=HTMLResponse)
def ui_infer(
    document_name: str = Form(...),
    target_tariff: str = Form(...),
    pdf_file: UploadFile = File(...),
    input_json: UploadFile = File(...),
) -> str:
    try:
        vessel_data = json.loads(input_json.file.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    document_record = upload_document(
        file=pdf_file,
        document_name=document_name,
        process=False,
    )
    request = InferenceRequest(
        vessel_data=vessel_data,
        target_tariffs=[target_tariff],
        document_hash=document_record["document_hash"],
        document_name=document_record["document_name"],
    )
    result = infer(request)
    pretty = json.dumps(result, ensure_ascii=False, indent=2)
    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Tariff Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.45; }}
    a {{ display: inline-block; margin-bottom: 18px; }}
    pre {{ background: #f4f4f4; padding: 14px; overflow: auto; }}
  </style>
</head>
<body>
  <a href="/">Back</a>
  <h1>Result</h1>
  <pre>{pretty}</pre>
</body>
</html>
""".strip()


@app.post("/documents")
def upload_document(
    file: UploadFile = File(...),
    document_name: str | None = Form(default=None),
    process: bool = Form(default=False),
) -> dict[str, Any]:
    os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)

    original_name = document_name or file.filename or "document.pdf"
    stored_name = safe_filename(original_name)
    tmp_path = os.path.join(DOCUMENT_STORE_DIR, stored_name)

    with open(tmp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    document_hash = file_sha256(tmp_path)
    document_dir = os.path.join(DOCUMENT_STORE_DIR, document_hash[:15])
    os.makedirs(document_dir, exist_ok=True)
    final_path = os.path.join(document_dir, stored_name)
    if os.path.abspath(tmp_path) != os.path.abspath(final_path):
        shutil.move(tmp_path, final_path)

    record: dict[str, Any] = {
        "document_hash": document_hash,
        "document_name": original_name,
        "stored_path": final_path,
        "processed": False,
    }

    if process:
        pages = run_ocr_pipeline(final_path)
        summary_result = summarize_pages(final_path, pages)
        record["processed"] = True
        record["page_count"] = len(pages)
        record["summary_count"] = len(summary_result.get("summaries", []))

    registry = read_json_file(DOCUMENT_REGISTRY_PATH, [])
    if not isinstance(registry, list):
        registry = []
    registry = [item for item in registry if item.get("document_hash") != document_hash]
    registry.append(record)
    write_json_file(DOCUMENT_REGISTRY_PATH, registry)

    return record


@app.post("/infer")
def infer(request: InferenceRequest) -> dict[str, Any]:
    if not request.target_tariffs:
        raise HTTPException(status_code=400, detail="target_tariffs must not be empty")

    calculator_record = find_calculator(request)
    calculate = load_calculator(calculator_record["code_path"])

    raw_results = calculate(request.vessel_data)
    if not isinstance(raw_results, dict):
        raise HTTPException(status_code=500, detail="Calculator returned a non-dict result")

    filtered_results = {
        tariff: float(raw_results[tariff])
        for tariff in request.target_tariffs
        if tariff in raw_results
    }
    missing = [tariff for tariff in request.target_tariffs if tariff not in filtered_results]
    if missing:
        raise HTTPException(status_code=500, detail=f"Calculator did not return tariffs: {missing}")

    return {
        "results": filtered_results,
        "calculator": calculator_record,
    }
