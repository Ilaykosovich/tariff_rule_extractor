from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
API_BASE_URL = "http://127.0.0.1:8001"
PDF_PATH = PROJECT_ROOT / "pdf_data" / "Port Tariff.pdf"
INPUT_JSON_PATH = PROJECT_ROOT / "input_param.json"
DOCUMENT_NAME = "Port Tariff.pdf"
REPORT_DIR = PROJECT_ROOT / "test_report"

EXPECTED_RESULTS: dict[str, float] = {
    "Light Dues": 60062.04,
    "Port Dues": 199549.22,
    "Towage Dues": 147074.38,
    "VTS Dues": 33315.75,
    "Pilotage Dues": 47189.94,
    "Running Lines": 19639.50,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run API tariff inference smoke tests.")
    parser.add_argument("--api", default=API_BASE_URL, help="API base URL.")
    parser.add_argument("--input", default=str(INPUT_JSON_PATH), help="Path to input_param.json.")
    parser.add_argument("--pdf", default=None, help="Path to the PDF used for the run.")
    parser.add_argument("--document-name", default=None, help="Document name stored in the report.")
    return parser


def write_report(path: str | os.PathLike[str], report: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def request_text(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def extract_ui_result(html_text: str) -> dict[str, Any]:
    match = re.search(r"<pre>(.*?)</pre>", html_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return {"detail": html_text}
    return json.loads(html.unescape(match.group(1)))


def multipart_form_data(
    fields: dict[str, str],
    files: dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----tariff-test-{uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files.items():
        filename = os.path.basename(path)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            content = f.read()
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def percent_error(expected: float, actual: float | None) -> float | None:
    if actual is None:
        return None
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / abs(expected) * 100


def run_tariff_test(
    api_base_url: str,
    input_json_path: Path,
    tariff: str,
) -> dict[str, Any]:
    body, boundary = multipart_form_data(
        fields={"target_tariff": tariff},
        files={"input_json": str(input_json_path)},
    )
    status_code, response_text = request_text(
        "POST",
        f"{api_base_url.rstrip('/')}/ui/infer",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        response = extract_ui_result(response_text)
    except (json.JSONDecodeError, TypeError) as exc:
        response = {"detail": f"Failed to parse UI response: {exc}", "raw_response": response_text}

    expected = EXPECTED_RESULTS[tariff]
    actual = None
    if status_code == 200:
        actual = response.get("results", {}).get(tariff)
        if actual is not None:
            actual = float(actual)

    error_percent = percent_error(expected, actual)
    return {
        "tariff": tariff,
        "ok": status_code == 200 and actual is not None,
        "status_code": status_code,
        "expected": expected,
        "actual": actual,
        "error_percent": error_percent,
        "response": response,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else PDF_PATH
    input_json_path = Path(args.input)
    document_name = args.document_name or DOCUMENT_NAME

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if not input_json_path.exists():
        print(f"Input JSON not found: {input_json_path}", file=sys.stderr)
        return 2

    started_at = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"test_report_{started_at}.json"
    report: dict[str, Any] = {
        "api_base_url": args.api,
        "pdf_path": str(pdf_path),
        "input_json_path": str(input_json_path),
        "document_name": document_name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_results": EXPECTED_RESULTS,
        "endpoint": "/ui/infer",
        "tests": [],
    }
    write_report(report_path, report)

    for tariff in EXPECTED_RESULTS:
        result = run_tariff_test(
            args.api,
            input_json_path,
            tariff,
        )
        report["tests"].append(result)
        write_report(report_path, report)

        actual = result["actual"]
        error = result["error_percent"]
        print(
            f"{tariff}: status={result['status_code']} "
            f"actual={actual} expected={result['expected']} error_percent={error}"
        )

    failed = [test for test in report["tests"] if not test["ok"]]
    print(f"Report written to: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
