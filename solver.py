from __future__ import annotations

import json
import os
import re
from typing import Any


def load_expected_results(path: str = "input.json") -> dict[str, float]:
    input_path = os.path.join(os.path.dirname(__file__), path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    expected_results: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} contains a non-string key: {key!r}")
        if not isinstance(value, int | float):
            raise ValueError(f"{path} value for {key!r} must be numeric")
        expected_results[key] = float(value)

    return expected_results


EXPECTED_RESULTS = load_expected_results()


def normalize_candidate_code(code: str) -> str:
    code = code.strip()

    python_block = re.search(r"```python\s*(.*?)```", code, flags=re.DOTALL | re.IGNORECASE)
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", code, flags=re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    return code


def _evaluation_label(expected_value: float, calculated_value: float) -> tuple[str, float]:
    if expected_value == 0:
        percent_diff = 0.0 if calculated_value == 0 else float("inf")
    else:
        percent_diff = abs(calculated_value - expected_value) / abs(expected_value) * 100

    if percent_diff < 1:
        return "within_1_percent", percent_diff
    if percent_diff <= 5:
        return "within_5_percent", percent_diff
    if percent_diff <= 10:
        return "within_10_percent", percent_diff
    if percent_diff <= 20:
        return "within_20_percent", percent_diff
    return "outside_20_percent", percent_diff


def _evalute_result_tool(key: str, calculated_value: float) -> dict[str, Any]:
    if key not in EXPECTED_RESULTS:
        return {
            "ok": False,
            "error": f"Unknown key: {key!r}. Available keys: {sorted(EXPECTED_RESULTS)}",
        }

    expected_value = EXPECTED_RESULTS[key]
    calculated = float(calculated_value)
    label, percent_diff = _evaluation_label(expected_value, calculated)

    return {
        "ok": True,
        "key": key,
        "expected_value": expected_value,
        "calculated_value": calculated,
        "percent_diff": percent_diff,
        "result": label,
    }


def evalute_results_tool(values: dict[str, float]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, value in values.items():
        try:
            result = _evalute_result_tool(key, value)
        except (TypeError, ValueError) as exc:
            errors[key] = str(exc)
            continue

        if result.get("ok"):
            results[key] = result
        else:
            errors[key] = str(result.get("error"))

    return {
        "ok": not errors,
        "results": results,
        "errors": errors,
    }


__all__ = ["evalute_results_tool", "normalize_candidate_code"]
