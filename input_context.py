from __future__ import annotations

import copy
import datetime as dt
from typing import Any


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


def describe_datetime(
    value: str,
    standard_work_start: int = 8,
    standard_work_end: int = 17,
    current_time: Any = None,
) -> dict[str, Any]:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return {"raw": value}

    now = parse_iso_datetime(current_time) if current_time is not None else dt.datetime.now()
    if now is None:
        now = dt.datetime.now()

    is_weekend = parsed.weekday() >= 5
    is_standard_working_hours = (
        not is_weekend
        and standard_work_start <= parsed.hour < standard_work_end
    )

    description: dict[str, Any] = {
        "raw": value,
        "date": parsed.date().isoformat(),
        "time": parsed.time().replace(microsecond=0).isoformat(),
        "weekday": parsed.strftime("%A"),
        "weekday_number": parsed.isoweekday(),
        "is_weekend": is_weekend,
        "is_standard_workday": not is_weekend,
        "is_standard_working_hours": is_standard_working_hours,
        "whole_hours": parsed.hour,
        "whole_minutes": parsed.minute,
        "standard_working_hours_assumption": f"Monday-Friday {standard_work_start:02d}:00-{standard_work_end:02d}:00 local port time",
    }

    current_is_standard_working_hours = (
        now.weekday() < 5
        and standard_work_start <= now.hour < standard_work_end
    )
    if current_is_standard_working_hours:
        description["hours_until_workday_start"] = 0
    else:
        next_work_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now.weekday() >= 5:
            next_work_start += dt.timedelta(days=(7 - now.weekday()) % 7)
        else:
            start_today = now.replace(hour=standard_work_start, minute=0, second=0, microsecond=0)
            if now < start_today:
                next_work_start = start_today
            else:
                next_work_start += dt.timedelta(days=1)
                while next_work_start.weekday() >= 5:
                    next_work_start += dt.timedelta(days=1)

        next_work_start = next_work_start.replace(
            hour=standard_work_start,
            minute=0,
            second=0,
            microsecond=0,
        )
        description["hours_until_workday_start"] = int(
            (next_work_start - now).total_seconds() // 3600
        )

    return description


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


def merge_missing_context(existing: Any, generated: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing, dict):
        return copy.deepcopy(generated)

    merged = copy.deepcopy(existing)
    for key, value in generated.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_missing_context(merged[key], value)
    return merged


def enrich_vessel_data_for_inference(vessel_data: dict[str, Any]) -> dict[str, Any]:
    enriched = copy.deepcopy(vessel_data)
    generated_context = deterministic_input_context(vessel_data)
    enriched["derived_context"] = merge_missing_context(
        enriched.get("derived_context"),
        generated_context,
    )
    return enriched
