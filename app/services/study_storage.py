from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_response(payload: Dict[str, Any], *, file_path: str) -> Dict[str, Any]:
    """Append a participant payload to the JSON file.

    The file stores a JSON array of records. Each call reads the existing
    array, appends the new record, and writes the whole file back.
    Returns a small metadata dict that can be returned to the client.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    record = {
        "receivedAt": _utc_now_iso(),
        "payload": payload,
    }

    # Read existing records
    records: List[Dict[str, Any]] = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except (json.JSONDecodeError, Exception):
                records = []

    records.append(record)

    # Write the full array back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return {"file_path": file_path, "entry_index": len(records) - 1, "receivedAt": record["receivedAt"]}


def read_all_payloads(*, file_path: str) -> List[Dict[str, Any]]:
    """Read all stored payloads from the JSON file.

    Returns the original Unity payloads (record['payload']).
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except (json.JSONDecodeError, Exception):
            return []

    payloads: List[Dict[str, Any]] = []
    for record in records:
        payload = record.get("payload") if isinstance(record, dict) else None
        if isinstance(payload, dict):
            payloads.append(payload)

    return payloads


def try_find_by_participant_id(*, file_path: str, participant_id: str) -> Optional[Dict[str, Any]]:
    """Return the most recent payload for a participantId, if present."""
    for payload in reversed(read_all_payloads(file_path=file_path)):
        pid = payload.get("participantId") or payload.get("participant_id")
        if pid == participant_id:
            return payload
    return None
