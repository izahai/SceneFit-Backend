from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_response(payload: Dict[str, Any], *, file_path: str) -> Dict[str, Any]:
    """Append a participant payload to the JSONL file.

    Returns a small metadata dict (including line_number) that can be returned
    to the client.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    record = {
        "receivedAt": _utc_now_iso(),
        "payload": payload,
    }

    line = json.dumps(record, ensure_ascii=False)

    # Append and compute line number in a lightweight way by counting existing lines.
    # (OK for small files. If it grows large, remove the line count.)
    line_number = 1
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line_number, _ in enumerate(f, start=1):
                pass
        line_number += 1

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return {"file_path": file_path, "line_number": line_number, "receivedAt": record["receivedAt"]}


def read_all_payloads(*, file_path: str) -> List[Dict[str, Any]]:
    """Read all stored payloads from JSONL.

    Returns the original Unity payloads (record['payload']).
    """
    if not os.path.exists(file_path):
        return []

    payloads: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except Exception:
                continue
            payload = record.get("payload")
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
