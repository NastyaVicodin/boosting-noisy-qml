"""Small JSON/CSV serialization helpers for experiment artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


def write_json(payload: Mapping, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv_rows(rows: Sequence[Mapping], path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path_obj.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path_obj.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
