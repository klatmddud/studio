from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def infer_num_classes_from_runtime_config(config: dict[str, Any]) -> int | None:
    annotation_paths = _collect_annotation_paths(config)
    if not annotation_paths:
        return None
    return infer_num_classes_from_annotation_paths(annotation_paths)


def infer_num_classes_from_annotation_paths(
    annotation_paths: list[str | Path],
) -> int:
    unique_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in annotation_paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen_paths:
            continue
        unique_paths.append(path)
        seen_paths.add(path)

    category_ids: set[int] = set()
    for path in unique_paths:
        category_ids.update(_load_category_ids_from_annotation(path))

    if not category_ids:
        raise ValueError("Could not infer num_classes because no category IDs were found.")

    # Detection targets keep COCO-style category IDs, so label 0 remains reserved
    # for background and the model needs capacity up to max(category_id).
    return max(category_ids) + 1


def _collect_annotation_paths(config: dict[str, Any]) -> list[str | Path]:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return []

    paths: list[str | Path] = []
    for key in ("train_annotations", "val_annotations", "annotations"):
        value = data.get(key)
        if value:
            paths.append(value)
    return paths


def _load_category_ids_from_annotation(path: Path) -> set[int]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Annotation file must contain a non-empty 'categories' list: {path}")

    category_ids: list[int] = []
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or "id" not in category:
            raise ValueError(
                f"Annotation category entry #{index} must be a mapping with an 'id': {path}"
            )
        category_id = int(category["id"])
        if category_id < 1:
            raise ValueError(
                f"Annotation category IDs must be >= 1 because 0 is reserved for background: {path}"
            )
        category_ids.append(category_id)

    unique_category_ids = set(category_ids)
    if len(unique_category_ids) != len(category_ids):
        raise ValueError(f"Annotation file contains duplicate category IDs: {path}")
    return unique_category_ids
