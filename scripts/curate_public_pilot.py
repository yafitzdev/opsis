"""Inventory the canonical archives and build a small, balanced parser pilot."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from rag_image_parser.public_data import (
    ai2d_description,
    chart_description,
    pubtabnet_markdown,
)

SEED = "rag-image-parser-public-pilot-v1"


def stable_rank(value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{value}".encode()).digest()[:8], "big")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def cache_pubtabnet_annotations(archive: Path, work_dir: Path) -> tuple[Path, dict[str, Any]]:
    annotation_path = work_dir / "pubtabnet.jsonl"
    inventory_path = work_dir / "pubtabnet_archive_inventory.json"
    if annotation_path.is_file() and inventory_path.is_file():
        return annotation_path, json.loads(inventory_path.read_text("utf-8"))

    work_dir.mkdir(parents=True, exist_ok=True)
    temporary = annotation_path.with_suffix(".jsonl.tmp")
    image_counts: Counter[str] = Counter()
    annotation_member = None
    members = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            members += 1
            parts = PurePosixPath(member.name).parts
            if member.isfile() and member.name.casefold().endswith(".png") and len(parts) >= 3:
                image_counts[parts[-2]] += 1
            if member.isfile() and member.name.casefold().endswith(".jsonl"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Could not read {member.name} from PubTabNet")
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(extracted, destination)
                annotation_member = member.name
    if annotation_member is None:
        raise RuntimeError("PubTabNet archive contains no JSONL annotation file")
    os.replace(temporary, annotation_path)
    inventory = {
        "archive_members": members,
        "image_counts": dict(sorted(image_counts.items())),
        "annotation_member": annotation_member,
        "annotation_bytes": annotation_path.stat().st_size,
    }
    write_json(inventory_path, inventory)
    return annotation_path, inventory


def offer_candidate(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]],
    bucket: str,
    identity: str,
    value: dict[str, Any],
    capacity: int,
) -> None:
    rank = stable_rank(identity)
    candidate = (-rank, identity, value)
    heap = heaps[bucket]
    if len(heap) < capacity:
        heapq.heappush(heap, candidate)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, candidate)


def take_balanced(
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]], count: int
) -> list[dict[str, Any]]:
    buckets = {
        key: sorted(((-rank, identity, value) for rank, identity, value in values))
        for key, values in heaps.items()
    }
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count and keys:
        remaining: list[str] = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < count:
                selected.append(values.pop(0)[2])
            if values:
                remaining.append(key)
        keys = remaining
    return selected


def select_pubtabnet(
    annotations: Path, train_count: int, eval_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    capacities = max((max(train_count, eval_count) + 2) // 3, 4)
    heaps: dict[str, dict[str, list[tuple[int, str, dict[str, Any]]]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
    }
    split_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    with annotations.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid PubTabNet JSON at line {line_number}") from exc
            split = str(record.get("split", ""))
            split_counts[split] += 1
            markdown, reason = pubtabnet_markdown(record)
            reasons[reason] += 1
            if markdown is None or split not in heaps:
                continue
            rows = markdown.count("\n")
            columns = markdown.splitlines()[0].count("|") - 1
            row_bucket = "short" if rows <= 6 else "medium" if rows <= 16 else "long"
            column_bucket = "narrow" if columns <= 4 else "wide"
            bucket = f"{row_bucket}-{column_bucket}"
            filename = str(record["filename"])
            value = {
                "source": "pubtabnet",
                "source_split": split,
                "filename": filename,
                "target": markdown,
                "bucket": bucket,
            }
            offer_candidate(heaps[split], bucket, filename, value, capacities)
    train = take_balanced(heaps["train"], train_count)
    evaluation = take_balanced(heaps["val"], eval_count)
    report = {
        "annotation_counts": dict(sorted(split_counts.items())),
        "curation_outcomes": dict(reasons.most_common()),
        "eligible_simple_tables": reasons["accepted"],
        "selected_train": len(train),
        "selected_evaluation": len(evaluation),
    }
    return train, evaluation, report


def extract_pubtabnet_images(
    archive: Path, selected: list[dict[str, Any]], destination: Path
) -> None:
    wanted: dict[str, tuple[Path, dict[str, Any]]] = {}
    for item in selected:
        source_name = f"pubtabnet/{item['source_split']}/{item['filename']}"
        target = destination / f"pubtabnet_{item['source_split']}_{item['filename']}"
        item["image"] = target
        if not target.is_file():
            wanted[source_name] = (target, item)
    if not wanted:
        return
    missing = set(wanted)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in missing:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target = wanted[member.name][0]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            missing.remove(member.name)
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Missing selected PubTabNet images: {sorted(missing)}")


def _zip_text(archive: zipfile.ZipFile, member: str) -> str:
    return archive.read(member).decode("utf-8-sig", errors="replace")


def select_chartqa(
    archive_path: Path, train_count: int, eval_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    heaps_by_split: dict[str, dict[str, list[tuple[int, str, dict[str, Any]]]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    outcomes: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        annotations = sorted(
            name
            for name in names
            if "/annotations/" in name and name.casefold().endswith(".json")
        )
        for name in annotations:
            parts = PurePosixPath(name).parts
            split = parts[-3]
            if split not in heaps_by_split:
                continue
            stem = PurePosixPath(name).stem
            table_member = str(PurePosixPath(*parts[:-2]) / "tables" / f"{stem}.csv")
            image_member = str(PurePosixPath(*parts[:-2]) / "png" / f"{stem}.png")
            if table_member not in names or image_member not in names:
                outcomes["missing_table_or_image"] += 1
                continue
            annotation = json.loads(_zip_text(archive, name))
            target = chart_description(annotation, _zip_text(archive, table_member))
            if target is None:
                outcomes["unusable_annotation"] += 1
                continue
            chart_type = str(annotation.get("type", "unknown"))
            type_counts[chart_type] += 1
            outcomes["accepted"] += 1
            value = {
                "source": "chartqa",
                "source_split": split,
                "filename": f"{stem}.png",
                "image_member": image_member,
                "target": target.text,
                "required_terms": list(target.required_terms),
                "required_relations": [],
                "bucket": chart_type,
            }
            offer_candidate(
                heaps_by_split[split], chart_type, f"{split}/{stem}", value, max(8, train_count)
            )
        train = take_balanced(heaps_by_split["train"], train_count)
        evaluation = take_balanced(heaps_by_split["test"], eval_count)
        for item in [*train, *evaluation]:
            target = Path(item["filename"])
            item["destination"] = f"chartqa_{item['source_split']}_{target.name}"
            with archive.open(item["image_member"]) as source:
                item["image_bytes"] = source.read()
    report = {
        "archive_entries": len(names),
        "curation_outcomes": dict(outcomes.most_common()),
        "eligible_by_type": dict(sorted(type_counts.items())),
        "selected_train": len(train),
        "selected_evaluation": len(evaluation),
    }
    return train, evaluation, report


def select_ai2d(
    archive_path: Path, train_count: int, eval_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    outcomes: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        categories = json.loads(_zip_text(archive, "ai2d/categories.json"))
        annotations = sorted(
            name
            for name in names
            if name.startswith("ai2d/annotations/") and name.endswith(".json")
        )
        for name in annotations:
            filename = PurePosixPath(name).name.removesuffix(".json")
            image_member = f"ai2d/images/{filename}"
            if image_member not in names:
                outcomes["missing_image"] += 1
                continue
            category = str(categories.get(filename, "unknown"))
            annotation = json.loads(_zip_text(archive, name))
            target = ai2d_description(annotation, category)
            if target is None:
                outcomes["no_resolvable_labeled_relation"] += 1
                continue
            outcomes["accepted"] += 1
            category_counts[category] += 1
            value = {
                "source": "ai2d",
                "source_split": "deterministic-local",
                "filename": filename,
                "image_member": image_member,
                "target": target.text,
                "required_terms": list(target.required_terms),
                "required_relations": [
                    {"source": source, "target": destination}
                    for source, destination in target.required_relations
                ],
                "bucket": category,
            }
            offer_candidate(heaps, category, filename, value, train_count + eval_count + 4)
        combined = take_balanced(heaps, train_count + eval_count)
        train = combined[:train_count]
        evaluation = combined[train_count:]
        for item in combined:
            item["destination"] = f"ai2d_{item['filename']}"
            with archive.open(item["image_member"]) as source:
                item["image_bytes"] = source.read()
    report = {
        "archive_entries": len(names),
        "image_count": sum(name.startswith("ai2d/images/") for name in names),
        "annotation_count": len(annotations),
        "curation_outcomes": dict(outcomes.most_common()),
        "eligible_by_category": dict(category_counts.most_common()),
        "selected_train": len(train),
        "selected_evaluation": len(evaluation),
    }
    return train, evaluation, report


def persist_zip_images(selected: list[dict[str, Any]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in selected:
        target = destination / item.pop("destination")
        target.write_bytes(item.pop("image_bytes"))
        item["image"] = target
        item.pop("image_member", None)


def concise_docci_description(value: str, maximum_words: int = 90) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(value.split()))
    chosen: list[str] = []
    words = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if chosen and words + len(sentence_words) > maximum_words:
            break
        chosen.append(sentence)
        words += len(sentence_words)
        if words >= 45:
            break
    description = " ".join(chosen).strip()
    if len(description.split()) > maximum_words:
        description = " ".join(description.split()[:maximum_words]).rstrip(" ,;:") + "."
    return description


def docci_required_terms(description: str) -> list[str]:
    stop = {
        "about",
        "against",
        "along",
        "angled",
        "behind",
        "close",
        "colored",
        "facing",
        "front",
        "image",
        "looking",
        "medium",
        "placed",
        "right",
        "shows",
        "slightly",
        "there",
        "towards",
        "visible",
        "wearing",
        "which",
        "white",
    }
    terms = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", description)
        if word.casefold() not in stop
    ]
    return list(dict.fromkeys(terms))[:8]


def select_docci(
    annotations_path: Path, train_count: int, eval_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    heaps: dict[str, dict[str, list[tuple[int, str, dict[str, Any]]]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }
    split_counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid DOCCI JSON at line {line_number}") from exc
            split = str(record.get("split", ""))
            split_counts[split] += 1
            if split not in heaps:
                continue
            filename = str(record.get("image_file", ""))
            description = concise_docci_description(str(record.get("description", "")))
            if not filename or len(description.split()) < 8:
                outcomes["missing_or_short_description"] += 1
                continue
            outcomes["accepted"] += 1
            value = {
                "source": "docci",
                "source_split": split,
                "filename": filename,
                "target": description,
                "required_terms": docci_required_terms(description),
                "required_relations": [],
                "bucket": "ordinary-image",
            }
            capacity = max(train_count, eval_count, 8)
            offer_candidate(heaps[split], "ordinary-image", filename, value, capacity)
    train = take_balanced(heaps["train"], train_count)
    evaluation = take_balanced(heaps["test"], eval_count)
    report = {
        "annotation_counts": dict(sorted(split_counts.items())),
        "curation_outcomes": dict(outcomes.most_common()),
        "selected_train": len(train),
        "selected_evaluation": len(evaluation),
        "target_maximum_words": 90,
    }
    return train, evaluation, report


def extract_docci_images(
    archive_path: Path, selected: list[dict[str, Any]], destination: Path
) -> None:
    wanted: dict[str, Path] = {}
    for item in selected:
        target = destination / f"docci_{item['filename']}"
        item["image"] = target
        if not target.is_file():
            wanted[item["filename"]] = target
    if not wanted:
        return
    missing = set(wanted)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            filename = PurePosixPath(member.name).name
            if filename not in missing or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target = wanted[filename]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            missing.remove(filename)
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Missing selected DOCCI images: {sorted(missing)[:10]}")


def manifest_image(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def build_manifests(
    output_dir: Path,
    training: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
) -> None:
    training_manifest = []
    validation_manifest = []
    benchmark_manifest = []
    metadata = []
    for item in training:
        kind = "table" if item["source"] == "pubtabnet" else "description"
        target = item["target"]
        tagged = f"<table>\n{target}\n</table>" if kind == "table" else (
            f"<description>{target}</description>"
        )
        training_manifest.append(
            {
                "image": manifest_image(item["image"], output_dir),
                "output": tagged,
                "kind": kind,
            }
        )
        metadata.append({**item, "image": str(item["image"]), "pilot_split": "train"})
    for item in evaluation:
        kind = "table" if item["source"] == "pubtabnet" else "description"
        target = item["target"]
        tagged = f"<table>\n{target}\n</table>" if kind == "table" else (
            f"<description>{target}</description>"
        )
        validation_manifest.append(
            {
                "image": manifest_image(item["image"], output_dir),
                "output": tagged,
                "kind": kind,
            }
        )
        case: dict[str, Any] = {
            "image": manifest_image(item["image"], output_dir),
            "expected_kind": kind,
        }
        if kind == "table":
            case["expected_text"] = item["target"]
        else:
            case["required_terms"] = item.get("required_terms", [])
            case["required_relations"] = item.get("required_relations", [])
        benchmark_manifest.append(case)
        metadata.append({**item, "image": str(item["image"]), "pilot_split": "evaluation"})
    write_jsonl(output_dir / "train.jsonl", training_manifest)
    write_jsonl(output_dir / "validation.jsonl", validation_manifest)
    write_jsonl(output_dir / "benchmark.jsonl", benchmark_manifest)
    write_jsonl(output_dir / "metadata.jsonl", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/public_data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/public_data/pilot"))
    parser.add_argument("--train-per-kind", type=int, default=8)
    parser.add_argument("--eval-per-kind", type=int, default=3)
    parser.add_argument("--train-tables", type=int)
    parser.add_argument("--train-charts", type=int)
    parser.add_argument("--train-diagrams", type=int)
    parser.add_argument("--train-images", type=int, default=0)
    parser.add_argument("--eval-tables", type=int)
    parser.add_argument("--eval-charts", type=int)
    parser.add_argument("--eval-diagrams", type=int)
    parser.add_argument("--eval-images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    work_dir = args.raw_dir.resolve().parent / "work"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    train_tables = args.train_tables if args.train_tables is not None else args.train_per_kind
    train_charts = args.train_charts if args.train_charts is not None else args.train_per_kind
    train_diagrams = (
        args.train_diagrams if args.train_diagrams is not None else args.train_per_kind
    )
    eval_tables = args.eval_tables if args.eval_tables is not None else args.eval_per_kind
    eval_charts = args.eval_charts if args.eval_charts is not None else args.eval_per_kind
    eval_diagrams = args.eval_diagrams if args.eval_diagrams is not None else args.eval_per_kind
    requested_counts = [
        train_tables,
        train_charts,
        train_diagrams,
        args.train_images,
        eval_tables,
        eval_charts,
        eval_diagrams,
        args.eval_images,
    ]
    if any(count < 0 for count in requested_counts):
        raise ValueError("Curation counts cannot be negative")

    pub_archive = args.raw_dir / "pubtabnet.tar.gz"
    chart_archive = args.raw_dir / "chartqa.zip"
    ai2d_archive = args.raw_dir / "ai2d-all.zip"
    docci_archive = args.raw_dir / "docci_images.tar.gz"
    docci_annotations = args.raw_dir / "docci_descriptions.jsonlines"
    for archive in (pub_archive, chart_archive, ai2d_archive):
        if not archive.is_file():
            raise FileNotFoundError(f"Missing complete source archive: {archive}")

    print("Caching PubTabNet annotations and inventory...", flush=True)
    pub_annotations, pub_inventory = cache_pubtabnet_annotations(pub_archive, work_dir)
    pub_train, pub_eval, pub_report = select_pubtabnet(
        pub_annotations, train_tables, eval_tables
    )
    if len(pub_train) != train_tables or len(pub_eval) != eval_tables:
        raise RuntimeError("PubTabNet did not contain enough eligible examples")
    print("Extracting selected PubTabNet table images...", flush=True)
    extract_pubtabnet_images(pub_archive, [*pub_train, *pub_eval], image_dir)

    print("Curating ChartQA charts...", flush=True)
    chart_train, chart_eval, chart_report = select_chartqa(
        chart_archive, train_charts, eval_charts
    )
    if len(chart_train) != train_charts or len(chart_eval) != eval_charts:
        raise RuntimeError("ChartQA did not contain enough eligible examples")
    persist_zip_images([*chart_train, *chart_eval], image_dir)

    print("Curating AI2D diagrams...", flush=True)
    diagram_train, diagram_eval, ai2d_report = select_ai2d(
        ai2d_archive, train_diagrams, eval_diagrams
    )
    if len(diagram_train) != train_diagrams or len(diagram_eval) != eval_diagrams:
        raise RuntimeError("AI2D did not contain enough eligible examples")
    persist_zip_images([*diagram_train, *diagram_eval], image_dir)

    docci_train: list[dict[str, Any]] = []
    docci_eval: list[dict[str, Any]] = []
    docci_report: dict[str, Any] | None = None
    if args.train_images or args.eval_images:
        for source in (docci_archive, docci_annotations):
            if not source.is_file():
                raise FileNotFoundError(f"Missing complete DOCCI source: {source}")
        print("Curating DOCCI ordinary images...", flush=True)
        docci_train, docci_eval, docci_report = select_docci(
            docci_annotations, args.train_images, args.eval_images
        )
        if len(docci_train) != args.train_images or len(docci_eval) != args.eval_images:
            raise RuntimeError("DOCCI did not contain enough eligible examples")
        extract_docci_images(docci_archive, [*docci_train, *docci_eval], image_dir)

    training = [*pub_train, *chart_train, *diagram_train, *docci_train]
    evaluation = [*pub_eval, *chart_eval, *diagram_eval, *docci_eval]
    build_manifests(output_dir, training, evaluation)
    selected_images = {item["image"].resolve() for item in [*training, *evaluation]}
    for existing in image_dir.iterdir():
        if existing.is_file() and existing.resolve() not in selected_images:
            existing.unlink()
    report = {
        "schema_version": 1,
        "seed": SEED,
        "selected": {"training": len(training), "evaluation": len(evaluation)},
        "pubtabnet": {"archive": pub_inventory, **pub_report},
        "chartqa": chart_report,
        "ai2d": ai2d_report,
        "docci": docci_report,
        "license_flags": [
            "AI2D is local research/evaluation-only and cannot be redistributed or used "
            "for a for-profit model without AllenAI permission.",
            "ChartQA is GPL-3.0; obtain legal guidance before distributing derived weights.",
            "PubTabNet is CDLA-Sharing-1.0 and carries source-data sharing obligations.",
            "DOCCI is CC-BY-4.0 and requires attribution when redistributed.",
        ],
        "known_scope_gaps": [
            "DOCCI supplies ordinary photographs, but screenshots and document figures "
            "remain absent.",
            "Simple-table pilot deliberately excludes merged cells and very large tables.",
            "Chart descriptions are deterministic targets, not human-written summaries.",
            "AI2D covers educational diagrams, not business or software architecture diagrams.",
            "The sources do not represent noisy scans, mobile photos, handwriting, or "
            "non-English text.",
        ],
    }
    write_json(output_dir / "curation_report.json", report)
    print(
        f"Pilot ready: {len(training)} training and {len(evaluation)} evaluation images at "
        f"{output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
