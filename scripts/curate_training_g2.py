"""Build the balanced 50k Opsis v1-nano-g2 training corpus."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import AutoTokenizer

from rag_image_parser.config import DEFAULT_MODEL_ID
from scripts.curate_public_pilot import (
    cache_pubtabnet_annotations,
    extract_docci_images,
    extract_pubtabnet_images,
    persist_zip_images,
    select_ai2d,
    select_chartqa,
    select_docci,
    select_pubtabnet,
    write_json,
    write_jsonl,
)

# Frozen under the original experiment label to reproduce the published g2 corpus exactly.
SEED = "pyrrho-opsis-v1-nano-g2-50k"
TRAIN_TOP_LEVEL = {
    "table": 12_500,
    "chart": 6_250,
    "diagram": 6_250,
    "image": 12_500,
    "pdf": 12_500,
}
TRAIN_PDF_CONTENT = {"table": 3_125, "chart": 3_125, "diagram": 3_125, "image": 3_125}
VALIDATION_TOP_LEVEL = {
    "table": 250,
    "chart": 125,
    "diagram": 125,
    "image": 250,
    "pdf": 250,
}
VALIDATION_PDF_CONTENT = {"table": 63, "chart": 63, "diagram": 62, "image": 62}


@dataclass(frozen=True, slots=True)
class RenderJob:
    source: Path
    destination: Path
    domain: str
    content: str
    variant: int
    identity: str


def stable_rank(value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{value}".encode()).digest()[:8], "big")


def source_identity(item: dict[str, Any]) -> str:
    return f"{item['source']}:{item.get('source_split', '')}:{item['filename']}"


def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: stable_rank(source_identity(item)))


def image_sha256(item: dict[str, Any]) -> str:
    return hashlib.sha256(Path(item["image"]).read_bytes()).hexdigest()


def deduplicate_image_splits(
    training: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    seen: set[str] = set()
    unique_training: list[dict[str, Any]] = []
    duplicate_training = 0
    for item in ordered(training):
        digest = image_sha256(item)
        if digest in seen:
            duplicate_training += 1
            continue
        seen.add(digest)
        unique_training.append(item)

    unique_evaluation: list[dict[str, Any]] = []
    duplicate_evaluation = 0
    cross_split_rejections = 0
    evaluation_seen: set[str] = set()
    for item in ordered(evaluation):
        digest = image_sha256(item)
        if digest in seen:
            cross_split_rejections += 1
            continue
        if digest in evaluation_seen:
            duplicate_evaluation += 1
            continue
        evaluation_seen.add(digest)
        unique_evaluation.append(item)
    return unique_training, unique_evaluation, {
        "duplicate_training_images_removed": duplicate_training,
        "duplicate_evaluation_images_removed": duplicate_evaluation,
        "cross_split_images_removed": cross_split_rejections,
    }


def filter_table_targets(
    items: list[dict[str, Any]], tokenizer: Any, maximum_tokens: int
) -> tuple[list[dict[str, Any]], list[int]]:
    accepted: list[dict[str, Any]] = []
    lengths: list[int] = []
    batch_size = 256
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        targets = [f"<table>\n{item['target']}\n</table>" for item in batch]
        encoded = tokenizer(targets, add_special_tokens=False)["input_ids"]
        for item, token_ids in zip(batch, encoded, strict=True):
            length = len(token_ids)
            if length <= maximum_tokens:
                accepted.append(item)
                lengths.append(length)
    return accepted, lengths


def allocate(
    base: list[dict[str, Any]], count: int, *, offset: int = 0
) -> list[tuple[dict[str, Any], int]]:
    if not base:
        raise ValueError("Cannot allocate variants from an empty source pool")
    values = ordered(base)
    allocated = []
    for index in range(count):
        position = index + offset
        allocated.append((values[position % len(values)], position // len(values)))
    return allocated


def _digest(job: RenderJob) -> bytes:
    return hashlib.sha256(
        f"{SEED}:{job.domain}:{job.content}:{job.variant}:{job.identity}".encode()
    ).digest()


def _cap_image(image: Image.Image, maximum: int = 2048) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    return ImageOps.contain(image, (maximum, maximum), Image.Resampling.LANCZOS)


def _render_clean(image: Image.Image, digest: bytes, variant: int) -> Image.Image:
    image = _cap_image(image)
    if variant == 0:
        return image
    contrast = 0.96 + (digest[0] / 255) * 0.08
    brightness = 0.97 + (digest[1] / 255) * 0.06
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    scale = 0.82 + (digest[2] / 255) * 0.17
    resized = (
        max(64, round(image.width * scale)),
        max(64, round(image.height * scale)),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def _render_pdf(image: Image.Image, digest: bytes) -> Image.Image:
    image = _cap_image(image, maximum=1800)
    scale = 0.58 + (digest[0] / 255) * 0.34
    resized = (
        max(64, round(image.width * scale)),
        max(64, round(image.height * scale)),
    )
    image = image.resize(resized, Image.Resampling.LANCZOS)
    if digest[1] % 4 == 0:
        image = ImageOps.grayscale(image).convert("RGB")
    contrast = 0.86 + (digest[2] / 255) * 0.22
    brightness = 0.94 + (digest[3] / 255) * 0.10
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    if digest[4] % 3 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.25 + digest[5] / 510))
    margin_x = max(8, round(image.width * (0.015 + digest[6] / 8500)))
    margin_y = max(8, round(image.height * (0.015 + digest[7] / 8500)))
    paper = 245 + digest[8] % 11
    canvas = Image.new(
        "RGB",
        (image.width + 2 * margin_x, image.height + 2 * margin_y),
        (paper, paper, max(238, paper - digest[9] % 5)),
    )
    canvas.paste(image, (margin_x, margin_y))
    return canvas


def render_job(job: RenderJob) -> None:
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    if job.destination.is_file():
        return
    with Image.open(job.source) as opened:
        image = opened.convert("RGB")
    digest = _digest(job)
    if job.domain == "pdf":
        image = _render_pdf(image, digest)
        image.save(job.destination, format="JPEG", quality=62 + digest[10] % 29, optimize=True)
    elif job.content == "image":
        image = _render_clean(image, digest, job.variant)
        image.save(job.destination, format="JPEG", quality=88 + digest[10] % 8, optimize=True)
    else:
        image = _render_clean(image, digest, job.variant)
        image.save(job.destination, format="PNG", optimize=True)


def tagged_target(item: dict[str, Any], content: str) -> tuple[str, str]:
    if content == "table":
        return "table", f"<table>\n{item['target']}\n</table>"
    return "description", f"<description>{item['target']}</description>"


def add_bucket(
    *,
    records: list[dict[str, Any]],
    jobs: list[RenderJob],
    output_dir: Path,
    split: str,
    domain: str,
    content: str,
    base: list[dict[str, Any]],
    count: int,
    offset: int,
) -> None:
    extension = ".jpg" if domain == "pdf" or content == "image" else ".png"
    for index, (item, variant) in enumerate(allocate(base, count, offset=offset)):
        identity = source_identity(item)
        relative = Path("images") / split / domain / content / f"{index:06d}{extension}"
        destination = output_dir / relative
        kind, output = tagged_target(item, content)
        records.append(
            {
                "image": relative.as_posix(),
                "output": output,
                "kind": kind,
                "content": content,
                "domain": domain,
                "source": item["source"],
                "source_id": identity,
                "variant": variant,
                "target": item["target"],
                "required_terms": item.get("required_terms", []),
                "required_relations": item.get("required_relations", []),
            }
        )
        jobs.append(
            RenderJob(
                source=Path(item["image"]),
                destination=destination,
                domain=domain,
                content=content,
                variant=variant,
                identity=identity,
            )
        )


def benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "image": record["image"],
        "expected_kind": record["kind"],
    }
    if record["kind"] == "table":
        value["expected_text"] = record["target"]
    else:
        value["required_terms"] = record["required_terms"]
        value["required_relations"] = record["required_relations"]
    return value


def manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("image", "output", "kind")}


def count_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_level = Counter(
        "pdf" if record["domain"] == "pdf" else record["content"] for record in records
    )
    content = Counter(record["content"] for record in records)
    domains = Counter(record["domain"] for record in records)
    unique_sources = defaultdict(set)
    for record in records:
        unique_sources[record["content"]].add(record["source_id"])
    return {
        "total": len(records),
        "top_level": dict(sorted(top_level.items())),
        "effective_content": dict(sorted(content.items())),
        "domains": dict(sorted(domains.items())),
        "unique_sources": {
            key: len(value) for key, value in sorted(unique_sources.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/public_data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/public_data/training_g2_50k")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--maximum-table-tokens", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = args.output_dir.resolve()
    if (output_dir / "train.jsonl").is_file():
        raise FileExistsError(f"Refusing to overwrite existing g2 corpus: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.raw_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Selecting token-bounded PubTabNet sources...", flush=True)
    annotations, pub_inventory = cache_pubtabnet_annotations(
        raw_dir / "pubtabnet.tar.gz", raw_dir.parent / "work"
    )
    pub_train_raw, pub_eval_raw, pub_report = select_pubtabnet(annotations, 22_000, 1_000)
    pub_train, train_table_lengths = filter_table_targets(
        pub_train_raw, tokenizer, args.maximum_table_tokens
    )
    pub_eval, eval_table_lengths = filter_table_targets(
        pub_eval_raw, tokenizer, args.maximum_table_tokens
    )
    needed_pub_train = TRAIN_TOP_LEVEL["table"] + TRAIN_PDF_CONTENT["table"]
    needed_pub_eval = VALIDATION_TOP_LEVEL["table"] + VALIDATION_PDF_CONTENT["table"]
    if len(pub_train) < needed_pub_train or len(pub_eval) < needed_pub_eval:
        raise RuntimeError(
            "Insufficient token-bounded PubTabNet records: "
            f"train {len(pub_train)}/{needed_pub_train}, eval {len(pub_eval)}/{needed_pub_eval}"
        )
    pub_train = pub_train[:needed_pub_train]
    pub_eval = pub_eval[:needed_pub_eval]
    extract_pubtabnet_images(
        raw_dir / "pubtabnet.tar.gz", [*pub_train, *pub_eval], source_dir / "tables"
    )

    print("Selecting all clean ChartQA sources...", flush=True)
    chart_train, chart_eval, chart_report = select_chartqa(
        raw_dir / "chartqa.zip", 25_000, 25_000
    )
    persist_zip_images([*chart_train, *chart_eval], source_dir / "charts")
    chart_train, chart_eval, chart_dedup_report = deduplicate_image_splits(
        chart_train, chart_eval
    )
    chart_report["image_hash_deduplication"] = chart_dedup_report

    print("Selecting and source-splitting AI2D diagrams...", flush=True)
    diagram_train, diagram_eval, diagram_report = select_ai2d(
        raw_dir / "ai2d-all.zip", 985, 188
    )
    persist_zip_images([*diagram_train, *diagram_eval], source_dir / "diagrams")

    print("Selecting all DOCCI ordinary-image sources...", flush=True)
    image_train, image_eval, image_report = select_docci(
        raw_dir / "docci_descriptions.jsonlines", 9_647, 5_000
    )
    needed_eval_images = VALIDATION_TOP_LEVEL["image"] + VALIDATION_PDF_CONTENT["image"]
    image_eval = ordered(image_eval)[:needed_eval_images]
    extract_docci_images(
        raw_dir / "docci_images.tar.gz",
        [*image_train, *image_eval],
        source_dir / "images",
    )

    pools = {
        "train": {
            "table": pub_train,
            "chart": chart_train,
            "diagram": diagram_train,
            "image": image_train,
        },
        "validation": {
            "table": pub_eval,
            "chart": chart_eval,
            "diagram": diagram_eval,
            "image": image_eval,
        },
    }
    for split, split_pools in pools.items():
        for content, pool in split_pools.items():
            if not pool:
                raise RuntimeError(f"Empty {split} source pool for {content}")

    records_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    jobs: list[RenderJob] = []
    for split, top_level, pdf_content in (
        ("train", TRAIN_TOP_LEVEL, TRAIN_PDF_CONTENT),
        ("validation", VALIDATION_TOP_LEVEL, VALIDATION_PDF_CONTENT),
    ):
        clean_offset = 0
        for content in ("table", "chart", "diagram", "image"):
            add_bucket(
                records=records_by_split[split],
                jobs=jobs,
                output_dir=output_dir,
                split=split,
                domain="clean",
                content=content,
                base=pools[split][content],
                count=top_level[content],
                offset=clean_offset,
            )
            add_bucket(
                records=records_by_split[split],
                jobs=jobs,
                output_dir=output_dir,
                split=split,
                domain="pdf",
                content=content,
                base=pools[split][content],
                count=pdf_content[content],
                offset=top_level[content],
            )

    train_sources = {record["source_id"] for record in records_by_split["train"]}
    validation_sources = {
        record["source_id"] for record in records_by_split["validation"]
    }
    overlap = train_sources.intersection(validation_sources)
    if overlap:
        raise RuntimeError(f"Source leakage across train/validation: {sorted(overlap)[:5]}")
    if len(records_by_split["train"]) != 50_000:
        raise RuntimeError("The g2 training corpus must contain exactly 50,000 files")
    if len(records_by_split["validation"]) != 1_000:
        raise RuntimeError("The g2 validation corpus must contain exactly 1,000 files")

    print(f"Rendering {len(jobs):,} deterministic image files...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, _ in enumerate(executor.map(render_job, jobs), 1):
            if completed % 2_500 == 0:
                print(f"  rendered {completed:,}/{len(jobs):,}", flush=True)

    for split, records in records_by_split.items():
        records.sort(key=lambda record: stable_rank(f"manifest:{record['image']}"))
        for record in records:
            if not (output_dir / record["image"]).is_file():
                raise RuntimeError(f"Missing rendered image: {record['image']}")
        write_jsonl(
            output_dir / f"{split}.jsonl",
            [manifest_record(record) for record in records],
        )
        write_jsonl(output_dir / f"{split}_metadata.jsonl", records)
    write_jsonl(
        output_dir / "benchmark.jsonl",
        [benchmark_record(record) for record in records_by_split["validation"]],
    )

    report = {
        "schema_version": 2,
        "release": "opsis-v1-nano-g2",
        "seed": SEED,
        "maximum_table_tokens": args.maximum_table_tokens,
        "training": count_report(records_by_split["train"]),
        "validation": count_report(records_by_split["validation"]),
        "table_token_lengths": {
            "candidate_train_min": min(train_table_lengths),
            "candidate_train_max": max(train_table_lengths),
            "candidate_eval_min": min(eval_table_lengths),
            "candidate_eval_max": max(eval_table_lengths),
        },
        "source_reports": {
            "pubtabnet": {"archive": pub_inventory, **pub_report},
            "chartqa": chart_report,
            "ai2d": diagram_report,
            "docci": image_report,
        },
        "notes": [
            "PDF is a rendering domain, not a fifth output type.",
            "PDF-domain records are evenly stratified across the four semantic content types.",
            "Charts, diagrams, and ordinary images use deterministic variants because their "
            "strictly curated unique source pools are smaller than their requested file counts.",
            "Train/validation separation is enforced at original source-image identity level.",
        ],
    }
    write_json(output_dir / "curation_report.json", report)
    print(f"G2 corpus ready at {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
