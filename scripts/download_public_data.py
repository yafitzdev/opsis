"""Download the complete public source archives used for data curation.

Downloads are resumable, source revisions are pinned, and completed files are
verified before they are recorded in the local manifest. Raw data lives under
``artifacts/`` so it cannot accidentally be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_BYTES = 4 * 1024 * 1024
REPORT_EVERY_BYTES = 256 * 1024 * 1024
USER_AGENT = "fitz-parse-public-data/0.1"


@dataclass(frozen=True)
class Source:
    id: str
    artifact: str
    url: str
    expected_bytes: int
    sha256: str | None
    license: str
    upstream: str
    purpose: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Source:
        return cls(**value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _open_download(source: Source, offset: int):
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(source.url, headers=headers)
    return urllib.request.urlopen(request, timeout=120)  # noqa: S310


def download_one(source: Source, output_dir: Path, retries: int = 5) -> dict[str, Any]:
    target = output_dir / source.artifact
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists() and target.stat().st_size == source.expected_bytes:
        digest = sha256_file(target)
        if source.sha256 is None or digest == source.sha256:
            print(f"[{source.id}] already verified: {target}", flush=True)
            return record(source, target, digest)
        raise RuntimeError(f"[{source.id}] existing file has the wrong SHA-256: {digest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with _open_download(source, offset) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    print(f"[{source.id}] server did not resume; restarting", flush=True)
                    offset = 0
                    partial.unlink(missing_ok=True)
                    response.close()
                    with _open_download(source, 0) as restarted:
                        _stream(source, restarted, partial, 0)
                else:
                    _stream(source, response, partial, offset)
            break
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"[{source.id}] download failed after {retries} attempts"
                ) from exc
            delay = min(2**attempt, 30)
            print(
                f"[{source.id}] attempt {attempt} failed ({exc}); retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)

    actual_bytes = partial.stat().st_size
    if actual_bytes != source.expected_bytes:
        raise RuntimeError(
            f"[{source.id}] byte count mismatch: expected {source.expected_bytes}, "
            f"got {actual_bytes}"
        )
    digest = sha256_file(partial)
    if source.sha256 is not None and digest != source.sha256:
        raise RuntimeError(
            f"[{source.id}] SHA-256 mismatch: expected {source.sha256}, got {digest}"
        )
    os.replace(partial, target)
    print(f"[{source.id}] verified {human_bytes(actual_bytes)} sha256={digest}", flush=True)
    return record(source, target, digest)


def _stream(source: Source, response: Any, partial: Path, offset: int) -> None:
    mode = "ab" if offset else "wb"
    downloaded = offset
    next_report = ((downloaded // REPORT_EVERY_BYTES) + 1) * REPORT_EVERY_BYTES
    with partial.open(mode) as handle:
        while chunk := response.read(CHUNK_BYTES):
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                percent = 100 * downloaded / source.expected_bytes
                print(
                    f"[{source.id}] {human_bytes(downloaded)} / "
                    f"{human_bytes(source.expected_bytes)} ({percent:.1f}%)",
                    flush=True,
                )
                next_report += REPORT_EVERY_BYTES


def record(source: Source, target: Path, digest: str) -> dict[str, Any]:
    return {
        "id": source.id,
        "artifact": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": digest,
        "license": source.license,
        "source_url": source.url,
        "upstream": source.upstream,
        "purpose": source.purpose,
    }


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest = output_dir / "download_manifest.json"
    merged = {}
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        merged.update({item["id"]: item for item in existing.get("artifacts", [])})
    merged.update({item["id"]: item for item in records})
    payload = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "artifacts": sorted(merged.values(), key=lambda item: item["id"]),
    }
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/public_data_sources.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/public_data/raw"),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--source", action="append", help="Only download a named source")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = [Source.from_dict(item) for item in config["sources"]]
    if args.source:
        selected = set(args.source)
        sources = [source for source in sources if source.id in selected]
        missing = selected - {source.id for source in sources}
        if missing:
            raise SystemExit(f"Unknown sources: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = sum(source.expected_bytes for source in sources)
    free = shutil.disk_usage(args.output_dir.resolve()).free
    partial = sum(
        (args.output_dir / source.artifact).with_suffix(
            (args.output_dir / source.artifact).suffix + ".part"
        ).stat().st_size
        for source in sources
        if (args.output_dir / source.artifact).with_suffix(
            (args.output_dir / source.artifact).suffix + ".part"
        ).exists()
    )
    remaining = max(0, required - partial)
    if free < remaining + 5 * 1024**3:
        raise SystemExit(
            f"Insufficient free disk: need {human_bytes(remaining)} plus 5 GiB headroom, "
            f"have {human_bytes(free)}"
        )
    print(
        f"Downloading {len(sources)} complete archives ({human_bytes(required)}) "
        f"with {human_bytes(free)} free",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(sources)))) as pool:
        futures = {pool.submit(download_one, source, args.output_dir): source for source in sources}
        for future in as_completed(futures):
            records.append(future.result())
            write_manifest(args.output_dir, records)
    manifest = write_manifest(args.output_dir, records)
    print(f"Manifest: {manifest.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
