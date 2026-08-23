# Public-data curation

The first public-data pool uses four complete, revision-pinned sources. The raw archives and
all derived images live below `artifacts/public_data/`, which is intentionally ignored by Git.

| Source | Parser role | Raw artifact | License warning |
|---|---|---:|---|
| PubTabNet | table image to Markdown | 11,244,059,914 bytes | CDLA-Sharing-1.0 |
| ChartQA | chart image to concise factual prose | 875,370,872 bytes | GPL-3.0 |
| AI2D | labeled diagram relationships to prose | 990,965,374 bytes | bundled research-only license |
| DOCCI | ordinary image to concise prose | 7,603,938,982 bytes | CC-BY-4.0 |

AI2D's license file permits academic, nonprofit, government-sponsored research and evaluation,
but forbids for-profit use and redistribution without AllenAI permission. Treat its records as an
evaluation/research pool, not production training data. ChartQA and PubTabNet also need a legal
review before distributing data or derived model weights. DOCCI requires attribution.

## Reproduce the download

```powershell
uv run python .\scripts\download_public_data.py
```

The downloader resumes `.part` files, pins source revisions, checks expected byte counts, verifies
published SHA-256 hashes when available, computes the AI2D hash locally, and writes
`artifacts/public_data/raw/download_manifest.json`.

## Build the pilot

```powershell
uv run python .\scripts\curate_public_pilot.py
```

The curator scans every annotation without modifying or fully unpacking the source archives. It
creates a deliberately small pilot by default:

- 8 training and 3 held-out table images;
- 8 training and 3 held-out chart images;
- 8 training and 3 held-out diagram images.

Selection is deterministic and balanced by table geometry, chart type, and diagram category.
Simple tables must be rectangular and fit Markdown's limitations; merged-cell tables are counted
but pruned from this first pass. Chart targets are generated from source tables and element
annotations. Diagram targets only include relationships whose endpoints can be resolved to visible
labels.

The first meaningful training corpus can be reproduced with:

```powershell
uv run python .\scripts\curate_public_pilot.py `
  --output-dir .\artifacts\public_data\training_v1 `
  --train-tables 10000 --train-charts 1500 `
  --train-diagrams 1000 --train-images 2500 `
  --eval-tables 200 --eval-charts 150 `
  --eval-diagrams 150 --eval-images 200
```

This produces exactly 15,000 training and 700 validation examples. DOCCI targets are truncated at
a sentence boundary to at most 90 words so they remain discovery descriptions rather than generic
long captions.

Outputs under `artifacts/public_data/pilot/` are:

- `train.jsonl`: model-ready tagged completions;
- `validation.jsonl`: held-out model-ready tagged completions;
- `benchmark.jsonl`: held-out exact-table and semantic-description checks;
- `metadata.jsonl`: provenance and curation bucket for every selected record;
- `curation_report.json`: full-pool counts, rejection reasons, licensing flags, and known gaps;
- `images/`: only the selected pilot images.

Run the CPU baseline on the held-out sample with:

```powershell
uv run rag-image-parse benchmark .\artifacts\public_data\pilot\benchmark.jsonl `
  --image-longest-edge 1024 --max-new-tokens 256 --json
```
