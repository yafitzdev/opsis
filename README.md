# Opsis

Opsis is a deliberately small local vision model and parser for one job:

```text
one image -> a concise RAG description OR a Markdown table
```

Normal images, illustrations, charts, and diagrams become factual, searchable prose. Images
whose primary content is a table become GitHub-flavored Markdown tables. PDF extraction and
page-layout reconstruction remain outside the visual model itself. An adaptive PDF router now
uses native PDF geometry to crop and dispatch visual regions without changing that model contract.

## Current status

This repository contains the complete source, training, evaluation, export, and PDF-routing path
used to produce Opsis v1 nano g2. Model weights, ONNX binaries, downloaded datasets, and generated
training artifacts are intentionally not stored in GitHub.

Research checkpoints:

- [`yafitzdev/opsis-v1-nano-g1`](https://huggingface.co/yafitzdev/opsis-v1-nano-g1)
- [`yafitzdev/opsis-v1-nano-g2`](https://huggingface.co/yafitzdev/opsis-v1-nano-g2), recommended

The codebase includes:

- lazy-loaded SmolVLM-256M inference on CPU;
- a balanced 50,000-file corpus spanning tables, charts, diagrams, ordinary images, and
  PDF-style visual crops;
- a completed 1-epoch LoRA run, merged checkpoint, and split ONNX CPU export;
- ONNX Runtime FP32 as the quality-preserving default, with experimental compressed variants;
- a strict, versioned prompt for the two output modes;
- Markdown-table extraction and validation;
- a CLI for parsing and benchmarking;
- relationship-aware diagram evaluation plus p95 latency and peak-memory reporting;
- adaptive PDF routing for clean text, tables, raster images, and vector diagrams;
- deterministic real-data ingestion from image/sidecar pairs;
- a JSONL training contract, LoRA-first training, and adapter merging;
- unit tests that do not download a model.

G2 keeps the same narrow capability as g1 and adds balanced data plus vision-attention LoRA. On a
source-held-out 50-case comparison, it retained 100% routing and valid Markdown, improved table
cell F1 from 31.0% to 50.5%, and improved directed-relation recall from 34.0% to 48.7%. It remains
a research checkpoint: exact table accuracy is 7.1%, and generated values must not be treated as
authoritative. See [`docs/TRAINING_G2.md`](docs/TRAINING_G2.md) for the complete evidence.

## Parse a PDF adaptively

```powershell
uv run rag-image-parse pdf .\document.pdf `
  --model yafitzdev/opsis-v1-nano-g2
```

The router extracts clean native text immediately and loads the visual model only if a page has a
table, substantial embedded image, or diagram-like vector region. On mixed pages, words inside
visual regions are suppressed before the parsed crop is inserted back into page order.

Structured output preserves page numbers, bounding boxes, page classification, quality flags,
and the visual inference count:

```powershell
uv run rag-image-parse pdf .\document.pdf --json
```

Ruled tables use the safer default line-based detector. For borderless tables with aligned text:

```powershell
uv run rag-image-parse pdf .\document.pdf --table-strategy text
```

Reliable digital table grids use PyMuPDF's native cells immediately and record
`native-table-structure-used`. Raster tables still go through our model. Pass
`--vision-for-detected-tables` to force a visual attempt for digital tables too; a malformed or
disagreeing result is recovered from the native cells. Full-page scans are flagged as
`scanned-page-needs-ocr-layout`; they are not silently reduced to incomplete image descriptions.
Use `--fail-on-scanned-pages` when that should stop a batch.

See [`docs/PDF_ROUTER.md`](docs/PDF_ROUTER.md) for the routing contract and programmatic text-parser
integration.

## Install

Python 3.10+ is supported. With `uv`:

```powershell
uv sync --extra dev --extra train --extra export
```

The first real inference downloads the selected model from Hugging Face. Model execution is
fully local after the files are cached. While the research checkpoints remain private, authenticate
with a Hugging Face account that has access before running these commands.

## Parse an image

```powershell
uv run rag-image-parse parse .\path\to\image.png `
  --model yafitzdev/opsis-v1-nano-g2
```

Machine-readable output:

```powershell
uv run rag-image-parse parse .\image.png `
  --model yafitzdev/opsis-v1-nano-g2 `
  --json
```

Use the first-generation checkpoint for a direct comparison:

```powershell
uv run rag-image-parse parse .\image.png `
  --model yafitzdev/opsis-v1-nano-g1
```

Useful options:

```text
--threads 8             CPU inference threads
--image-longest-edge 2048  Vision resolution; 512/1024 are faster training targets
--max-new-tokens 768    Output ceiling; descriptions stop much earlier
--backend onnx          Optimized CPU runtime (default)
--onnx-variant fp32     Quality-preserving default; hybrid/int8 are experimental
--backend pytorch       Reference runtime; use with --dtype float32
```

## Output contract

The public result contains `kind`, `text`, `latency_seconds`, and `model_id`. `kind` is either
`description` or `table`. Plain CLI output prints only `text`.

Description outputs are concise prose. Diagram descriptions must name important labels and
relationships. Table outputs contain only a valid Markdown table. See
[`docs/CONTRACT.md`](docs/CONTRACT.md) for the complete contract.

## Benchmark

Create JSONL such as:

```json
{"image":"samples/diagram.png","expected_kind":"description","required_terms":["Parser","Index"],"required_relations":[{"source":"Parser","target":"Index"}]}
{"image":"samples/table.png","expected_kind":"table","expected_text":"| A | B |\n|---|---|\n| 1 | 2 |"}
```

Then run:

```powershell
uv run rag-image-parse benchmark .\benchmark.jsonl --json
```

The report includes kind accuracy, valid-table rate, exact table matches, required-label recall,
directional relationship recall, mean/median/p95 latency, peak resident memory, and output length.
`forbidden_terms` can mark unsupported claims. Paths are resolved relative to the manifest.

## Fine-tuning data

Training examples use the same fixed instruction as inference:

```json
{"image":"images/dog.jpg","output":"<description>A brown dog catches a red frisbee in a grassy park.</description>"}
{"image":"images/results.png","output":"<table>\n| Model | Score |\n|---|---:|\n| A | 91.2 |\n</table>"}
```

Generate varied, exactly labeled synthetic table and process-diagram examples:

```powershell
uv run python .\scripts\generate_synthetic_data.py `
  --output-dir .\artifacts\synthetic `
  --tables 1000 `
  --diagrams 1000
```

Synthetic data teaches serialization and simple visual relationships. It must eventually be
mixed with real image captions and real-world table/diagram crops so the model does not overfit
to clean rendered styles.

Download the pinned PubTabNet, ChartQA, AI2D, and DOCCI sources, then build the balanced public
pilot:

```powershell
uv run python .\scripts\download_public_data.py
uv run python .\scripts\curate_public_pilot.py
```

See [`docs/PUBLIC_DATA.md`](docs/PUBLIC_DATA.md) for reproducibility and licensing notes, and
[`docs/DATA_ASSESSMENT.md`](docs/DATA_ASSESSMENT.md) for the measured CPU baseline and recommended
training-data scope. AI2D's bundled license is research-only; do not use it for a commercial model
without permission.

For real data, place each image beside exactly one sidecar label. A `.txt` sidecar is a plain
description; a `.md` sidecar contains only the exact Markdown table. Do not add training tags.

```text
data/raw/
  bicycle.jpg
  bicycle.txt
  quarterly-results.png
  quarterly-results.md
```

Build validated, deterministic, kind-stratified manifests:

```powershell
uv run rag-image-build-dataset .\data\raw .\artifacts\real-manifests `
  --validation-fraction 0.1 `
  --seed 17
```

Validate and fine-tune:

```powershell
uv run rag-image-train `
  --train-file .\artifacts\synthetic\train.jsonl `
  --validation-file .\artifacts\synthetic\validation.jsonl `
  --output-dir .\outputs\smolvlm-rag-parser `
  --image-longest-edge 1024
```

LoRA over the text attention layers and vision-to-text connector is the default and trains about
0.40% of this checkpoint's parameters at rank 8. Pass `--full-finetune` for ordinary SFT or
`--freeze-vision` when doing a full fine-tune. Train and infer with the same
`--image-longest-edge`; 1024 is the current speed target.

Pass `--vision-lora` to additionally adapt all vision-attention Q/K/V/output projections. Opsis g2
uses this mode and trains 1,614,336 parameters, about 0.625% of the model.

Merge a trained adapter into a standalone checkpoint for PyTorch inference:

```powershell
uv run rag-image-merge-adapter `
  .\outputs\smolvlm-rag-parser `
  .\outputs\smolvlm-rag-parser-merged
```

Export the merged checkpoint into the split FP32 and INT8 ONNX layouts used by the optimized CPU
backend:

```powershell
uv run rag-image-export-onnx `
  .\outputs\smolvlm-rag-parser-merged `
  .\outputs\smolvlm-rag-parser-onnx
```

The exporter transplants the learned connector and text-attention tensors into the official
SmolVLM split graphs and validates their shapes. It emits FP32, mixed INT8, and hybrid graphs. The
mixed INT8 variant leaves all 121 trained projections in FP32; the hybrid additionally leaves the
whole decoder in FP32. Both compressed variants are experimental because neither beat FP32 on the
v1 latency/quality benchmark. Training uses a GPU; every exported variant runs entirely on CPU.

## Smoke fixtures

Generate two synthetic images for manual tests:

```powershell
uv run python .\scripts\make_smoke_images.py --output-dir .\artifacts\smoke
```

## Tests

```powershell
uv run pytest
uv run ruff check .
```
