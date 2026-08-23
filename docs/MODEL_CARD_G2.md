---
license: other
license_name: mixed-source-research-preview
base_model: yafitzdev/opsis-v1-nano-g1
library_name: transformers
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - rag
  - visual-ingestion
  - image-to-text
  - document-ai
  - table-recognition
  - opsis
  - onnx
  - cpu
---

# opsis-v1-nano-g2

`opsis-v1-nano-g2` is a small local RAG visual-ingestion co-processor. It
runs on one image or document crop and returns either concise discovery text or
a GitHub-flavored Markdown table for downstream indexing and retrieval.

It is not a general-purpose assistant, not a replacement for native PDF text
extraction, and not an authoritative OCR system. It sits before indexing, or
beside a PDF routing pipeline, so visual content can become searchable without
a hosted VLM API or inference GPU.

Compared with `opsis-v1-nano-g1`, g2 keeps the same v1 capability and
output contract while moving to a balanced 50,000-file corpus and adapting the
vision encoder in addition to text attention and the vision-to-text connector.
G2 is the recommended Opsis v1 nano generation.

## Native V1 Output Modes

| Mode | Native output | Intended use |
|---|---|---|
| `description` | `<description>concise factual text</description>` | Ordinary images, charts, and diagrams that need searchable RAG metadata. |
| `table` | `<table>GitHub-flavored Markdown table</table>` | Table images that need recoverable rows and cells rather than prose. |

Charts stay in `description` mode and should include visible labels, values,
and trends. Diagrams stay in `description` mode and should include visible
nodes and relationships.

## Output Contract

The raw Hugging Face model output is one tagged string. It is not unrestricted
assistant prose. Decode it by wrapper:

| Wrapper | Parsed kind | Decoding |
|---|---|---|
| `<description>...</description>` | `description` | Strip the wrapper and retain concise factual text. |
| `<table>...</table>` | `table` | Strip the wrapper and retain the complete Markdown table. |

The companion parser normalizes either result into one object:

```json
{
  "kind": "description",
  "text": "A line chart shows quarterly revenue rising from $12M in Q1 to $18M in Q4.",
  "latency_seconds": 3.72,
  "model_id": "yafitzdev/opsis-v1-nano-g2",
  "prompt_version": "rag-image-v3"
}
```

The model does not return bounding boxes, OCR confidence, source coordinates,
PDF text blocks, or verified numeric facts. Preserve the source image or page
crop beside generated text when exact visual evidence matters.

## Intended Use

Use this model when a RAG or retrieval system needs local visual signals for:

- indexing ordinary images with concise discovery descriptions,
- converting compact table images into Markdown,
- recording visible chart labels, values, and trends,
- recording labeled diagram nodes and relationships,
- enriching visual regions routed out of otherwise native-text PDFs,
- keeping document ingestion local on CPU-only systems.

This model is not intended to replace reliable native table extraction, parse
full scanned pages as a layout engine, verify high-stakes measurements, or
replace source review when exact cells and labels matter.

## Input Format

The model accepts one raster image per inference. For direct Transformers use,
pair the image with the v1 parser prompt:

```text
Parse this image for RAG search. If it is a table, output only the complete
table as Markdown. Otherwise output only a concise factual description. Include
meaningful visible text. For a diagram, name its labels and relationships. For
a chart, mention its labels, values, and trend. Do not speculate or add a heading.
```

For PDFs, extract reliable native text normally and send only table, image,
chart, or diagram regions to Opsis. Crop quality and readable resolution have a
direct effect on output quality.

## Quick Start

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "yafitzdev/opsis-v1-nano-g2"
PROMPT = """Parse this image for RAG search. If it is a table, output only the complete
table as Markdown. Otherwise output only a concise factual description. Include meaningful
visible text. For a diagram, name its labels and relationships. For a chart, mention its labels,
values, and trend. Do not speculate or add a heading."""

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    size={"longest_edge": 1024},
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
).eval()
image = Image.open("image.png").convert("RGB")

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ],
}]
prompt_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=768,
        repetition_penalty=1.05,
    )

generated = output_ids[:, inputs["input_ids"].shape[-1]:]
text = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(text)
```

## CPU ONNX

The repository includes the quality-preserving split FP32 ONNX runtime:

- `onnx/vision_encoder.onnx`
- `onnx/embed_tokens.onnx`
- `onnx/decoder_model_merged.onnx`

With the `rag-image-parser` project installed, run:

```powershell
uv run rag-image-parse parse .\image.png `
  --model yafitzdev/opsis-v1-nano-g2 `
  --backend onnx `
  --onnx-variant fp32 `
  --threads 8 `
  --image-longest-edge 1024
```

Mixed INT8 reduced routing and Markdown validity in evaluation and is not
included in this repository.

## Evaluation

Source-held-out automatic evaluation on 50 cases containing ten examples each
from tables, charts, diagrams, ordinary images, and PDF-domain crops:

| Metric | G1 | G2 |
|---|---:|---:|
| kind accuracy | 1.0000 | 1.0000 |
| valid Markdown tables | 1.0000 | 1.0000 |
| exact tables | 0.0000 | 0.0714 |
| correct table shape | 0.2143 | 0.4286 |
| mean table cell F1 | 0.3103 | 0.5047 |
| required-term recall | 0.5382 | 0.5773 |
| directed-relation recall | 0.3397 | 0.4872 |
| median CPU latency | 3.71 s | 3.72 s |
| p95 CPU latency | 16.74 s | 10.95 s |
| peak process RSS | 2,377 MiB | 2,382 MiB |

Bucket-level checkpoint comparison:

| Benchmark slice | G1 | G2 |
|---|---:|---:|
| clean-table cell F1 | 0.3928 | 0.5459 |
| PDF-domain table cell F1 | 0.1040 | 0.4017 |
| chart required-term recall | 0.7989 | 0.8178 |
| diagram required-term recall | 0.5207 | 0.6443 |
| diagram directed-relation recall | 0.3583 | 0.4917 |
| ordinary-image required-term recall | 0.3375 | 0.3250 |

The benchmark is automatically labeled and too small for a product-quality
claim. G2 materially improved tables, diagrams, and PDF-domain crops; ordinary-
image term recall was approximately flat.

## Training Data

| Training bucket | Files | Validation files | Role |
|---|---:|---:|---|
| clean tables | 12,500 | 250 | Compact table image to Markdown. |
| clean charts | 6,250 | 125 | Chart labels, values, and trends. |
| clean diagrams | 6,250 | 125 | Diagram nodes and relationships. |
| clean ordinary images | 12,500 | 250 | Ordinary-image discovery descriptions. |
| PDF-style visual crops | 12,500 | 250 | Document-domain versions balanced across all four content types. |
| **Total** | **50,000** | **1,000** | Source-held-out balanced visual parsing. |

Sources include PubTabNet, ChartQA, AI2D, and DOCCI. After distributing the PDF
bucket, the effective training mix is 15,625 tables, 9,375 charts, 9,375
diagrams, and 15,625 ordinary images. Deterministic label-preserving variants
expand source pools that contain fewer strict unique examples than the rendered
file targets.

Training used LoRA rank 8 with alpha 16 for one epoch at a 1024-pixel longest
edge. It updated 1,614,336 of 258,099,264 parameters (0.625%), including text
attention, vision attention, and the vision-to-text connector.

## Artifacts

This repository contains:

- `model.safetensors`: standalone merged Transformers checkpoint,
- `onnx/vision_encoder.onnx`: FP32 vision encoder and connector,
- `onnx/embed_tokens.onnx`: FP32 token embedding graph,
- `onnx/decoder_model_merged.onnx`: FP32 autoregressive decoder,
- tokenizer, processor, generation, and model configuration files,
- `rag_image_parser_config.json`: Opsis release and training metadata,
- `SHA256SUMS`: checksums for the packaged ONNX graphs.

## Limitations

1. **Exact table recovery remains low.** G2 improved table structure and cell
   F1, but exact table accuracy was 7.1% on the small held-out benchmark.
2. **Evaluation is small and automatically labeled.** The 50-case benchmark is
   useful for checkpoint comparison, not broad product certification.
3. **Generated labels and values can be wrong.** Descriptions and cells are RAG
   discovery metadata, not verified evidence.
4. **English-centric mixed-source training.** Multilingual behavior is not
   established, and some source pools use deterministic augmentation.
5. **Crop quality matters.** Tiny text, dense pages, poor scans, and incorrect
   visual-region routing reduce performance.
6. **CPU latency varies by image complexity.** Dense tables and long outputs can
   be substantially slower than the reported median.

## License

Mixed-source research preview. The training mixture includes research-restricted
AI2D data and sources with separate attribution, redistribution, or underlying-
image terms. This package must not be represented as commercially cleared.
Review and satisfy every source obligation before making the repository public
or using the weights commercially.
