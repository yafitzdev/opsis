# Training experiment v1

## Purpose

This is the first meaningful fine-tuning experiment for the narrow parser contract: one image in,
either a concise searchable description or a GitHub-flavored Markdown table out. It is a research
checkpoint, not a commercially distributable release, because its training mixture includes
AI2D's research-only data and sources with share-alike or attribution requirements.

## Corpus

The deterministic corpus under `artifacts/public_data/training_v1/` contains 15,000 training and
700 source-held-out validation records.

| Source | Train | Validation | Role |
|---|---:|---:|---|
| PubTabNet | 10,000 | 200 | simple table to Markdown |
| ChartQA | 1,500 | 150 | chart to factual description |
| AI2D | 1,000 | 150 | labeled diagram relationships |
| DOCCI | 2,500 | 200 | ordinary-image replay descriptions |

Selection is balanced by source-specific structure rather than taking the first records. Complex
tables with merged cells are excluded because Markdown cannot express row or column spans without
a flattening policy. DOCCI descriptions end at a sentence boundary and are capped at 90 words.
The exact selection counts and rejection reasons are in `curation_report.json` beside the corpus.

## Training configuration

| Setting | Value |
|---|---|
| Base | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Resolution | 1024-pixel longest edge |
| Method | LoRA rank 8, alpha 16, dropout 0.05 |
| Updated modules | text-attention Q/K/V/O projections and vision-to-text connector |
| Trainable parameters | 1,024,512 of 257,509,440 (0.398%) |
| Precision/device | bfloat16, NVIDIA RTX 5090 |
| Effective batch | 16 (batch 1, accumulation 16) |
| Epochs | 1 |
| Optimizer steps | 938 |
| Learning rate | 2e-4 with 3% warmup and linear decay |
| Weight decay | 0.01 |
| Seed | 17 |

Validation loss was measured at steps 250, 500, and 750. Each candidate was retained independently
so final artifact selection could use held-out loss rather than defaulting to the last step.

## Deployment contract

The selected adapter is merged into a standalone Transformers checkpoint. The local ONNX exporter
then replaces exactly 120 text-attention tensors and one connector tensor in the official split
SmolVLM graphs. It emits FP32, mixed INT8, and hybrid CPU graphs for `OnnxBackend`. A direct held-out
generation check produced the same token prefix from merged PyTorch and FP32 ONNX, establishing
that the tensor transplant preserved the learned behavior.

## Results

Training completed in 1:48:08 for 938 optimizer steps. Final train loss was 0.5174. Validation loss
improved throughout the run, so the final adapter was selected.

| Checkpoint | Validation loss |
|---|---:|
| Step 250 | 1.1513 |
| Step 500 | 1.0267 |
| Step 750 | 0.9988 |
| Final, step 938 | 0.9937 |

The deployment pilot contains three tables, three charts, and three diagrams. It is useful for
regression detection, but far too small for a product claim.

| Metric | Untuned INT8 | Tuned FP32 | Tuned hybrid |
|---|---:|---:|---:|
| Kind accuracy | 66.7% | **100%** | **100%** |
| Valid Markdown tables | 0% | **100%** | **100%** |
| Exact table shape | 0% | 0% | 0% |
| Mean table cell F1 | 0% | **10.0%** | 8.7% |
| Required-term recall | 41.3% | 60.9% | **75.5%** |
| Directed-relation recall | 50.0% | 66.7% | **75.0%** |
| Median latency, 8 CPU threads | **2.37 s** | 5.18 s | 16.62 s |
| p95 latency | **10.01 s** | 17.03 s | 20.43 s |
| Peak RSS | **1,646 MiB** | 2,381 MiB | 2,049 MiB |

FP32 is the v1 release candidate: it is the only fine-tuned variant that preserves quality while
also beating the hybrid on latency. Whole-model dynamic INT8 collapsed routing to 22.2%. Keeping
the 121 trained projections in FP32 recovered valid tables to 66.7%, but kind accuracy remained
55.6%. Keeping the decoder FP32 restored routing, at the cost of worse latency than full FP32.

The fine-tune clearly learned the narrow schema and improved discovery descriptions. It did not
learn reliable table OCR: correct structure and exact cell recovery remain the dominant blocker.
Some long table targets also exceeded the 768-token inference budget and caused unusually slow
training steps. V2 should cap targets by tokenizer length, oversample compact high-resolution
tables, add OCR-corruption augmentation, and score a much larger manually checked gold set.

Reports:

- `artifacts/public_data/pilot/baseline_v2_report.json`
- `artifacts/public_data/pilot/finetuned_fp32_report.json`
- `artifacts/public_data/pilot/finetuned_hybrid_report.json`
- `artifacts/public_data/pilot/finetuned_mixed_int8_report.json`

## Artifact layout

- `outputs/rag-image-parser-v1-lora/`: selected adapter, about 4 MiB of learned weights;
- `outputs/rag-image-parser-v1-merged/`: standalone Transformers checkpoint;
- `outputs/rag-image-parser-v1-onnx/`: CPU deployment directory;
- FP32 ONNX graphs: about 981 MiB total;
- hybrid ONNX graphs: about 653 MiB total;
- mixed INT8 ONNX graphs: about 344 MiB total.

The corpus is not commercially clean. AI2D is research-only, PubTabNet and ChartQA carry
redistribution/share-alike constraints, and DOCCI requires attribution. Replace or relicense those
sources before publishing model weights for commercial use.
