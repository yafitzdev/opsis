# Opsis v1 nano g2

## Outcome

`opsis-v1-nano-g2` is the second generation of the same Opsis v1 capability: one image in,
either concise RAG-discovery prose or a GitHub-flavored Markdown table out. The version remains v1
because no output heads or fundamental capabilities changed. G2 changes the data balance and adds
vision-attention adaptation.

G2 is the selected research checkpoint. It improves substantially over g1 on the larger
source-held-out comparison, but it is not an authoritative OCR system.

## Corpus

The deterministic corpus is under `artifacts/public_data/training_g2_50k/` and contains exactly
50,000 training files plus 1,000 validation files.

| Top-level training bucket | Files |
|---|---:|
| Clean tables | 12,500 |
| Clean charts | 6,250 |
| Clean diagrams | 6,250 |
| Clean ordinary images | 12,500 |
| PDF-domain visual crops | 12,500 |

PDF is a rendering domain, not a fifth semantic output type. Its 12,500 examples are divided
equally among tables, charts, diagrams, and ordinary images. The effective semantic mixture is
15,625 tables, 9,375 charts, 9,375 diagrams, and 15,625 ordinary images.

| Content | Unique training sources | Rendered files |
|---|---:|---:|
| Tables | 15,625 | 15,625 |
| Charts | 1,606 | 9,375 |
| Diagrams | 985 | 9,375 |
| Ordinary images | 9,647 | 15,625 |

Charts and diagrams therefore depend heavily on deterministic, label-preserving transforms. That
is acceptable for g2 but is the clearest data-diversity target for g3. PDF transforms include
downsampling, JPEG compression, grayscale, mild contrast/brightness shifts, light blur, and page
margins.

All table targets are capped at 640 tokenizer tokens. The final corpus has 51,000 readable rendered
files, 50,999 distinct byte hashes, and zero train/validation image-hash overlap. One byte-identical
PubTabNet image occurs twice within training under two different source article IDs.

## Training

| Setting | Value |
|---|---|
| Starting checkpoint | merged Opsis g1 |
| Resolution | 1024-pixel longest edge |
| Method | LoRA rank 8, alpha 16, dropout 0.05 |
| Updated modules | text Q/K/V/O, connector, vision Q/K/V/output |
| Trainable parameters | 1,614,336 of 258,099,264 (0.625%) |
| Precision/device | bfloat16, NVIDIA RTX 5090 |
| Batch | 4, accumulation 4, effective 16 |
| Epochs / optimizer steps | 1 / 3,125 |
| Learning rate | 1e-4, 3% warmup, linear decay |
| Runtime | 2:50:23 |
| Throughput | 4.891 images/s including validation |
| Final train loss | 0.6605 |

Validation improved at every checkpoint:

| Step | Validation loss |
|---|---:|
| 625 | 0.7184 |
| 1,250 | 0.6951 |
| 1,875 | 0.6828 |
| 2,500 | 0.6763 |
| 3,125 | **0.6732** |

The final checkpoint was therefore selected.

## G1 versus g2

The comparison uses 50 source-held-out records: ten each from clean tables, charts, diagrams,
ordinary images, and PDF-domain crops. It is automatically labeled and useful for model selection;
it is not a substitute for a manually reviewed product benchmark.

| Metric | G1 FP32 | G2 FP32 |
|---|---:|---:|
| Kind accuracy | 100% | 100% |
| Valid Markdown tables | 100% | 100% |
| Exact tables | 0% | **7.1%** |
| Correct table shape | 21.4% | **42.9%** |
| Mean table cell F1 | 31.0% | **50.5%** |
| Required-term recall | 53.8% | **57.7%** |
| Directed-relation recall | 34.0% | **48.7%** |
| Median CPU latency | **3.71 s** | 3.72 s |
| p95 CPU latency | 16.74 s | **10.95 s** |
| Peak RSS | **2,377 MiB** | 2,382 MiB |

Clean-table cell F1 improved from 39.3% to 54.6%. PDF-domain table cell F1 improved from 10.4%
to 40.2%. Diagram relation recall improved from 35.8% to 49.2%. Ordinary-image required-term
recall was essentially flat and slightly lower in this small slice.

The older nine-case pilot is noisier: it shows stronger discovery-term recall but lower table and
diagram scores for g2. The larger 50-case comparison is the selection signal, while both remain too
small for production claims.

## CPU artifacts

- `outputs/pyrrho-opsis-v1-nano-g2-lora/`: selected LoRA adapter;
- `outputs/pyrrho-opsis-v1-nano-g2-merged/`: standalone Transformers checkpoint;
- `outputs/pyrrho-opsis-v1-nano-g2-onnx/`: split CPU ONNX checkpoint.

The exporter patches 48 learned vision-attention tensors, 120 text-attention tensors, and the
vision-to-text connector into the split graphs. A held-out 50-token generation prefix matched
exactly between merged PyTorch and FP32 ONNX.

FP32 remains the release default. Mixed INT8 reduced peak RSS to about 1.83 GiB but achieved only
88.9% routing and 66.7% valid tables on the nine-case pilot, while also running slower at the
median. It remains experimental.

Reports:

- `artifacts/public_data/training_g2_50k/g1_balanced_50_report.json`
- `artifacts/public_data/training_g2_50k/g2_balanced_50_report.json`
- `artifacts/public_data/pilot/opsis_g2_fp32_report.json`
- `artifacts/public_data/pilot/opsis_g2_mixed_int8_report.json`

## Release and licensing

The current mixture includes AI2D's research-only data, ChartQA, PubTabNet/PMC images, and DOCCI.
Do not publish it as an unrestricted commercial model. A Hugging Face package should initially be
private or gated, marked as a research preview, and use a custom/mixed-source license declaration
until the source obligations have been reviewed or the restrictive sources have been replaced.
