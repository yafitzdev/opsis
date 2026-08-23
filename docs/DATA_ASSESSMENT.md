# Public-data pilot assessment

## Outcome

The downloaded data supported both the 15,000-example g1 experiment and the balanced 50,000-file
Opsis g2 experiment. G2 improves table cell F1 from 31.0% to 50.5% and relation recall from 34.0%
to 48.7% on the 50-case source-held-out comparison. It remains insufficient for authoritative
extraction; exact table accuracy is only 7.1%. A commercially usable diagram set and a larger
manually reviewed gold evaluation set are the next priorities.

This assessment was generated from the revision-pinned archives in
`artifacts/public_data/raw/download_manifest.json` and the deterministic pilot in
`artifacts/public_data/pilot/`.

## What was downloaded and inspected

| Source | Complete raw contents | Full-pool curation result |
|---|---|---|
| PubTabNet | 519,030 images; 500,777 train, 9,115 validation, 9,138 unlabeled test | 242,766 simple Markdown-compatible labels |
| ChartQA | 62,665 archive entries covering images, tables, element annotations, and QA | 1,910 clean chart-description candidates |
| AI2D | 4,904 images and 4,903 annotation files | 1,173 diagrams with label-resolvable connections |
| DOCCI | 15,103 images plus long human descriptions | 2,700 concise ordinary-image replay targets selected for train/validation |

PubTabNet also contains 234,540 tables with row or column spans. GitHub-flavored Markdown has no
rowspan/colspan semantics, so those records are intentionally excluded until the product chooses
a deterministic flattening rule. Other PubTabNet rejections are 17,714 oversized tables, 12,704
long-cell tables, 1,999 non-rectangular tables, 168 empty-header tables, and one one-row table.

ChartQA targets are regenerated from source tables plus chart annotations. Strict numeric parsing,
missing-value limits, annotation-based visible labels, and corrupt-punctuation checks reject 17,916
annotations rather than training on noisy values. AI2D targets resolve text labels to objects and
distinguish directed arrows from undirected connections.

## Small held-out CPU test

The pilot contains 24 training-format examples and 9 held-out examples, balanced across tables,
charts, and diagrams. The 24 records remain a pipeline smoke test; the real adapter was trained on
a separate 15,000-record source-balanced corpus. Both the untouched baseline and fine-tuned ONNX
artifacts used 1024-pixel preprocessing, 8 CPU threads, and a 768-token table ceiling.

| Metric | Untouched INT8 | Fine-tuned FP32 |
|---|---:|---:|
| Cases | 9 | 9 |
| Kind accuracy | 66.7% | **100%** |
| Valid Markdown table rate | 0% | **100%** |
| Exact table shape | 0% | 0% |
| Mean table cell F1 | 0% | **10.0%** |
| Mean description required-term recall | 41.3% | **60.9%** |
| Mean directed-relation recall | 50.0% | **66.7%** |
| Median latency | **2.37 s/image** | 5.18 s/image |
| p95 latency | **10.01 s/image** | 17.03 s/image |
| Peak process RSS | **1,646 MiB** | 2,381 MiB |

The trained artifact proves the schema can be specialized with roughly one million updated
parameters. Quality is still not adequate: all tables are syntactically valid but most cells are
wrong or missing, and no table matches the reference shape. Charts and diagrams recover more
search terms and relationships, but the nine-case pilot cannot establish robustness.

## What is still needed

### Completed first fine-tuning run

V1 used the proposed 15,000-example mixture:

| Pool | Target count | Reason |
|---|---:|---|
| Simple PubTabNet tables | 8,000-10,000 | teach reliable table classification and exact Markdown serialization |
| Clean ChartQA descriptions | 1,500 | use nearly all currently eligible chart targets |
| AI2D diagrams | 1,000 research-only | establish whether relation descriptions improve with SFT |
| General-image replay | 2,500 DOCCI examples | prevent catastrophic loss of ordinary image captioning |

Selection used source-specific shape/type/category balancing and source-held-out validation. The
next run should additionally cap every target by tokenizer length: V1 included table labels as long
as 929 words, well beyond the 768-token inference budget.

### Product-quality data target

Expect roughly 40,000-60,000 curated examples, not hundreds of thousands:

- 20,000-25,000 simple real tables;
- 5,000 complex tables after defining a merged-cell-to-Markdown flattening policy;
- 5,000-8,000 charts with checked factual descriptions;
- 5,000-8,000 commercially usable process, architecture, and business diagrams;
- 8,000-12,000 ordinary figures, screenshots, and photographs with concise RAG descriptions;
- 3,000-5,000 noisy PDF crops: low resolution, scans, rotations, compression, and small text.

A manually reviewed gold set of 600-1,000 images should remain permanently outside training. It
should report table cell accuracy/TEDS in addition to exact match, chart fact precision/recall,
diagram node/edge precision/recall, hallucination rate, latency, and peak memory.

## Blocking decisions and risks

1. **AI2D cannot be product training data as downloaded.** Its bundled license limits use to
   research/evaluation and forbids redistribution and for-profit use without AllenAI permission.
   Replace it with a commercially usable diagram pool or obtain permission before a product run.
2. **Complex-table output is undefined.** Markdown cannot preserve merged cells. Choose whether to
   repeat parent labels, leave covered cells blank, or allow HTML before curating the 234,540
   spanning-table records.
3. **Ordinary-image replay is now present but not product-cleared.** DOCCI fills this training role
   and requires attribution; verify the intended model-weight distribution against all source terms.
4. **Chart summaries are deterministic, not human-authored.** Manually review at least 500 and add
   1,000-2,000 human or carefully teacher-written summaries to teach salience without hallucination.
5. **Naive quantization is not quality preserving.** Whole-model dynamic INT8 collapsed routing.
   The FP32 export is currently both faster and more accurate than the quality-preserving hybrid.

## Go/no-go checkpoint

Do not scale blindly to every PubTabNet record. Build V2 around OCR fidelity and a 600-1,000-image
gold set, and scale only when it reaches all of these thresholds:

- at least 90% table-kind accuracy and 80% structurally valid Markdown;
- at least 85% chart key-fact recall with under 2% unsupported-fact rate;
- at least 80% diagram node recall and 65% directed-edge recall;
- median under 3 seconds and p95 under 8 seconds on the target CPU at the chosen resolution.
