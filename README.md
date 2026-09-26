<div align="center">

# Opsis

### A CPU-runnable visual co-processor for retrieval systems

**Turn an image into searchable text or a Markdown table.**

[Why it exists](#why-it-exists) · [How it works](#how-it-works) · [What it enables](#what-it-enables) · [Boundaries](#boundaries)

[Model collection](https://huggingface.co/collections/yafitzdev/opsis-v1) · [Training-data provenance](https://huggingface.co/datasets/yafitzdev/opsis-v1) · [All datasets](https://huggingface.co/yafitzdev/datasets)

</div>

---

## Why it exists

A retrieval system can index a document's text while leaving its charts, diagrams, tables, and embedded images effectively invisible. Sending every page to a general vision model adds cost and can discard reliable text already present in the PDF.

Opsis handles the visual part of ingestion. It gives an index a compact description of an image, or a Markdown table when the image is primarily tabular. The original image remains the source of truth.

---

## How it works

Opsis takes **one raster image or document crop** at a time. A small vision-language model produces one tagged result:

| Image content | Output | Use |
| --- | --- | --- |
| Photograph, illustration, chart, or diagram | `<description>…</description>` | Searchable discovery text with visible labels and relationships. |
| Table | `<table>…</table>` | Rows and cells represented as GitHub-flavored Markdown. |

A parser strips the tag and returns the content with its output kind. In a PDF pipeline, native text extraction handles readable text; visual regions are cropped and routed to Opsis. The model can run locally on CPU through its FP32 ONNX export, without a hosted vision API.

```text
PDF or image
     ↓
native text extraction + visual-region routing
     ↓
Opsis describes each visual crop or reconstructs its table
     ↓
retrieval index stores the result beside the original source
```

---

## What it enables

**Search across visual content.** Charts, diagrams, and ordinary images contribute terms to retrieval instead of disappearing from the index.

**Recover table structure.** Table images become Markdown that can be inspected and searched alongside surrounding document text.

**Keep ingestion local.** The published checkpoint includes a CPU runtime, so visual processing can stay on the user's machine.

**Preserve source context.** Opsis supplies discovery metadata while the calling system retains page locations and the original image for review.

---

## Boundaries

Generated descriptions and table cells can be wrong. Opsis is useful for finding and organizing visual evidence, but its output is not verified evidence or a substitute for inspecting the source image. It does not extract reliable native PDF text, reconstruct full page layout, or guarantee exact table transcription.

The [published model card](https://huggingface.co/yafitzdev/opsis-v1-nano-g3) contains the checkpoint, measured results, runtime instructions, and limitations. This repository is the public overview of the project.

## License

The repository text is licensed under [CC BY-NC 4.0](LICENSE). The model and its training sources have separate terms described on the model card.
