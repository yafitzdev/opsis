# Adaptive PDF router

`PdfRouter` is an orchestration layer around the one-image visual parser. It keeps native text
cheap and invokes vision only for page regions that need visual understanding.

## Routing states

| Page kind | Meaning | Action |
|---|---|---|
| `clean` | Native text, no material visual regions | Text parser only; vision stays unloaded |
| `mixed` | Native text plus table/image/diagram regions | Text parser plus cropped visual inference |
| `visual` | Visual regions without useful native text | Cropped visual inference |
| `scanned` | Near-full-page raster image and little native text | Flag for OCR/layout extraction |
| `review` | Native text is fragmented, encoded badly, or multi-column | Preserve text and flag it |

The router does not send scanned mixed-layout pages through the description model. A description
would discard their body text and create a deceptively successful result. Set
`fail_on_scanned_pages=True` or use `--fail-on-scanned-pages` to make this condition fatal.

## Detection order

1. Extract positioned native text blocks.
2. Detect ruled tables, or aligned borderless tables when `table_strategy="text"`.
3. Detect substantial embedded raster images.
4. Cluster vector paths and retain diagram-like clusters with enough paths and/or text labels.
5. De-duplicate overlapping regions, with tables taking precedence.
6. Remove native words whose centers fall inside a visual region.
7. Render each region at 200 DPI and call the visual parser serially.
8. Merge native and visual elements by page, vertical position, and horizontal position.

The visual backend is prepared on the first visual region, not when the PDF is opened. A document
containing only clean text therefore never loads ONNX Runtime sessions or model weights.

## Table recovery

For a detected digital table, PyMuPDF supplies both a bounding box and structured cells. The fast
default serializes reliable native cells directly and emits `native-table-structure-used`, without
loading vision. Pass `vision_for_detected_tables=True` or `--vision-for-detected-tables` to force
a visual retry. If that result is not a Markdown table or disagrees with the canonical native
cells, the router uses the native structure and emits
`table-vision-recovered-from-native-structure`.

This recovery does not apply to rasterized tables because they have no native cells. Those depend
entirely on the visual parser.

## Programmatic integration

Pass any object with `parse(text: str) -> str`, or a callable, as the text parser:

```python
from rag_image_parser import ImageParser, PdfRouter
from rag_image_parser.backend import OnnxBackend


class RetrievalTextParser:
    def parse(self, text: str) -> str:
        return normalize_and_annotate(text)


router = PdfRouter(
    visual_parser=ImageParser(OnnxBackend()),
    text_parser=RetrievalTextParser(),
    render_dpi=200,
)
result = router.parse("document.pdf")

print(result.to_markdown())
metadata = result.to_dict()
```

Each element contains:

```json
{
  "page_number": 2,
  "kind": "table",
  "source": "table",
  "bbox": [72.0, 190.0, 372.0, 292.0],
  "text": "| System | Latency |\n| --- | --- |\n| Parser-A | 42 ms |",
  "latency_seconds": 5.8
}
```

The bounding box uses PDF page coordinates in points. The PDF path, page dimensions, page kind,
quality flags, visual model usage, and number of visual inferences are included in the document
result.

## Known boundary

Scanned pages need an OCR/layout stage capable of separating text, tables, and figures before this
router can provide complete page extraction. The current router reports that condition explicitly.
It never claims that an image description is a substitute for the missing page text.
