from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import pymupdf
from PIL import Image

from rag_image_parser.backend import OnnxBackend
from rag_image_parser.markdown import canonical_table
from rag_image_parser.parser import ImageParser
from rag_image_parser.types import ParseResult

RectTuple = tuple[float, float, float, float]


class PageKind(str, Enum):
    CLEAN = "clean"
    MIXED = "mixed"
    VISUAL = "visual"
    SCANNED = "scanned"
    REVIEW = "review"


class PdfElementKind(str, Enum):
    TEXT = "text"
    DESCRIPTION = "description"
    TABLE = "table"


class VisualParser(Protocol):
    def prepare(self) -> None: ...

    def parse(self, image: Image.Image) -> ParseResult: ...


class TextParser(Protocol):
    def parse(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PdfElement:
    page_number: int
    kind: PdfElementKind
    source: str
    bbox: RectTuple
    text: str
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True, slots=True)
class PdfPageResult:
    page_number: int
    kind: PageKind
    width: float
    height: float
    native_text_characters: int
    visual_regions: int
    quality_flags: tuple[str, ...]
    elements: tuple[PdfElement, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["elements"] = [element.to_dict() for element in self.elements]
        return data


@dataclass(frozen=True, slots=True)
class PdfParseResult:
    source: str
    pages: tuple[PdfPageResult, ...]
    visual_model_used: bool
    visual_inferences: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "visual_model_used": self.visual_model_used,
            "visual_inferences": self.visual_inferences,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_markdown(self) -> str:
        sections: list[str] = []
        for page in self.pages:
            content: list[str] = []
            for element in page.elements:
                if element.kind is PdfElementKind.DESCRIPTION:
                    label = element.source.capitalize()
                    content.append(f"[{label}: {element.text}]")
                else:
                    content.append(element.text)
            body = "\n\n".join(part for part in content if part.strip())
            if not body and page.kind is PageKind.SCANNED:
                body = "[Scanned page: OCR or page-layout extraction required.]"
            sections.append(f"## Page {page.page_number}\n\n{body}".rstrip())
        return "\n\n".join(sections)


@dataclass(frozen=True, slots=True)
class _TextBlock:
    bbox: pymupdf.Rect
    text: str


@dataclass(frozen=True, slots=True)
class _VisualRegion:
    bbox: pymupdf.Rect
    source: str
    native_table: str | None = None


class _IdentityTextParser:
    def parse(self, text: str) -> str:
        return text


class PdfRouter:
    """Route native PDF text and rendered visual regions to specialized parsers."""

    def __init__(
        self,
        visual_parser: VisualParser | None = None,
        *,
        text_parser: TextParser | Callable[[str], str] | None = None,
        render_dpi: int = 200,
        min_visual_area_ratio: float = 0.01,
        full_page_image_ratio: float = 0.75,
        table_strategy: str = "lines",
        vision_for_detected_tables: bool = False,
        detect_vector_graphics: bool = True,
        vector_min_paths: int = 3,
        visual_padding_points: float = 3.0,
        fail_on_scanned_pages: bool = False,
    ) -> None:
        if render_dpi < 72:
            raise ValueError("render_dpi must be at least 72")
        if not 0 < min_visual_area_ratio < 1:
            raise ValueError("min_visual_area_ratio must be between 0 and 1")
        if not 0 < full_page_image_ratio <= 1:
            raise ValueError("full_page_image_ratio must be in (0, 1]")
        if vector_min_paths < 1:
            raise ValueError("vector_min_paths must be at least 1")
        if visual_padding_points < 0:
            raise ValueError("visual_padding_points cannot be negative")
        if table_strategy not in {"lines", "lines_strict", "text"}:
            raise ValueError("table_strategy must be lines, lines_strict, or text")

        self.visual_parser = visual_parser or ImageParser(OnnxBackend())
        self.text_parser = text_parser or _IdentityTextParser()
        self.render_dpi = render_dpi
        self.min_visual_area_ratio = min_visual_area_ratio
        self.full_page_image_ratio = full_page_image_ratio
        self.table_strategy = table_strategy
        self.vision_for_detected_tables = vision_for_detected_tables
        self.detect_vector_graphics = detect_vector_graphics
        self.vector_min_paths = vector_min_paths
        self.visual_padding_points = visual_padding_points
        self.fail_on_scanned_pages = fail_on_scanned_pages
        self._visual_prepared = False

    def parse(self, pdf_path: str | Path) -> PdfParseResult:
        path = Path(pdf_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {path}")

        pages: list[PdfPageResult] = []
        visual_inferences = 0
        try:
            document = pymupdf.open(path)
        except pymupdf.FileDataError as exc:
            raise ValueError(f"Cannot open PDF: {path}") from exc

        try:
            if not document.is_pdf:
                raise ValueError(f"Input is not a PDF: {path}")
            for page_index, page in enumerate(document):
                page_result, page_inferences = self._parse_page(page, page_index + 1)
                pages.append(page_result)
                visual_inferences += page_inferences
        finally:
            document.close()

        return PdfParseResult(
            source=str(path),
            pages=tuple(pages),
            visual_model_used=visual_inferences > 0,
            visual_inferences=visual_inferences,
        )

    def _parse_page(
        self,
        page: pymupdf.Page,
        page_number: int,
    ) -> tuple[PdfPageResult, int]:
        text_blocks = _native_text_blocks(page)
        native_characters = sum(len(block.text) for block in text_blocks)
        regions, region_flags, scanned = self._visual_regions(page, text_blocks)
        non_visual_text_blocks = [
            block
            for block in text_blocks
            if not any(_intersection_ratio(block.bbox, region.bbox) >= 0.5 for region in regions)
        ]
        quality_flags = [
            *region_flags,
            *_native_quality_flags(non_visual_text_blocks),
            *_layout_quality_flags(page, non_visual_text_blocks),
        ]

        if scanned:
            quality_flags.append("scanned-page-needs-ocr-layout")
            if self.fail_on_scanned_pages:
                raise RuntimeError(
                    f"Page {page_number} appears scanned and needs OCR/layout extraction"
                )
            regions = [
                region
                for region in regions
                if _area(region.bbox) / _area(page.rect) < self.full_page_image_ratio
            ]

        text_elements, excluded_words = self._text_elements(
            page,
            page_number,
            text_blocks,
            regions,
        )
        if excluded_words:
            quality_flags.append("native-text-overlaps-visual")

        visual_elements: list[PdfElement] = []
        visual_inferences = 0
        for region in regions:
            if region.native_table is not None and not self.vision_for_detected_tables:
                visual_elements.append(
                    PdfElement(
                        page_number=page_number,
                        kind=PdfElementKind.TABLE,
                        source=region.source,
                        bbox=_rect_tuple(region.bbox),
                        text=region.native_table,
                    )
                )
                quality_flags.append("native-table-structure-used")
                continue
            image = self._render_region(page, region.bbox)
            self._prepare_visual_parser()
            parsed = self.visual_parser.parse(image)
            visual_inferences += 1
            parsed_kind = PdfElementKind(parsed.kind.value)
            parsed_text = parsed.text
            if region.native_table is not None:
                predicted_table = canonical_table(parsed.text)
                native_table = canonical_table(region.native_table)
                if parsed_kind is not PdfElementKind.TABLE or predicted_table != native_table:
                    parsed_kind = PdfElementKind.TABLE
                    parsed_text = region.native_table
                    quality_flags.append("table-vision-recovered-from-native-structure")
            visual_elements.append(
                PdfElement(
                    page_number=page_number,
                    kind=parsed_kind,
                    source=region.source,
                    bbox=_rect_tuple(region.bbox),
                    text=parsed_text,
                    latency_seconds=parsed.latency_seconds,
                )
            )

        elements = sorted(
            [*text_elements, *visual_elements],
            key=lambda element: (element.bbox[1], element.bbox[0]),
        )
        suspicious_native = any(
            flag.startswith("native-text-") or flag == "multi-column-layout"
            for flag in quality_flags
        )
        if scanned:
            kind = PageKind.SCANNED
        elif regions and text_elements:
            kind = PageKind.MIXED
        elif regions:
            kind = PageKind.VISUAL
        elif suspicious_native:
            kind = PageKind.REVIEW
        else:
            kind = PageKind.CLEAN

        result = PdfPageResult(
            page_number=page_number,
            kind=kind,
            width=float(page.rect.width),
            height=float(page.rect.height),
            native_text_characters=native_characters,
            visual_regions=len(regions),
            quality_flags=tuple(dict.fromkeys(quality_flags)),
            elements=tuple(elements),
        )
        return result, visual_inferences

    def _visual_regions(
        self,
        page: pymupdf.Page,
        text_blocks: list[_TextBlock],
    ) -> tuple[list[_VisualRegion], list[str], bool]:
        page_area = _area(page.rect)
        minimum_area = page_area * self.min_visual_area_ratio
        regions: list[_VisualRegion] = []
        flags: list[str] = []

        # Avoid PyMuPDF's optional layout-model path: this router owns layout routing,
        # and the optional-package warning would corrupt JSONL CLI output on stdout.
        table_messages = StringIO()
        with redirect_stdout(table_messages), redirect_stderr(table_messages):
            table_finder = page.find_tables(strategy=self.table_strategy, use_layout=False)
        if table_finder is None:
            message = table_messages.getvalue().strip()
            detail = f": {message}" if message else ""
            raise RuntimeError(f"PDF table detection failed{detail}")
        table_regions = [
            _VisualRegion(
                pymupdf.Rect(table.bbox),
                "table",
                native_table=_native_markdown_table(table.extract()),
            )
            for table in table_finder.tables
            if _area(pymupdf.Rect(table.bbox)) >= minimum_area
        ]
        if table_regions:
            flags.append("table-detected")
        regions.extend(table_regions)

        image_regions: list[_VisualRegion] = []
        full_page_image = False
        for info in page.get_image_info(xrefs=True):
            bbox = pymupdf.Rect(info["bbox"])
            area_ratio = _area(bbox) / page_area
            if (
                _area(bbox) < minimum_area
                or bbox.width < 24
                or bbox.height < 24
                or _duplicates_region(bbox, regions)
            ):
                continue
            full_page_image = full_page_image or area_ratio >= self.full_page_image_ratio
            image_regions.append(_VisualRegion(bbox, "image"))
        if image_regions:
            flags.append("image-detected")
        regions.extend(image_regions)

        if self.detect_vector_graphics:
            drawings = page.get_drawings()
            for cluster in page.cluster_drawings(drawings=drawings):
                bbox = pymupdf.Rect(cluster)
                if (
                    _area(bbox) < minimum_area
                    or _area(bbox) / page_area >= self.full_page_image_ratio
                    or _duplicates_region(bbox, regions)
                ):
                    continue
                paths = [
                    drawing
                    for drawing in drawings
                    if pymupdf.Rect(drawing["rect"]).intersects(bbox)
                ]
                contains_text = any(
                    _intersection_ratio(block.bbox, bbox) >= 0.2 for block in text_blocks
                )
                if len(paths) >= self.vector_min_paths and (contains_text or len(paths) >= 8):
                    regions.append(_VisualRegion(bbox, "diagram"))
                    flags.append("vector-graphic-detected")

        scanned = full_page_image and sum(len(block.text) for block in text_blocks) < 24
        return regions, list(dict.fromkeys(flags)), scanned

    def _text_elements(
        self,
        page: pymupdf.Page,
        page_number: int,
        text_blocks: list[_TextBlock],
        regions: list[_VisualRegion],
    ) -> tuple[list[PdfElement], int]:
        if not regions:
            elements = [
                self._make_text_element(page_number, block.bbox, block.text)
                for block in text_blocks
                if block.text.strip()
            ]
            return elements, 0

        grouped_words: dict[int, dict[int, list[tuple[Any, ...]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        excluded = 0
        for word in page.get_text("words", sort=True):
            bbox = pymupdf.Rect(word[:4])
            if any(_center_inside(bbox, region.bbox) for region in regions):
                excluded += 1
                continue
            grouped_words[int(word[5])][int(word[6])].append(word)

        elements: list[PdfElement] = []
        for lines in grouped_words.values():
            block_words: list[tuple[Any, ...]] = []
            rendered_lines: list[str] = []
            for words in lines.values():
                ordered = sorted(words, key=lambda word: int(word[7]))
                block_words.extend(ordered)
                rendered_lines.append(" ".join(str(word[4]) for word in ordered))
            text = "\n".join(rendered_lines).strip()
            if not text:
                continue
            bbox = _union_rects([pymupdf.Rect(word[:4]) for word in block_words])
            elements.append(self._make_text_element(page_number, bbox, text))
        return elements, excluded

    def _make_text_element(
        self,
        page_number: int,
        bbox: pymupdf.Rect,
        text: str,
    ) -> PdfElement:
        parser = self.text_parser
        parsed = parser(text) if callable(parser) else parser.parse(text)
        if not isinstance(parsed, str):
            raise TypeError("text_parser must return a string")
        return PdfElement(
            page_number=page_number,
            kind=PdfElementKind.TEXT,
            source="native",
            bbox=_rect_tuple(bbox),
            text=parsed.strip(),
        )

    def _prepare_visual_parser(self) -> None:
        if not self._visual_prepared:
            self.visual_parser.prepare()
            self._visual_prepared = True

    def _render_region(self, page: pymupdf.Page, bbox: pymupdf.Rect) -> Image.Image:
        padding = self.visual_padding_points
        clip = pymupdf.Rect(
            bbox.x0 - padding,
            bbox.y0 - padding,
            bbox.x1 + padding,
            bbox.y1 + padding,
        ) & page.rect
        pixmap = page.get_pixmap(
            dpi=self.render_dpi,
            clip=clip,
            colorspace=pymupdf.csRGB,
            alpha=False,
            annots=False,
        )
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _native_text_blocks(page: pymupdf.Page) -> list[_TextBlock]:
    blocks: list[_TextBlock] = []
    for raw in page.get_text("blocks", sort=True):
        if len(raw) < 7 or int(raw[6]) != 0:
            continue
        text = str(raw[4]).strip()
        if text:
            blocks.append(_TextBlock(pymupdf.Rect(raw[:4]), text))
    return blocks


def _native_quality_flags(blocks: list[_TextBlock]) -> list[str]:
    if not blocks:
        return []
    text = "".join(block.text for block in blocks)
    flags: list[str] = []
    if text.count("\ufffd") / max(len(text), 1) > 0.005:
        flags.append("native-text-broken-encoding")
    short_blocks = sum(len(block.text.split()) <= 1 for block in blocks)
    if len(blocks) >= 8 and short_blocks / len(blocks) > 0.6:
        flags.append("native-text-fragmented")
    return flags


def _layout_quality_flags(page: pymupdf.Page, blocks: list[_TextBlock]) -> list[str]:
    width = float(page.rect.width)
    left = [
        block
        for block in blocks
        if block.bbox.x1 <= width * 0.58 and block.bbox.width <= width * 0.55
    ]
    right = [
        block
        for block in blocks
        if block.bbox.x0 >= width * 0.42 and block.bbox.width <= width * 0.55
    ]
    if not left or not right:
        return []
    left_range = (min(block.bbox.y0 for block in left), max(block.bbox.y1 for block in left))
    right_range = (min(block.bbox.y0 for block in right), max(block.bbox.y1 for block in right))
    overlap = min(left_range[1], right_range[1]) - max(left_range[0], right_range[0])
    if overlap > 24:
        return ["multi-column-layout"]
    return []


def _area(rect: pymupdf.Rect) -> float:
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def _intersection_ratio(first: pymupdf.Rect, second: pymupdf.Rect) -> float:
    intersection = first & second
    return _area(intersection) / max(_area(first), 1.0)


def _duplicates_region(bbox: pymupdf.Rect, regions: list[_VisualRegion]) -> bool:
    for region in regions:
        intersection = bbox & region.bbox
        smaller_area = min(_area(bbox), _area(region.bbox))
        if smaller_area and _area(intersection) / smaller_area >= 0.8:
            return True
    return False


def _center_inside(inner: pymupdf.Rect, outer: pymupdf.Rect) -> bool:
    return outer.contains(pymupdf.Point((inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2))


def _union_rects(rects: list[pymupdf.Rect]) -> pymupdf.Rect:
    if not rects:
        return pymupdf.Rect()
    result = pymupdf.Rect(rects[0])
    for rect in rects[1:]:
        result.include_rect(rect)
    return result


def _rect_tuple(rect: pymupdf.Rect) -> RectTuple:
    return tuple(round(float(value), 3) for value in (rect.x0, rect.y0, rect.x1, rect.y1))


def _native_markdown_table(rows: list[list[str | None]]) -> str | None:
    rows = [
        row
        for row in rows
        if any(value is not None and str(value).strip() for value in row)
    ]
    if len(rows) < 2:
        return None
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return None
    normalized = [
        ["" if value is None else _escape_markdown_cell(str(value)) for value in row]
        + [""] * (width - len(row))
        for row in rows
    ]
    non_empty = sum(bool(value.strip()) for row in normalized for value in row)
    if non_empty / (len(normalized) * width) < 0.5:
        return None
    lines = [
        "| " + " | ".join(normalized[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>").strip()
