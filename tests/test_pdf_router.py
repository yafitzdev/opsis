from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from rag_image_parser.cli import main as cli_main
from rag_image_parser.pdf_router import PageKind, PdfElementKind, PdfRouter
from rag_image_parser.types import OutputKind, ParseResult


class FakeVisualParser:
    def __init__(self, kind: OutputKind = OutputKind.TABLE) -> None:
        self.kind = kind
        self.prepare_calls = 0
        self.images: list[Image.Image] = []

    def prepare(self) -> None:
        self.prepare_calls += 1

    def parse(self, image: Image.Image) -> ParseResult:
        self.images.append(image.copy())
        text = (
            "| Item | Value |\n| --- | --- |\n| Alpha | 10 |\n| Beta | 20 |"
            if self.kind is OutputKind.TABLE
            else "A blue chart with two rising bars."
        )
        return ParseResult(
            kind=self.kind,
            text=text,
            latency_seconds=0.25,
            model_id="fake/visual",
            prompt_version="test",
        )


def _text_and_table_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=letter)
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 720, "This first page contains only ordinary native text.")
    canvas.drawString(72, 700, "It should never initialize the visual model.")
    canvas.showPage()

    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 730, "Quarterly results appear in the table below.")
    left, bottom, width, row_height = 72, 500, 300, 32
    columns = [left, left + 150, left + width]
    rows = [bottom, bottom + row_height, bottom + row_height * 2, bottom + row_height * 3]
    for x in columns:
        canvas.line(x, rows[0], x, rows[-1])
    for y in rows:
        canvas.line(columns[0], y, columns[-1], y)
    values = [("Item", "Value"), ("Alpha", "10"), ("Beta", "20")]
    for index, (first, second) in enumerate(values):
        y = rows[-1] - 21 - index * row_height
        canvas.drawString(left + 8, y, first)
        canvas.drawString(left + 158, y, second)
    canvas.save()


def _image_only_pdf(path: Path) -> None:
    image = Image.new("RGB", (612, 792), "white")
    canvas = Canvas(str(path), pagesize=letter)
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=792)
    canvas.save()


def _image_and_text_pdf(path: Path) -> None:
    image = Image.new("RGB", (300, 180), "#dbeafe")
    canvas = Canvas(str(path), pagesize=letter)
    canvas.drawString(72, 730, "The figure below summarizes the trend.")
    canvas.drawImage(ImageReader(image), 72, 480, width=300, height=180)
    canvas.save()


def _borderless_table_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=letter)
    rows = [
        ("Region", "Revenue", "Growth"),
        ("North", "$120k", "+5%"),
        ("South", "$95k", "+2%"),
        ("West", "$150k", "+8%"),
    ]
    for row_index, row in enumerate(rows):
        y = 700 - row_index * 28
        for x, value in zip((72, 240, 400), row, strict=True):
            canvas.drawString(x, y, value)
    canvas.save()


def _two_column_pdf(path: Path) -> None:
    canvas = Canvas(str(path), pagesize=letter)
    for x, prefix in ((72, "Left"), (330, "Right")):
        text = canvas.beginText(x, 700)
        text.setLeading(20)
        for index in range(4):
            text.textLine(f"{prefix} column paragraph line {index + 1}.")
        canvas.drawText(text)
    canvas.save()


def test_routes_clean_text_and_mixed_table_without_duplicate_cells(tmp_path: Path) -> None:
    pdf = tmp_path / "mixed.pdf"
    _text_and_table_pdf(pdf)
    visual = FakeVisualParser()
    router = PdfRouter(
        visual,
        text_parser=lambda text: text.upper(),
        detect_vector_graphics=True,
        vision_for_detected_tables=True,
    )

    result = router.parse(pdf)

    assert [page.kind for page in result.pages] == [PageKind.CLEAN, PageKind.MIXED]
    assert result.visual_model_used is True
    assert result.visual_inferences == 1
    assert visual.prepare_calls == 1
    assert len(visual.images) == 1
    assert visual.images[0].width > 700
    assert "ORDINARY NATIVE TEXT" in result.pages[0].elements[0].text
    mixed_text = "\n".join(element.text for element in result.pages[1].elements)
    assert "QUARTERLY RESULTS" in mixed_text
    assert mixed_text.count("Alpha") == 1
    assert "native-text-overlaps-visual" in result.pages[1].quality_flags
    assert any(element.kind is PdfElementKind.TABLE for element in result.pages[1].elements)


def test_clean_pdf_never_prepares_visual_parser(tmp_path: Path) -> None:
    pdf = tmp_path / "clean.pdf"
    canvas = Canvas(str(pdf), pagesize=letter)
    canvas.drawString(72, 720, "A clean text-only document.")
    canvas.save()
    visual = FakeVisualParser()

    result = PdfRouter(visual).parse(pdf)

    assert result.pages[0].kind is PageKind.CLEAN
    assert result.visual_model_used is False
    assert visual.prepare_calls == 0
    assert not visual.images


def test_routes_embedded_image_crop_to_description_parser(tmp_path: Path) -> None:
    pdf = tmp_path / "image.pdf"
    _image_and_text_pdf(pdf)
    visual = FakeVisualParser(OutputKind.DESCRIPTION)

    result = PdfRouter(visual, detect_vector_graphics=False).parse(pdf)

    page = result.pages[0]
    assert page.kind is PageKind.MIXED
    assert page.visual_regions == 1
    description = next(
        element for element in page.elements if element.kind is PdfElementKind.DESCRIPTION
    )
    assert description.source == "image"
    assert "rising bars" in description.text


def test_recovers_table_from_native_structure_when_visual_retry_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "mixed.pdf"
    _text_and_table_pdf(pdf)
    visual = FakeVisualParser(OutputKind.DESCRIPTION)

    result = PdfRouter(visual, vision_for_detected_tables=True).parse(pdf)

    page = result.pages[1]
    table = next(element for element in page.elements if element.kind is PdfElementKind.TABLE)
    assert table.text == (
        "| Item | Value |\n"
        "| --- | --- |\n"
        "| Alpha | 10 |\n"
        "| Beta | 20 |"
    )
    assert "table-vision-recovered-from-native-structure" in page.quality_flags


def test_flags_full_page_scan_without_sending_it_to_visual_parser(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf)
    visual = FakeVisualParser(OutputKind.DESCRIPTION)

    result = PdfRouter(visual).parse(pdf)

    page = result.pages[0]
    assert page.kind is PageKind.SCANNED
    assert page.visual_regions == 0
    assert "scanned-page-needs-ocr-layout" in page.quality_flags
    assert visual.prepare_calls == 0
    assert "OCR or page-layout extraction required" in result.to_markdown()


def test_can_fail_fast_on_scanned_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _image_only_pdf(pdf)

    with pytest.raises(RuntimeError, match="needs OCR/layout extraction"):
        PdfRouter(FakeVisualParser(), fail_on_scanned_pages=True).parse(pdf)


def test_rejects_unknown_table_strategy() -> None:
    with pytest.raises(ValueError, match="table_strategy"):
        PdfRouter(FakeVisualParser(), table_strategy="guess")


def test_text_strategy_detects_and_recovers_borderless_table(tmp_path: Path) -> None:
    pdf = tmp_path / "borderless.pdf"
    _borderless_table_pdf(pdf)
    visual = FakeVisualParser(OutputKind.DESCRIPTION)

    result = PdfRouter(
        visual,
        table_strategy="text",
        vision_for_detected_tables=True,
    ).parse(pdf)

    table = next(
        element
        for element in result.pages[0].elements
        if element.kind is PdfElementKind.TABLE
    )
    assert "| Region | Revenue | Growth |" in table.text
    assert "| South | $95k | +2% |" in table.text
    assert "|  |  |  |" not in table.text


def test_uses_native_table_fast_path_without_preparing_visual_model(tmp_path: Path) -> None:
    pdf = tmp_path / "mixed.pdf"
    _text_and_table_pdf(pdf)
    visual = FakeVisualParser(OutputKind.DESCRIPTION)

    result = PdfRouter(visual).parse(pdf)

    table = next(
        element
        for element in result.pages[1].elements
        if element.kind is PdfElementKind.TABLE
    )
    assert "| Alpha | 10 |" in table.text
    assert result.visual_inferences == 0
    assert visual.prepare_calls == 0
    assert "native-table-structure-used" in result.pages[1].quality_flags


def test_flags_multi_column_native_page_for_review_without_loading_vision(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "columns.pdf"
    _two_column_pdf(pdf)
    visual = FakeVisualParser()

    result = PdfRouter(visual).parse(pdf)

    assert result.pages[0].kind is PageKind.REVIEW
    assert "multi-column-layout" in result.pages[0].quality_flags
    assert result.visual_model_used is False
    assert visual.prepare_calls == 0


def test_clean_pdf_cli_emits_structured_json_without_loading_model(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = tmp_path / "clean.pdf"
    canvas = Canvas(str(pdf), pagesize=letter)
    canvas.drawString(72, 720, "CLI clean-page routing.")
    canvas.save()

    exit_code = cli_main(["pdf", str(pdf), "--json"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"kind": "clean"' in output
    assert '"visual_model_used": false' in output
