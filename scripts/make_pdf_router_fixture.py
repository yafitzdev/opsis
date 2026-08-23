from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    if candidate.is_file():
        return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _chart() -> Image.Image:
    image = Image.new("RGB", (560, 300), "white")
    draw = ImageDraw.Draw(image)
    title = _font(25)
    label = _font(18)
    draw.text((150, 18), "Monthly signups", fill="#111827", font=title)
    baseline = 245
    values = [("Jan", 72), ("Feb", 118), ("Mar", 165)]
    for index, (month, height) in enumerate(values):
        left = 85 + index * 145
        draw.rectangle((left, baseline - height, left + 72, baseline), fill="#2563eb")
        draw.text((left + 15, baseline + 12), month, fill="#111827", font=label)
    draw.line((55, 55, 55, baseline), fill="#111827", width=3)
    draw.line((55, baseline, 510, baseline), fill="#111827", width=3)
    return image


def make_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(path), pagesize=letter)
    canvas.setTitle("Adaptive PDF Router Smoke Fixture")

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(72, 730, "Page 1 - clean native text")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(72, 695, "This page contains ordinary paragraphs and no visual regions.")
    canvas.drawString(
        72,
        674,
        "The PDF router should return this text without loading the vision model.",
    )
    canvas.showPage()

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(72, 730, "Page 2 - mixed content")
    canvas.setFont("Helvetica", 12)
    canvas.drawString(
        72,
        702,
        "The table and chart below should be routed as separate image crops.",
    )

    left, bottom, width, row_height = 72, 500, 300, 34
    columns = [left, left + 150, left + width]
    rows = [bottom + index * row_height for index in range(4)]
    canvas.setFillColorRGB(0.86, 0.92, 1.0)
    canvas.rect(left, rows[-2], width, row_height, fill=1, stroke=0)
    canvas.setFillColorRGB(0, 0, 0)
    for x in columns:
        canvas.line(x, rows[0], x, rows[-1])
    for y in rows:
        canvas.line(columns[0], y, columns[-1], y)
    values = [("System", "Latency"), ("Parser-A", "42 ms"), ("Parser-B", "19 ms")]
    for index, (first, second) in enumerate(values):
        y = rows[-1] - 22 - index * row_height
        canvas.drawString(left + 8, y, first)
        canvas.drawString(left + 158, y, second)

    canvas.drawImage(ImageReader(_chart()), 72, 160, width=448, height=240)
    canvas.setFont("Helvetica-Oblique", 10)
    canvas.drawString(72, 140, "Figure 1. Signups increase from January through March.")
    canvas.save()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="tmp/pdfs/router-smoke.pdf")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    make_fixture(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
