from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_diagram(path: Path) -> None:
    image = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    body_font = _font(26)
    draw.text((330, 35), "Local RAG Pipeline", fill="#111827", font=title_font)

    boxes = [
        ((70, 230, 260, 350), "PDF parser"),
        ((405, 230, 595, 350), "Image model"),
        ((740, 230, 930, 350), "Vector store"),
    ]
    for box, label in boxes:
        draw.rounded_rectangle(box, radius=16, fill="#dbeafe", outline="#2563eb", width=4)
        text_box = draw.textbbox((0, 0), label, font=body_font)
        width = text_box[2] - text_box[0]
        draw.text(
            ((box[0] + box[2] - width) / 2, box[1] + 43),
            label,
            fill="#111827",
            font=body_font,
        )

    for start_x, end_x in [(260, 405), (595, 740)]:
        draw.line((start_x, 290, end_x, 290), fill="#111827", width=5)
        draw.polygon(
            [(end_x, 290), (end_x - 20, 278), (end_x - 20, 302)],
            fill="#111827",
        )
    image.save(path)


def make_table(path: Path) -> None:
    image = Image.new("RGB", (900, 460), "white")
    draw = ImageDraw.Draw(image)
    font = _font(28)
    headers = ["Model", "Accuracy", "Latency"]
    rows = [
        ["Tiny-A", "91.2%", "42 ms"],
        ["Tiny-B", "88.7%", "19 ms"],
        ["Tiny-C", "90.1%", "27 ms"],
    ]
    x_positions = [40, 360, 620, 860]
    y_positions = [50, 140, 230, 320, 410]
    for x in x_positions:
        draw.line((x, y_positions[0], x, y_positions[-1]), fill="black", width=3)
    for y in y_positions:
        draw.line((x_positions[0], y, x_positions[-1], y), fill="black", width=3)

    for column, value in enumerate(headers):
        draw.text((x_positions[column] + 18, 78), value, fill="black", font=font)
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            draw.text(
                (x_positions[column] + 18, 168 + row_index * 90),
                value,
                fill="black",
                font=font,
            )
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/smoke")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_diagram(output_dir / "diagram.png")
    make_table(output_dir / "table.png")
    benchmark_records = [
        {
            "image": "diagram.png",
            "expected_kind": "description",
            "required_terms": ["Local RAG Pipeline", "PDF parser", "Image model", "Vector store"],
            "required_relations": [
                {"source": "PDF parser", "target": "Image model"},
                {"source": "Image model", "target": "Vector store"},
            ],
        },
        {
            "image": "table.png",
            "expected_kind": "table",
            "expected_text": (
                "| Model | Accuracy | Latency |\n"
                "| --- | --- | --- |\n"
                "| Tiny-A | 91.2% | 42 ms |\n"
                "| Tiny-B | 88.7% | 19 ms |\n"
                "| Tiny-C | 90.1% | 27 ms |"
            ),
        },
    ]
    training_records = [
        {
            "image": "diagram.png",
            "output": (
                "<description>A diagram titled Local RAG Pipeline shows a left-to-right flow "
                "from PDF parser to Image model to Vector store.</description>"
            ),
        },
        {
            "image": "table.png",
            "output": (
                "<table>\n| Model | Accuracy | Latency |\n| --- | --- | --- |\n"
                "| Tiny-A | 91.2% | 42 ms |\n| Tiny-B | 88.7% | 19 ms |\n"
                "| Tiny-C | 90.1% | 27 ms |\n</table>"
            ),
        },
    ]
    for filename, records in [
        ("benchmark.jsonl", benchmark_records),
        ("train.jsonl", training_records),
    ]:
        serialized = "\n".join(json.dumps(record) for record in records) + "\n"
        (output_dir / filename).write_text(serialized, encoding="utf-8")
    print(output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
