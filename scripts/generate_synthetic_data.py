from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class Example:
    filename: str
    output: str
    kind: str


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_name = "arialbd.ttf" if bold else "arial.ttf"
    linux_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("C:/Windows/Fonts") / windows_name,
        Path("/usr/share/fonts/truetype/dejavu") / linux_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(_escape_markdown(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_markdown(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _table_values(rng: random.Random) -> tuple[list[str], list[list[str]]]:
    def item_row(index: int) -> list[str]:
        quantity = rng.randint(1, 20)
        price = round(rng.uniform(2, 90), 2)
        return [
            ["Adapter", "Cable", "Sensor", "Bracket", "Switch", "Case"][index],
            str(quantity),
            f"${price:.2f}",
            f"${quantity * price:.2f}",
        ]

    schemas = [
        (
            ["Model", "Accuracy", "Latency"],
            lambda index: [
                f"Tiny-{chr(65 + index)}",
                f"{rng.uniform(78, 98):.1f}%",
                f"{rng.randint(8, 95)} ms",
            ],
        ),
        (
            ["Region", "Revenue", "Growth"],
            lambda index: [
                ["North", "South", "East", "West", "Central", "Global"][index],
                f"${rng.randint(40, 950):,}k",
                f"{rng.uniform(-8, 24):+.1f}%",
            ],
        ),
        (
            ["Month", "Orders", "Returns", "Rating"],
            lambda index: [
                ["January", "February", "March", "April", "May", "June"][index],
                str(rng.randint(80, 2500)),
                str(rng.randint(0, 90)),
                f"{rng.uniform(3.2, 5):.1f}",
            ],
        ),
        (
            ["Item", "Quantity", "Unit price", "Total"],
            item_row,
        ),
    ]
    headers, row_factory = rng.choice(schemas)
    rows = [row_factory(index) for index in range(rng.randint(3, 6))]
    return list(headers), rows


def make_table_example(rng: random.Random, path: Path) -> str:
    headers, rows = _table_values(rng)
    column_width = rng.randint(180, 225)
    row_height = rng.randint(62, 82)
    margin = rng.randint(28, 55)
    width = margin * 2 + column_width * len(headers)
    height = margin * 2 + row_height * (len(rows) + 1)
    image = Image.new("RGB", (width, height), rng.choice(["white", "#f8fafc", "#fffdf5"]))
    draw = ImageDraw.Draw(image)
    font_size = rng.randint(21, 28)
    body_font = _font(font_size)
    header_font = _font(font_size, bold=True)
    header_fill = rng.choice(["#dbeafe", "#e2e8f0", "#dcfce7", "#1e3a8a"])
    header_text = "white" if header_fill == "#1e3a8a" else "#111827"
    grid_color = rng.choice(["#111827", "#475569", "#94a3b8"])
    line_width = rng.randint(1, 3)
    zebra = rng.choice([True, False])
    x0 = margin
    y0 = margin

    draw.rectangle(
        (x0, y0, x0 + len(headers) * column_width, y0 + row_height),
        fill=header_fill,
    )
    if zebra:
        for row_index in range(len(rows)):
            if row_index % 2:
                top = y0 + row_height * (row_index + 1)
                draw.rectangle(
                    (x0, top, x0 + len(headers) * column_width, top + row_height),
                    fill="#f1f5f9",
                )
    for column in range(len(headers) + 1):
        x = x0 + column * column_width
        draw.line(
            (x, y0, x, y0 + row_height * (len(rows) + 1)),
            fill=grid_color,
            width=line_width,
        )
    for row in range(len(rows) + 2):
        y = y0 + row * row_height
        draw.line(
            (x0, y, x0 + len(headers) * column_width, y),
            fill=grid_color,
            width=line_width,
        )

    for column, value in enumerate(headers):
        draw.text(
            (x0 + column * column_width + 14, y0 + (row_height - font_size) // 2 - 3),
            value,
            fill=header_text,
            font=header_font,
        )
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            draw.text(
                (
                    x0 + column * column_width + 14,
                    y0 + (row_index + 1) * row_height + (row_height - font_size) // 2 - 3,
                ),
                value,
                fill="#111827",
                font=body_font,
            )
    image.save(path, optimize=True)
    return f"<table>\n{_markdown_table(headers, rows)}\n</table>"


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((box[0] + box[2] - width) / 2, (box[1] + box[3] - height) / 2 - bounds[1]),
        text,
        fill=fill,
        font=font,
    )


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
) -> None:
    draw.line((*start, *end), fill=color, width=5)
    draw.polygon(
        [(end[0], end[1]), (end[0] - 18, end[1] - 11), (end[0] - 18, end[1] + 11)],
        fill=color,
    )


def make_diagram_example(rng: random.Random, path: Path) -> str:
    pipelines = [
        ("RAG ingestion", ["PDF parser", "Image parser", "Embeddings", "Vector store"]),
        ("Request flow", ["Client", "API gateway", "Application", "Database"]),
        ("Training pipeline", ["Images", "Augmentation", "Vision model", "Checkpoint"]),
        ("Data processing", ["Source", "Validator", "Transformer", "Index"]),
        ("Deployment", ["Upload", "Worker", "Model", "Search API"]),
    ]
    title, all_labels = rng.choice(pipelines)
    labels = all_labels[: rng.randint(3, 4)]
    width = 1200
    height = 520
    image = Image.new("RGB", (width, height), rng.choice(["white", "#f8fafc", "#fffdf5"]))
    draw = ImageDraw.Draw(image)
    title_font = _font(rng.randint(30, 38), bold=True)
    node_font = _font(rng.randint(22, 28), bold=rng.choice([True, False]))
    _draw_centered(draw, (300, 30, 900, 100), title, title_font, "#111827")

    node_width = 210
    node_height = 105
    gap = (width - 120 - node_width * len(labels)) // (len(labels) - 1)
    boxes: list[tuple[int, int, int, int]] = []
    for index, label in enumerate(labels):
        left = 60 + index * (node_width + gap)
        top = 220 + rng.randint(-18, 18)
        box = (left, top, left + node_width, top + node_height)
        boxes.append(box)
        fill = rng.choice(["#dbeafe", "#dcfce7", "#fef3c7", "#f3e8ff"])
        draw.rounded_rectangle(box, radius=15, fill=fill, outline="#334155", width=4)
        _draw_centered(draw, box, label, node_font, "#111827")
    for source, target in zip(boxes, boxes[1:], strict=False):
        _arrow(
            draw,
            (source[2], (source[1] + source[3]) // 2),
            (target[0], (target[1] + target[3]) // 2),
            "#334155",
        )
    image.save(path, optimize=True)

    relationship = f"{labels[0]} leads to {labels[1]}, which leads to {labels[2]}"
    if len(labels) == 4:
        relationship += f", and finally {labels[3]}"
    return (
        f"<description>A left-to-right diagram titled {title} shows "
        f"{relationship}.</description>"
    )


def _write_manifest(path: Path, examples: list[Example]) -> None:
    lines = [
        json.dumps(
            {"image": example.filename, "output": example.output, "kind": example.kind},
            ensure_ascii=False,
        )
        for example in examples
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate exact synthetic supervision")
    parser.add_argument("--output-dir", default="artifacts/synthetic")
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--diagrams", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.tables < 0 or args.diagrams < 0 or args.tables + args.diagrams == 0:
        raise ValueError("Generate at least one table or diagram")
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0, 1)")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    for index in range(args.tables):
        filename = f"images/table_{index:06d}.png"
        output = make_table_example(rng, output_dir / filename)
        examples.append(Example(filename=filename, output=output, kind="table"))
    for index in range(args.diagrams):
        filename = f"images/diagram_{index:06d}.png"
        output = make_diagram_example(rng, output_dir / filename)
        examples.append(Example(filename=filename, output=output, kind="description"))

    rng.shuffle(examples)
    validation_count = round(len(examples) * args.validation_fraction)
    _write_manifest(output_dir / "validation.jsonl", examples[:validation_count])
    _write_manifest(output_dir / "train.jsonl", examples[validation_count:])
    print(
        f"Generated {len(examples) - validation_count} train and "
        f"{validation_count} validation examples"
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
