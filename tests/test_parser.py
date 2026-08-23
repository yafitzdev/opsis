from pathlib import Path

from PIL import Image

from rag_image_parser.parser import ImageParser
from rag_image_parser.prompts import PARSER_PROMPT
from rag_image_parser.types import OutputKind


class FakeBackend:
    def __init__(self, output: str) -> None:
        self.output = output
        self.seen_prompt = ""

    @property
    def model_id(self) -> str:
        return "fake/test-model"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        assert image.mode == "RGB"
        assert max_new_tokens == 123
        self.seen_prompt = prompt
        return self.output


def _image(path: Path) -> None:
    Image.new("RGB", (16, 16), "white").save(path)


def test_parser_returns_description(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    _image(image)
    backend = FakeBackend("<description>A red bicycle beside a brick wall.</description>")

    result = ImageParser(backend, max_new_tokens=123).parse(image)

    assert result.kind is OutputKind.DESCRIPTION
    assert result.text == "A red bicycle beside a brick wall."
    assert result.model_id == "fake/test-model"
    assert backend.seen_prompt == PARSER_PROMPT


def test_parser_returns_only_markdown_table(tmp_path: Path) -> None:
    image = tmp_path / "table.png"
    _image(image)
    backend = FakeBackend(
        "<table>\n| Name | Value |\n|---|---:|\n| Alpha | 4 |\n</table>"
    )

    result = ImageParser(backend, max_new_tokens=123).parse(image)

    assert result.kind is OutputKind.TABLE
    assert result.text == "| Name | Value |\n|---|---:|\n| Alpha | 4 |"


def test_parser_rejects_missing_image(tmp_path: Path) -> None:
    backend = FakeBackend("unused")
    parser = ImageParser(backend)

    try:
        parser.parse(tmp_path / "missing.png")
    except FileNotFoundError as exc:
        assert "missing.png" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
