from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from rag_image_parser.backend import GenerationBackend, TransformersBackend
from rag_image_parser.markdown import (
    clean_description,
    extract_markdown_table,
    extract_structured_table,
)
from rag_image_parser.prompts import PARSER_PROMPT, PROMPT_VERSION
from rag_image_parser.types import OutputKind, ParseResult


class ImageParser:
    def __init__(
        self,
        backend: GenerationBackend | None = None,
        *,
        max_new_tokens: int = 768,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        self.backend = backend or TransformersBackend()
        self.max_new_tokens = max_new_tokens

    def prepare(self) -> None:
        prepare = getattr(self.backend, "prepare", None)
        if prepare is not None:
            prepare()

    def parse(self, image: str | Path | Image.Image) -> ParseResult:
        pil_image = self._load_image(image)
        started = time.perf_counter()
        generated = self.backend.generate(
            pil_image,
            PARSER_PROMPT,
            self.max_new_tokens,
        )
        latency = time.perf_counter() - started

        table = extract_markdown_table(generated) or extract_structured_table(generated)
        if table is not None:
            kind = OutputKind.TABLE
            text = table
        else:
            kind = OutputKind.DESCRIPTION
            text = clean_description(generated)

        if not text:
            raise RuntimeError("The model returned an empty result")

        return ParseResult(
            kind=kind,
            text=text,
            latency_seconds=latency,
            model_id=self.backend.model_id,
            prompt_version=PROMPT_VERSION,
        )

    @staticmethod
    def _load_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as opened:
            return opened.convert("RGB")
