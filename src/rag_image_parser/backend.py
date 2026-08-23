from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from rag_image_parser.config import DEFAULT_MODEL_ID


class GenerationBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str: ...


class OnnxBackend:
    """Greedy ONNX Runtime inference using the official split SmolVLM exports."""

    _VARIANT_SUFFIXES = {
        "fp32": "",
        "hybrid": "_hybrid",
        "int8": "_int8",
        "q4": "_q4",
    }

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        variant: str = "fp32",
        threads: int | None = None,
        image_longest_edge: int = 2048,
        repetition_penalty: float = 1.05,
    ) -> None:
        if variant not in self._VARIANT_SUFFIXES:
            choices = ", ".join(sorted(self._VARIANT_SUFFIXES))
            raise ValueError(f"Unsupported ONNX variant {variant!r}; choose {choices}")
        if threads is not None and threads < 1:
            raise ValueError("threads must be at least 1")
        _validate_image_longest_edge(image_longest_edge)
        self._model_id = model_id
        self.variant = variant
        self.threads = threads
        self.image_longest_edge = image_longest_edge
        self.repetition_penalty = repetition_penalty
        self._processor = None
        self._vision_session = None
        self._embed_session = None
        self._decoder_session = None
        self._config = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _onnx_file(self, stem: str) -> str:
        suffix = self._VARIANT_SUFFIXES[self.variant]
        filename = f"{stem}{suffix}.onnx"
        local_model = Path(self._model_id)
        if local_model.exists():
            path = local_model / "onnx" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing ONNX component: {path}")
            return str(path)

        from huggingface_hub import hf_hub_download

        return hf_hub_download(self._model_id, filename=f"onnx/{filename}")

    def prepare(self) -> None:
        if self._decoder_session is not None:
            return

        import onnxruntime
        from transformers import AutoConfig, AutoProcessor

        options = onnxruntime.SessionOptions()
        # The split model opens three sessions. Independent CPU arenas retain large prompt
        # intermediates in each session and nearly double peak RSS on high-resolution images.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        if self.threads is not None:
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1

        self._config = AutoConfig.from_pretrained(self._model_id)
        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            size={"longest_edge": self.image_longest_edge},
        )
        session_args = {
            "sess_options": options,
            "providers": ["CPUExecutionProvider"],
        }
        self._vision_session = onnxruntime.InferenceSession(
            self._onnx_file("vision_encoder"),
            **session_args,
        )
        self._embed_session = onnxruntime.InferenceSession(
            self._onnx_file("embed_tokens"),
            **session_args,
        )
        self._decoder_session = onnxruntime.InferenceSession(
            self._onnx_file("decoder_model_merged"),
            **session_args,
        )

    def _penalize_repetition(self, logits: np.ndarray, generated: np.ndarray) -> None:
        if self.repetition_penalty == 1.0 or generated.size == 0:
            return
        for batch_index in range(logits.shape[0]):
            for token_id in np.unique(generated[batch_index]):
                score = logits[batch_index, -1, token_id]
                if score < 0:
                    logits[batch_index, -1, token_id] *= self.repetition_penalty
                else:
                    logits[batch_index, -1, token_id] /= self.repetition_penalty

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        self.prepare()
        assert self._config is not None
        assert self._processor is not None
        assert self._vision_session is not None
        assert self._embed_session is not None
        assert self._decoder_session is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=prompt_text,
            images=[image],
            return_tensors="np",
        )

        text_config = self._config.text_config
        batch_size = inputs["input_ids"].shape[0]
        past_key_values = {
            f"past_key_values.{layer}.{key_or_value}": np.zeros(
                [batch_size, text_config.num_key_value_heads, 0, text_config.head_dim],
                dtype=np.float32,
            )
            for layer in range(text_config.num_hidden_layers)
            for key_or_value in ("key", "value")
        }
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = np.cumsum(attention_mask, axis=-1, dtype=np.int64)
        generated = np.empty((batch_size, 0), dtype=np.int64)
        image_features = None
        stop_sequences = [
            np.asarray(
                self._processor.tokenizer.encode(tag, add_special_tokens=False),
                dtype=np.int64,
            )
            for tag in ("</description>", "</table>")
        ]
        description_prefix = np.asarray(
            self._processor.tokenizer.encode("<description>", add_special_tokens=False),
            dtype=np.int64,
        )

        for _ in range(max_new_tokens):
            inputs_embeds = self._embed_session.run(None, {"input_ids": input_ids})[0]
            if image_features is None:
                image_features = self._vision_session.run(
                    ["image_features"],
                    {
                        "pixel_values": inputs["pixel_values"],
                        "pixel_attention_mask": inputs["pixel_attention_mask"].astype(np.bool_),
                    },
                )[0]
                image_tokens = inputs["input_ids"] == self._config.image_token_id
                inputs_embeds[image_tokens] = image_features.reshape(
                    -1,
                    image_features.shape[-1],
                )

            logits, *present_values = self._decoder_session.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **past_key_values,
                },
            )
            self._penalize_repetition(logits, generated)
            input_ids = logits[:, -1].argmax(-1, keepdims=True).astype(np.int64)
            generated = np.concatenate([generated, input_ids], axis=-1)

            for key, value in zip(past_key_values, present_values, strict=True):
                past_key_values[key] = value
            attention_mask = np.concatenate(
                [attention_mask, np.ones_like(input_ids)],
                axis=-1,
            )
            position_ids = position_ids[:, -1:] + 1

            if (input_ids == self._processor.tokenizer.eos_token_id).all():
                break
            if any(_rows_end_with(generated, sequence) for sequence in stop_sequences):
                break
            # Description targets are intentionally short. A mode-specific ceiling prevents
            # relation-list degeneration while leaving long Markdown tables untouched.
            if generated.shape[1] >= 160 and _rows_start_with(
                generated, description_prefix
            ):
                break

        return self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class TransformersBackend:
    """Lazy-loaded Hugging Face backend suitable for a CPU correctness baseline."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        threads: int | None = None,
        image_longest_edge: int = 2048,
    ) -> None:
        _validate_image_longest_edge(image_longest_edge)
        self._model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.threads = threads
        self.image_longest_edge = image_longest_edge
        self._processor = None
        self._model = None
        self._torch = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def prepare(self) -> None:
        """Load model artifacts before a timed benchmark."""
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if self.threads is not None:
            if self.threads < 1:
                raise ValueError("threads must be at least 1")
            torch.set_num_threads(self.threads)

        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        try:
            dtype = dtypes[self.dtype_name]
        except KeyError as exc:
            supported = ", ".join(sorted(dtypes))
            raise ValueError(f"Unsupported dtype {self.dtype_name!r}; choose {supported}") from exc

        if self.device == "cpu" and dtype is torch.float16:
            raise ValueError("float16 CPU inference is poorly supported; use float32 or bfloat16")

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            size={"longest_edge": self.image_longest_edge},
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_id,
            dtype=dtype,
        ).to(self.device)
        self._model.eval()
        self._torch = torch

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=prompt_text,
            images=[image],
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
                use_cache=True,
            )

        generated_ids = output_ids[:, input_length:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()


def _validate_image_longest_edge(value: int) -> None:
    if value < 512 or value % 512 != 0:
        raise ValueError("image_longest_edge must be a multiple of 512 and at least 512")


def _rows_start_with(values: np.ndarray, prefix: np.ndarray) -> bool:
    return values.shape[1] >= prefix.size and bool(
        np.all(values[:, : prefix.size] == prefix, axis=1).all()
    )


def _rows_end_with(values: np.ndarray, suffix: np.ndarray) -> bool:
    return values.shape[1] >= suffix.size and bool(
        np.all(values[:, -suffix.size :] == suffix, axis=1).all()
    )
