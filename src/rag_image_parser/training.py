from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_image_parser.config import DEFAULT_MODEL_ID
from rag_image_parser.dataset import ImageCompletionDataset
from rag_image_parser.prompts import PARSER_PROMPT, PROMPT_VERSION

TEXT_LORA_TARGETS = (
    r"(?:model\.text_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|model\.connector\.modality_projection\.proj)"
)
VISION_LORA_TARGETS = (
    r"(?:model\.text_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|model\.connector\.modality_projection\.proj"
    r"|model\.vision_model\.encoder\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|out_proj))"
)


@dataclass(slots=True)
class ImageCompletionCollator:
    processor: Any

    def __call__(self, examples: list[dict[str, object]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        outputs = [str(example["output"]) for example in examples]

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PARSER_PROMPT},
                ],
            }
        ]
        full_messages = [
            [
                *user_messages,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output}],
                },
            ]
            for output in outputs
        ]
        prompt_text = self.processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_texts = [
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            for messages in full_messages
        ]
        prompt_texts = [prompt_text] * len(examples)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="rag-image-train")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional optimizer-step ceiling; useful for smoke runs",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Defaults to 2e-4 for LoRA and 2e-5 for full fine-tuning",
    )
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument(
        "--image-longest-edge",
        type=int,
        choices=(512, 1024, 1536, 2048),
        default=2048,
        help="Train at the same vision resolution intended for inference",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Update base weights instead of the default text-attention LoRA adapters",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--vision-lora",
        action="store_true",
        help="Also adapt vision-attention Q/K/V/output projections",
    )
    return parser.parse_args()


def add_text_lora(
    model: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    include_vision: bool = False,
) -> Any:
    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank and alpha must be at least 1")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=VISION_LORA_TARGETS if include_vision else TEXT_LORA_TARGETS,
        bias="none",
    )
    return get_peft_model(model, config)


def main() -> int:
    args = _parse_args()

    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is not available in this PyTorch runtime")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model,
        size={"longest_edge": args.image_longest_edge},
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype)

    if not args.full_finetune:
        model = add_text_lora(
            model,
            rank=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            include_vision=args.vision_lora,
        )

    if args.freeze_vision:
        for name, parameter in model.named_parameters():
            if "vision_model" in name:
                parameter.requires_grad = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
                "trainable_fraction": trainable_parameters / total_parameters,
            },
            indent=2,
        )
    )

    train_dataset = ImageCompletionDataset(args.train_file)
    validation_dataset = (
        ImageCompletionDataset(args.validation_file) if args.validation_file else None
    )
    output_dir = Path(args.output_dir).resolve()
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2e-5 if args.full_finetune else 2e-4
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=learning_rate,
        # Transformers 5 accepts a fractional warmup_steps value and expands it against
        # the final optimizer-step count in TrainingArguments.get_warmup_steps().
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if validation_dataset is not None else "no",
        eval_steps=args.eval_steps if validation_dataset is not None else None,
        load_best_model_at_end=validation_dataset is not None,
        metric_for_best_model="eval_loss" if validation_dataset is not None else None,
        greater_is_better=False if validation_dataset is not None else None,
        per_device_eval_batch_size=args.batch_size,
        prediction_loss_only=True,
        remove_unused_columns=False,
        bf16=args.bf16,
        report_to="none",
        save_total_limit=2,
        seed=17,
        dataloader_num_workers=args.dataloader_workers,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=ImageCompletionCollator(processor),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if validation_dataset is not None:
        validation_metrics = trainer.evaluate()
        trainer.save_metrics("validation", validation_metrics)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    metadata = {
        "base_model": args.model,
        "prompt_version": PROMPT_VERSION,
        "image_longest_edge": args.image_longest_edge,
        "fine_tuning": "full" if args.full_finetune else "lora",
        "freeze_vision": args.freeze_vision,
        "learning_rate": learning_rate,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lora": (
            None
            if args.full_finetune
            else {
                "rank": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "targets": VISION_LORA_TARGETS if args.vision_lora else TEXT_LORA_TARGETS,
                "vision_lora": args.vision_lora,
            }
        ),
    }
    (output_dir / "rag_image_parser_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
