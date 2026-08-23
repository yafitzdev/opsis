from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_adapter(
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
) -> Path:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter = Path(adapter_path).resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")
    destination = Path(output_dir).resolve()
    if destination == adapter:
        raise ValueError("Merged output directory must differ from the adapter directory")

    peft_config = PeftConfig.from_pretrained(str(adapter))
    resolved_base = base_model or peft_config.base_model_name_or_path
    if not resolved_base:
        raise ValueError("The adapter does not identify a base model; pass --base-model")

    model = AutoModelForImageTextToText.from_pretrained(
        resolved_base,
        dtype=torch.float32,
    )
    adapter_model = PeftModel.from_pretrained(model, str(adapter))
    merged = adapter_model.merge_and_unload(safe_merge=True)
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(destination), safe_serialization=True)

    processor_source = (
        str(adapter) if (adapter / "processor_config.json").is_file() else resolved_base
    )
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.save_pretrained(str(destination))

    source_metadata = adapter / "rag_image_parser_config.json"
    metadata = (
        json.loads(source_metadata.read_text(encoding="utf-8"))
        if source_metadata.is_file()
        else {}
    )
    metadata.update(
        {
            "base_model": resolved_base,
            "merged_from_adapter": str(adapter),
            "fine_tuning": "merged_lora",
        }
    )
    (destination / "rag_image_parser_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-image-merge-adapter",
        description="Merge a trained LoRA adapter into a standalone PyTorch checkpoint",
    )
    parser.add_argument("adapter_path")
    parser.add_argument("output_dir")
    parser.add_argument("--base-model")
    args = parser.parse_args()
    destination = merge_adapter(
        args.adapter_path,
        args.output_dir,
        base_model=args.base_model,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
