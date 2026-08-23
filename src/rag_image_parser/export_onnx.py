from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rag_image_parser.config import DEFAULT_MODEL_ID

DECODER_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
VISION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _check_model_allowing_ort_contrib(model: Any) -> None:
    """Accept the ORT contrib layer-normalization node used by the official decoder."""
    import onnx

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as exc:
        if "SimplifiedLayerNormalization" not in str(exc):
            raise


def _replace_initializer(model: Any, name: str, value: Any) -> None:
    import numpy as np
    from onnx import numpy_helper

    matches = [item for item in model.graph.initializer if item.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ONNX initializer named {name!r}, found {len(matches)}"
        )
    current = matches[0]
    array = np.asarray(value, dtype=numpy_helper.to_array(current).dtype)
    if tuple(array.shape) != tuple(current.dims):
        expected_shape = tuple(current.dims)
        raise ValueError(
            f"Shape mismatch for {name}: checkpoint has {array.shape}, "
            f"ONNX expects {expected_shape}"
        )
    current.CopyFrom(numpy_helper.from_array(array, name=name))


def _find_connector_initializer(model: Any, shape: tuple[int, ...]) -> str:
    candidates = [item.name for item in model.graph.initializer if tuple(item.dims) == shape]
    if len(candidates) != 1:
        raise ValueError(
            "Could not identify the connector projection uniquely; "
            f"expected one initializer with shape {shape}, found {candidates}"
        )
    return candidates[0]


def _consumer_node_names(model: Any, initializer_names: set[str]) -> list[str]:
    """Return named nodes that consume weights which must remain in FP32."""
    consumers = {
        node.name
        for node in model.graph.node
        if node.name and initializer_names.intersection(node.input)
    }
    return sorted(consumers)


def _weight_initializer_for_node(model: Any, node_name: str) -> str:
    nodes = [node for node in model.graph.node if node.name == node_name]
    if len(nodes) != 1:
        raise ValueError(f"Expected exactly one ONNX node named {node_name!r}, found {len(nodes)}")
    initializer_names = {item.name for item in model.graph.initializer}
    weights = [name for name in nodes[0].input if name in initializer_names]
    if len(weights) != 1:
        raise ValueError(
            f"Expected exactly one initializer input for ONNX node {node_name!r}, "
            f"found {weights}"
        )
    return weights[0]


def _onnx_component(model_id: str, filename: str) -> Path:
    local_model = Path(model_id)
    if local_model.is_dir():
        candidate = local_model / "onnx" / filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing base ONNX component: {candidate}")
        return candidate

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, filename=f"onnx/{filename}"))


def export_split_onnx(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    base_onnx_model: str = DEFAULT_MODEL_ID,
    quantize: bool = True,
) -> Path:
    """Patch a merged SmolVLM checkpoint into the official split ONNX graph layout."""
    import onnx
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    source = Path(checkpoint).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Merged checkpoint directory does not exist: {source}")
    destination = Path(output_dir).resolve()
    if source == destination:
        raise ValueError("ONNX output directory must differ from the checkpoint directory")
    onnx_dir = destination / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(str(source))
    processor = AutoProcessor.from_pretrained(str(source))
    model = AutoModelForImageTextToText.from_pretrained(str(source), dtype=torch.float32)
    model.eval()
    state = model.state_dict()

    vision_path = _onnx_component(base_onnx_model, "vision_encoder.onnx")
    embed_path = _onnx_component(base_onnx_model, "embed_tokens.onnx")
    decoder_path = _onnx_component(base_onnx_model, "decoder_model_merged.onnx")

    vision = onnx.load(str(vision_path))
    connector_key = "model.connector.modality_projection.proj.weight"
    connector = state[connector_key].detach().float().cpu().numpy().T
    connector_name = _find_connector_initializer(vision, tuple(connector.shape))
    _replace_initializer(vision, connector_name, connector)
    patched_vision_names = {connector_name}
    patched_vision_weights = 0
    for layer in range(config.vision_config.num_hidden_layers):
        for projection in VISION_PROJECTIONS:
            state_name = (
                f"model.vision_model.encoder.layers.{layer}.self_attn.{projection}.weight"
            )
            node_name = (
                f"/vision_model/encoder/layers.{layer}/self_attn/{projection}/MatMul"
            )
            onnx_name = _weight_initializer_for_node(vision, node_name)
            weight = state[state_name].detach().float().cpu().numpy().T
            _replace_initializer(vision, onnx_name, weight)
            patched_vision_names.add(onnx_name)
            patched_vision_weights += 1
    vision_quantization_exclusions = _consumer_node_names(vision, patched_vision_names)
    _check_model_allowing_ort_contrib(vision)
    fp32_vision = onnx_dir / "vision_encoder.onnx"
    onnx.save(vision, str(fp32_vision))
    del vision

    fp32_embed = onnx_dir / "embed_tokens.onnx"
    shutil.copy2(embed_path, fp32_embed)

    decoder = onnx.load(str(decoder_path))
    patched_decoder_weights = 0
    patched_decoder_names: set[str] = set()
    for layer in range(config.text_config.num_hidden_layers):
        for projection in DECODER_PROJECTIONS:
            state_name = f"model.text_model.layers.{layer}.self_attn.{projection}.weight"
            onnx_name = f"model.layers.{layer}.attn.{projection}.MatMul.weight"
            weight = state[state_name].detach().float().cpu().numpy().T
            _replace_initializer(decoder, onnx_name, weight)
            patched_decoder_names.add(onnx_name)
            patched_decoder_weights += 1
    decoder_quantization_exclusions = _consumer_node_names(
        decoder, patched_decoder_names
    )
    _check_model_allowing_ort_contrib(decoder)
    fp32_decoder = onnx_dir / "decoder_model_merged.onnx"
    onnx.save(decoder, str(fp32_decoder))
    del decoder, model, state

    config.save_pretrained(str(destination))
    processor.save_pretrained(str(destination))

    variants = ["fp32"]
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantization_exclusions = {
            fp32_vision: vision_quantization_exclusions,
            fp32_embed: [],
            fp32_decoder: decoder_quantization_exclusions,
        }
        for fp32_path in (fp32_vision, fp32_embed, fp32_decoder):
            int8_path = fp32_path.with_name(f"{fp32_path.stem}_int8.onnx")
            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
                nodes_to_exclude=quantization_exclusions[fp32_path],
            )
            _check_model_allowing_ort_contrib(
                onnx.load(str(int8_path), load_external_data=False)
            )
        # Decoder quantization caused the largest quality regression in the v1 held-out set.
        # The hybrid keeps it in FP32 while compressing vision and token embeddings.
        shutil.copy2(
            fp32_vision.with_name("vision_encoder_int8.onnx"),
            onnx_dir / "vision_encoder_hybrid.onnx",
        )
        shutil.copy2(
            fp32_embed.with_name("embed_tokens_int8.onnx"),
            onnx_dir / "embed_tokens_hybrid.onnx",
        )
        shutil.copy2(fp32_decoder, onnx_dir / "decoder_model_merged_hybrid.onnx")
        variants.extend(("int8", "hybrid"))

    metadata = {
        "format": "rag-image-parser-split-onnx-v1",
        "source_checkpoint": str(source),
        "base_onnx_graph": base_onnx_model,
        "patched_connector_weights": 1,
        "patched_vision_weights": patched_vision_weights,
        "patched_decoder_weights": patched_decoder_weights,
        "int8_strategy": "dynamic-backbone-with-trained-projections-fp32",
        "int8_fp32_nodes": {
            "vision_encoder": vision_quantization_exclusions,
            "decoder": decoder_quantization_exclusions,
        },
        "hybrid_strategy": "dynamic-vision-and-embeddings-with-decoder-fp32",
        "variants": variants,
    }
    source_metadata = source / "rag_image_parser_config.json"
    if source_metadata.is_file():
        metadata["training"] = json.loads(source_metadata.read_text(encoding="utf-8"))
    (destination / "rag_image_parser_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag-image-export-onnx",
        description="Export a merged SmolVLM parser to split FP32 and INT8 ONNX artifacts",
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--base-onnx-model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    destination = export_split_onnx(
        args.checkpoint,
        args.output_dir,
        base_onnx_model=args.base_onnx_model,
        quantize=not args.no_quantize,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
