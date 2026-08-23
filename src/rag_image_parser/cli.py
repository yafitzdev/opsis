from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_image_parser.backend import OnnxBackend, TransformersBackend
from rag_image_parser.benchmark import run_benchmark
from rag_image_parser.config import DEFAULT_MODEL_ID
from rag_image_parser.parser import ImageParser
from rag_image_parser.pdf_router import PdfRouter


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("onnx", "pytorch"),
        default="onnx",
        help="Inference runtime; ONNX FP32 is the quality-preserving CPU default",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Local path or Hugging Face model ID",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--image-longest-edge",
        type=int,
        default=2048,
        help="Vision preprocessing cap in 512px steps; lower is faster but needs fine-tuning",
    )
    parser.add_argument(
        "--onnx-variant",
        choices=("int8", "hybrid", "q4", "fp32"),
        default="fp32",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-image-parse")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_command = subparsers.add_parser("parse", help="Parse one or more images")
    parse_command.add_argument("images", nargs="+")
    parse_command.add_argument("--json", action="store_true", dest="as_json")
    _add_backend_arguments(parse_command)

    benchmark_command = subparsers.add_parser("benchmark", help="Run a JSONL benchmark")
    benchmark_command.add_argument("manifest")
    benchmark_command.add_argument("--limit", type=int)
    benchmark_command.add_argument("--json", action="store_true", dest="as_json")
    benchmark_command.add_argument("--output", type=Path, help="Write the JSON report to a file")
    _add_backend_arguments(benchmark_command)

    pdf_command = subparsers.add_parser(
        "pdf",
        help="Adaptively route native text and visual PDF regions",
    )
    pdf_command.add_argument("pdfs", nargs="+")
    pdf_command.add_argument("--json", action="store_true", dest="as_json")
    pdf_command.add_argument("--render-dpi", type=int, default=200)
    pdf_command.add_argument("--min-visual-area-ratio", type=float, default=0.01)
    pdf_command.add_argument(
        "--table-strategy",
        choices=("lines", "lines_strict", "text"),
        default="lines",
        help="Use text for borderless tables; lines is the safer default",
    )
    pdf_command.add_argument("--no-vector-graphics", action="store_true")
    pdf_command.add_argument(
        "--vision-for-detected-tables",
        action="store_true",
        help="Force visual parsing even when a digital table has reliable native cells",
    )
    pdf_command.add_argument("--fail-on-scanned-pages", action="store_true")
    _add_backend_arguments(pdf_command)
    return parser


def _image_parser(args: argparse.Namespace) -> ImageParser:
    if args.backend == "onnx":
        backend = OnnxBackend(
            model_id=args.model,
            variant=args.onnx_variant,
            threads=args.threads,
            image_longest_edge=args.image_longest_edge,
        )
    else:
        backend = TransformersBackend(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            threads=args.threads,
            image_longest_edge=args.image_longest_edge,
        )
    return ImageParser(backend, max_new_tokens=args.max_new_tokens)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        parser = _image_parser(args)
        if args.command == "parse":
            results = [parser.parse(image) for image in args.images]
            if args.as_json:
                for result in results:
                    print(json.dumps(result.to_dict(), ensure_ascii=False))
            elif len(results) == 1:
                print(results[0].text)
            else:
                for image, result in zip(args.images, results, strict=True):
                    print(f"## {image}\n\n{result.text}\n")
            return 0

        if args.command == "pdf":
            router = PdfRouter(
                parser,
                render_dpi=args.render_dpi,
                min_visual_area_ratio=args.min_visual_area_ratio,
                table_strategy=args.table_strategy,
                vision_for_detected_tables=args.vision_for_detected_tables,
                detect_vector_graphics=not args.no_vector_graphics,
                fail_on_scanned_pages=args.fail_on_scanned_pages,
            )
            results = [router.parse(pdf) for pdf in args.pdfs]
            if args.as_json:
                for result in results:
                    print(json.dumps(result.to_dict(), ensure_ascii=False))
            elif len(results) == 1:
                print(results[0].to_markdown())
            else:
                for pdf, result in zip(args.pdfs, results, strict=True):
                    print(f"# {pdf}\n\n{result.to_markdown()}\n")
            return 0

        report = run_benchmark(parser, args.manifest, limit=args.limit)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.as_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            summary = report["summary"]
            print(f"Cases: {summary['cases']}")
            print(f"Kind accuracy: {summary['kind_accuracy']:.1%}")
            if summary["exact_text_accuracy"] is not None:
                print(f"Exact text accuracy: {summary['exact_text_accuracy']:.1%}")
            if summary["valid_table_rate"] is not None:
                print(f"Valid table rate: {summary['valid_table_rate']:.1%}")
            if summary["semantic_requirement_accuracy"] is not None:
                print(
                    "Semantic requirement accuracy: "
                    f"{summary['semantic_requirement_accuracy']:.1%}"
                )
            if summary["mean_required_relation_recall"] is not None:
                print(
                    "Mean required relation recall: "
                    f"{summary['mean_required_relation_recall']:.1%}"
                )
            print(f"Median latency: {summary['median_latency_seconds']:.3f}s")
            print(f"p95 latency: {summary['p95_latency_seconds']:.3f}s")
            print(f"Peak RSS: {summary['peak_rss_megabytes']:.1f} MiB")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
