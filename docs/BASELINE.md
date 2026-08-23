# CPU baseline: 2026-08-21

This is an engineering smoke measurement, not a representative quality benchmark.

Hardware:

- AMD Ryzen 5 9600, 6 physical cores / 12 logical processors
- six inference threads
- SmolVLM-256M-Instruct
- two generated fixtures: one three-node diagram and one 3x3 data table
- greedy decoding, maximum 220 new tokens
- model loading excluded from timed cases

## Results

| Runtime | Precision | Mean latency/image | Table result | Diagram result |
|---|---:|---:|---|---|
| PyTorch | FP32 | 10.72 s | Exact cells | Incomplete/reversed relationship |
| ONNX Runtime | INT8 | 5.64 s | Exact cells | Incomplete/reversed relationship |
| ONNX Runtime | Q4 | 8.71 s | Failed | Repetitive prompt-like output |

INT8 is therefore the initial CPU default. Q4 remains available for experimentation but is not a
release candidate on this runtime. Table equality is measured on canonical cell contents, so
cosmetic differences in Markdown delimiter dashes do not count as errors.

### Image-resolution sweep

The stock processor upscales images to a 2048-pixel longest edge. On the 900x460 table fixture,
this produces 13 vision tiles and 832 image tokens. The same fixture produces five tiles / 320
tokens at 1024 and one tile / 64 tokens at 512.

| Longest edge | Mean latency/image | Output contract |
|---:|---:|---|
| 512 | 0.78 s | Failed table and diagram |
| 1024 | 2.11 s | Failed table and diagram |
| 1536 | 3.91 s | Failed table and diagram |
| 2048 | 5.53 s | Exact table; diagram direction still wrong |

This makes 1024 the first fine-tuning target: it is roughly 2.6x faster than 2048 on this smoke
set, but it must not become the inference default until a trained checkpoint recovers quality.

Peak RSS initially reached 3.98 GiB at 2048 because each of the three split ONNX sessions retained
its own CPU allocation arena. Disabling those per-session arenas reduced the measured peak to
about 2.13 GiB with no regression in this smoke test (5.61 s mean versus 5.85 s immediately
before the allocator change). This is now the default ONNX configuration.

The relationship-aware evaluation of the 2048 INT8 baseline reports 75% required-label recall,
50% directional relation recall, and 0% all-requirements accuracy on the diagram. This explicitly
captures the reversed flow that kind accuracy alone missed.

## Immediate implications

1. The base 256M model can already read a clean table correctly.
2. Diagram relationship descriptions need task-specific supervision.
3. Quantization alone is not enough to meet an aggressive latency target.
4. The next benchmark must contain real photographs, screenshots, charts, tables, and diagrams,
   with p50/p95 latency and peak memory measured on at least two CPU classes.
