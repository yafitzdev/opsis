# Parser contract v1

## Scope

The visual parser accepts one raster image. It never accepts a PDF or reconstructs a whole page.
The separate `PdfRouter` accepts a PDF, uses native page geometry to locate regions, renders those
regions, and calls the visual parser one crop at a time.

## Output modes

### `description`

Return roughly 40-100 words of literal, retrieval-friendly prose.

- Name the main subjects, setting, actions, and visible text that affect meaning.
- For charts, mention the chart type, labels, units, major values, and apparent trend.
- For diagrams, name the components and explain the visible relationships or flow.
- Do not speculate about identity, intent, causes, or details that are not visible.
- Do not add headings, analysis, confidence statements, or conversational filler.

### `table`

Return only one GitHub-flavored Markdown table.

- Preserve every legible header, row, value, unit, sign, and decimal.
- Preserve blank cells as blank.
- Use the second row as the Markdown delimiter row.
- Do not add a caption, explanation, or fenced code block.
- A table may exceed the description word target.

Merged cells cannot be represented losslessly in ordinary Markdown. During data creation they
must be expanded into repeated cells or flattened according to one documented dataset policy.

## Training serialization

Descriptions are wrapped in `<description>...</description>` and tables in
`<table>...</table>`. These tags are training-only delimiters and are removed from public output.

## Required evaluation dimensions

- output-kind accuracy;
- Markdown validity and consistent column count;
- exact and normalized table transcription accuracy;
- diagram relationship coverage;
- unsupported-detail/hallucination rate;
- p50/p95 latency and peak resident memory on named CPU hardware.

Benchmark manifests may express description checks with `required_terms`, directional
`required_relations` (`source` and `target`), and `forbidden_terms`. These are intentionally
deterministic lexical checks; ambiguous examples still require human review.
