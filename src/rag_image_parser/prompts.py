PROMPT_VERSION = "rag-image-v3"

PARSER_PROMPT = """Parse this image for RAG search. If it is a table, output only the complete
table as Markdown. Otherwise output only a concise factual description. Include meaningful
visible text. For a diagram, name its labels and relationships. For a chart, mention its labels,
values, and trend. Do not speculate or add a heading."""
