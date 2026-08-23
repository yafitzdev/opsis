from rag_image_parser.markdown import (
    canonical_table,
    clean_description,
    extract_markdown_table,
    extract_structured_table,
)


def test_extracts_table_and_removes_fence_and_preface() -> None:
    generated = """Here is the table:
```markdown
| Model | Score |
|:---|---:|
| Tiny | 91.2 |
```
This should not be retained.
"""

    assert extract_markdown_table(generated) == (
        "| Model | Score |\n|:---|---:|\n| Tiny | 91.2 |"
    )


def test_rejects_inconsistent_table_rows() -> None:
    generated = "| A | B |\n|---|---|\n| only one |"

    assert extract_markdown_table(generated) == "| A | B |\n|---|---|"


def test_rejects_non_table() -> None:
    assert extract_markdown_table("A diagram with three connected services.") is None


def test_clean_description_removes_training_tags() -> None:
    assert clean_description("<description>\nA labeled diagram.\n</description>") == (
        "A labeled diagram."
    )


def test_converts_python_table_object_to_markdown() -> None:
    generated = (
        "{'header': ['Model', 'Score'], "
        "'rows': [['Tiny-A', '91.2%'], ['Tiny-B', '88.7%']]}"
    )

    assert extract_structured_table(generated) == (
        "| Model | Score |\n"
        "| --- | --- |\n"
        "| Tiny-A | 91.2% |\n"
        "| Tiny-B | 88.7% |"
    )


def test_rejects_ragged_structured_table() -> None:
    generated = "{\"header\": [\"A\", \"B\"], \"rows\": [[1]]}"

    assert extract_structured_table(generated) is None


def test_canonical_table_ignores_delimiter_style() -> None:
    first = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    second = "|A|B|\n|:------|------:|\n|1|2|"

    assert canonical_table(first) == canonical_table(second)
