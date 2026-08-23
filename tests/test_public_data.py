from rag_image_parser.public_data import (
    ai2d_description,
    chart_description,
    pubtabnet_markdown,
)


def test_pubtabnet_simple_table_becomes_markdown() -> None:
    record = {
        "html": {
            "structure": {
                "tokens": [
                    "<thead>",
                    "<tr>",
                    "<td>",
                    "</td>",
                    "<td>",
                    "</td>",
                    "</tr>",
                    "</thead>",
                    "<tbody>",
                    "<tr>",
                    "<td>",
                    "</td>",
                    "<td>",
                    "</td>",
                    "</tr>",
                    "</tbody>",
                ]
            },
            "cells": [
                {"tokens": ["Model"]},
                {"tokens": ["Score"]},
                {"tokens": ["Tiny"]},
                {"tokens": ["91.2"]},
            ],
        }
    }

    markdown, reason = pubtabnet_markdown(record)

    assert reason == "accepted"
    assert markdown == "| Model | Score |\n| --- | --- |\n| Tiny | 91.2 |"


def test_pubtabnet_spans_are_pruned() -> None:
    record = {
        "html": {
            "structure": {"tokens": ["<tr>", '<td rowspan="2">', "</td>", "</tr>"]},
            "cells": [{"tokens": ["Merged"]}],
        }
    }
    assert pubtabnet_markdown(record) == (None, "spanning_cells")


def test_chart_description_uses_title_and_extrema() -> None:
    annotation = {
        "type": "h_bar",
        "general_figure_info": {"title": {"text": "Waste share"}},
    }
    target = chart_description(annotation, "Country,Share\nA,1.2\nB,5.4\n")

    assert target is not None
    assert "Waste share" in target.text
    assert "minimum is 1.2 at A" in target.text
    assert "maximum is 5.4 at B" in target.text
    assert "Country" in target.required_terms


def test_chart_description_transposes_single_series_time_row() -> None:
    annotation = {
        "type": "line",
        "general_figure_info": {"title": {"text": "Organic land"}},
    }
    target = chart_description(annotation, "Entity,2005,2006,2007\nBenin,0.01,0.03,0.02\n")

    assert target is not None
    assert "Benin" in target.text
    assert "minimum is 0.01 at 2005" in target.text
    assert "maximum is 0.03 at 2006" in target.text


def test_chart_description_rejects_corrupt_numeric_cells() -> None:
    annotation = {
        "type": "pie",
        "general_figure_info": {"title": {"text": "Opinion"}},
    }
    assert chart_description(annotation, "Entity,Value\nYes,....3955\nNo,..55\n") is None


def test_chart_description_prefers_chart_annotations_for_pie_values() -> None:
    annotation = {
        "type": "pie",
        "general_figure_info": {"title": {"text": "Admissions"}},
        "models": [
            {"name": "Christian ,", "value": "3292"},
            {"name": "Muslim ,", "value": "3410"},
        ],
    }
    target = chart_description(annotation, "Entity,Value\nChristian,3410\nMuslim,3410\n")

    assert target is not None
    assert "3292 at Christian" in target.text
    assert "3410 at Muslim" in target.text
    assert "Entity" not in target.required_terms
    assert "Value" not in target.required_terms


def test_ai2d_description_resolves_labeled_directed_link() -> None:
    annotation = {
        "text": {
            "T0": {"value": "Parser"},
            "T1": {"value": "Index"},
        },
        "relationships": {
            "label0": {
                "category": "intraObjectLabel",
                "origin": "T0",
                "destination": "B0",
            },
            "label1": {
                "category": "intraObjectLabel",
                "origin": "T1",
                "destination": "B1",
            },
            "link": {
                "category": "interObjectLinkage",
                "origin": "B0",
                "destination": "B1",
                "hasDirectionality": True,
            },
        },
    }

    target = ai2d_description(annotation, "processFlow")

    assert target is not None
    assert target.required_relations == (("Parser", "Index"),)
    assert "Parser leads to Index" in target.text


def test_ai2d_description_does_not_call_undirected_link_a_flow() -> None:
    annotation = {
        "text": {"T0": {"value": "Earth"}, "T1": {"value": "Sun"}},
        "relationships": {
            "label0": {
                "category": "intraObjectLabel",
                "origin": "T0",
                "destination": "B0",
            },
            "label1": {
                "category": "intraObjectLabel",
                "origin": "T1",
                "destination": "B1",
            },
            "link": {
                "category": "interObjectLinkage",
                "origin": "B0",
                "destination": "B1",
                "hasDirectionality": False,
            },
        },
    }

    target = ai2d_description(annotation, "solarSystem")

    assert target is not None
    assert target.required_relations == ()
    assert "Earth connects with Sun" in target.text
