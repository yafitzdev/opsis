from types import SimpleNamespace

import numpy as np
import pytest

from rag_image_parser.export_onnx import (
    _consumer_node_names,
    _find_connector_initializer,
    _replace_initializer,
    _weight_initializer_for_node,
)


class FakeTensor:
    def __init__(self, name: str, value: np.ndarray) -> None:
        self.name = name
        self.dims = value.shape
        self.value = value

    def CopyFrom(self, replacement: "FakeTensor") -> None:
        self.name = replacement.name
        self.dims = replacement.dims
        self.value = replacement.value


class FakeNode:
    def __init__(self, name: str, inputs: list[str]) -> None:
        self.name = name
        self.input = inputs


def test_find_connector_initializer_requires_unique_shape() -> None:
    model = SimpleNamespace(
        graph=SimpleNamespace(
            initializer=[FakeTensor("connector", np.zeros((12, 3), dtype=np.float32))]
        )
    )
    assert _find_connector_initializer(model, (12, 3)) == "connector"
    with pytest.raises(ValueError, match="identify the connector"):
        _find_connector_initializer(model, (3, 12))


def test_replace_initializer_rejects_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx.numpy_helper

    monkeypatch.setattr(onnx.numpy_helper, "to_array", lambda tensor: tensor.value)
    monkeypatch.setattr(
        onnx.numpy_helper,
        "from_array",
        lambda value, name: FakeTensor(name, value),
    )
    tensor = FakeTensor("weight", np.zeros((2, 3), dtype=np.float32))
    model = SimpleNamespace(graph=SimpleNamespace(initializer=[tensor]))
    with pytest.raises(ValueError, match="Shape mismatch"):
        _replace_initializer(model, "weight", np.zeros((3, 2), dtype=np.float32))

    replacement = np.ones((2, 3), dtype=np.float32)
    _replace_initializer(model, "weight", replacement)
    np.testing.assert_array_equal(tensor.value, replacement)


def test_consumer_node_names_returns_only_named_weight_consumers() -> None:
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[
                FakeNode("trained_mm", ["x", "trained"]),
                FakeNode("other_add", ["y", "other"]),
                FakeNode("", ["z", "trained"]),
            ]
        )
    )

    assert _consumer_node_names(model, {"trained"}) == ["trained_mm"]


def test_weight_initializer_for_node_requires_one_weight() -> None:
    model = SimpleNamespace(
        graph=SimpleNamespace(
            node=[FakeNode("projection", ["activation", "weight"])],
            initializer=[FakeTensor("weight", np.zeros((2, 2), dtype=np.float32))],
        )
    )

    assert _weight_initializer_for_node(model, "projection") == "weight"
    with pytest.raises(ValueError, match="exactly one ONNX node"):
        _weight_initializer_for_node(model, "missing")
