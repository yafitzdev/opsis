import numpy as np
import pytest

from rag_image_parser.backend import OnnxBackend, _rows_end_with, _rows_start_with


def test_rejects_unknown_onnx_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported ONNX variant"):
        OnnxBackend(variant="tiny")


def test_token_sequence_boundaries() -> None:
    values = np.asarray([[44, 10079, 46, 12, 9617, 10079, 46]])
    assert _rows_start_with(values, np.asarray([44, 10079, 46]))
    assert _rows_end_with(values, np.asarray([9617, 10079, 46]))
    assert not _rows_end_with(values, np.asarray([9617, 6413, 46]))


def test_rejects_invalid_thread_count() -> None:
    with pytest.raises(ValueError, match="threads must be at least 1"):
        OnnxBackend(threads=0)


@pytest.mark.parametrize("edge", [0, 511, 900, 1025])
def test_rejects_invalid_image_edge(edge: int) -> None:
    with pytest.raises(ValueError, match="multiple of 512"):
        OnnxBackend(image_longest_edge=edge)


@pytest.mark.parametrize("edge", [512, 1024, 1536, 2048])
def test_accepts_supported_image_edge(edge: int) -> None:
    backend = OnnxBackend(image_longest_edge=edge)

    assert backend.image_longest_edge == edge
