import pytest

from rag_image_parser.training import add_text_lora


def test_rejects_invalid_lora_hyperparameters() -> None:
    with pytest.raises(ValueError, match="rank and alpha"):
        add_text_lora(object(), rank=0, alpha=32, dropout=0.05)
    with pytest.raises(ValueError, match="dropout"):
        add_text_lora(object(), rank=16, alpha=32, dropout=1.0)
