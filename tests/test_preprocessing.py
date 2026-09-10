from src.preprocessing.cleaner import (
    clean_for_traditional_ml,
    clean_for_transformer,
    normalize_whitespace,
)
from src.preprocessing.tokenizer import simple_tokenize


def test_normalize_whitespace_collapses_and_trims():
    assert normalize_whitespace("  a   b\t\nc  ") == "a b c"


def test_clean_for_traditional_ml_lowercases_and_strips_punct():
    result = clean_for_traditional_ml("Set the Temperature to 22 degrees!")
    assert result == "set the temperature to 22 degrees"


def test_clean_for_traditional_ml_can_keep_case():
    result = clean_for_traditional_ml("Hello World", lowercase=False, strip_punct=False)
    assert result == "Hello World"


def test_clean_for_transformer_preserves_case_and_punctuation():
    text = "Navigate to Berlin Brandenburg Airport!"
    result = clean_for_transformer(text)
    assert "Berlin" in result
    assert "!" in result


def test_simple_tokenize_keeps_contractions():
    tokens = simple_tokenize("I'm cold, make it warmer!")
    assert "i'm" in tokens
    assert "warmer" in tokens
    assert "," not in tokens


def test_empty_string_handling():
    assert normalize_whitespace("") == ""
    assert clean_for_traditional_ml("") == ""
    assert simple_tokenize("") == []
