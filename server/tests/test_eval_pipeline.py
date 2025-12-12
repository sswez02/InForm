from src.core.text_utils import normalise, tokenise


def test_normalise_basic():
    assert normalise("Hello, WORLD!?") == "hello world"


def test_tokenise_removes_stopwords():
    tokens = tokenise("This is a test of the system.")
    assert "this" not in tokens
    assert "is" not in tokens
