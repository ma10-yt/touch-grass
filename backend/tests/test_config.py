import pytest

from app.config import Settings


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ollama-internal:11434", "http://ollama-internal:11434"),
        ("ollama-internal:11434/", "http://ollama-internal:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("https://ollama.example:11434/", "https://ollama.example:11434"),
    ],
)
def test_ollama_url_normalization(value, expected):
    settings = Settings.model_validate({"OLLAMA_URL": value})

    assert settings.OLLAMA_URL == expected


def test_ollama_url_rejects_empty_value():
    with pytest.raises(ValueError, match="OLLAMA_URL must not be empty"):
        Settings.model_validate({"OLLAMA_URL": " / "})


def test_default_settings_keep_local_ollama_url_and_production_model_name():
    settings = Settings.model_validate({})

    assert settings.OLLAMA_URL == "http://localhost:11434"
    assert settings.GEMMA_MODEL == "gemma3:1b"


def test_ollama_url_rejects_scheme_without_host():
    with pytest.raises(ValueError, match="OLLAMA_URL must include a host"):
        Settings.model_validate({"OLLAMA_URL": "http://"})
