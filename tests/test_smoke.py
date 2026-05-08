"""
Smoke tests — verify imports and config structure without requiring API keys.
Run with: python -m pytest tests/ -v
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_module_imports():
    """All modules must be importable without errors."""
    import audio_handler   # noqa: F401
    import translation     # noqa: F401
    import tts             # noqa: F401
    import websocket_client  # noqa: F401
    import config          # noqa: F401


def test_config_attributes_exist():
    """Config must expose all required attributes."""
    import config
    required_attrs = [
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
        "AZURE_TTS_KEY",
        "AZURE_REGION",
        "SOURCE_LANGUAGE",
        "TARGET_LANGUAGE",
        "SAMPLE_RATE",
        "ENCODING",
        "CHANNELS",
        "GROQ_MODEL",
    ]
    for attr in required_attrs:
        assert hasattr(config, attr), f"config.{attr} is missing"


def test_config_defaults():
    """Default values must be sensible even without a .env file."""
    import config
    assert config.SAMPLE_RATE == 16000
    assert config.CHANNELS == 1
    assert config.ENCODING == "linear16"
    assert config.GROQ_MODEL == "llama3-8b-8192"
    assert config.GROQ_TEMPERATURE == 0.2


def test_audiohandler_class_structure():
    """AudioHandler must have start/finish methods."""
    from audio_handler import AudioHandler
    assert hasattr(AudioHandler, "start")
    assert hasattr(AudioHandler, "finish")


def test_translation_functions_exist():
    """Translation module must expose both sync and async functions."""
    from translation import groq_translate, groq_translate_async
    import asyncio
    assert callable(groq_translate)
    assert asyncio.iscoroutinefunction(groq_translate_async)
