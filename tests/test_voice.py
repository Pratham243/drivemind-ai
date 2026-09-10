from src.voice.stt import UnavailableSTTProvider, get_stt_provider


def test_default_provider_reports_unavailable():
    provider = get_stt_provider()
    assert isinstance(provider, UnavailableSTTProvider)
    assert provider.is_available() is False


def test_unavailable_provider_raises_clear_error_not_silent_failure():
    provider = UnavailableSTTProvider()
    try:
        provider.transcribe("some.wav")
        assert False, "should have raised"
    except RuntimeError as exc:
        assert "not available" in str(exc)
