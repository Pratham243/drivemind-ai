"""Speech-to-text abstraction (PHASE 18 — optional extension).

Voice is treated as an optional input modality layered *in front of* the
existing text pipeline: a working STT backend simply produces text that
is handed to :class:`~src.nlu.router.NLURouter` exactly like a typed
message. No part of the core system (agent, tools, safety, API) depends
on this module, and its absence must never break text interaction — see
:func:`get_stt_provider`.

No real speech recognition dependency (e.g. `speech_recognition`,
`whisper`) is installed by default (it would add a large, often
platform-fiddly dependency for a text-first assistant); this module
defines the seam a real backend would plug into, matching the same
mock/real provider pattern used for the LLM layer in
:mod:`src.nlu.llm_provider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class SpeechToTextProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Return the transcribed text for an audio file."""


class UnavailableSTTProvider(SpeechToTextProvider):
    """Default provider: voice input is not configured in this environment.

    Any caller must check :meth:`is_available` and fall back to text
    input — it must never crash the application.
    """

    name = "unavailable"

    def is_available(self) -> bool:
        return False

    def transcribe(self, audio_path: str) -> str:
        raise RuntimeError(
            "Speech-to-text is not available in this environment. "
            "Install a real STT backend (e.g. `openai-whisper` or "
            "`SpeechRecognition`) and implement a SpeechToTextProvider "
            "subclass to enable voice input. Text input remains fully "
            "functional without this."
        )


def get_stt_provider() -> SpeechToTextProvider:
    """Return the active STT provider.

    Always returns :class:`UnavailableSTTProvider` in this build — no
    optional STT dependency is installed by default. This function is
    the single place a future real backend (e.g. wrapping `whisper`)
    would be wired in, keeping every caller unaware of the swap.
    """
    return UnavailableSTTProvider()
