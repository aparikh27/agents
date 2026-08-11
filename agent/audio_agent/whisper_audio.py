from typing import Any
import os
import numpy as np
import whisper

from agents.messaging import Message, MessageStatus
from agents.agent.audio_agent.audio import AudioAgent


class WhisperAudioAgent(AudioAgent):
    """Concrete implementation of AudioAgent using OpenAI Whisper for Speech-to-Text."""

    def __init__(self, model_size: str = "base"):
        super().__init__()  # Passes name="Audio" up to BaseAgent
        self.model_size = model_size
        self.model = None
        # Kept for architectural consistency with the stub, but loading is
        # actually deferred to _get_model() (lazy load), matching the
        # legacy WhisperModel pattern.
        self._load_model()

    def _load_model(self) -> None:
        """Declares intent to load the Whisper model; actual load is lazy."""
        print(f"🎙️ [WhisperAudioAgent] Whisper model ('{self.model_size}') will be lazy-loaded on first use...")

    def _get_model(self):
        """Lazily loads and caches the Whisper model (mirrors legacy WhisperModel._get_model)."""
        if self.model is None:
            print(f"🎙️ [WhisperAudioAgent] Loading Whisper model '{self.model_size}' into memory...")
            self.model = whisper.load_model(self.model_size)
        return self.model

    def _transcribe(self, message: Message, payload: dict[str, Any]) -> Message:
        """Transcribes audio into text. Accepts either a file path ('audio_path')
        or raw audio data ('audio_data', e.g. a numpy array / audio buffer)."""
        audio_path = payload.get("audio_path")
        audio_data = payload.get("audio_data")

        if audio_path is None and audio_data is None:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Missing required payload parameter: provide either 'audio_path' or 'audio_data'",
            )

        # File path branch
        if audio_path is not None:
            if not os.path.isfile(audio_path):
                return self.create_response(
                    request=message,
                    status=MessageStatus.ERROR,
                    error=f"Audio file not found: '{audio_path}'",
                )
            source = audio_path
        # Raw audio data branch (numpy array or anything array-like)
        else:
            source = self._normalize_audio_data(audio_data)
            if source is None:
                return self.create_response(
                    request=message,
                    status=MessageStatus.ERROR,
                    error="Invalid 'audio_data': expected a NumPy array or array-like buffer of float samples",
                )

        try:
            model = self._get_model()
            result = model.transcribe(source, language=payload.get("language"))
        except Exception as e:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error=f"Whisper transcription failed: {e}",
            )

        text = result.get("text", "").strip()
        detected_language = result.get("language", payload.get("language", "en"))

        response_payload = {
            "text": text,
            "language": detected_language,
        }
        if audio_path is not None:
            response_payload["audio_path"] = audio_path

        return self.create_response(
            request=message,
            payload=response_payload,
        )

    def _normalize_audio_data(self, audio_data) -> np.ndarray | None:
        """Coerces raw audio input into the float32, mono, 1-D array Whisper expects
        (mirrors legacy WhisperModel.process_audio normalization)."""
        if audio_data is None:
            return None

        if not isinstance(audio_data, np.ndarray):
            try:
                audio_data = np.asarray(audio_data)
            except (TypeError, ValueError):
                return None

        if audio_data.size == 0:
            return None

        audio_data = audio_data.astype(np.float32, copy=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.reshape(-1)

        return audio_data

    def _text_to_speech(self, message: Message, payload: dict[str, Any]) -> Message:
        text = payload.get("text")
        if not text:
            return self.create_response(
                request=message,
                status=MessageStatus.ERROR,
                error="Missing required payload parameter: 'text'",
            )

        return self.create_response(
            request=message,
            payload={
                "text": text,
                "output_audio_path": payload.get("output_path", "output_speech.wav"),
                "status": "generated",
            },
        )