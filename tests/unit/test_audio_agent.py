import pytest
import numpy as np
from unittest.mock import MagicMock
from messaging import Message, MessageType, MessageStatus
from agent.audio_agent.audio import AudioAgent
from agent.audio_agent.whisper_audio import WhisperAudioAgent


class DummyAudioAgent(AudioAgent):
    """Concrete dummy class for testing abstract AudioAgent base routing."""

    def _transcribe(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"text": "dummy transcription"})

    def _text_to_speech(self, message: Message, payload: dict) -> Message:
        return self.create_response(request=message, payload={"status": "speech generated"})


@pytest.mark.unit
class TestAudioAgentBase:

    def test_audio_agent_routing_transcribe(self, message_factory):
        agent = DummyAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"text": "dummy transcription"}
        assert resp.parent_id == msg.request_id

    def test_audio_agent_routing_text_to_speech(self, message_factory):
        agent = DummyAudioAgent()
        msg = message_factory(receiver="Audio", action="text_to_speech")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload == {"status": "speech generated"}

    def test_audio_agent_unsupported_action(self, message_factory):
        agent = DummyAudioAgent()
        msg = message_factory(receiver="Audio", action="unsupported_action")
        resp = agent.handle_message(msg)
        assert resp.status == MessageStatus.ERROR
        assert "AudioAgent does not support action 'unsupported_action'." in resp.error


@pytest.mark.unit
class TestWhisperAudioAgent:

    def test_transcribe_missing_payload_params(self, mock_whisper_model, message_factory):
        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing required payload parameter" in resp.error

    def test_transcribe_nonexistent_audio_path(self, mock_whisper_model, message_factory):
        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe", payload={"audio_path": "/nonexistent/file.wav"})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Audio file not found" in resp.error

    def test_transcribe_valid_audio_path(self, mock_whisper_model, message_factory, tmp_path):
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"RIFF dummy audio content")

        mock_whisper_model.transcribe.return_value = {
            "text": "Open the door",
            "language": "en",
        }

        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe", payload={"audio_path": str(audio_file)})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["text"] == "Open the door"
        assert resp.payload["language"] == "en"
        assert resp.payload["audio_path"] == str(audio_file)

    def test_transcribe_valid_audio_data(self, mock_whisper_model, message_factory):
        raw_samples = np.zeros(16000, dtype=np.float32)
        mock_whisper_model.transcribe.return_value = {
            "text": "Turn left",
            "language": "en",
        }

        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe", payload={"audio_data": raw_samples})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["text"] == "Turn left"
        assert "audio_path" not in resp.payload

    def test_transcribe_invalid_audio_data(self, mock_whisper_model, message_factory):
        agent = WhisperAudioAgent()
        # Empty audio data array
        msg = message_factory(receiver="Audio", action="transcribe", payload={"audio_data": np.array([])})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Invalid 'audio_data'" in resp.error

    def test_transcribe_model_exception(self, mock_whisper_model, message_factory, tmp_path):
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"dummy content")

        mock_whisper_model.transcribe.side_effect = RuntimeError("CUDA memory error")

        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="transcribe", payload={"audio_path": str(audio_file)})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Whisper transcription failed: CUDA memory error" in resp.error

    def test_text_to_speech_missing_text(self, message_factory):
        agent = WhisperAudioAgent()
        msg = message_factory(receiver="Audio", action="text_to_speech", payload={})
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.ERROR
        assert "Missing required payload parameter: 'text'" in resp.error

    def test_text_to_speech_success(self, message_factory):
        agent = WhisperAudioAgent()
        msg = message_factory(
            receiver="Audio",
            action="text_to_speech",
            payload={"text": "Hello world", "output_path": "custom_speech.wav"}
        )
        resp = agent.handle_message(msg)

        assert resp.status == MessageStatus.SUCCESS
        assert resp.payload["text"] == "Hello world"
        assert resp.payload["output_audio_path"] == "custom_speech.wav"
        assert resp.payload["status"] == "generated"
