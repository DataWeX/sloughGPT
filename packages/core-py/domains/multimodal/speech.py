"""
Speech Recognition Module

Speech-to-text using Web Speech API (browser) with server fallback.
"""

from typing import Optional, Protocol
from dataclasses import dataclass
import logging

logger = logging.getLogger("man.speech")


@dataclass
class TranscriptionResult:
    """Result from speech recognition."""
    text: str
    confidence: float
    language: str
    duration: Optional[float] = None


class SpeechRecognizer(Protocol):
    """Protocol for speech recognition implementations."""
    
    def recognize(self, audio_data: bytes, language: str) -> TranscriptionResult:
        """Convert audio to text."""
        ...
    
    def recognize_stream(self, audio_chunk: bytes) -> TranscriptionResult:
        """Stream recognition."""
        ...


class BrowserSpeechRecognizer:
    """
    Client-side speech recognizer using Web Speech API.
    
    Used via JavaScript in browser - this is the protocol definition.
    """
    
    def __init__(self, language: str = "en-US"):
        self.language = language
        self.continuous = False
        self.interim_results = True
    
    def get_config(self) -> dict:
        """Return config for browser Web Speech API."""
        return {
            "language": self.language,
            "continuous": self.continuous,
            "interimResults": self.interim_results,
        }


class ServerSpeechRecognizer:
    """
    Server-side speech recognition.
    
    Requires whisper or similar model loaded.
    """
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._processor = None
    
    def load_model(self):
        """Load whisper model."""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            import torch
            
            self._processor = WhisperProcessor.from_pretrained(f"openai/{self.model_name}")
            self._model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/{self.model_name}"
            )
            self._model.eval()
            logger.info(f"Loaded whisper model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load whisper: {e}")
            self._model = None
    
    def recognize(self, audio_data: bytes, language: str = "en") -> TranscriptionResult:
        """Recognize speech from audio bytes."""
        import torch
        import io
        from scipy.io import wavfile
        
        if self._model is None:
            self.load_model()
        
        if self._model is None:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
            )
        
        try:
            sample_rate, audio = wavfile.read(io.BytesIO(audio_data))
            
            # Process with whisper
            input_features = self._processor(
                audio, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features
            
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=language
            )
            
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )
            
            text = self._processor.batch_decode(predicted_ids)[0]
            
            return TranscriptionResult(
                text=text,
                confidence=0.9,
                language=language,
            )
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=language,
            )


def get_speech_recognizer(
    use_server: bool = False,
    model_name: str = "base",
) -> SpeechRecognizer:
    """
    Get speech recognizer instance.
    
    Args:
        use_server: Use server-side model instead of browser
        model_name: Server model (whisper base/medium/large)
    
    Returns:
        SpeechRecognizer implementation
    """
    if use_server:
        return ServerSpeechRecognizer(model_name)
    return BrowserSpeechRecognizer()


__all__ = [
    "TranscriptionResult",
    "SpeechRecognizer", 
    "BrowserSpeechRecognizer",
    "ServerSpeechRecognizer",
    "get_speech_recognizer",
]