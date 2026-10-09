import io
from dataclasses import dataclass
from typing import Union

import numpy as np
import scipy.io.wavfile

@dataclass
class AudioChunk:
    data: Union[bytes, np.ndarray]
    sample_rate: int
    chunk_index: int
    sentence_index: int
    timestamp_ms: float
    is_final: bool
    is_sentence_final: bool


class AudioFormatConverter:

    @staticmethod
    def to_pcm16_le(audio_np: np.ndarray) -> bytes:
        """ Vectorized conversion to Little-Endian 16-bit PCM."""
        audio_clipped = np.clip(audio_np, -1.0, 1.0)
        audio_clipped *= 32767.0
        pcm_16 = audio_clipped.astype('<i2')
        return pcm_16.tobytes()

    @staticmethod
    def to_wav(audio_np: np.ndarray, sample_rate: int) -> bytes:
        """ Vectorized WAV header + PCM16 data."""
        audio_clipped = np.clip(audio_np, -1.0, 1.0)
        audio_clipped *= 32767.0
        pcm_16 = audio_clipped.astype('<i2')
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, sample_rate, pcm_16)
        return wav_buffer.getvalue()

    @staticmethod
    def to_float32(audio_np: np.ndarray) -> bytes:
        """Little-Endian Float32 conversion."""
        return audio_np.astype('<f4').tobytes()

    @staticmethod
    def convert(audio_np: np.ndarray, sample_rate: int, output_format: str) -> Union[bytes, np.ndarray]:
        if output_format in ["pcm16_le", "pcm16"]:
            return AudioFormatConverter.to_pcm16_le(audio_np)
        elif output_format == "wav":
            return AudioFormatConverter.to_wav(audio_np, sample_rate)
        elif output_format == "float32":
            return AudioFormatConverter.to_float32(audio_np)
        elif output_format == "numpy":
            return audio_np
        else:
            raise ValueError(f"Desteklenmeyen çıktı formatı: {output_format}")
