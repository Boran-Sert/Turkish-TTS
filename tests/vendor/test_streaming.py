import unittest
import numpy as np
from turkish_tts._vendor.voxcpm.streaming import AudioFormatConverter

class TestStreamingComponents(unittest.TestCase):
    """Unit tests for the streaming infrastructure."""
    
    def test_audio_format_converter_pcm16_le(self):
        """Tests Little-Endian PCM16 vectorized conversion accuracy."""
        audio_np = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        pcm_bytes = AudioFormatConverter.to_pcm16_le(audio_np)
        
        expected_array = np.array([-32767, 0, 32767], dtype=np.int16)
        expected_bytes = expected_array.astype('<i2').tobytes()
        
        self.assertEqual(pcm_bytes, expected_bytes)

    def test_audio_format_converter_clipping(self):
        """Tests that out-of-bound float signals are clipped properly."""
        audio_np = np.array([-2.0, 5.0], dtype=np.float32)
        pcm_bytes = AudioFormatConverter.to_pcm16_le(audio_np)
        
        expected_array = np.array([-32767, 32767], dtype=np.int16)
        expected_bytes = expected_array.astype('<i2').tobytes()
        
        self.assertEqual(pcm_bytes, expected_bytes)

if __name__ == '__main__':
    unittest.main()
