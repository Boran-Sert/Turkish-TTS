import re
import time
import queue
from typing import Iterator, List, Optional
import sys
import os

import logging
logger = logging.getLogger("TTS_SYSTEM")

class TextBuffer:
    def __init__(
        self,
        sentence_delimiters: str = ".?!…\n",
        flush_timeout_ms: float = 300.0,
        min_chars: int = 5,
    ):
        self.flush_timeout_s = flush_timeout_ms / 1000.0
        self.min_chars = min_chars
        self._buffer: List[str] = []
        self._last_push_time = time.time()
        
        escaped_delims = re.escape(sentence_delimiters)
        self._sentence_regex = re.compile(f'([{escaped_delims}]+(?:\\s+|$))')

    def push(self, fragment: str) -> List[str]:
        """Yeni gelen LLM token'ını O(1) ekler, tamamlanan cümleleri döndürür."""
        if not fragment:
            return []
            
        self._buffer.append(fragment)
        self._last_push_time = time.time()
        
        logger.debug(f"[TextBuffer] Push: '{fragment}' -> Buffer boyutu: {len(''.join(self._buffer))} bytes")
        
        return self._extract_sentences()

    def _extract_sentences(self) -> List[str]:
        """Tamponu kontrol eder ve noktalama işaretlerinden böler."""
        if not self._buffer:
            return []

        current_text = "".join(self._buffer)

        parts = self._sentence_regex.split(current_text)
        
        sentences = []
        new_buffer_parts = []
        
        i = 0
        while i < len(parts) - 1:
            text_part = parts[i]
            delim_part = parts[i+1]
            sentence = text_part + delim_part
            
            # Daha önceden birikmiş (min_chars'tan küçük olduğu için bekletilen)
            # kısımlar varsa, onları anlam bütünlüğü için bu parçanın başına ekle.
            if new_buffer_parts:
                sentence = "".join(new_buffer_parts) + sentence
                new_buffer_parts = []
            
            if len(sentence.strip()) >= self.min_chars:
                stripped_sentence = sentence.strip()
                sentences.append(stripped_sentence)
                logger.debug(f"[TextBuffer] Cümle ayrıştırıldı: '{stripped_sentence}' ({len(stripped_sentence.encode('utf-8'))} bytes)")
            else:
                new_buffer_parts.append(sentence)
            i += 2
            
        if i < len(parts):
            new_buffer_parts.append(parts[i])
            
        if new_buffer_parts:

            self._buffer = ["".join(new_buffer_parts)]
        else:
            self._buffer = []
            
        return sentences

    def flush(self) -> Optional[str]:
        """Zaman aşımına uğramış eksik cümleleri zorla fırlatır."""
        if self._buffer and (time.time() - self._last_push_time) >= self.flush_timeout_s:
            return self.finalize()
        return None

    def finalize(self) -> Optional[str]:
        """Akış bittiğinde arta kalan her şeyi döner."""
        if self._buffer:
            text = "".join(self._buffer).strip()
            self._buffer = []
            if text:
                return text
        return None


class StreamingTextSource:

    def __init__(self, buffer: TextBuffer):
        self.buffer = buffer
        self.sentence_queue = queue.Queue()
        self.finished = False

    def push_text(self, fragment: str) -> None:
        """LLM iş parçacığı tarafından çağrılır (Producer)."""
        sentences = self.buffer.push(fragment)
        for s in sentences:
            self.sentence_queue.put(s)

    def finish(self) -> None:
        """LLM iş parçacığı bittiğinde çağrılır."""
        self.finished = True
        final_sentence = self.buffer.finalize()
        if final_sentence:
            self.sentence_queue.put(final_sentence)
        self.sentence_queue.put(None)  # Sentinel (Bitiş sinyali)

    def __iter__(self) -> Iterator[str]:
        """TTS iş parçacığı tarafından çağrılır (Consumer)."""
        while True:
            try:
                sentence = self.sentence_queue.get(timeout=0.1)
                if sentence is None:
                    break
                yield sentence
            except queue.Empty:
                if self.finished:
                    break

                flushed_sentence = self.buffer.flush()
                if flushed_sentence:
                    yield flushed_sentence
