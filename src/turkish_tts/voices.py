"""Reference voices for cloning, discovered from a directory of wav and txt pairs."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .settings import settings

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")


class VoiceNotFound(LookupError):
    """Raised when a named voice is not present in the voices directory."""


@dataclass(frozen=True)
class Voice:
    """A reference voice: an audio sample and, optionally, its transcript."""

    name: str
    audio: Path
    text: Optional[str] = None

    @property
    def has_transcript(self) -> bool:
        """Whether a sibling .txt transcript was found next to the audio."""
        return bool(self.text)

    def as_prompt(self, continuation: bool = False) -> dict:
        """Returns the model arguments for this voice, cloning its timbre by default.

        Continuation mode additionally conditions on the transcript, so it is only
        correct when the .txt really transcribes the audio.
        """
        prompt = {"reference_wav_path": str(self.audio)}
        if continuation:
            if not self.text:
                raise ValueError(f"'{self.name}' icin transkript yok, devam modu kullanilamaz.")
            prompt["prompt_wav_path"] = str(self.audio)
            prompt["prompt_text"] = self.text
        return prompt


def voices_dir() -> Path:
    """Returns the directory voices are loaded from."""
    return Path(settings.voices_dir)


def _read_transcript(audio: Path) -> Optional[str]:
    """Reads the sibling .txt transcript of an audio file, if there is one."""
    transcript = audio.with_suffix(".txt")
    if not transcript.is_file():
        return None
    text = transcript.read_text(encoding="utf-8").strip()
    return text or None


def list_voices(directory: Optional[Union[str, Path]] = None) -> List[Voice]:
    """Lists every reference voice found in the voices directory."""
    root = Path(directory) if directory else voices_dir()
    if not root.is_dir():
        return []
    found = [
        Voice(name=path.stem, audio=path, text=_read_transcript(path))
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    return found


def resolve_voice(voice: Union[str, Path, Voice, None]) -> Optional[Voice]:
    """Turns a name, an audio path or a Voice into a Voice, or None when nothing is given."""
    if voice is None:
        return None
    if isinstance(voice, Voice):
        return voice

    candidate = Path(voice)
    if candidate.suffix.lower() in AUDIO_SUFFIXES:
        if not candidate.is_file():
            raise VoiceNotFound(f"Ses dosyası bulunamadı: {candidate}")
        return Voice(name=candidate.stem, audio=candidate, text=_read_transcript(candidate))

    name = str(voice)
    for known in list_voices():
        if known.name == name:
            return known

    available = ", ".join(v.name for v in list_voices()) or "yok"
    raise VoiceNotFound(f"'{name}' sesi {voices_dir()} içinde yok. Mevcut sesler: {available}")
