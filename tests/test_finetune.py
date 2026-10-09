"""Tests for the fine-tuning wrapper: dataset validation and trainer wiring."""

import json
import wave
from pathlib import Path

import pytest

from turkish_tts import TurkishTTS
from turkish_tts.finetune import DatasetError, FinetuneConfig, check_dataset, finetune

SAMPLE_RATE = 16000


def write_wav(path: Path, seconds: float = 1.0) -> Path:
    """Writes a silent mono WAV so the validator sees a real audio file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return path


def write_manifest(path: Path, rows: list) -> Path:
    """Writes a JSONL manifest."""
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def dataset(tmp_path):
    """Builds a two-sample dataset with real audio files."""
    audio_one = write_wav(tmp_path / "audio" / "bir.wav")
    audio_two = write_wav(tmp_path / "audio" / "iki.wav", seconds=2.0)
    return write_manifest(
        tmp_path / "train.jsonl",
        [
            {"audio": str(audio_one), "text": "Merhaba dünya."},
            {"audio": str(audio_two), "text": "Bugün hava çok güzel."},
        ],
    )


def test_valid_dataset_reports_statistics(dataset):
    """A well-formed manifest validates and reports durations."""
    result = check_dataset(str(dataset))

    assert result.is_valid
    assert result.total_samples == 2
    assert result.valid_samples == 2
    assert pytest.approx(sum(result.audio_durations), abs=0.1) == 3.0


def test_missing_manifest_is_rejected(tmp_path):
    """A missing manifest names the path it looked for."""
    with pytest.raises(DatasetError, match="bulunamadi"):
        check_dataset(str(tmp_path / "yok.jsonl"))


def test_missing_audio_is_rejected(tmp_path):
    """A manifest pointing at absent audio fails with the offending entry."""
    manifest = write_manifest(
        tmp_path / "bad.jsonl", [{"audio": str(tmp_path / "yok.wav"), "text": "Merhaba."}]
    )
    with pytest.raises(DatasetError) as exc:
        check_dataset(str(manifest))

    assert "kullanilamaz" in str(exc.value)


def test_empty_text_is_rejected(tmp_path):
    """A manifest row without text fails validation."""
    audio = write_wav(tmp_path / "a.wav")
    manifest = write_manifest(tmp_path / "bad.jsonl", [{"audio": str(audio), "text": "   "}])

    with pytest.raises(DatasetError):
        check_dataset(str(manifest))


def test_dry_run_validates_without_training(dataset, tmp_path):
    """dry_run returns the resolved config and starts no training."""
    config = finetune(str(dataset), str(tmp_path / "out"), dry_run=True, steps=10)

    assert isinstance(config, FinetuneConfig)
    assert config.steps == 10
    assert not (tmp_path / "out").exists()


def test_defaults_match_the_trendyol_recipe():
    """The defaults are the values the shipped Trendyol checkpoint was trained with."""
    config = FinetuneConfig(dataset="d.jsonl", output_dir="out")

    assert config.lora_rank == 64
    assert config.lora_alpha == 64
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.steps == 2000
    assert config.batch_size == 2
    assert config.grad_accum_steps == 8
    assert config.sample_rate == 16000
    assert config.out_sample_rate == 48000


def test_trainer_kwargs_match_the_trainer_signature():
    """Every argument the wrapper passes must exist on the vendored trainer."""
    import inspect

    from turkish_tts._vendor.voxcpm.training import runner

    train = getattr(runner.train, "__wrapped__", runner.train)
    accepted = set(inspect.signature(train).parameters)
    passed = set(FinetuneConfig(dataset="d.jsonl", output_dir="out").trainer_kwargs("base"))

    assert passed <= accepted, f"trainer bu argumanlari tanimiyor: {sorted(passed - accepted)}"


def test_required_trainer_arguments_are_supplied():
    """The trainer's mandatory arguments are all provided."""
    import inspect

    from turkish_tts._vendor.voxcpm.training import runner

    train = getattr(runner.train, "__wrapped__", runner.train)
    required = {
        name
        for name, param in inspect.signature(train).parameters.items()
        if param.default is inspect.Parameter.empty
    }
    passed = set(FinetuneConfig(dataset="d.jsonl", output_dir="out").trainer_kwargs("base"))

    assert required <= passed, f"eksik zorunlu arguman: {sorted(required - passed)}"


def test_lora_block_is_shaped_as_the_trainer_expects():
    """The LoRA sub-dictionary uses the keys the trainer reads."""
    kwargs = FinetuneConfig(
        dataset="d.jsonl", output_dir="out", lora_rank=32, lora_target_proj=True
    ).trainer_kwargs("base")

    assert kwargs["lora"] == {
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": True,
        "r": 32,
        "alpha": 64,
        "dropout": 0.0,
    }


def test_finetune_is_reachable_from_the_public_class():
    """TurkishTTS.finetune is the documented entry point."""
    assert TurkishTTS.finetune.__doc__
    assert callable(TurkishTTS.finetune)
