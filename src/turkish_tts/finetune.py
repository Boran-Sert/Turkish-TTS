"""LoRA fine-tuning on a local Turkish dataset, wrapping the vendored trainer."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .logger import get_logger

logger = get_logger()

TRENDYOL_BASE = "Trendyol/Trendyol-TTS"
FINETUNE_EXTRA_HINT = (
    "Ince ayar bagimliliklari eksik. Kurulum: pip install 'turkish-tts[finetune]'"
)


@dataclass
class FinetuneConfig:
    """Fine-tuning settings, defaulting to the values the Trendyol checkpoint used."""

    dataset: str
    output_dir: str
    val_dataset: str = ""
    base_model: str = TRENDYOL_BASE
    steps: int = 2000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    batch_size: int = 2
    grad_accum_steps: int = 8
    max_batch_tokens: int = 8192
    num_workers: int = 2
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_lm: bool = True
    lora_target_dit: bool = True
    lora_target_proj: bool = False
    log_interval: int = 10
    valid_interval: int = 250
    save_interval: int = 250
    tensorboard_dir: str = ""
    lambdas: Dict[str, float] = field(
        default_factory=lambda: {"loss/diff": 1.0, "loss/stop": 1.0}
    )

    def trainer_kwargs(self, pretrained_path: str) -> dict:
        """Translates these settings into the keyword arguments the trainer expects."""
        return {
            "pretrained_path": pretrained_path,
            "train_manifest": self.dataset,
            "val_manifest": self.val_dataset,
            "sample_rate": self.sample_rate,
            "out_sample_rate": self.out_sample_rate,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "num_workers": self.num_workers,
            "num_iters": self.steps,
            "log_interval": self.log_interval,
            "valid_interval": self.valid_interval,
            "save_interval": self.save_interval,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.steps,
            "max_batch_tokens": self.max_batch_tokens,
            "save_path": self.output_dir,
            "tensorboard": self.tensorboard_dir or str(Path(self.output_dir) / "logs"),
            "lambdas": self.lambdas,
            "lora": {
                "enable_lm": self.lora_target_lm,
                "enable_dit": self.lora_target_dit,
                "enable_proj": self.lora_target_proj,
                "r": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
            },
        }


class DatasetError(ValueError):
    """Raised when the training manifest is unusable."""


def _blank_text_errors(path: Path) -> list:
    """Finds rows whose text is only whitespace, which the vendored check lets through."""
    errors = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip() == "":
            errors.append(f"satir {number}: text yalnizca bosluk")
    return errors


def check_dataset(manifest: str, sample_rate: int = 16000, max_samples: int = 0):
    """Validates a JSONL manifest and returns the vendored validation result."""
    from ._vendor.voxcpm.training.validate import validate_manifest

    path = Path(manifest)
    if not path.is_file():
        raise DatasetError(f"Veri seti bulunamadi: {path}")

    result = validate_manifest(str(path), sample_rate=sample_rate, max_samples=max_samples)
    result.errors.extend(_blank_text_errors(path))
    if not result.is_valid:
        head = "\n  - ".join(result.errors[:10])
        more = f"\n  ... ve {len(result.errors) - 10} hata daha" if len(result.errors) > 10 else ""
        raise DatasetError(
            f"{path} kullanilamaz ({result.valid_samples}/{result.total_samples} gecerli):"
            f"\n  - {head}{more}"
        )
    return result


def _resolve_base_model(base_model: str) -> str:
    """Returns a local directory for the base model, downloading it when needed."""
    local = Path(base_model)
    if local.is_dir():
        return str(local)

    from huggingface_hub import snapshot_download

    logger.info(f"Temel model indiriliyor: {base_model}")
    return snapshot_download(repo_id=base_model)


def _write_manifest(config: FinetuneConfig, pretrained_path: str, stats) -> Path:
    """Records what produced this checkpoint, mirroring the Trendyol manifest."""
    path = Path(config.output_dir) / "finetune_manifest.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "voxcpm2_lora_adapter",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_model": config.base_model,
                "base_model_path": pretrained_path,
                "dataset": config.dataset,
                "val_dataset": config.val_dataset or None,
                "samples": stats.valid_samples,
                "audio_hours": round(sum(stats.audio_durations) / 3600, 3),
                "config": asdict(config),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def finetune(
    dataset: str,
    output_dir: str,
    *,
    dry_run: bool = False,
    **overrides,
) -> FinetuneConfig:
    """Fine-tunes a LoRA adapter on the given JSONL dataset and writes it to output_dir."""
    config = FinetuneConfig(dataset=dataset, output_dir=output_dir, **overrides)

    stats = check_dataset(config.dataset, sample_rate=config.sample_rate)
    logger.info(
        f"Veri seti dogrulandi: {stats.valid_samples} ornek, "
        f"{sum(stats.audio_durations) / 3600:.2f} saat ses"
    )
    if config.val_dataset:
        check_dataset(config.val_dataset, sample_rate=config.sample_rate)

    if dry_run:
        logger.info("dry_run: egitim baslatilmadi.")
        return config

    pretrained_path = _resolve_base_model(config.base_model)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        from ._vendor.voxcpm.training.runner import train
    except ImportError as exc:
        raise ImportError(f"{FINETUNE_EXTRA_HINT} ({exc})") from exc

    logger.info(f"Ince ayar basliyor: {config.steps} adim, cikti {config.output_dir}")
    train(**config.trainer_kwargs(pretrained_path))

    manifest = _write_manifest(config, pretrained_path, stats)
    logger.info(f"Ince ayar bitti. Manifest: {manifest}")
    return config
