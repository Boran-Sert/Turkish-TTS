"""Downloads the model weights into the Hugging Face cache before first use."""

import argparse

from .logger import get_logger
from .settings import settings

logger = get_logger()

REQUIRED_FILES = ("config.json", "model.safetensors")


def fetch(model: str | None = None, revision: str | None = None) -> str:
    """Downloads the model repo and returns the local snapshot directory."""
    from huggingface_hub import snapshot_download

    repo_id = model or settings.model.model_path
    logger.info(f"Agirliklar indiriliyor: {repo_id}")
    path = snapshot_download(repo_id=repo_id, revision=revision)
    logger.info(f"Hazir: {path}")
    return path


def main() -> None:
    """Command line entry point for fetching weights."""
    parser = argparse.ArgumentParser(description="Turkish TTS model agirliklarini indirir")
    parser.add_argument("--model", help="Hugging Face repo id veya yerel dizin")
    parser.add_argument("--revision", help="Sabitlenecek commit veya etiket")
    args = parser.parse_args()
    fetch(args.model, args.revision)


if __name__ == "__main__":
    main()
