"""Keeps the documentation honest: links, config defaults and endpoints must match the code."""

import json
import re
from pathlib import Path

import pytest

from turkish_tts.api.app import app
from turkish_tts.settings import Settings, load_settings

REPO = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "API_INTEGRATION_GUIDE.md",
    "SYSTEM_ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "CLAUDE.md",
]

LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CONFIG_ROW = re.compile(r"^\| `([a-z_]+\.[a-z_]+)` \| `([^`]+)` \|", re.MULTILINE)
ENDPOINT = re.compile(r"`(?:GET|POST|WS) (/[a-z0-9/_]*)`")

# Kullanıcıyı yanlış yönlendirecek, artık var olmayan isimler.
STALE_NAMES = [
    "/ws/tts",
    "python -m service.main",
    "setup_env.py",
    "requirements.txt",
    "VoxCPMEnginePool",
    "/tts/synthesize",
    "/ai/chat",
    "piper_fallback",
]


def doc_text(name: str) -> str:
    """Reads one documentation file."""
    return (REPO / name).read_text(encoding="utf-8")


def served_paths() -> set:
    """Returns every path the app serves, descending into included routers."""
    paths = set()
    pending = list(app.routes)
    while pending:
        route = pending.pop()
        path = getattr(route, "path", None)
        if path:
            paths.add(path)
        pending.extend(getattr(route, "routes", None) or [])
        included = getattr(route, "original_router", None)
        if included is not None:
            pending.extend(included.routes)
    return paths


@pytest.mark.parametrize("name", DOCS)
def test_relative_links_resolve(name):
    """Every relative link in the docs points at a file that exists."""
    missing = []
    for target in LINK.findall(doc_text(name)):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = REPO / target.split("#")[0]
        if not path.exists():
            missing.append(target)

    assert not missing, f"{name} içindeki bozuk bağlantılar: {missing}"


@pytest.mark.parametrize("name", DOCS)
def test_no_stale_names(name):
    """The docs do not mention modules, scripts or routes that were removed."""
    text = doc_text(name)
    found = [stale for stale in STALE_NAMES if stale in text]

    assert not found, f"{name} kaldırılmış isimlere atıf yapıyor: {found}"


def shipped_settings(monkeypatch) -> dict:
    """Loads the settings a developer gets from the repo's own tts_config.json."""
    monkeypatch.setenv("TTS_CONFIG_FILE", str(REPO / "tts_config.json"))
    for name in list(__import__("os").environ):
        if name.startswith("TTS_") and name != "TTS_CONFIG_FILE":
            monkeypatch.delenv(name, raising=False)
    return load_settings().model_dump()


def test_documented_config_keys_exist():
    """Each config key in the README table exists on the settings model."""
    defaults = Settings().model_dump()
    unknown = []
    for key, _ in CONFIG_ROW.findall(doc_text("README.md")):
        section, field = key.split(".")
        if section not in defaults or field not in defaults[section]:
            unknown.append(key)

    assert not unknown, f"README'de olup ayarlarda olmayan anahtarlar: {unknown}"


def test_documented_config_defaults_match(monkeypatch):
    """Each documented value equals what the shipped config actually produces."""
    defaults = shipped_settings(monkeypatch)
    rows = CONFIG_ROW.findall(doc_text("README.md"))
    assert rows, "README'de yapılandırma tablosu bulunamadı"

    wrong = []
    for key, documented in rows:
        section, field = key.split(".")
        actual = defaults[section][field]
        try:
            expected = json.loads(documented)
        except json.JSONDecodeError:
            expected = documented
        if expected != actual:
            wrong.append(f"{key}: belgede {documented!r}, kodda {actual!r}")

    assert not wrong, "Yanlış belgelenmiş ayarlar: " + "; ".join(wrong)


def test_documented_endpoints_exist():
    """Every endpoint path named in the docs is served by the app."""
    served = served_paths()
    missing = []
    for name in DOCS:
        for path in set(ENDPOINT.findall(doc_text(name))):
            if path not in served:
                missing.append(f"{name}: {path}")

    assert not missing, f"Belgelenip sunulmayan yollar: {missing}"


def test_readme_documents_every_route():
    """Every public route is mentioned in the README."""
    readme = doc_text("README.md")
    public = {
        path
        for path in served_paths()
        if not path.startswith(("/openapi", "/docs", "/redoc")) and path != "/"
    }
    undocumented = [path for path in public if f"`{path}`" not in readme and path not in readme]

    assert not undocumented, f"README'de geçmeyen rotalar: {sorted(undocumented)}"


WRONG_RATE = re.compile(r"sample_rate[\"']?\s*[:=]\s*24000")


@pytest.mark.parametrize("name", DOCS)
def test_sample_rate_is_never_documented_as_24000(name):
    """No document presents 24000 Hz as the output rate."""
    assert not WRONG_RATE.search(doc_text(name)), f"{name} ornekleme hizini 24000 gosteriyor"
