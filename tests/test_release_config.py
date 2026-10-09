"""Pins the release rules: which commit prefix is allowed to bump which version part.

These read the configuration rather than running semantic-release, so the suite needs
no release tooling. CI additionally runs the real CLI against this config.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MAJOR_MARKERS = ("feat!", "BREAKING CHANGE")
MINOR_TYPES = {"feat"}
PATCH_TYPES = {"fix", "perf", "docs", "refactor", "style", "build"}
NO_RELEASE_TYPES = {"chore", "ci", "test"}


def release_config() -> dict:
    """Reads the semantic-release section of pyproject.toml."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["semantic_release"]


def test_only_feat_bumps_the_minor_version():
    """feat is the single minor-bumping commit type."""
    assert set(release_config()["commit_parser_options"]["minor_tags"]) == MINOR_TYPES


def test_fixes_and_docs_bump_the_patch_version():
    """Corrections, docs and similar housekeeping produce a patch release."""
    patch = set(release_config()["commit_parser_options"]["patch_tags"])

    assert patch == PATCH_TYPES
    assert "docs" in patch, "docs bir yama surumu uretmeli"


def test_maintenance_commits_do_not_release():
    """chore, ci and test must not publish a new version."""
    options = release_config()["commit_parser_options"]
    silent = set(options["other_allowed_tags"])

    assert silent == NO_RELEASE_TYPES
    assert not silent & set(options["patch_tags"])
    assert not silent & set(options["minor_tags"])


def test_breaking_change_in_zero_version_reaches_one_point_zero():
    """A breaking change while on 0.x must produce 1.0.0, not 0.x+1."""
    assert release_config()["major_on_zero"] is True


def test_every_commit_type_has_exactly_one_rule():
    """No commit type appears in two bump categories."""
    options = release_config()["commit_parser_options"]
    groups = [
        set(options["minor_tags"]),
        set(options["patch_tags"]),
        set(options["other_allowed_tags"]),
    ]
    flat = [tag for group in groups for tag in group]

    assert len(flat) == len(set(flat)), "bir commit turu iki kategoride"


def test_version_is_stamped_in_both_places():
    """The version lives in pyproject.toml and in the package itself."""
    release = release_config()

    assert release["version_toml"] == ["pyproject.toml:project.version"]
    assert release["version_variables"] == ["src/turkish_tts/__init__.py:__version__"]


def test_release_commits_are_kept_out_of_the_changelog():
    """The automated release commit does not pollute its own changelog."""
    import re

    patterns = release_config()["changelog"]["exclude_commit_patterns"]
    assert any(re.match(p, "chore(release): 1.2.3 [skip ci]") for p in patterns)


def test_release_commit_message_stops_ci_recursion():
    """The release commit carries [skip ci] so pushing it cannot retrigger a release."""
    assert "[skip ci]" in release_config()["commit_message"]


def test_package_version_matches_pyproject():
    """The two stamped version locations agree."""
    import turkish_tts

    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert turkish_tts.__version__ == config["project"]["version"]
