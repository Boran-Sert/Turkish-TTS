"""Bridges the vendored engine to the package settings object."""

from ...settings import StreamingSettings as StreamingConfigModel
from ...settings import settings as config_instance

__all__ = ["config_instance", "StreamingConfigModel"]
