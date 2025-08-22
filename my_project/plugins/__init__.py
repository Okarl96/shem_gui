"""Plugin system for extensible visualizations."""
from __future__ import annotations

from .base import ITabPlugin
from .loader import PluginLoader

__all__ = ["ITabPlugin", "PluginLoader"]
