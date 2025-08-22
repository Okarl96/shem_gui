"""Plugin system for extensible visualizations"""

from .base import ITabPlugin
from .loader import PluginLoader

__all__ = ['ITabPlugin', 'PluginLoader']
