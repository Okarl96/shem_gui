"""Plugin interface definition."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QWidget


class ITabPlugin(Protocol):
    """Interface for tab plugins."""

    @property
    def name(self) -> str:
        """Display name for the tab."""
        ...

    @property
    def icon(self) -> str:
        """Path to icon file (optional)."""
        ...

    def create_widget(self, data_source) -> QWidget:
        """Create the plugin's widget.

        Args:
            data_source: Reference to data processor or reconstructor

        Returns
        -------
            QWidget to be added as a tab
        """
        ...

    def on_data_update(self, data_point) -> None:
        """Handle new data points.

        Args:
            data_point: DataPoint object
        """
        ...

    def on_scan_started(self) -> None:
        """Called when scan starts."""
        ...

    def on_scan_completed(self) -> None:
        """Called when scan completes."""
        ...
