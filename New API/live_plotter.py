"""
LivePlotter - Real-time Visualization

Event-driven plotter that updates when individual pixels complete.
No continuous refresh - only updates on new data arrival.

Supports:
- 2D images (heatmap visualization)
- 1D lines (profile plots)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, Union
import threading


class LivePlotter:
    """
    Event-driven visualization for scan data.
    
    Updates only when new data arrives (per-pixel), not continuously.
    
    Usage:
        # 2D scan
        plotter = LivePlotter(plot_type='2D', shape=(100, 100))
        
        # In scan loop callback:
        def on_point_complete(idx, pos, value):
            plotter.update_point(idx, value)
        
        # After scan
        plotter.save_image('scan_result.png')
        plotter.close()
        
        # 1D scan
        plotter = LivePlotter(plot_type='1D', shape=(100,))
        plotter.update_point(idx, value)
    """
    
    def __init__(self,
                 plot_type: str,
                 shape: Tuple[int, ...],
                 colormap: str = 'viridis',
                 title: Optional[str] = None,
                 interactive: bool = True,
                 extent: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize plotter.

        Args:
            plot_type: '2D' or '1D'
            shape:
                - For 2D: (height, width) or (rows, cols)
                - For 1D: (length,)
            colormap: Matplotlib colormap name for 2D images
            title: Plot title
            interactive: Show interactive window (True) or headless (False)
            extent: For 2D plots, (x_min, x_max, y_min, y_max) in real units
                   If provided, axes will show real positions instead of pixels
        """
        self.plot_type = plot_type
        self.shape = shape
        self.colormap = colormap
        self.title = title
        self.interactive = interactive
        self.extent = extent

        # Data buffer - initialize with NaN
        if plot_type == '2D':
            if len(shape) != 2:
                raise ValueError("2D plot requires shape (height, width)")
            self.data = np.full(shape, np.nan)
        elif plot_type == '1D':
            if len(shape) != 1:
                raise ValueError("1D plot requires shape (length,)")
            self.data = np.full(shape[0], np.nan)
        else:
            raise ValueError("plot_type must be '2D' or '1D'")

        # Setup matplotlib
        self._setup_plot()

        # Thread safety for updates
        self._lock = threading.Lock()

        print(f"âœ“ LivePlotter initialized: {plot_type} {shape}")

    def _setup_plot(self):
        """Initialize matplotlib figure and axes"""
        # Use interactive backend or Agg for headless
        if not self.interactive:
            plt.switch_backend('Agg')

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        if self.title:
            self.fig.suptitle(self.title, fontsize=14)

        if self.plot_type == '2D':
            # Setup 2D image plot
            self.im = self.ax.imshow(
                self.data,
                cmap=self.colormap,
                interpolation='nearest',
                aspect='auto',
                origin='upper',
                extent=self.extent  # Maps pixels to real positions
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax)
            self.cbar.set_label('Signal (nA)', rotation=270, labelpad=20)

            # Set axis labels based on whether extent is provided
            if self.extent:
                self.ax.set_xlabel('X position (nm)')
                self.ax.set_ylabel('Y position (nm)')
            else:
                self.ax.set_xlabel('X pixel')
                self.ax.set_ylabel('Y pixel')

        elif self.plot_type == '1D':
            # Setup 1D line plot
            self.line, = self.ax.plot([], [], 'o-', linewidth=2, markersize=4)
            self.ax.set_xlim(0, self.shape[0] - 1)
            self.ax.set_xlabel('Point index')
            self.ax.set_ylabel('Signal (nA)')
            self.ax.grid(True, alpha=0.3)

        if self.interactive:
            plt.ion()  # Interactive mode
            self.fig.show()

    def update_point(self, index: Union[int, Tuple[int, int]], value: float):
        """
        Update single data point and refresh display.

        This is called once per acquired pixel - no rate limiting needed
        since it's already naturally rate-limited by acquisition speed.

        Args:
            index:
                - For 2D: linear index OR (row, col) tuple
                - For 1D: point index
            value: Signal value to display
        """
        with self._lock:
            if self.plot_type == '2D':
                # Handle 2D indexing
                if isinstance(index, int):
                    # Convert linear index to 2D
                    row = index // self.shape[1]
                    col = index % self.shape[1]
                else:
                    row, col = index

                self.data[row, col] = value

            elif self.plot_type == '1D':
                self.data[index] = value

            # Update visualization
            self._refresh_display()

    def update_batch(self, indices: list, values: list):
        """
        Update multiple points at once.

        Args:
            indices: List of indices (format depends on plot_type)
            values: List of corresponding signal values
        """
        with self._lock:
            for idx, val in zip(indices, values):
                if self.plot_type == '2D':
                    if isinstance(idx, int):
                        row = idx // self.shape[1]
                        col = idx % self.shape[1]
                    else:
                        row, col = idx
                    self.data[row, col] = val
                else:
                    self.data[idx] = val

            self._refresh_display()

    def _refresh_display(self):
        """Update the matplotlib display"""
        try:
            if self.plot_type == '2D':
                # Update image data
                self.im.set_data(self.data)

                # Auto-scale color limits based on valid (non-NaN) data
                valid_data = self.data[~np.isnan(self.data)]
                if len(valid_data) > 0:
                    vmin, vmax = np.min(valid_data), np.max(valid_data)
                    self.im.set_clim(vmin, vmax)

            elif self.plot_type == '1D':
                # Update line data
                valid_mask = ~np.isnan(self.data)
                x_data = np.arange(len(self.data))[valid_mask]
                y_data = self.data[valid_mask]

                self.line.set_data(x_data, y_data)

                # Auto-scale Y axis
                if len(y_data) > 0:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    margin = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
                    self.ax.set_ylim(y_min - margin, y_max + margin)

            # Redraw
            if self.interactive:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

        except Exception as e:
            # Fail silently to avoid breaking the scan
            pass

    def set_colormap(self, cmap: str):
        """
        Change colormap for 2D plots.

        Args:
            cmap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'gray')
        """
        if self.plot_type == '2D':
            self.im.set_cmap(cmap)
            self.colormap = cmap
            self._refresh_display()

    def set_title(self, title: str):
        """Update plot title"""
        self.title = title
        self.fig.suptitle(title, fontsize=14)
        if self.interactive:
            self.fig.canvas.draw_idle()

    def get_data(self) -> np.ndarray:
        """
        Get current data array.

        Returns:
            Copy of internal data buffer
        """
        with self._lock:
            return self.data.copy()

    def save_image(self, filepath: str, dpi: int = 300):
        """
        Save current plot to file.

        Args:
            filepath: Output path (e.g., 'scan.png', 'result.pdf')
            dpi: Resolution for raster formats
        """
        try:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"âœ“ Saved plot to: {filepath}")
        except Exception as e:
            print(f"âœ— Failed to save plot: {e}")

    def close(self):
        """Close the plot window and cleanup"""
        plt.close(self.fig)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto cleanup"""
        self.close()
        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def create_2d_plotter(rows: int, cols: int, **kwargs) -> LivePlotter:
    """
    Convenience constructor for 2D image plots.

    Args:
        rows: Number of rows (Y pixels)
        cols: Number of columns (X pixels)
        **kwargs: Additional arguments for LivePlotter

    Returns:
        Configured LivePlotter instance
    """
    return LivePlotter(plot_type='2D', shape=(rows, cols), **kwargs)


def create_1d_plotter(num_points: int, **kwargs) -> LivePlotter:
    """
    Convenience constructor for 1D line plots.

    Args:
        num_points: Number of points in the line
        **kwargs: Additional arguments for LivePlotter

    Returns:
        Configured LivePlotter instance
    """
    return LivePlotter(plot_type='1D', shape=(num_points,), **kwargs)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    import time

    print("Testing 2D plotter...")
    with create_2d_plotter(50, 50, title='2D Scan Test') as plotter:
        # Simulate scan data
        for i in range(50 * 50):
            row = i // 50
            col = i % 50
            # Simulate signal with some pattern
            value = 100 * np.sin(row / 5) * np.cos(col / 5) + 50
            plotter.update_point(i, value)

            if i % 100 == 0:
                time.sleep(0.01)  # Slow down for visualization

        plotter.save_image('test_2d.png')
        time.sleep(2)

    print("\nTesting 1D plotter...")
    with create_1d_plotter(100, title='1D Line Scan Test') as plotter:
        # Simulate line scan
        for i in range(100):
            value = 50 + 30 * np.sin(i / 10)
            plotter.update_point(i, value)
            time.sleep(0.02)

        plotter.save_image('test_1d.png')
        time.sleep(2)

    print("\nTests complete!")