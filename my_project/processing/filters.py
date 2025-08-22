"""Image filtering and processing utilities"""

import numpy as np
from scipy import ndimage
from typing import Optional


class ImageFilters:
    """Image filtering utilities"""
    
    @staticmethod
    def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian filter"""
        return ndimage.gaussian_filter(image, sigma)
    
    @staticmethod
    def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
        """Apply median filter"""
        return ndimage.median_filter(image, size)
    
    @staticmethod
    def edge_detection(image: np.ndarray) -> np.ndarray:
        """Apply edge detection"""
        return ndimage.sobel(image)
    
    @staticmethod
    def remove_outliers(image: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using z-score method"""
        mean = np.mean(image)
        std = np.std(image)
        z_scores = np.abs((image - mean) / std)
        filtered = image.copy()
        filtered[z_scores > threshold] = mean
        return filtered
