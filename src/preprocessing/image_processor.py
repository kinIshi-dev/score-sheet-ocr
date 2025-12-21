"""Image preprocessing module for score sheet OCR.

This module provides functions to preprocess score sheet images
before OCR processing. Includes grayscale conversion, thresholding,
noise removal, and other image enhancement techniques.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union


class ImageProcessor:
    """Handles image preprocessing for OCR."""

    def __init__(self):
        """Initialize ImageProcessor."""
        pass

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from file.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array (BGR format)

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return image

    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            image: Input image (BGR or RGB)

        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            # Already grayscale
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_threshold(
        self,
        image: np.ndarray,
        method: str = "adaptive",
        block_size: int = 11,
        c: int = 2
    ) -> np.ndarray:
        """Apply thresholding to binarize the image.

        Args:
            image: Input grayscale image
            method: Thresholding method ("binary", "otsu", "adaptive")
            block_size: Size of pixel neighborhood for adaptive threshold
            c: Constant subtracted from mean for adaptive threshold

        Returns:
            Binary image
        """
        if method == "binary":
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            return binary
        elif method == "otsu":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c
            )
            return binary
        else:
            raise ValueError(f"Unknown threshold method: {method}")

    def remove_noise(
        self,
        image: np.ndarray,
        method: str = "median",
        kernel_size: int = 3
    ) -> np.ndarray:
        """Remove noise from image.

        Args:
            image: Input image
            method: Noise removal method ("median", "gaussian", "bilateral")
            kernel_size: Size of the kernel (must be odd)

        Returns:
            Denoised image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        if method == "median":
            return cv2.medianBlur(image, kernel_size)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            raise ValueError(f"Unknown noise removal method: {method}")

    def morphological_operations(
        self,
        image: np.ndarray,
        operation: str = "close",
        kernel_size: Tuple[int, int] = (3, 3)
    ) -> np.ndarray:
        """Apply morphological operations.

        Args:
            image: Input binary image
            operation: Operation type ("erode", "dilate", "open", "close")
            kernel_size: Size of the morphological kernel

        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        if operation == "erode":
            return cv2.erode(image, kernel, iterations=1)
        elif operation == "dilate":
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == "open":
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")

    def resize_image(
        self,
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None
    ) -> np.ndarray:
        """Resize image.

        Args:
            image: Input image
            width: Target width (if None, calculated from height or scale)
            height: Target height (if None, calculated from width or scale)
            scale: Scale factor (if provided, width/height are ignored)

        Returns:
            Resized image
        """
        if scale is not None:
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
        elif width is not None and height is not None:
            new_width = width
            new_height = height
        elif width is not None:
            aspect_ratio = image.shape[0] / image.shape[1]
            new_width = width
            new_height = int(width * aspect_ratio)
        elif height is not None:
            aspect_ratio = image.shape[1] / image.shape[0]
            new_height = height
            new_width = int(height * aspect_ratio)
        else:
            return image  # No resizing

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def preprocess_pipeline(
        self,
        image: np.ndarray,
        denoise: bool = True,
        threshold: bool = True,
        threshold_method: str = "adaptive",
        resize_scale: Optional[float] = None
    ) -> np.ndarray:
        """Complete preprocessing pipeline.

        Args:
            image: Input image
            denoise: Whether to apply denoising
            threshold: Whether to apply thresholding
            threshold_method: Thresholding method to use
            resize_scale: Optional scale factor for resizing

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        processed = self.convert_to_grayscale(image)

        # Resize if requested
        if resize_scale is not None:
            processed = self.resize_image(processed, scale=resize_scale)

        # Denoise
        if denoise:
            processed = self.remove_noise(processed, method="bilateral")

        # Threshold
        if threshold:
            processed = self.apply_threshold(processed, method=threshold_method)

        return processed

    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save image to file.

        Args:
            image: Image to save
            output_path: Path to save the image

        Raises:
            ValueError: If image cannot be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise ValueError(f"Failed to save image: {output_path}")


def process_scoresheet(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> np.ndarray:
    """Convenience function to process a scoresheet image.

    Args:
        input_path: Path to input image
        output_path: Optional path to save processed image
        **kwargs: Additional arguments passed to preprocess_pipeline

    Returns:
        Preprocessed image
    """
    processor = ImageProcessor()
    image = processor.load_image(input_path)
    processed = processor.preprocess_pipeline(image, **kwargs)

    if output_path is not None:
        processor.save_image(processed, output_path)

    return processed
