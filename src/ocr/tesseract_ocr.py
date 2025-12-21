"""Tesseract OCR integration module for score sheet text extraction.

This module provides functions to extract text from preprocessed
score sheet images using Tesseract OCR with Japanese language support.
"""

import pytesseract
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OCRResult:
    """Container for OCR results."""

    text: str
    confidence: float
    boxes: List[Dict[str, Union[str, int, float]]]

    def __repr__(self) -> str:
        return f"OCRResult(text={self.text[:50]}..., confidence={self.confidence:.2f})"


class TesseractOCR:
    """Handles OCR processing using Tesseract."""

    def __init__(
        self,
        lang: str = "jpn+eng",
        psm: int = 6,
        oem: int = 3
    ):
        """Initialize TesseractOCR.

        Args:
            lang: Language(s) for OCR (e.g., "jpn", "eng", "jpn+eng")
            psm: Page Segmentation Mode (0-13)
                 6 = Assume a single uniform block of text (default)
                 3 = Fully automatic page segmentation, but no OSD
                 11 = Sparse text. Find as much text as possible
            oem: OCR Engine Mode (0-3)
                 3 = Default, based on what is available (LSTM + Legacy)
        """
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.config = self._build_config()

    def _build_config(self) -> str:
        """Build Tesseract configuration string.

        Returns:
            Configuration string for pytesseract
        """
        return f"--oem {self.oem} --psm {self.psm}"

    def extract_text(
        self,
        image: np.ndarray,
        config: Optional[str] = None
    ) -> str:
        """Extract text from image.

        Args:
            image: Input image (grayscale or binary)
            config: Optional custom Tesseract config string

        Returns:
            Extracted text
        """
        if config is None:
            config = self.config

        text = pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=config
        )
        return text.strip()

    def extract_with_confidence(
        self,
        image: np.ndarray,
        config: Optional[str] = None
    ) -> OCRResult:
        """Extract text with confidence scores.

        Args:
            image: Input image (grayscale or binary)
            config: Optional custom Tesseract config string

        Returns:
            OCRResult containing text, confidence, and bounding boxes
        """
        if config is None:
            config = self.config

        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text and calculate average confidence
        text_parts = []
        confidences = []
        boxes = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            if text and conf > 0:  # Valid text with confidence
                text_parts.append(text)
                confidences.append(conf)

                boxes.append({
                    'text': text,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': conf
                })

        full_text = ' '.join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes
        )

    def extract_numbers(
        self,
        image: np.ndarray,
        whitelist: str = "0123456789"
    ) -> str:
        """Extract only numbers from image.

        Args:
            image: Input image (grayscale or binary)
            whitelist: Characters to recognize (default: digits only)

        Returns:
            Extracted numbers as string
        """
        config = f"{self.config} -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=config
        )
        return text.strip()

    def extract_boxes(
        self,
        image: np.ndarray,
        config: Optional[str] = None
    ) -> List[Dict[str, Union[str, int]]]:
        """Extract text with bounding box information.

        Args:
            image: Input image (grayscale or binary)
            config: Optional custom Tesseract config string

        Returns:
            List of dictionaries containing text and box coordinates
        """
        if config is None:
            config = self.config

        data = pytesseract.image_to_boxes(
            image,
            lang=self.lang,
            config=config
        )

        boxes = []
        for line in data.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    boxes.append({
                        'char': parts[0],
                        'x1': int(parts[1]),
                        'y1': int(parts[2]),
                        'x2': int(parts[3]),
                        'y2': int(parts[4])
                    })

        return boxes

    def get_available_languages(self) -> List[str]:
        """Get list of available Tesseract languages.

        Returns:
            List of language codes
        """
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            print(f"Error getting languages: {e}")
            return []


class ScoresheetOCR:
    """Specialized OCR for JPA score sheets."""

    def __init__(self):
        """Initialize ScoresheetOCR."""
        self.ocr = TesseractOCR(lang="jpn+eng", psm=6)

    def extract_player_name(
        self,
        name_roi: np.ndarray
    ) -> Tuple[str, float]:
        """Extract player name from ROI.

        Args:
            name_roi: Region of interest containing player name

        Returns:
            Tuple of (name, confidence)
        """
        result = self.ocr.extract_with_confidence(name_roi)
        return result.text, result.confidence

    def extract_score(
        self,
        score_roi: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """Extract numeric score from ROI.

        Args:
            score_roi: Region of interest containing score

        Returns:
            Tuple of (score, confidence)
        """
        result = self.ocr.extract_with_confidence(score_roi)

        # Try to parse as integer
        try:
            score = int(result.text.strip())
            return score, result.confidence
        except ValueError:
            # If not a valid number, try extracting digits only
            numbers = self.ocr.extract_numbers(score_roi)
            try:
                score = int(numbers.strip())
                return score, result.confidence * 0.8  # Lower confidence
            except ValueError:
                return None, 0.0

    def extract_date(
        self,
        date_roi: np.ndarray
    ) -> Tuple[str, float]:
        """Extract date from ROI.

        Args:
            date_roi: Region of interest containing date

        Returns:
            Tuple of (date_string, confidence)
        """
        result = self.ocr.extract_with_confidence(date_roi)
        return result.text, result.confidence

    def extract_full_scoresheet(
        self,
        image: np.ndarray
    ) -> Dict[str, Union[str, float]]:
        """Extract all data from a complete scoresheet.

        Args:
            image: Full scoresheet image (preprocessed)

        Returns:
            Dictionary containing extracted data
        """
        result = self.ocr.extract_with_confidence(image)

        return {
            'full_text': result.text,
            'confidence': result.confidence,
            'boxes': result.boxes
        }


def check_tesseract_installation() -> bool:
    """Check if Tesseract is properly installed.

    Returns:
        True if Tesseract is available, False otherwise
    """
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"Tesseract not found or not properly installed: {e}")
        return False


def list_available_languages() -> None:
    """Print available Tesseract languages."""
    ocr = TesseractOCR()
    langs = ocr.get_available_languages()
    print(f"Available languages ({len(langs)}):")
    for lang in sorted(langs):
        print(f"  - {lang}")
