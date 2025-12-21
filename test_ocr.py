#!/usr/bin/env python3
"""Quick OCR test script for JPA score sheets."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.image_processor import ImageProcessor
from ocr.tesseract_ocr import TesseractOCR, check_tesseract_installation


def test_basic_ocr():
    """Test basic OCR on a sample scoresheet."""

    # Check Tesseract installation
    print("=" * 60)
    print("Checking Tesseract installation...")
    print("=" * 60)
    if not check_tesseract_installation():
        print("ERROR: Tesseract is not installed or not accessible.")
        print("\nPlease install Tesseract:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-jpn")
        print("  macOS: brew install tesseract tesseract-lang")
        return

    print("\n" + "=" * 60)
    print("Loading and preprocessing image...")
    print("=" * 60)

    # Load sample image
    sample_path = Path("data/raw/LINE_ALBUM_2025秋_251221_1.jpg")
    if not sample_path.exists():
        print(f"ERROR: Sample image not found: {sample_path}")
        return

    processor = ImageProcessor()
    image = processor.load_image(sample_path)
    print(f"✓ Loaded image: {sample_path}")
    print(f"  Size: {image.shape[1]}x{image.shape[0]} pixels")

    # Preprocess
    preprocessed = processor.preprocess_pipeline(
        image,
        denoise=True,
        threshold=True,
        threshold_method="adaptive"
    )
    print(f"✓ Preprocessed image")

    # Save preprocessed image for inspection
    output_path = Path("data/processed/test_preprocessed.jpg")
    processor.save_image(preprocessed, output_path)
    print(f"✓ Saved preprocessed image: {output_path}")

    print("\n" + "=" * 60)
    print("Running OCR...")
    print("=" * 60)

    # Initialize OCR
    ocr = TesseractOCR(lang="jpn+eng", psm=6)

    # Extract text with confidence
    result = ocr.extract_with_confidence(preprocessed)

    print(f"\nConfidence: {result.confidence:.2f}%")
    print(f"\nExtracted Text ({len(result.text)} characters):")
    print("-" * 60)
    print(result.text)
    print("-" * 60)

    # Show some bounding boxes
    print(f"\nDetected {len(result.boxes)} text elements")
    if result.boxes:
        print("\nFirst 10 detected elements:")
        for i, box in enumerate(result.boxes[:10]):
            print(f"  {i+1}. '{box['text']}' (confidence: {box['confidence']:.1f}%)")

    # Try extracting just numbers
    print("\n" + "=" * 60)
    print("Testing number extraction...")
    print("=" * 60)
    numbers = ocr.extract_numbers(preprocessed)
    print(f"Numbers found: {numbers}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_ocr()
