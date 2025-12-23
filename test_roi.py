#!/usr/bin/env python3
"""Test script for ROI extraction."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.image_processor import ImageProcessor
from preprocessing.roi_extractor import ROIExtractor, visualize_rois
from ocr.tesseract_ocr import TesseractOCR


def test_roi_extraction():
    """Test ROI extraction on sample scoresheet."""

    print("=" * 60)
    print("ROI Extraction Test")
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

    # Initialize ROI extractor
    extractor = ROIExtractor()

    print("\n" + "=" * 60)
    print("Step 1: Separating sheets...")
    print("=" * 60)

    sheets = extractor.separate_sheets(image)
    print(f"✓ Found {len(sheets)} sheets")

    for i, sheet in enumerate(sheets):
        print(f"  Sheet {i+1}: {sheet.shape[1]}x{sheet.shape[0]} pixels")
        # Save separated sheets
        output_path = Path(f"data/processed/sheet_{i+1}.jpg")
        processor.save_image(sheet, output_path)
        print(f"    Saved to: {output_path}")

    if len(sheets) == 0:
        print("ERROR: No sheets detected!")
        return

    # Process first sheet
    sheet = sheets[0]

    print("\n" + "=" * 60)
    print("Step 2: Detecting match areas...")
    print("=" * 60)

    match_rois = extractor.detect_match_areas(sheet, num_matches=5)
    print(f"✓ Detected {len(match_rois)} match areas")

    for match_roi in match_rois:
        print(f"  {match_roi.label}: y={match_roi.y}, height={match_roi.height}")

    # Visualize match areas
    vis_path = Path("data/processed/match_areas_visualization.jpg")
    visualize_rois(sheet, match_rois, str(vis_path))
    print(f"✓ Saved visualization: {vis_path}")

    print("\n" + "=" * 60)
    print("Step 3: Extracting field ROIs from first match...")
    print("=" * 60)

    # Extract first match area
    first_match = match_rois[0]
    match_area = first_match.extract(sheet)

    # Extract all fields
    fields = extractor.extract_all_fields(match_area)

    print(f"✓ Extracted {len(fields)} fields:")
    for field_name, roi in fields.items():
        print(f"  {field_name}: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

        # Save ROI image
        if roi.image is not None:
            roi_path = Path(f"data/processed/roi_{field_name}.jpg")
            processor.save_image(roi.image, roi_path)
            print(f"    Saved: {roi_path}")

    print("\n" + "=" * 60)
    print("Step 4: Running OCR on extracted ROIs...")
    print("=" * 60)

    # Try different PSM modes for better results
    psm_modes = [7, 8, 6, 13]  # 7=single line, 8=single word, 6=uniform block, 13=raw line

    for field_name, roi in fields.items():
        if roi.image is None:
            continue

        print(f"\n{field_name}:")

        # Save preprocessed version for debugging
        preprocessed = processor.preprocess_pipeline(
            roi.image,
            denoise=True,
            threshold=True,
            threshold_method="adaptive"
        )
        debug_path = Path(f"data/processed/roi_{field_name}_preprocessed.jpg")
        processor.save_image(preprocessed, debug_path)
        print(f"  Preprocessed saved: {debug_path}")

        # Try different PSM modes
        best_text = ""
        best_conf = 0

        for psm in psm_modes:
            ocr = TesseractOCR(lang="jpn+eng", psm=psm)

            if field_name in ['player_number', 'skill_level', 'total_score']:
                # Try with digit-only configuration
                ocr_digits = TesseractOCR(lang="eng", psm=psm)
                text = ocr_digits.extract_numbers(preprocessed)
            else:
                text = ocr.extract_text(preprocessed)

            result = ocr.extract_with_confidence(preprocessed)

            if result.confidence > best_conf:
                best_conf = result.confidence
                best_text = text if text else result.text

        print(f"  Best text: '{best_text}' (confidence: {best_conf:.1f}%)")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nCheck data/processed/ for extracted ROIs and visualizations")


if __name__ == "__main__":
    test_roi_extraction()
