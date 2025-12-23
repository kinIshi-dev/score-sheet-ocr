#!/usr/bin/env python3
"""Evaluate OCR accuracy against ground truth data."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.image_processor import ImageProcessor
from preprocessing.roi_extractor import ROIExtractor
from ocr.tesseract_ocr import TesseractOCR
from extract_player_rows import extract_player_rows_from_image


def load_ground_truth(json_path: Path) -> Dict:
    """Load ground truth data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predicted: str, actual: str) -> bool:
    """Calculate if prediction matches actual value."""
    if not actual:  # Skip empty ground truth
        return None

    # Normalize strings (remove spaces, convert to lowercase)
    pred_norm = str(predicted).strip().lower()
    actual_norm = str(actual).strip().lower()

    return pred_norm == actual_norm


def evaluate_single_image(
    image_path: Path,
    ground_truth_path: Path
) -> Dict:
    """Evaluate OCR on a single image."""

    # Load ground truth
    gt_data = load_ground_truth(ground_truth_path)

    # Extract player rows using improved method
    player_rows = extract_player_rows_from_image(image_path)

    # Initialize processor and extractor
    processor = ImageProcessor()
    extractor = ROIExtractor()

    # Initialize OCR
    ocr = TesseractOCR(lang="eng", psm=7)

    # Results storage
    results = {
        'total_fields': 0,
        'correct_fields': 0,
        'by_field_type': {
            'player_number': {'total': 0, 'correct': 0},
            'skill_level': {'total': 0, 'correct': 0},
            'total_score': {'total': 0, 'correct': 0}
        },
        'details': []
    }

    # Process each player row
    gt_sheet = gt_data['sheets'][0]  # Only process first sheet for now

    for row_idx, player_row in enumerate(player_rows):
        match_idx = row_idx // 2
        player_idx = row_idx % 2

        if match_idx >= len(gt_sheet['matches']):
            break

        gt_match = gt_sheet['matches'][match_idx]
        gt_player = gt_match['players'][player_idx]

        # Extract field ROIs from this player row
        fields = extractor.extract_all_fields(player_row)

        # Test each field
        for field_name in ['player_number', 'skill_level', 'total_score']:
            if field_name not in fields:
                continue

            roi = fields[field_name]
            if roi.image is None:
                continue

            # Preprocess
            preprocessed = processor.preprocess_pipeline(
                roi.image,
                denoise=True,
                threshold=True,
                threshold_method="adaptive"
            )

            # OCR
            ocr_text = ocr.extract_numbers(preprocessed)
            gt_value = gt_player.get(field_name, '')

            # Calculate accuracy
            is_correct = calculate_accuracy(ocr_text, gt_value)

            if is_correct is not None:
                results['total_fields'] += 1
                results['by_field_type'][field_name]['total'] += 1

                if is_correct:
                    results['correct_fields'] += 1
                    results['by_field_type'][field_name]['correct'] += 1

                results['details'].append({
                    'sheet': 0,
                    'match': match_idx,
                    'player': player_idx,
                    'field': field_name,
                    'predicted': ocr_text,
                    'actual': gt_value,
                    'correct': is_correct
                })

    # Calculate overall accuracy
    if results['total_fields'] > 0:
        results['accuracy'] = results['correct_fields'] / results['total_fields']
    else:
        results['accuracy'] = 0.0

    # Calculate per-field accuracy
    for field_type in results['by_field_type']:
        field_stats = results['by_field_type'][field_type]
        if field_stats['total'] > 0:
            field_stats['accuracy'] = field_stats['correct'] / field_stats['total']
        else:
            field_stats['accuracy'] = 0.0

    return results


def print_results(results: Dict):
    """Print evaluation results."""

    print("\n" + "=" * 70)
    print("OCR Evaluation Results")
    print("=" * 70)

    print(f"\nOverall Accuracy: {results['accuracy']:.1%}")
    print(f"Correct: {results['correct_fields']} / {results['total_fields']}")

    print("\n" + "-" * 70)
    print("Accuracy by Field Type:")
    print("-" * 70)

    for field_type, stats in results['by_field_type'].items():
        if stats['total'] > 0:
            print(f"{field_type:20s}: {stats['accuracy']:6.1%}  ({stats['correct']}/{stats['total']})")
        else:
            print(f"{field_type:20s}: N/A")

    print("\n" + "-" * 70)
    print("Detailed Results:")
    print("-" * 70)

    for detail in results['details']:
        status = "✓" if detail['correct'] else "✗"
        print(f"{status} Sheet{detail['sheet']} Match{detail['match']} P{detail['player']} "
              f"{detail['field']:15s}: '{detail['predicted']:10s}' "
              f"(expected: '{detail['actual']}')")

    print("\n" + "=" * 70)


def main():
    """Main evaluation function."""

    # Paths
    image_path = Path("data/raw/LINE_ALBUM_2025秋_251221_1.jpg")
    gt_path = Path("data/ground_truth/LINE_ALBUM_2025秋_251221_1.json")

    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        return

    print(f"Evaluating: {image_path.name}")
    print(f"Ground truth: {gt_path.name}")

    # Run evaluation
    results = evaluate_single_image(image_path, gt_path)

    # Print results
    print_results(results)

    # Save results
    output_path = Path("data/processed/evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
