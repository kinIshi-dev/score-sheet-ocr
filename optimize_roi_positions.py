#!/usr/bin/env python3
"""Optimize ROI positions using grid search."""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.image_processor import ImageProcessor
from preprocessing.roi_extractor import ROI
from ocr.tesseract_ocr import TesseractOCR


def load_ground_truth(json_path: Path) -> Dict:
    """Load ground truth data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predicted: str, actual: str) -> bool:
    """Calculate if prediction matches actual value."""
    if not actual:
        return None
    pred_norm = str(predicted).strip().lower()
    actual_norm = str(actual).strip().lower()
    return pred_norm == actual_norm


def test_roi_position(
    match_area,
    processor,
    ocr,
    x_percent: float,
    y_percent: float,
    w_percent: float,
    h_percent: float,
    field_name: str,
    actual_value: str
) -> Tuple[bool, str]:
    """Test a specific ROI position and return if it's correct."""

    height, width = match_area.shape[:2]

    # Create ROI
    roi = ROI(
        x=int(width * x_percent),
        y=int(height * y_percent),
        width=int(width * w_percent),
        height=int(height * h_percent),
        label=field_name
    )

    # Extract ROI
    roi_image = roi.extract(match_area)

    if roi_image is None or roi_image.size == 0:
        return None, ""

    # Preprocess
    preprocessed = processor.preprocess_pipeline(
        roi_image,
        denoise=True,
        threshold=True,
        threshold_method="adaptive"
    )

    # OCR
    ocr_text = ocr.extract_numbers(preprocessed)

    # Check accuracy
    is_correct = calculate_accuracy(ocr_text, actual_value)

    return is_correct, ocr_text


def optimize_field_position(
    match_areas: List,
    processor: ImageProcessor,
    ocr: TesseractOCR,
    gt_values: List[str],
    field_name: str,
    x_range: List[float],
    y_range: List[float],
    w_range: List[float],
    h_range: List[float]
) -> Dict:
    """Optimize position for a single field type using grid search."""

    best_accuracy = 0.0
    best_params = None
    best_details = None

    total_combinations = len(x_range) * len(y_range) * len(w_range) * len(h_range)
    tested = 0

    print(f"\n  Testing {total_combinations} combinations for {field_name}...")

    # Grid search
    for x, y, w, h in itertools.product(x_range, y_range, w_range, h_range):
        tested += 1

        if tested % 100 == 0:
            print(f"    Progress: {tested}/{total_combinations} ({tested/total_combinations*100:.1f}%)")

        correct = 0
        total = 0
        predictions = []

        # Test on all match areas
        for match_area, gt_value in zip(match_areas, gt_values):
            if not gt_value:
                continue

            is_correct, ocr_text = test_roi_position(
                match_area, processor, ocr,
                x, y, w, h,
                field_name, gt_value
            )

            if is_correct is not None:
                total += 1
                if is_correct:
                    correct += 1
                predictions.append({
                    'predicted': ocr_text,
                    'actual': gt_value,
                    'correct': is_correct
                })

        # Calculate accuracy
        if total > 0:
            accuracy = correct / total

            # Update best if improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'x': x, 'y': y, 'w': w, 'h': h}
                best_details = {
                    'correct': correct,
                    'total': total,
                    'predictions': predictions
                }

    print(f"    Complete! Best accuracy: {best_accuracy:.1%}")

    return {
        'field_name': field_name,
        'best_accuracy': best_accuracy,
        'best_params': best_params,
        'details': best_details
    }


def main():
    """Main optimization function."""

    # Paths
    image_path = Path("data/raw/LINE_ALBUM_2025秋_251221_1.jpg")
    gt_path = Path("data/ground_truth/LINE_ALBUM_2025秋_251221_1.json")

    print("=" * 70)
    print("ROI Position Optimization")
    print("=" * 70)

    # Load ground truth
    gt_data = load_ground_truth(gt_path)

    # Load image
    processor = ImageProcessor()
    image = processor.load_image(image_path)

    # For now, manually extract the top sheet
    # TODO: Fix sheet separation to detect 2 sheets
    height, width = image.shape[:2]
    top_sheet = image[0:int(height*0.48), :]

    print(f"\nUsing top sheet: {top_sheet.shape[1]}x{top_sheet.shape[0]}")

    # Manually divide into 5 match areas, then split each into 2 player rows
    match_height = top_sheet.shape[0] // 5
    player_rows = []  # List of individual player row images

    for i in range(5):
        y_start = i * match_height
        match_area = top_sheet[y_start:y_start+match_height, :]

        # Skip header/hatching area (top 25%)
        data_start = int(match_area.shape[0] * 0.25)
        data_area = match_area[data_start:, :]

        # Split data area into 2 player rows (top and bottom)
        row_height = data_area.shape[0] // 2
        player1_row = data_area[0:row_height, :]
        player2_row = data_area[row_height:, :]

        player_rows.append(player1_row)
        player_rows.append(player2_row)

    print(f"Extracted {len(player_rows)} player rows (5 matches × 2 players)")

    # Initialize OCR
    ocr = TesseractOCR(lang="eng", psm=7)

    # Prepare ground truth values for all 10 players (5 matches × 2 players)
    gt_sheet = gt_data['sheets'][0]
    gt_player_numbers = []
    gt_skill_levels = []
    gt_total_scores = []

    for match in gt_sheet['matches']:
        for player in match['players']:
            gt_player_numbers.append(player['player_number'])
            gt_skill_levels.append(player['skill_level'])
            gt_total_scores.append(player['total_score'])

    # Define search ranges (as percentages of match area size)
    # Format: [start, end, step]

    # Narrow search ranges for faster optimization
    search_params = {
        'player_number': {
            'x_range': [0.02, 0.03, 0.04],               # 2%, 3%, 4%
            'y_range': [0.20, 0.30],                     # 20%, 30%
            'w_range': [0.05, 0.06],                     # 5%, 6%
            'h_range': [0.40, 0.50],                     # 40%, 50%
        },
        'skill_level': {
            'x_range': [0.20, 0.22, 0.24],               # 20%, 22%, 24%
            'y_range': [0.20, 0.30],                     # 20%, 30%
            'w_range': [0.04, 0.05, 0.06],               # 4%, 5%, 6%
            'h_range': [0.40, 0.50],                     # 40%, 50%
        },
        'total_score': {
            'x_range': [0.85, 0.87, 0.89],               # 85%, 87%, 89%
            'y_range': [0.20, 0.30],                     # 20%, 30%
            'w_range': [0.06, 0.07, 0.08],               # 6%, 7%, 8%
            'h_range': [0.40, 0.50],                     # 40%, 50%
        }
    }

    # Optimize each field
    results = []

    print("\n" + "=" * 70)
    print("Optimizing Field Positions")
    print("=" * 70)

    # Player Number
    print("\n[1/3] Optimizing player_number...")
    result = optimize_field_position(
        player_rows, processor, ocr,
        gt_player_numbers, 'player_number',
        **search_params['player_number']
    )
    results.append(result)

    # Skill Level
    print("\n[2/3] Optimizing skill_level...")
    result = optimize_field_position(
        player_rows, processor, ocr,
        gt_skill_levels, 'skill_level',
        **search_params['skill_level']
    )
    results.append(result)

    # Total Score
    print("\n[3/3] Optimizing total_score...")
    result = optimize_field_position(
        player_rows, processor, ocr,
        gt_total_scores, 'total_score',
        **search_params['total_score']
    )
    results.append(result)

    # Print results
    print("\n" + "=" * 70)
    print("Optimization Results")
    print("=" * 70)

    for result in results:
        print(f"\n{result['field_name']}:")
        print(f"  Best Accuracy: {result['best_accuracy']:.1%}")
        if result['best_params']:
            params = result['best_params']
            print(f"  Best Position:")
            print(f"    x: {params['x']:.3f} ({params['x']*100:.1f}%)")
            print(f"    y: {params['y']:.3f} ({params['y']*100:.1f}%)")
            print(f"    w: {params['w']:.3f} ({params['w']*100:.1f}%)")
            print(f"    h: {params['h']:.3f} ({params['h']*100:.1f}%)")
            if result['details']:
                print(f"  Results: {result['details']['correct']}/{result['details']['total']} correct")

    # Save results
    output_path = Path("data/processed/optimization_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
