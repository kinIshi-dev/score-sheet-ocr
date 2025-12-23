#!/usr/bin/env python3
"""
最終版: プレイヤー行抽出モジュール

太い罫線検出により試合エリアを分割し、
各試合から2人のプレイヤーデータを抽出します。
"""

import sys
from pathlib import Path
from typing import List
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocessing.image_processor import ImageProcessor


def extract_all_players(image_path: Path) -> List[np.ndarray]:
    """
    画像から全プレイヤー行を抽出

    Args:
        image_path: スコアシート画像のパス

    Returns:
        プレイヤー行画像のリスト（10個: 5試合 × 2プレイヤー）
    """
    processor = ImageProcessor()
    image = processor.load_image(image_path)

    # 上のシートを抽出
    height, width = image.shape[:2]
    top_sheet = image[0:int(height*0.48), :]

    # 太い罫線で試合を分割
    match_areas = split_by_thick_lines(top_sheet)

    # 各試合から2人のプレイヤーを抽出
    player_rows = []
    for match_area in match_areas:
        players = extract_players_from_match(match_area)
        player_rows.extend(players)

    return player_rows


def split_by_thick_lines(sheet: np.ndarray) -> List[np.ndarray]:
    """
    太い罫線を検出して試合エリアに分割

    Args:
        sheet: シート画像

    Returns:
        試合エリアのリスト（5個）
    """
    processor = ImageProcessor()

    # グレースケール + 二値化
    gray = processor.convert_to_grayscale(sheet)
    binary = processor.apply_threshold(gray, method='otsu')

    # 水平投影
    h_projection = np.sum(binary == 0, axis=1)

    # 太い線を検出（黒ピクセルが600以上）
    threshold = 600
    thick_line_candidates = []

    for y in range(len(h_projection)):
        if h_projection[y] > threshold:
            thick_line_candidates.append(y)

    # 連続する行をグループ化
    grouped_lines = []
    if thick_line_candidates:
        current_group = [thick_line_candidates[0]]
        for y in thick_line_candidates[1:]:
            if y - current_group[-1] <= 3:
                current_group.append(y)
            else:
                grouped_lines.append(int(np.median(current_group)))
                current_group = [y]
        grouped_lines.append(int(np.median(current_group)))

    # 線の間を試合エリアとして抽出
    match_areas = []
    for i in range(len(grouped_lines) - 1):
        y_start = grouped_lines[i]
        y_end = grouped_lines[i + 1]
        match_area = sheet[y_start:y_end, :]
        match_areas.append(match_area)

    return match_areas


def extract_players_from_match(match_area: np.ndarray) -> List[np.ndarray]:
    """
    試合エリアから2人のプレイヤー行を抽出

    各試合エリアの構造:
    - 上部 0-40%: ヘッダーと試合情報
    - 中央 40-70%: プレイヤー1のデータ
    - 中央 60-90%: プレイヤー2のデータ（一部重複）
    - 下部 90-100%: 下部ヘッダー

    Args:
        match_area: 試合エリア画像

    Returns:
        [player0_row, player1_row]
    """
    h = match_area.shape[0]

    # プレイヤーデータの位置（比率）
    player0_start = int(h * 0.40)
    player0_end = int(h * 0.70)
    player1_start = int(h * 0.60)
    player1_end = int(h * 0.90)

    player0_row = match_area[player0_start:player0_end, :]
    player1_row = match_area[player1_start:player1_end, :]

    return [player0_row, player1_row]


def main():
    """テスト実行"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_players_final.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"Extracting players from: {image_path.name}")

    player_rows = extract_all_players(image_path)

    print(f"Extracted {len(player_rows)} player rows")

    # 保存
    processor = ImageProcessor()
    output_dir = Path("data/processed/extracted_players")
    output_dir.mkdir(exist_ok=True)

    for i, row in enumerate(player_rows):
        match_idx = i // 2
        player_idx = i % 2
        output_path = output_dir / f"match{match_idx}_player{player_idx}.jpg"
        processor.save_image(row, output_path)
        print(f"  Match {match_idx} Player {player_idx}: {row.shape[1]}x{row.shape[0]} → {output_path.name}")

    print(f"\nAll player rows saved to: {output_dir}")


if __name__ == "__main__":
    main()
