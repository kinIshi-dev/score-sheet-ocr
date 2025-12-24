#!/usr/bin/env python3
"""
縦罫線を検出してフィールド境界を特定するスクリプト
line_based_match画像から選手番号、名前、スキルレベルなどのROI座標を抽出
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def detect_vertical_lines(image: np.ndarray, threshold_factor: float = 0.8) -> List[int]:
    """
    縦罫線を検出

    Args:
        image: グレースケール画像
        threshold_factor: 罫線判定の閾値係数

    Returns:
        縦罫線のx座標リスト
    """
    h, w = image.shape

    # 二値化
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 垂直投影（各列の黒ピクセル数）
    v_projection = np.sum(binary == 0, axis=0)

    # 統計
    mean_val = np.mean(v_projection)
    std_val = np.std(v_projection)
    threshold = mean_val + threshold_factor * std_val

    # 罫線候補
    line_candidates = np.where(v_projection > threshold)[0]

    if len(line_candidates) == 0:
        return []

    # 連続する列をグループ化
    groups = []
    current_group = [line_candidates[0]]

    for i in range(1, len(line_candidates)):
        if line_candidates[i] - line_candidates[i-1] <= 3:  # 3px以内なら同じ罫線
            current_group.append(line_candidates[i])
        else:
            groups.append(current_group)
            current_group = [line_candidates[i]]
    groups.append(current_group)

    # 各罫線の位置（中央値）
    line_positions = [int(np.median(group)) for group in groups]

    return line_positions

def identify_field_regions(v_lines: List[int], image_width: int) -> Dict[str, Tuple[int, int]]:
    """
    縦罫線から各フィールドの領域を特定

    スコアシート構造（左から右へ）:
    1. 左端（0-v_lines[0]）: JPAロゴなど
    2. プレイヤー情報エリア（v_lines[0]-v_lines[1]）
       - この中にさらに縦罫線があるはず（番号|名前|スキル の区切り）
    3. スコア記録エリア（v_lines[1]-右端の前）
    4. 右端の統計エリア

    Args:
        v_lines: 縦罫線のx座標リスト
        image_width: 画像の幅

    Returns:
        各フィールドのx座標範囲 {フィールド名: (start_x, end_x)}
    """
    regions = {}

    if len(v_lines) < 2:
        print("Warning: Not enough vertical lines detected")
        return regions

    print(f"\nDetected {len(v_lines)} vertical lines:")
    for i, x in enumerate(v_lines):
        print(f"  Line {i}: x={x}")

    # 主要な区切りを特定
    # 最初の大きなギャップを探す（プレイヤー情報エリアの終わり）
    gaps = []
    for i in range(len(v_lines) - 1):
        gap = v_lines[i+1] - v_lines[i]
        gaps.append((i, gap))

    # ギャップの大きい順にソート
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 gaps:")
    for i, (idx, gap) in enumerate(gaps_sorted[:5]):
        print(f"  {i+1}. Between line {idx} and {idx+1}: {gap}px")

    # 仮定: 最初の2-4本の罫線がプレイヤー情報エリア内
    # その後の大きなギャップがスコアエリアとの境界

    if len(v_lines) >= 4:
        # プレイヤー情報エリアを細分化
        player_info_start = v_lines[0]
        player_info_end = v_lines[3] if len(v_lines) > 3 else v_lines[-1]

        # プレイヤー情報エリア内の罫線
        player_lines = [x for x in v_lines if player_info_start <= x <= player_info_end]

        if len(player_lines) >= 3:
            # 典型的には: [左端, 番号と名前の境界, 名前とスキルの境界, 右端]
            regions['player_number'] = (player_lines[0], player_lines[1])
            regions['player_name'] = (player_lines[1], player_lines[2])
            if len(player_lines) >= 4:
                regions['skill_level'] = (player_lines[2], player_lines[3])
            else:
                regions['skill_level'] = (player_lines[2], player_info_end)

        # スコアエリアとその先
        if len(v_lines) > 4:
            regions['score_area'] = (player_info_end, v_lines[-3] if len(v_lines) > 5 else image_width)

            # 右端の統計エリア
            if len(v_lines) >= 6:
                regions['safety_count'] = (v_lines[-3], v_lines[-2])
                regions['total_score'] = (v_lines[-2], v_lines[-1])
                regions['match_score'] = (v_lines[-1], image_width)

    return regions

def visualize_field_boundaries(image_path: str, match_idx: int):
    """フィールド境界を可視化"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print(f"\n{'='*70}")
    print(f"Match {match_idx} - Size: {w}x{h}px")
    print(f"{'='*70}")

    # 縦罫線検出
    v_lines = detect_vertical_lines(gray)

    # フィールド領域特定
    regions = identify_field_regions(v_lines, w)

    # 可視化用の画像作成
    vis_image = image.copy()

    # すべての縦罫線を薄い赤で描画
    for x in v_lines:
        cv2.line(vis_image, (x, 0), (x, h), (0, 0, 200), 1)

    # フィールド領域を色分けして描画
    colors = {
        'player_number': (255, 0, 0),    # 青
        'player_name': (0, 255, 0),       # 緑
        'skill_level': (0, 255, 255),     # 黄
        'safety_count': (255, 0, 255),    # マゼンタ
        'total_score': (255, 128, 0),     # オレンジ
        'match_score': (128, 0, 255),     # 紫
    }

    overlay = vis_image.copy()

    print(f"\nIdentified field regions:")
    for field_name, (x_start, x_end) in regions.items():
        print(f"  {field_name:15s}: x={x_start:4d}-{x_end:4d} (width: {x_end-x_start:3d}px)")

        if field_name in colors:
            color = colors[field_name]
            cv2.rectangle(overlay, (x_start, 0), (x_end, h), color, -1)

            # ラベル
            cv2.putText(vis_image, field_name, (x_start + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis_image, field_name, (x_start + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 半透明オーバーレイ
    cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)

    # 保存
    output_path = Path('data/processed') / f'field_boundaries_match{match_idx}.jpg'
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nVisualization saved: {output_path}")

    return regions

def analyze_all_matches():
    """すべての試合画像を分析"""
    all_regions = {}

    for match_idx in range(5):
        image_path = f"data/processed/line_based_match{match_idx}.jpg"

        if not Path(image_path).exists():
            print(f"Warning: {image_path} not found")
            continue

        regions = visualize_field_boundaries(image_path, match_idx)
        if regions:
            all_regions[match_idx] = regions

    # サマリー - すべての試合で共通する平均座標を計算
    print(f"\n{'='*70}")
    print("SUMMARY - Average field positions across all matches:")
    print(f"{'='*70}")

    if all_regions:
        # 各フィールドの平均位置を計算
        field_names = set()
        for regions in all_regions.values():
            field_names.update(regions.keys())

        avg_regions = {}
        for field_name in field_names:
            positions = [regions[field_name] for regions in all_regions.values()
                        if field_name in regions]

            if positions:
                avg_start = int(np.mean([pos[0] for pos in positions]))
                avg_end = int(np.mean([pos[1] for pos in positions]))
                std_start = int(np.std([pos[0] for pos in positions]))
                std_end = int(np.std([pos[1] for pos in positions]))

                avg_regions[field_name] = (avg_start, avg_end)

                print(f"{field_name:15s}: x={avg_start:4d}-{avg_end:4d} "
                      f"(±{std_start}/{std_end}px)")

        return avg_regions

    return None

if __name__ == "__main__":
    avg_regions = analyze_all_matches()

    if avg_regions:
        print(f"\n{'='*70}")
        print("Recommended ROI coordinates for implementation:")
        print(f"{'='*70}")
        print("\nPython dict format:")
        print("ROI_COORDS = {")
        for field_name, (x_start, x_end) in avg_regions.items():
            print(f"    '{field_name}': ({x_start}, {x_end}),")
        print("}")
