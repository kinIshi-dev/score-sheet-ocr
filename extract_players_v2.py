#!/usr/bin/env python3
"""
改良版プレイヤー行抽出モジュール
analyze_match_structure.pyのアルゴリズムを使用
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

def split_by_thick_lines(sheet: np.ndarray, threshold: int = 600) -> List[np.ndarray]:
    """
    太い罫線でシートを試合エリアに分割

    Args:
        sheet: スコアシートの画像（グレースケール）
        threshold: 太い罫線判定の閾値（黒ピクセル数）

    Returns:
        試合エリアのリスト
    """
    h, w = sheet.shape

    # 二値化
    _, binary = cv2.threshold(sheet, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 水平投影
    h_projection = np.sum(binary == 0, axis=1)

    # 太い罫線を検出
    thick_lines = np.where(h_projection > threshold)[0]

    if len(thick_lines) == 0:
        return [sheet]

    # 連続する行をグループ化
    groups = []
    current_group = [thick_lines[0]]

    for i in range(1, len(thick_lines)):
        if thick_lines[i] - thick_lines[i-1] <= 5:
            current_group.append(thick_lines[i])
        else:
            groups.append(current_group)
            current_group = [thick_lines[i]]
    groups.append(current_group)

    # 各罫線の位置（中央値）
    line_positions = [int(np.median(group)) for group in groups]

    # 罫線間を抽出
    matches = []
    for i in range(len(line_positions) - 1):
        start = line_positions[i]
        end = line_positions[i + 1]
        match_area = sheet[start:end, :]
        matches.append(match_area)

    return matches

def detect_lines_in_match(match_area: np.ndarray, threshold_factor: float = 2.0) -> List[Dict]:
    """試合エリア内の罫線を検出"""
    h, w = match_area.shape

    _, binary = cv2.threshold(match_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h_projection = np.sum(binary == 0, axis=1)

    mean_val = np.mean(h_projection)
    std_val = np.std(h_projection)
    threshold = mean_val + threshold_factor * std_val

    line_candidates = np.where(h_projection > threshold)[0]

    if len(line_candidates) == 0:
        return []

    # グループ化
    groups = []
    current_group = [line_candidates[0]]

    for i in range(1, len(line_candidates)):
        if line_candidates[i] - line_candidates[i-1] <= 3:
            current_group.append(line_candidates[i])
        else:
            groups.append(current_group)
            current_group = [line_candidates[i]]
    groups.append(current_group)

    # 罫線情報
    lines = []
    for group in groups:
        pos = int(np.median(group))
        thickness = len(group)
        strength = int(np.mean([h_projection[i] for i in group]))
        lines.append({
            'position': pos,
            'thickness': thickness,
            'strength': strength
        })

    return lines

def analyze_regions(match_area: np.ndarray, lines: List[Dict]) -> List[Dict]:
    """罫線間の領域を分析"""
    h, w = match_area.shape
    regions = []

    for i in range(len(lines) - 1):
        start = lines[i]['position'] + lines[i]['thickness']
        end = lines[i+1]['position']

        if end - start < 5:
            continue

        region_height = end - start
        region = match_area[start:end, :]

        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_density = np.sum(binary == 0) / (region_height * w)

        h_proj = np.sum(binary == 0, axis=1)
        h_variance = np.var(h_proj)

        regions.append({
            'start': start,
            'end': end,
            'height': region_height,
            'text_density': text_density,
            'h_variance': h_variance
        })

    return regions

def extract_player_rows_from_match(match_area: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    試合エリアから2つのプレイヤー行を抽出

    Returns:
        (player0_row, player1_row) または (None, None)
    """
    h, w = match_area.shape

    # 罫線検出
    lines = detect_lines_in_match(match_area)
    regions = analyze_regions(match_area, lines)

    # 大きな領域を細分化
    refined_regions = []
    for region in regions:
        if region['height'] > 80:
            sub_h = region['height'] // 3
            p0_start = region['start'] + int(sub_h * 0.5)
            p0_end = region['start'] + int(sub_h * 1.8)
            p1_start = region['start'] + int(sub_h * 1.2)
            p1_end = region['start'] + int(sub_h * 2.5)

            refined_regions.append({
                'start': p0_start,
                'end': p0_end,
                'height': p0_end - p0_start,
                'text_density': region['text_density'],
                'h_variance': region['h_variance']
            })
            refined_regions.append({
                'start': p1_start,
                'end': p1_end,
                'height': p1_end - p1_start,
                'text_density': region['text_density'],
                'h_variance': region['h_variance']
            })
        else:
            refined_regions.append(region)

    # フィルタリングとスコアリング
    candidates = []
    for region in refined_regions:
        if 15 <= region['height'] <= 80:
            if 0.03 <= region['text_density'] <= 0.35:
                if region['h_variance'] > 50:
                    score = 0

                    # 高さスコア
                    if 35 <= region['height'] <= 50:
                        score += 3
                    elif 25 <= region['height'] <= 60:
                        score += 2
                    elif 20 <= region['height'] <= 70:
                        score += 1

                    # 密度スコア
                    if 0.10 <= region['text_density'] <= 0.18:
                        score += 3
                    elif 0.07 <= region['text_density'] <= 0.22:
                        score += 2
                    elif 0.05 <= region['text_density'] <= 0.28:
                        score += 1

                    # 変動スコア
                    if region['h_variance'] > 1000:
                        score += 3
                    elif region['h_variance'] > 500:
                        score += 2
                    elif region['h_variance'] > 200:
                        score += 1

                    region['score'] = score
                    candidates.append(region)

    if len(candidates) < 2:
        return None, None

    candidates.sort(key=lambda x: x['score'], reverse=True)

    # 最高スコアを選択
    player0_region = candidates[0]

    # 2つ目を選択（重複しないもの）
    player1_region = None
    for candidate in candidates[1:]:
        if abs(candidate['start'] - player0_region['end']) > 10 or \
           abs(player0_region['start'] - candidate['end']) > 10:
            player1_region = candidate
            break

    if player1_region is None:
        return None, None

    # 位置順にソート
    if player0_region['start'] > player1_region['start']:
        player0_region, player1_region = player1_region, player0_region

    # 画像を抽出
    player0_row = match_area[player0_region['start']:player0_region['end'], :]
    player1_row = match_area[player1_region['start']:player1_region['end'], :]

    return player0_row, player1_row

def extract_all_player_rows(image_path: str, save_dir: Optional[str] = None) -> Dict[int, Dict[int, np.ndarray]]:
    """
    スコアシート全体からすべてのプレイヤー行を抽出

    Returns:
        {match_idx: {0: player0_row, 1: player1_row}}
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 上半分（Sheet 0）
    top_sheet = gray[0:h//2, :]

    # 試合に分割
    matches = split_by_thick_lines(top_sheet)

    # プレイヤー行を抽出
    results = {}
    for match_idx, match_area in enumerate(matches[:5]):  # 最初の5試合
        player0, player1 = extract_player_rows_from_match(match_area)

        if player0 is not None and player1 is not None:
            results[match_idx] = {0: player0, 1: player1}

            # 保存
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(save_path / f'match{match_idx}_player0.jpg'), player0)
                cv2.imwrite(str(save_path / f'match{match_idx}_player1.jpg'), player1)

    return results

if __name__ == "__main__":
    results = extract_all_player_rows(
        "data/raw/LINE_ALBUM_2025秋_251221_1.jpg",
        save_dir="data/processed/player_rows_v2"
    )

    print(f"Extracted player rows from {len(results)} matches:")
    for match_idx, players in results.items():
        print(f"  Match {match_idx}: Player 0 ({players[0].shape[0]}px), Player 1 ({players[1].shape[0]}px)")
