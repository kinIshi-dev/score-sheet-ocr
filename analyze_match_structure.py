#!/usr/bin/env python3
"""
試合構造をより詳細に分析し、プレイヤーデータ行を特定する

戦略：
1. すべての太い罫線を検出
2. 罫線間の領域を分析（テキスト密度、高さなど）
3. データ行っぽい領域を特定
"""

import cv2
import numpy as np
from pathlib import Path

def detect_all_lines(match_area: np.ndarray, threshold_factor=2.0):
    """すべての水平罫線を検出"""
    h, w = match_area.shape

    # 二値化
    _, binary = cv2.threshold(match_area, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 水平投影
    h_projection = np.sum(binary == 0, axis=1)

    # 統計
    mean_val = np.mean(h_projection)
    std_val = np.std(h_projection)

    # 罫線候補（平均 + threshold_factor * σ 以上）
    threshold = mean_val + threshold_factor * std_val
    line_candidates = np.where(h_projection > threshold)[0]

    # 連続する行をグループ化
    if len(line_candidates) == 0:
        return []

    groups = []
    current_group = [line_candidates[0]]

    for i in range(1, len(line_candidates)):
        if line_candidates[i] - line_candidates[i-1] <= 3:  # 3px以内なら同じ罫線
            current_group.append(line_candidates[i])
        else:
            groups.append(current_group)
            current_group = [line_candidates[i]]
    groups.append(current_group)

    # 各罫線の位置（中央値）と太さ
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

def analyze_regions_between_lines(match_area: np.ndarray, lines):
    """罫線間の領域を分析"""
    h, w = match_area.shape

    regions = []
    for i in range(len(lines) - 1):
        start = lines[i]['position'] + lines[i]['thickness']
        end = lines[i+1]['position']

        if end - start < 5:  # 5px未満の領域はスキップ
            continue

        region_height = end - start
        region = match_area[start:end, :]

        # 二値化
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # テキスト密度（黒ピクセルの割合）
        text_density = np.sum(binary == 0) / (region_height * w)

        # 水平投影の変動（テキストがあると変動が大きい）
        h_proj = np.sum(binary == 0, axis=1)
        h_variance = np.var(h_proj)

        regions.append({
            'start': start,
            'end': end,
            'height': region_height,
            'text_density': text_density,
            'h_variance': h_variance,
            'line_before': i,
            'line_after': i + 1
        })

    return regions

def identify_player_rows(lines, regions, match_area):
    """プレイヤーデータ行を特定

    戦略：
    1. 大きな領域（>80px）がある場合、その中を細分化する
    2. 通常サイズの領域から2つのプレイヤー行を選ぶ
    """
    h, w = match_area.shape

    # 大きな領域（>80px）を細分化
    refined_regions = []
    for region in regions:
        if region['height'] > 80:
            # この領域を3等分して中央2つを使う（上下はヘッダー）
            sub_h = region['height'] // 3

            # Player 0候補（上1/3から中1/3）
            p0_start = region['start'] + int(sub_h * 0.5)
            p0_end = region['start'] + int(sub_h * 1.8)

            # Player 1候補（中1/3から下1/3）
            p1_start = region['start'] + int(sub_h * 1.2)
            p1_end = region['start'] + int(sub_h * 2.5)

            refined_regions.append({
                'start': p0_start,
                'end': p0_end,
                'height': p0_end - p0_start,
                'text_density': region['text_density'],
                'h_variance': region['h_variance'],
                'from_subdivision': True
            })

            refined_regions.append({
                'start': p1_start,
                'end': p1_end,
                'height': p1_end - p1_start,
                'text_density': region['text_density'],
                'h_variance': region['h_variance'],
                'from_subdivision': True
            })
        else:
            refined_regions.append(region)

    # データ行の特徴でフィルタリング
    candidate_regions = []
    for region in refined_regions:
        # 高さフィルター：15-80px
        if 15 <= region['height'] <= 80:
            # 密度フィルター：緩めに
            if 0.03 <= region['text_density'] <= 0.35:
                # 変動フィルター：緩めに
                if region['h_variance'] > 50:
                    candidate_regions.append(region)

    # スコア付け
    for region in candidate_regions:
        score = 0

        # 高さスコア（30-50pxが理想）
        if 35 <= region['height'] <= 50:
            score += 3
        elif 25 <= region['height'] <= 60:
            score += 2
        elif 20 <= region['height'] <= 70:
            score += 1

        # 密度スコア（0.10-0.18が理想）
        if 0.10 <= region['text_density'] <= 0.18:
            score += 3
        elif 0.07 <= region['text_density'] <= 0.22:
            score += 2
        elif 0.05 <= region['text_density'] <= 0.28:
            score += 1

        # 変動スコア（高いほど良い）
        if region['h_variance'] > 1000:
            score += 3
        elif region['h_variance'] > 500:
            score += 2
        elif region['h_variance'] > 200:
            score += 1

        region['score'] = score

    # スコアでソート
    candidate_regions.sort(key=lambda x: x['score'], reverse=True)

    # 上位候補から2つ選ぶ（なるべく離れている2つ）
    if len(candidate_regions) >= 2:
        # 最高スコアを1つ目として選択
        player0 = candidate_regions[0]

        # 2つ目は、1つ目と十分離れているもの
        player1 = None
        for candidate in candidate_regions[1:]:
            # 重複チェック：少なくとも10px離れている
            if abs(candidate['start'] - player0['end']) > 10 or \
               abs(player0['start'] - candidate['end']) > 10:
                player1 = candidate
                break

        if player1:
            # 位置順にソート
            players = sorted([player0, player1], key=lambda x: x['start'])
            return players

    return []

def analyze_match(match_area: np.ndarray, match_idx: int):
    """試合エリアを総合分析"""
    h, w = match_area.shape

    print(f"\n{'='*60}")
    print(f"Match {match_idx} - Height: {h}px")
    print(f"{'='*60}")

    # 罫線検出
    lines = detect_all_lines(match_area, threshold_factor=2.0)
    print(f"\nDetected {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  Line {i}: y={line['position']:3d}, thickness={line['thickness']:2d}px, strength={line['strength']:4d}")

    # 領域分析
    regions = analyze_regions_between_lines(match_area, lines)
    print(f"\nRegions between lines ({len(regions)}):")
    for i, region in enumerate(regions):
        print(f"  Region {i}: y={region['start']:3d}-{region['end']:3d} "
              f"(h={region['height']:2d}px, density={region['text_density']:.3f}, "
              f"variance={region['h_variance']:.0f})")

    # プレイヤー行特定
    player_rows = identify_player_rows(lines, regions, match_area)

    if len(player_rows) >= 2:
        print(f"\n✓ Identified player rows:")
        print(f"  Player 0: y={player_rows[0]['start']}-{player_rows[0]['end']} "
              f"(score: {player_rows[0]['score']})")
        print(f"  Player 1: y={player_rows[1]['start']}-{player_rows[1]['end']} "
              f"(score: {player_rows[1]['score']})")

        # 可視化
        visualize_result(match_area, match_idx, lines, player_rows)

        return {
            'player0': (player_rows[0]['start'], player_rows[0]['end']),
            'player1': (player_rows[1]['start'], player_rows[1]['end'])
        }
    else:
        print(f"\n✗ Could not identify 2 player rows (found {len(player_rows)})")
        return None

def visualize_result(match_area: np.ndarray, match_idx: int, lines, player_rows):
    """結果を可視化"""
    h, w = match_area.shape
    vis_image = cv2.cvtColor(match_area, cv2.COLOR_GRAY2BGR)

    # すべての罫線を薄い赤で描画
    for line in lines:
        cv2.line(vis_image, (0, line['position']), (w, line['position']),
                (0, 0, 200), 1)

    # プレイヤー領域を強調
    overlay = vis_image.copy()

    cv2.rectangle(overlay, (0, player_rows[0]['start']), (w, player_rows[0]['end']),
                 (255, 0, 0), -1)  # 青
    cv2.rectangle(overlay, (0, player_rows[1]['start']), (w, player_rows[1]['end']),
                 (0, 255, 0), -1)  # 緑

    cv2.addWeighted(overlay, 0.25, vis_image, 0.75, 0, vis_image)

    # ラベル
    cv2.putText(vis_image, f'P0: y={player_rows[0]["start"]}-{player_rows[0]["end"]}',
                (10, player_rows[0]['start'] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis_image, f'P0: y={player_rows[0]["start"]}-{player_rows[0]["end"]}',
                (10, player_rows[0]['start'] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(vis_image, f'P1: y={player_rows[1]["start"]}-{player_rows[1]["end"]}',
                (10, player_rows[1]['start'] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis_image, f'P1: y={player_rows[1]["start"]}-{player_rows[1]["end"]}',
                (10, player_rows[1]['start'] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_path = Path('data/processed') / f'structure_analysis_match{match_idx}.jpg'
    cv2.imwrite(str(output_path), vis_image)
    print(f"  → Visualization: {output_path}")

def main():
    from extract_players_final import split_by_thick_lines

    image_path = "data/raw/LINE_ALBUM_2025秋_251221_1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top_sheet = gray[0:h//2, :]

    matches = split_by_thick_lines(top_sheet)
    print(f"Processing {len(matches)} matches...\n")

    results = {}
    for i, match_area in enumerate(matches[:5]):  # 最初の5試合のみ
        result = analyze_match(match_area, i)
        if result:
            results[i] = result

    # サマリー
    print(f"\n{'='*60}")
    print("SUMMARY - Player extraction regions:")
    print(f"{'='*60}")
    for match_idx, result in results.items():
        p0_start, p0_end = result['player0']
        p1_start, p1_end = result['player1']
        print(f"Match {match_idx}:")
        print(f"  Player 0: y={p0_start}:{p0_end} (height: {p0_end-p0_start}px)")
        print(f"  Player 1: y={p1_start}:{p1_end} (height: {p1_end-p1_start}px)")

if __name__ == "__main__":
    main()
