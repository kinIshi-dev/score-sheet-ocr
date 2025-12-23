#!/usr/bin/env python3
"""
改良版プレイヤー行抽出を使ったOCRテスト
"""

import cv2
import sys
import json
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ocr.tesseract_ocr import TesseractOCR
from extract_players_v2 import extract_all_player_rows

def load_ground_truth(json_path: str):
    """正解データを読み込む（フラットなプレイヤーリストに変換）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sheet 0のすべてのプレイヤーをフラットなリストに
    players = []
    for sheet in data['sheets']:
        if sheet['sheet_index'] == 0:
            for match in sheet['matches']:
                for player in match['players']:
                    players.append(player)
    return players

def simple_ocr_test(player_row: cv2.Mat, ocr: TesseractOCR):
    """プレイヤー行全体をOCRして結果を返す"""
    # 日本語+英語でOCR（単一行モード）
    return ocr.extract_text(player_row)

def calculate_accuracy(predicted: str, ground_truth_player: dict) -> float:
    """
    簡易的な精度計算
    正解データの各フィールドが予測テキストに含まれているかチェック
    """
    matches = 0
    total = 0

    fields_to_check = ['player_number', 'player_name', 'skill_level',
                      'safety_count', 'total_score', 'match_score']

    for field in fields_to_check:
        value = ground_truth_player.get(field, '')
        if value and value != '':
            total += 1
            # 値が予測テキストに含まれているかチェック
            if str(value) in predicted:
                matches += 1

    return (matches / total * 100) if total > 0 else 0

def main():
    # プレイヤー行を抽出
    print("Extracting player rows...")
    player_rows = extract_all_player_rows(
        "data/raw/LINE_ALBUM_2025秋_251221_1.jpg"
    )

    # 正解データを読み込む
    ground_truth = load_ground_truth("data/ground_truth/LINE_ALBUM_2025秋_251221_1.json")

    # OCRエンジン初期化
    ocr = TesseractOCR()

    print("\n" + "="*80)
    print("OCR Test Results with Improved Player Row Extraction")
    print("="*80)

    total_accuracy = 0
    test_count = 0

    for match_idx in sorted(player_rows.keys()):
        print(f"\nMatch {match_idx}:")
        print("-" * 80)

        for player_idx in [0, 1]:
            # プレイヤー行を取得
            player_row = player_rows[match_idx][player_idx]

            # 対応する正解データを取得
            global_player_idx = match_idx * 2 + player_idx
            if global_player_idx >= len(ground_truth):
                continue

            gt_player = ground_truth[global_player_idx]

            # OCR実行
            ocr_text = simple_ocr_test(player_row, ocr)

            # 精度計算
            accuracy = calculate_accuracy(ocr_text, gt_player)

            # 結果表示
            print(f"  Player {player_idx} (Global: {global_player_idx}):")
            print(f"    Ground Truth:")
            print(f"      Number: {gt_player.get('player_number', 'N/A')}")
            print(f"      Name: {gt_player.get('player_name', 'N/A')}")
            print(f"      Skill: {gt_player.get('skill_level', 'N/A')}")
            print(f"      Safety: {gt_player.get('safety_count', 'N/A')}")
            print(f"      Total: {gt_player.get('total_score', 'N/A')}")
            print(f"      Match Score: {gt_player.get('match_score', 'N/A')}")
            print(f"    OCR Result: {ocr_text}")
            print(f"    Accuracy: {accuracy:.1f}%")

            total_accuracy += accuracy
            test_count += 1

    # 全体の精度
    print("\n" + "="*80)
    print(f"Overall Accuracy: {total_accuracy / test_count:.1f}%")
    print(f"Tested {test_count} player rows")
    print("="*80)

if __name__ == "__main__":
    main()
