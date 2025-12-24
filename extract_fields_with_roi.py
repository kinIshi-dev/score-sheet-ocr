#!/usr/bin/env python3
"""
手動特定したROI座標を使用してフィールドを抽出

座標はdata/ground_truth/zahyou.mdから取得
line_based_match4.jpgをベースに測定された座標
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from ocr.tesseract_ocr import TesseractOCR

# 手動測定したROI座標（data/ground_truth/zahyou.mdより）
# Player 1の座標
PLAYER1_ROI = {
    'player_number': {'x': 139, 'y': 21, 'w': 164, 'h': 24},
    'player_name': {'x': 139, 'y': 40, 'w': 164, 'h': 25},
    'skill_level': {'x': 303, 'y': 25, 'w': 33, 'h': 39},
    'safety_count': {'x': 851, 'y': 21, 'w': 96, 'h': 29},
    'total_score': {'x': 946, 'y': 15, 'w': 35, 'h': 31},
    'match_score': {'x': 980, 'y': 0, 'w': 37, 'h': 51},
}

# Player 2の座標
PLAYER2_ROI = {
    'player_number': {'x': 138, 'y': 61, 'w': 165, 'h': 23},
    'player_name': {'x': 137, 'y': 80, 'w': 166, 'h': 25},
    'skill_level': {'x': 302, 'y': 64, 'w': 34, 'h': 41},
    'safety_count': {'x': 852, 'y': 70, 'w': 95, 'h': 30},
    'total_score': {'x': 947, 'y': 69, 'w': 35, 'h': 31},
    'match_score': {'x': 981, 'y': 69, 'w': 38, 'h': 52},
}

def extract_roi(image: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    """
    ROI座標を使って画像から領域を切り出す

    Args:
        image: 入力画像
        roi: ROI座標 {'x', 'y', 'w', 'h'}

    Returns:
        切り出された領域
    """
    x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']

    # 画像境界チェック
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return image[y:y+h, x:x+w]

def extract_all_fields(player_row: np.ndarray, player_idx: int) -> Dict[str, np.ndarray]:
    """
    プレイヤー行から全フィールドを抽出

    Args:
        player_row: プレイヤー行の画像
        player_idx: プレイヤー番号（0または1）

    Returns:
        各フィールドの画像 {field_name: roi_image}
    """
    roi_coords = PLAYER1_ROI if player_idx == 0 else PLAYER2_ROI

    fields = {}
    for field_name, roi in roi_coords.items():
        field_img = extract_roi(player_row, roi)
        fields[field_name] = field_img

    return fields

def preprocess_for_ocr(field_image: np.ndarray) -> np.ndarray:
    """
    OCR用に画像を前処理（シンプルバージョン）

    Args:
        field_image: 入力画像

    Returns:
        前処理済み画像
    """
    # リサイズ（3倍に拡大）
    h, w = field_image.shape[:2]
    resized = cv2.resize(field_image, (w*3, h*3), interpolation=cv2.INTER_CUBIC)

    # グレースケール確認
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return resized

def ocr_field(field_image: np.ndarray, field_name: str, ocr: TesseractOCR) -> str:
    """
    フィールド画像からOCRでテキストを抽出

    Args:
        field_image: フィールドの画像
        field_name: フィールド名
        ocr: OCRエンジン

    Returns:
        抽出されたテキスト
    """
    # 前処理
    processed = preprocess_for_ocr(field_image)

    # フィールドの種類に応じて最適なOCR設定を使用
    if field_name in ['player_number', 'skill_level', 'safety_count',
                       'total_score', 'match_score']:
        # 数字のみ
        text = ocr.extract_numbers(processed)
    else:
        # 名前（英字+数字）
        text = ocr.extract_text(processed)

    return text.strip()

def visualize_roi_extraction(player_row: np.ndarray, player_idx: int,
                             match_idx: int, output_dir: str = "data/processed/roi_vis"):
    """ROI抽出を可視化"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # カラー画像に変換
    if len(player_row.shape) == 2:
        vis_image = cv2.cvtColor(player_row, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = player_row.copy()

    roi_coords = PLAYER1_ROI if player_idx == 0 else PLAYER2_ROI

    colors = {
        'player_number': (255, 0, 0),    # 青
        'player_name': (0, 255, 0),       # 緑
        'skill_level': (0, 255, 255),     # 黄
        'safety_count': (255, 0, 255),    # マゼンタ
        'total_score': (255, 128, 0),     # オレンジ
        'match_score': (128, 0, 255),     # 紫
    }

    overlay = vis_image.copy()

    for field_name, roi in roi_coords.items():
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']

        # 矩形描画
        color = colors.get(field_name, (200, 200, 200))
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)

        # ラベル
        cv2.putText(vis_image, field_name, (x+2, y+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        cv2.putText(vis_image, field_name, (x+2, y+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 半透明オーバーレイ
    cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

    # 保存
    output_path = Path(output_dir) / f'roi_match{match_idx}_player{player_idx}.jpg'
    cv2.imwrite(str(output_path), vis_image)
    print(f"  Visualization saved: {output_path}")

def test_roi_extraction():
    """ROI抽出のテスト"""
    # line_based_match4.jpgを読み込む
    test_image_path = "data/processed/line_based_match4.jpg"

    if not Path(test_image_path).exists():
        print(f"Error: {test_image_path} not found")
        return

    image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"Testing ROI extraction on: {test_image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}px\n")

    # OCRエンジン初期化（PSM=7: 単一行のテキスト）
    ocr = TesseractOCR(lang='eng', psm=7)  # 英語のみ（数字と英字名前用）

    for player_idx in [0, 1]:
        print(f"{'='*70}")
        print(f"Player {player_idx}")
        print(f"{'='*70}")

        # フィールド抽出
        fields = extract_all_fields(gray, player_idx)

        # 可視化
        visualize_roi_extraction(gray, player_idx, 4)

        # 各フィールドをOCR
        for field_name, field_img in fields.items():
            h, w = field_img.shape
            text = ocr_field(field_img, field_name, ocr)
            print(f"  {field_name:15s} ({w:3d}x{h:2d}px): '{text}'")

            # フィールド画像を保存（デバッグ用）
            field_path = Path('data/processed/roi_fields') / f'match4_p{player_idx}_{field_name}.jpg'
            field_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(field_path), field_img)

            # 前処理後の画像も保存
            processed = preprocess_for_ocr(field_img)
            processed_path = Path('data/processed/roi_fields') / f'match4_p{player_idx}_{field_name}_processed.jpg'
            cv2.imwrite(str(processed_path), processed)

        print()

if __name__ == "__main__":
    test_roi_extraction()
