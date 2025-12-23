# ROI位置最適化アプローチ

JPAスコアシートOCRプロジェクトにおけるROI（関心領域）位置最適化の様々なアプローチをまとめた資料。

**最終更新**: 2025-12-23

---

## 目次

1. [現在の状況](#現在の状況)
2. [問題の定義](#問題の定義)
3. [試したアプローチ](#試したアプローチ)
4. [今後のアプローチ候補](#今後のアプローチ候補)
5. [推奨アプローチ](#推奨アプローチ)
6. [実装優先度](#実装優先度)

---

## 現在の状況

### 達成済み
- ✅ ROI抽出モジュール実装（`src/preprocessing/roi_extractor.py`）
- ✅ 正解データ作成（10人分、3フィールド）
- ✅ 評価スクリプト実装（`evaluate_ocr.py`）
- ✅ スコアシート構造の正しい理解
  - 各試合エリアの上25%はヘッダー
  - 残り75%を2分割して2人のプレイヤー行

### 現在の精度
- **player_number**: 0% (5桁の数字、検出困難)
- **skill_level**: 10% (1/10正解)
- **total_score**: 10% (1/10正解)

### ボトルネック
- ROI位置が不正確
- OCR処理が遅い（最適化に時間がかかる）
- 前処理が不十分（罫線、ノイズが残る）

---

## 問題の定義

### 主要課題
各フィールド（player_number, skill_level, total_score）に対して、最適なROI位置（x, y, width, height）をプレイヤー行画像のサイズに対する**パーセンテージ**で特定する。

### 制約
- プレイヤー行画像サイズ: 約1090x54ピクセル
- パラメータ範囲: 0.0 ≤ x,y,w,h ≤ 1.0
- 評価指標: 正解データとの一致率
- 実行時間: 合理的な時間内（数分〜数十分）

---

## 試したアプローチ

### 1. グリッドサーチ（単純）⚠️ 失敗

**実装**: `optimize_roi_positions.py`

**手法**:
- 各パラメータ（x, y, w, h）に対して離散的な値の組み合わせをすべて試す
- 各組み合わせでOCRを実行して精度を測定
- 最も精度が高い位置を採用

**パラメータ例**:
```python
x_range = [0.02, 0.03, 0.04]           # 3通り
y_range = [0.20, 0.30]                 # 2通り
w_range = [0.05, 0.06]                 # 2通り
h_range = [0.40, 0.50]                 # 2通り
# 合計: 3×2×2×2 = 24通り
```

**結果**:
- ✅ 実装完了、実行成功
- ❌ 精度向上せず（0-10%のまま）
- ⚠️ 探索範囲が狭すぎる可能性

**問題点**:
- 探索範囲を広げるとOCR回数が爆発的に増加
- 10人分のデータ × 100通りの位置 = 1000回のOCR → 非常に遅い

---

### 2. 2段階グリッドサーチ ⚠️ タイムアウト

**実装**: `optimize_roi_2stage.py`

**手法**:
1. **Phase 1（粗探索）**: 広い範囲を大きなステップで探索
2. **Phase 2（細探索）**: Phase 1で見つけた良い位置の周辺を細かく探索

**パラメータ例**:
```python
# Phase 1: Coarse
x_range = [0%, 5%, 10%]                # 3通り
y_range = [10%, 20%, 30%, 40%]         # 4通り
w_range = [3%, 6%, 9%]                 # 3通り
h_range = [20%, 30%, 40%, 50%, 60%]    # 5通り
# 合計: 3×4×3×5 = 180通り

# Phase 2: Fine (±2%範囲を0.5%刻み)
# 周辺9×9×9×9 = 約6000通り
```

**結果**:
- ✅ 実装完了
- ❌ タイムアウト（5分以内に終わらない）
- ❌ 動作未確認

**問題点**:
- OCRが遅すぎる（1回あたり数秒）
- Phase 1だけで180×10人 = 1800回のOCR
- Phase 2まで到達できない

---

## 今後のアプローチ候補

### アプローチA: 手動位置特定 ⭐ **最も確実**

**手法**:
1. プレイヤー行画像を目視で確認
2. 画像編集ツールでピクセル座標を測定
3. パーセンテージに変換
4. ハードコードしてテスト
5. 精度を確認

**利点**:
- ✅ 最も速い（数分で完了）
- ✅ 確実に動作する位置が見つかる
- ✅ 他の手法のベースラインになる

**欠点**:
- ❌ 手動作業が必要
- ❌ 他の画像に汎化しない可能性
- ❌ スケーラブルでない

**実装方法**:
```python
# 1. プレイヤー行画像を保存
processor.save_image(player_row, 'debug_player_row.jpg')

# 2. GIMPやPaintなどで座標を測定
# 例: skill_levelが x=240, y=15, w=50, h=30 (画像サイズ1090x54)

# 3. パーセンテージに変換
x_percent = 240 / 1090 = 0.220
y_percent = 15 / 54 = 0.278
w_percent = 50 / 1090 = 0.046
h_percent = 30 / 54 = 0.556

# 4. ハードコードしてテスト
roi = ROI(
    x=int(width * 0.220),
    y=int(height * 0.278),
    width=int(width * 0.046),
    height=int(height * 0.556),
    label="skill_level"
)
```

**推奨度**: ⭐⭐⭐⭐⭐ (5/5)

---

### アプローチB: ランダムサーチ + 早期終了

**手法**:
1. パラメータ空間からランダムに位置をサンプリング
2. 各位置でOCRを実行
3. 一定精度（例: 50%）に達したら終了

**利点**:
- ✅ グリッドサーチより効率的
- ✅ 広い範囲を探索できる
- ✅ 早期終了で時間を節約

**欠点**:
- ❌ 最適解の保証なし
- ❌ 運に左右される
- ❌ それでも遅い可能性

**実装例**:
```python
import random

best_accuracy = 0
best_params = None
max_iterations = 500
target_accuracy = 0.5

for i in range(max_iterations):
    # ランダムにパラメータを生成
    x = random.uniform(0.15, 0.30)
    y = random.uniform(0.10, 0.40)
    w = random.uniform(0.03, 0.10)
    h = random.uniform(0.30, 0.60)

    # テスト
    accuracy = test_position(x, y, w, h)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = (x, y, w, h)

    # 早期終了
    if accuracy >= target_accuracy:
        break
```

**推奨度**: ⭐⭐⭐ (3/5)

---

### アプローチC: ベイズ最適化

**手法**:
1. ガウス過程回帰でパラメータ空間をモデル化
2. 獲得関数（Expected Improvement など）を使って次の探索点を選択
3. 効率的に最適解に収束

**利点**:
- ✅ サンプル効率が非常に高い
- ✅ 理論的に優れた手法
- ✅ グローバル最適解に近い解が得られる

**欠点**:
- ❌ 実装が複雑
- ❌ 新しいライブラリが必要（scikit-optimize, Optuna など）
- ❌ それでも数百回のOCRは必要

**実装例**:
```python
from skopt import gp_minimize
from skopt.space import Real

# パラメータ空間の定義
space = [
    Real(0.15, 0.30, name='x'),
    Real(0.10, 0.40, name='y'),
    Real(0.03, 0.10, name='w'),
    Real(0.30, 0.60, name='h')
]

# 目的関数（最小化するので負の精度を返す）
def objective(params):
    x, y, w, h = params
    accuracy = test_position(x, y, w, h)
    return -accuracy  # 最小化問題に変換

# 最適化実行
result = gp_minimize(
    objective,
    space,
    n_calls=100,      # 最大100回の評価
    random_state=42
)

best_params = result.x
best_accuracy = -result.fun
```

**推奨度**: ⭐⭐⭐⭐ (4/5) - 時間があれば試す価値あり

---

### アプローチD: テンプレートマッチング

**手法**:
1. 各フィールドの「理想的な画像パターン」をテンプレートとして保存
2. OpenCVのテンプレートマッチング（`cv2.matchTemplate`）で位置を検出
3. 検出された位置でOCRを実行

**利点**:
- ✅ 高速（OCR不要で位置検出）
- ✅ ROI位置が動的に変化する場合にも対応
- ✅ OpenCVの標準機能

**欠点**:
- ❌ 手書き文字の多様性に対応困難
- ❌ 良いテンプレートの作成が難しい
- ❌ 罫線・ノイズに影響されやすい

**実装例**:
```python
import cv2

# テンプレート画像（skill_levelの理想的な見た目）
template = cv2.imread('templates/skill_level.jpg', 0)

# プレイヤー行でテンプレートマッチング
result = cv2.matchTemplate(player_row_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 最も一致した位置
x, y = max_loc
w, h = template.shape[::-1]

# この位置でOCR
roi = player_row[y:y+h, x:x+w]
text = ocr.extract_numbers(roi)
```

**推奨度**: ⭐⭐ (2/5) - スコアシートには不向き

---

### アプローチE: 機械学習による位置検出

**手法**:
1. アノテーションデータでYOLO/Faster R-CNNなどの物体検出モデルを学習
2. モデルが各フィールドの位置を自動検出
3. 検出された位置でOCRを実行

**利点**:
- ✅ 様々なフォーマットに汎化
- ✅ 一度学習すれば高速
- ✅ 位置の多様性に対応

**欠点**:
- ❌ 大量のアノテーションデータが必要（数百〜数千枚）
- ❌ 学習に時間とGPUリソースが必要
- ❌ MVPには過剰

**推奨度**: ⭐ (1/5) - 将来的な選択肢

---

### アプローチF: 前処理の改善

**手法**:
OCR精度を上げるための前処理を強化：
1. **罫線除去**: Hough変換で直線を検出して除去
2. **コントラスト調整**: CLAHEなどで局所的に強調
3. **文字領域の二値化改善**: Sauvola, Niblackなどの適応的手法
4. **ノイズ除去強化**: モルフォロジー処理、メディアンフィルタ

**利点**:
- ✅ ROI位置に依存しない
- ✅ すべてのフィールドの精度向上に寄与
- ✅ 比較的実装が簡単

**欠点**:
- ❌ ROI位置問題は解決しない
- ❌ 過度な前処理で文字が劣化する可能性

**実装例**:
```python
def remove_lines(image):
    """罫線除去"""
    # エッジ検出
    edges = cv2.Canny(image, 50, 150)

    # Hough変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=10)

    # 検出された直線を白で塗りつぶす
    mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    # 元画像から罫線部分を除去
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

def enhance_contrast(image):
    """コントラスト強化（CLAHE）"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
```

**推奨度**: ⭐⭐⭐⭐ (4/5) - ROI最適化と並行して実施

---

## 推奨アプローチ

### 短期的（MVP達成）

**優先度1: アプローチA（手動位置特定）**
- 今すぐ実施可能
- 確実に動作する位置が得られる
- 他のアプローチのベースラインになる
- **実装時間**: 30分〜1時間

**優先度2: アプローチF（前処理改善）**
- 手動位置特定と並行して実施
- すべてのフィールドの精度向上に寄与
- **実装時間**: 1〜2時間

### 中期的（精度向上）

**優先度3: アプローチC（ベイズ最適化）**
- 手動位置をベースラインとして、さらなる最適化
- より汎用的な位置の発見
- **実装時間**: 2〜3時間

**優先度4: アプローチB（ランダムサーチ）**
- ベイズ最適化がうまくいかない場合の代替案
- **実装時間**: 1時間

### 長期的（プロダクション品質）

**優先度5: アプローチE（機械学習）**
- 十分なデータが集まってから検討
- より汎用的で堅牢なシステム
- **実装時間**: 数週間

---

## 実装優先度

### 次のセッションで実施すべきこと

1. **手動位置特定（アプローチA）** - 30分
   - `debug_player_row.jpg`を詳細に観察
   - GIMPでピクセル座標を測定
   - ハードコードしてテスト
   - 精度50%以上を目指す

2. **前処理改善（アプローチF）** - 1時間
   - 罫線除去の実装
   - CLAHEによるコントラスト強化
   - 新しい前処理パイプラインでテスト

3. **評価と調整** - 30分
   - 手動位置 + 前処理改善の効果を測定
   - 目標: 70%以上の精度達成

### 成功の定義

- ✅ skill_level: 70%以上
- ✅ total_score: 70%以上
- ⚠️ player_number: 50%以上（5桁は難しい）

---

## 参考資料

### 関連ファイル
- `src/preprocessing/roi_extractor.py` - ROI抽出モジュール
- `evaluate_ocr.py` - 評価スクリプト
- `optimize_roi_positions.py` - グリッドサーチ（単純）
- `optimize_roi_2stage.py` - 2段階グリッドサーチ（未完成）
- `data/ground_truth/LINE_ALBUM_2025秋_251221_1.json` - 正解データ

### 外部リソース
- [OpenCV Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
- [scikit-optimize (ベイズ最適化)](https://scikit-optimize.github.io/stable/)
- [Tesseract PSM Modes](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options)
- [CLAHE (Contrast Limited Adaptive Histogram Equalization)](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

---

**次のアクション**: アプローチA（手動位置特定）から開始
