# Ground Truth Data（正解データ）

このディレクトリには、OCR精度評価のための正解データを格納します。

## ファイル構成

- `template.json` - 正解データのテンプレート
- `LINE_ALBUM_2025秋_251221_1.json` - 各画像の正解データ
- `README.md` - このファイル

## データフォーマット

各JSONファイルは以下の構造を持ちます：

```json
{
  "image_name": "画像ファイル名",
  "sheets": [
    {
      "sheet_index": 0,  // 0=上のシート、1=下のシート
      "team_name": "チーム名",
      "matches": [
        {
          "match_index": 0,  // 0-4（5試合）
          "player_number": "選手番号",
          "player_name": "選手名",
          "skill_level": "スキルレベル（1-9）",
          "total_innings": "合計イニング",
          "safety_count": "セーフティ数",
          "total_score": "合計得点",
          "match_score": "試合得点",
          "match_time": "試合時間"
        }
      ]
    }
  ]
}
```

## フィールド説明

### 必須フィールド（🔴 優先度高）
- `player_number`: 選手番号（主キー）
- `skill_level`: スキルレベル（斜めマスの左上）
- `total_score`: 合計得点

### オプションフィールド
- `player_name`: 選手名
- `total_innings`: 合計イニング
- `safety_count`: セーフティ数
- `match_score`: 試合得点
- `match_time`: 試合時間

## 記入ガイド

1. `template.json` をコピーして新しいファイルを作成
2. 画像を見ながら各フィールドに正解値を記入
3. 読み取れない場合は `null` を設定
4. 空白の場合は `""` を設定

## 注意事項

- 手書き文字が不明瞭な場合は、最も可能性が高い値を記入
- 確信が持てない場合は、コメントを追加するか `null` にする
- 数値フィールドは文字列として記入（"75" など）
