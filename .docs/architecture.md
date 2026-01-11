# アーキテクチャ設計

## 設計原則

- **追記専用JSONL**: cases.jsonl は常に末尾追記のみ
- **自動ID生成**: スクリプトがID衝突を自動回避
- **書き込み時バリデーション**: Pydantic Caseモデルで検証
- **CSVは派生データ**: 真のデータソースはJSONL

## データ保存場所

| パス | 用途 |
|------|------|
| `data/raw/cases.jsonl` | 主データ（JSONL形式） |
| `data/raw/cases.csv` | CSV派生物 |
| `data/by_scale/` | スケール別分類 |
| `data/import/` | バッチ追加用入力 |
| `data/archive/` | バックアップ |

## Python環境

```bash
source .venv/bin/activate  # Python 3.13
```

依存: pydantic>=2.0.0
