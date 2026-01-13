# success_level 算出仕様書

## 概要

success_levelは、特定の条件（卦×爻、パターン等）における成功確率を
実測データから算出した値です。ベイズ平滑化と信頼区間を用いて、
サンプルサイズが小さい場合でも安定した推定値を提供します。

## 算出式

### 1. ベイズ平滑化（Laplace補正）

```
smoothed_rate = (success_sum + α × prior) / (total + α)
```

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| success_sum | 実測値 | Success=1.0, PartialSuccess=0.7, Mixed=0.5, Failure=0.0 |
| total | 実測値 | サンプル数 |
| α (alpha) | 1.0 | 事前観測数（平滑化強度） |
| prior | 0.5 | 事前成功率（中立値） |

**効果**: n=0でも50%を返し、nが増えるほど実測値に収束

### 2. 信頼区間（Wilson score interval）

95%信頼区間を Wilson score interval で算出:

```python
z = 1.96  # 95% confidence
p = success_sum / total
denominator = 1 + z^2 / total
centre = (p + z^2 / (2 * total)) / denominator
margin = z * sqrt((p * (1-p) + z^2 / (4*total)) / total) / denominator
CI = [centre - margin, centre + margin]
```

**採用理由**: 二項分布の信頼区間として、nが小さくても適切に機能

### 3. 信頼性フラグ

| sample_count | 信頼性 | 用途 |
|--------------|--------|------|
| n < 5 | 表示不可 | 統計として不使用 |
| 5 ≤ n < 10 | 低 | 参考値として表示 |
| 10 ≤ n < 20 | 中 | 警告フラグ付きで使用 |
| n ≥ 20 | 高 | 統計として信頼可能 |

設定値:
- `MIN_SAMPLE_RELIABLE = 10` （is_reliable=true の閾値）

## 統計キーの粒度

以下の複数粒度で統計を算出:

| キー形式 | 例 | 用途 |
|----------|-----|------|
| `hex:{id}` | `hex:11` | 卦単位の成功率 |
| `hex:{id}:yao:{pos}` | `hex:11:yao:3` | 卦×爻の成功率 |
| `pattern:{type}` | `pattern:Steady_Growth` | パターン単位 |
| `hex:{id}:pattern:{type}` | `hex:11:pattern:Steady_Growth` | 卦×パターン |

## データソース

- **使用データ**: Gold + Silver事例（v3分類）
- **現在のサンプルサイズ**: 3,086件（2026-01-13時点）
- **出力先**: `data/analysis/success_rate_table.json`

## 出力フォーマット

```json
{
  "hex:11": {
    "sample_count": 358,
    "raw_success_rate": 0.962,
    "smoothed_success_rate": 0.962,
    "confidence_interval": [0.934, 0.976],
    "is_reliable": true,
    "outcome_distribution": {
      "Success": 340,
      "Failure": 10,
      "Mixed": 8
    }
  }
}
```

## 注意事項

1. **パターン名バイアス**: パターン名に結果が含まれるため（例: Failed_Attempt）、
   パターン別成功率は参考値として扱う

2. **企業公式ソースの自己呈示バイアス**:
   - tier4_corporate（企業IR/プレスリリース）は一次情報として事実性は高い
   - ただし、企業は自社に有利な情報を選択的に開示する傾向がある
   - **対策**:
     - tier1-2（政府・主要メディア）ソースとの併用を推奨
     - 企業公式のみの事例はSuccess偏重の可能性を考慮
     - Failure事例は主要メディア報道を優先的に参照

3. **地域バイアス**:
   - 現在のデータは日本事例が77.6%を占める
   - success_levelは日本の事業環境・文脈に最適化されている可能性
   - グローバル用途では外的妥当性に注意が必要

2. **再計算タイミング**: Gold/Silver事例が追加された場合、
   `python3 scripts/quality/phase4_success_level.py` を再実行

3. **旧success_level値との関係**:
   - 旧: 固定値 85/65/50/15（outcome依存）
   - 新: 実測統計ベース（卦×パターン依存）

## 実装ファイル

- `scripts/quality/phase4_success_level.py` - 算出スクリプト
- `scripts/quality/quality_config_v2.py` - 設定値（SUCCESS_LEVEL_RULES）
