# 診断ツール詳細

## 易経変化診断エンジン（メイン）

```bash
python scripts/diagnostic_engine.py
```

位相（八卦）× 勢（momentum）× 時（timing）に基づく行動推奨システム。
8つの質問に回答すると、実例データに基づいて最適な行動を推奨。

## 実例ベース診断ツール（レガシー）

```bash
python scripts/diagnose.py
```

対話形式で以下を入力：
1. スケール（individual/company/family/country/other）
2. 現在の状態
3. きっかけのタイプ（オプション）

→ 類似する歴史的事例を最大10件表示

### マッチングアルゴリズム

優先度の高い順にスコアリング：
1. **必須（100点）**: scale完全一致
2. **重要（50点）**: before_state完全一致
3. **補助（30点）**: trigger_type完全一致
4. **八卦（20点）**: before_hex完全一致
5. **信頼性ボーナス**: S=10, A=7, B=4, C=2

## 384爻分析

```bash
python scripts/analyze_384_lines.py
```

変爻パターンの統計分析：
- ユニークな(卦ペア, 爻番号)組み合わせ
- トランジション別の使用頻度
- 八卦ペアの網羅率

## データ収集目標

- 各(scale × before_state)で最低20事例
- individualスケール優先
- 合計10,000事例目標
