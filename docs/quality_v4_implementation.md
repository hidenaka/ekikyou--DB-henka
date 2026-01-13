# 品質分類v4 実装ドキュメント

## 概要

LLMディベート（Claude + Codex GPT-5.2）3回を経て、Codex批評5点を反映した品質分類システムv4を実装。

## Codex批評5点への対応状況

| 批評点 | 対応 | 実装 |
|--------|------|------|
| 1. Gold二階建て化 | ✅ | Verified=Gold、Verifiable=Silver |
| 2. 証拠独立性スキーマ | ✅ | シンジケーション重複検出 |
| 3. COI手動レジストリ優先 | ✅ | coi_registry.py |
| 4. 統計KPI変更 | ❌ | 今後対応（校正・スコア評価） |
| 5. スコープ縮小 | ✅ | 主張単位は今後、事例単位で運用 |

## 新ファイル

### coi_registry.py
```
scripts/quality/coi_registry.py
```

- **DOMAIN_OWNER_REGISTRY**: ドメイン→所有企業マッピング（80+企業）
- **CORPORATE_GROUPS**: 企業グループ関係（子会社・関連会社）
- **TARGET_NAME_NORMALIZATION**: 企業名の揺らぎ対応（日本語/英語）
- **evaluate_coi()**: COI総合評価関数

### phase1_2_v4.py
```
scripts/quality/phase1_2_v4.py
```

- **check_evidence_independence()**: 証拠独立性チェック
- **determine_verification_status()**: 検証状態判定
- **evaluate_case_v4()**: v4評価関数

## 分類結果比較

| 指標 | v3 | v4 | 変化 |
|------|----|----|------|
| Gold | ~980件 | 717件 | -27% |
| Silver | ~2,170件 | 5,325件 | +145% |
| Bronze | ~1,900件 | 1,873件 | ≒ |
| Quarantine | ~4,900件 | 4,956件 | ≒ |

## v4分類ロジック

### Gold判定条件（厳格化）
```python
Gold = (trust_level = "verified") AND
       (best_tier in GOLD_TIERS) AND
       (coi != "self" OR has_non_coi_source)
```

### Verifiable（未検証）の扱い
- v3: Gold候補
- v4: **Silver扱い**（Codex指摘: 未検証をGoldと呼ぶのは矛盾）

### COI=self検出
- tier4_corporate（企業公式）が自社について報じている場合
- 361件がCOI=selfのみでSilverに降格

### 証拠独立性
| 状態 | 件数 | 説明 |
|------|------|------|
| independent | 294件 | 複数の独立ソース |
| single_source | 7,548件 | 単一ソース |
| same_domain | 73件 | 同一ドメイン複数URL |
| syndication_overlap | 0件 | シンジケーション重複 |

## Tier別成功率

| Tier | Success/Failure | 比率 |
|------|-----------------|------|
| Gold | 367/113 | 3.2:1 |
| Silver | 2,507/1,389 | 1.8:1 |
| Bronze | 1,300/137 | 9.5:1 |

※Bronze異常値は調査要

## 今後の課題

1. **統計KPI変更**: CI幅ではなく校正（calibration）と予測スコアで評価
2. **主張単位評価**: 現行の事例単位から拡張（長期課題）
3. **COIレジストリ拡充**: 提携関係（affiliated）の追加
4. **Bronze異常値調査**: Success比が高すぎる原因究明

## 実行方法

```bash
cd scripts/quality
python3 phase1_2_v4.py
```

## 生成ファイル

| ファイル | 内容 |
|----------|------|
| data/gold/gold_cases_v4.jsonl | Gold事例（717件） |
| data/gold/silver_cases_v4.jsonl | Silver事例（5,325件） |
| data/raw/bronze_cases_v4.jsonl | Bronze事例（1,873件） |
| data/quarantine/quarantine_v4.jsonl | Quarantine事例（4,956件） |
| data/gold/extraction_report_v4.json | 分類レポート |

---
*作成日: 2026-01-13*
*LLMディベート: 3回実施（v1→v2→v3）*
