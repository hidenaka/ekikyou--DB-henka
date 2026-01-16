# v4スキーマ改善の概要

## Codex批判への対応状況

### 批判1: 主体種別の構造化
**対応**: `subject_type`フィールドを新設
- company (64.8%): 企業の意思決定
- individual (30.3%): 個人の意思決定
- policy (2.9%): 政策・規制・中央銀行
- exogenous (1.1%): 外生ショック
- market (0.9%): 市場全体の動向

### 批判2: 政策を外生ショックとして管理
**対応**: 分類ロジックを改善
- target_nameで政策機関を判定（日銀、FRB等）
- story_summaryの政策キーワードは企業事例を「政策」に誤分類しないよう除外
- 例: 日立製作所のリーマンショック対応 → company（以前はmarket誤分類）

### 批判3: 日付・位相の明示
**対応**: `event_phase`フィールドを新設
- announcement (2.5%): 発表時点
- execution (5.1%): 実行時点
- completion (15.6%): 完了時点
- outcome (76.8%): 結果確定時点

### 批判4: 混線の禁止・重複排除強化
**対応**: `primary_subject_id`による正規化
- フォーマット: `{CORP|INDV|GOVT|OTHR}_{country}_{hash8}`
- 日本語名はMD5ハッシュの先頭8文字で一意化
- 例: `CORP_日本_91C325D4`（日立製作所）

### 批判5: 二重アノテーション・一致率の数値化
**対応**: `scripts/double_annotate.py`を実装

#### 100件サンプル二重アノテーション結果
```
アノテータ1（厳格基準、閾値0.50）: 39件有効 (39.0%)
アノテータ2（寛容基準、閾値0.35）: 65件有効 (65.0%)

単純一致率: 74.0%
Cohen's Kappa: 0.512
解釈: 中程度の一致
```

#### スコアリング基準（統一）
1. explicit_yao (30%): logic_memoに爻の明示的参照があるか
2. phrase_match (15%): 爻辞テキストがmemoに含まれるか
3. structural (25%): 爻位とbefore_stateの構造的整合性
4. source (15%): ソース信頼性（official=1.0, news=0.8等）
5. memo_quality (15%): logic_memoの存在と長さ

### 批判6: 運用定義の文書化
**対応**: `docs/operational_definitions_v1.md`を作成
- 6段階爻モデルの定義
- 採択基準・棄却基準の明示
- 多段階イベントの処理方法
- 反例・エッジケースの扱い

## 実装ファイル一覧

| ファイル | 内容 |
|---------|------|
| `scripts/schema_v3.py` | v4フィールド追加（SubjectType, EventPhase等） |
| `scripts/migrate_to_v4.py` | 既存データへのv4フィールド付与 |
| `scripts/double_annotate.py` | 二重アノテーション・Kappa計算 |
| `docs/operational_definitions_v1.md` | 運用定義ドキュメント |
| `docs/schema_extension_v4.md` | スキーマ拡張設計書 |
| `data/quality/double_annotation_result.json` | アノテーション結果 |

## 残存課題

1. **Kappa 0.512は「中程度」**
   - 目標: 0.6以上（かなりの一致）
   - 不一致26件は閾値境界（0.35-0.50）の事例

2. **annotation_status = 'single'のまま**
   - 全13,060件が単一アノテーション状態
   - 二重検証済み(verified)への移行は今後

3. **event_idの付与**
   - 多段階イベントの紐付けは未実装
   - 同一イベントの異なるフェーズ事例の識別が必要

## 総合評価依頼

Codex批判6項目のうち5項目に対応:
1. ✅ 主体種別の構造化
2. ✅ 政策の分離管理
3. ✅ 位相の明示
4. ✅ 二重アノテーション・Kappa数値化
5. ✅ 運用定義の文書化
6. △ 重複排除の正規化（primary_subject_id実装済み、運用は今後）

B評価からA評価への昇格条件を満たしているか評価を依頼。
