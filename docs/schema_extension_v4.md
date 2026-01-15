# スキーマ拡張設計 v4

## 背景・目的

Codexからの批判を受けて、メタデータスキーマを拡張する。

**Codex指摘事項**:
1. 「事例メタデータ（対象主体の種別=企業/政策、日付、位相、主要論点）を構造化し、混線を禁止せよ」
2. 「企業行動のロジックDBに政策を入れるなら、政策を『外生ショック』として別カテゴリで管理せよ」

**現在の問題**:
- `scale` フィールドに "company" / "country" が混在し、企業意思決定と政策決定の区別が曖昧
- 同一主体（例：トヨタ）の複数フェーズ事例が、重複排除ロジックで識別しづらい
- M&Aなど多段階イベントの「どの時点か」が不明確

---

## 1. 新規フィールド定義

### 1.1 subject_type（主体種別）

**目的**: 意思決定主体の種別を明確に分類し、企業ロジックと外生要因を分離

```python
class SubjectType(str, Enum):
    COMPANY = "company"           # 企業の意思決定事例
    POLICY = "policy"             # 政策・規制・中央銀行の決定
    INDIVIDUAL = "individual"     # 個人の意思決定事例
    MARKET = "market"             # 市場全体の動向（株価、業界トレンド等）
    EXOGENOUS = "exogenous"       # 外生ショック（自然災害、パンデミック等）
```

**使い分けガイドライン**:

| subject_type | 例 | 特徴 |
|--------------|-----|------|
| company | トヨタのEV戦略転換 | 企業内部の意思決定 |
| policy | FRBの利上げ決定 | 政府・中央銀行の規制・政策 |
| individual | イーロン・マスクのTwitter買収 | 個人としての意思決定 |
| market | 2020年コロナショック相場 | 市場全体の動き（主体なし） |
| exogenous | 東日本大震災 | 外生的ショック（人為的意思決定なし） |

**既存 `scale` との関係**:
- `scale` は「対象のスケール（規模）」を表す → 維持
- `subject_type` は「意思決定の主体種別」を表す → 新設
- 両者は直交する属性

```
例: 「日本のアベノミクス」
  - scale = "country"（国家規模の事象）
  - subject_type = "policy"（政策決定が主体）

例: 「トヨタのEV戦略」
  - scale = "company"（企業規模の事象）
  - subject_type = "company"（企業の意思決定）

例: 「東日本大震災」
  - scale = "country"（国家規模の影響）
  - subject_type = "exogenous"（外生ショック）
```

---

### 1.2 event_phase（事象位相）

**目的**: M&A、危機対応など多段階事象の「どの時点か」を明示

```python
class EventPhase(str, Enum):
    ANNOUNCEMENT = "announcement"     # 発表時点
    EXECUTION = "execution"           # 実行時点
    COMPLETION = "completion"         # 完了時点
    OUTCOME = "outcome"               # 結果確定時点
```

**使用例**:

| 事例 | event_phase | 説明 |
|------|-------------|------|
| ソフトバンクArm買収発表（2016/7） | announcement | 買収意向発表 |
| ソフトバンクArm買収完了（2016/9） | completion | 買収手続き完了 |
| ソフトバンクArm上場（2023/9） | outcome | 投資結果確定 |

**多段階イベントの記録方法**:
- 同一イベントの異なるフェーズは、別事例として記録
- `event_id` で同一イベントを紐付け（後述）

---

### 1.3 primary_subject_id（主体ID）

**目的**: 重複排除を「主体ID × 事象ID × 日付」で確実に行う

```python
# フィールド定義
primary_subject_id: Optional[str] = None  # 例: "CORP_JP_TOYOTA"
event_id: Optional[str] = None            # 例: "EVT_2016_SBG_ARM"
event_date: Optional[str] = None          # 例: "2016-07-18"
```

**ID命名規則**:

```
primary_subject_id:
  CORP_{COUNTRY}_{SHORTNAME}     # 企業: CORP_JP_TOYOTA, CORP_US_APPLE
  GOV_{COUNTRY}_{AGENCY}         # 政府機関: GOV_JP_BOJ, GOV_US_FED
  PERS_{COUNTRY}_{NAME}          # 個人: PERS_US_MUSK, PERS_JP_SON
  MKT_{REGION}_{TYPE}            # 市場: MKT_GLOBAL_STOCK, MKT_JP_NIKKEI

event_id:
  EVT_{YEAR}_{SUBJECT}_{KEYWORD} # イベント: EVT_2016_SBG_ARM
```

**重複排除ロジック**:
```python
# 従来: target_name × period で重複判定 → 曖昧
# 新規: primary_subject_id × event_id × event_phase で厳密判定

def is_duplicate(new_case, existing_cases):
    key = (
        new_case.get('primary_subject_id'),
        new_case.get('event_id'),
        new_case.get('event_phase')
    )
    for case in existing_cases:
        existing_key = (
            case.get('primary_subject_id'),
            case.get('event_id'),
            case.get('event_phase')
        )
        if key == existing_key:
            return True
    return False
```

---

## 2. スキーマ拡張の完全定義

### 2.1 schema_v3.py への追加

```python
# 新規Enum追加
class SubjectType(str, Enum):
    COMPANY = "company"
    POLICY = "policy"
    INDIVIDUAL = "individual"
    MARKET = "market"
    EXOGENOUS = "exogenous"

class EventPhase(str, Enum):
    ANNOUNCEMENT = "announcement"
    EXECUTION = "execution"
    COMPLETION = "completion"
    OUTCOME = "outcome"

# Caseモデルへの追加フィールド
class Case(BaseModel):
    # ... 既存フィールド ...

    # === v4 新規フィールド ===
    subject_type: Optional[SubjectType] = None      # 主体種別
    event_phase: Optional[EventPhase] = None        # 事象位相
    primary_subject_id: Optional[str] = None        # 主体ID
    event_id: Optional[str] = None                  # イベントID
    event_date: Optional[str] = None                # イベント日付（YYYY-MM-DD）
```

### 2.2 JSONスキーマ定義

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HaQei Case Schema v4",
  "type": "object",
  "properties": {
    "subject_type": {
      "type": "string",
      "enum": ["company", "policy", "individual", "market", "exogenous"],
      "description": "意思決定主体の種別"
    },
    "event_phase": {
      "type": "string",
      "enum": ["announcement", "execution", "completion", "outcome"],
      "description": "多段階事象の位相"
    },
    "primary_subject_id": {
      "type": "string",
      "pattern": "^(CORP|GOV|PERS|MKT)_[A-Z]{2,}_[A-Z0-9_]+$",
      "description": "主体の一意ID"
    },
    "event_id": {
      "type": "string",
      "pattern": "^EVT_[0-9]{4}_[A-Z0-9_]+$",
      "description": "イベントの一意ID"
    },
    "event_date": {
      "type": "string",
      "pattern": "^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$",
      "description": "イベント日付（YYYY, YYYY-MM, YYYY-MM-DD）"
    }
  }
}
```

---

## 3. 既存データのマイグレーション方針

### 3.1 マイグレーション戦略

**フェーズ1: 自動推論（即時）**
```python
def infer_subject_type(case):
    """既存データからsubject_typeを推論"""
    scale = case.get('scale', '')
    name = case.get('target_name', '')
    summary = case.get('story_summary', '')

    # 政策キーワードチェック
    policy_keywords = ['規制', '政策', 'FRB', 'Fed', '金融政策',
                       '中央銀行', '日銀', '法改正', 'GDPR', '消費税']
    if any(kw in name or kw in summary for kw in policy_keywords):
        return 'policy'

    # 外生ショックチェック
    exogenous_keywords = ['地震', '津波', '災害', 'パンデミック',
                          'コロナ', '台風', '洪水']
    if any(kw in summary for kw in exogenous_keywords):
        return 'exogenous'

    # scaleからの推論
    mapping = {
        'company': 'company',
        'individual': 'individual',
        'country': 'policy',  # デフォルト（要レビュー）
        'family': 'individual',
        'other': 'market'
    }
    return mapping.get(scale, 'company')
```

**フェーズ2: 手動レビュー（後日）**
- `scale=country` かつ `subject_type=policy` の事例をレビュー
- 市場動向（subject_type=market）への再分類

**フェーズ3: ID付与（後日）**
- 高頻度主体（トヨタ、任天堂等）に `primary_subject_id` を付与
- 重要イベント（M&A、危機対応）に `event_id` を付与

### 3.2 マイグレーションスクリプト

```python
#!/usr/bin/env python3
"""
マイグレーションスクリプト: v3 → v4
新規フィールドをOptionalとして追加、推論可能なものは自動付与
"""

import json
from pathlib import Path

def migrate_case(case):
    """1件のcaseをv4形式にマイグレーション"""
    # subject_typeの推論
    if 'subject_type' not in case:
        case['subject_type'] = infer_subject_type(case)

    # event_phaseはデフォルトでoutcome（結果確定時点として記録されている前提）
    if 'event_phase' not in case:
        case['event_phase'] = 'outcome'

    # ID系は空のまま（後日付与）
    if 'primary_subject_id' not in case:
        case['primary_subject_id'] = None
    if 'event_id' not in case:
        case['event_id'] = None
    if 'event_date' not in case:
        case['event_date'] = None

    return case

def main():
    cases_path = Path("data/raw/cases.jsonl")
    cases = []
    with open(cases_path, 'r') as f:
        for line in f:
            case = json.loads(line.strip())
            cases.append(migrate_case(case))

    # バックアップ後に上書き
    backup_path = cases_path.with_suffix('.jsonl.v3.bak')
    cases_path.rename(backup_path)

    with open(cases_path, 'w') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')

    print(f"Migrated {len(cases)} cases to v4 schema")

if __name__ == "__main__":
    main()
```

---

## 4. 入口ゲートへの影響

### 4.1 重複チェックロジックの拡張

```python
def check_exact_duplicate(new_case, existing_cases):
    """完全一致重複チェック（v4対応）"""

    # 優先: primary_subject_id × event_id × event_phase
    if new_case.get('primary_subject_id') and new_case.get('event_id'):
        new_key = (
            new_case.get('primary_subject_id'),
            new_case.get('event_id'),
            new_case.get('event_phase')
        )
        for case in existing_cases:
            if case.get('primary_subject_id') and case.get('event_id'):
                existing_key = (
                    case.get('primary_subject_id'),
                    case.get('event_id'),
                    case.get('event_phase')
                )
                if new_key == existing_key:
                    return case

    # フォールバック: 従来のtarget_name × period
    new_key = (new_case.get('target_name', ''), new_case.get('period', ''))
    for case in existing_cases:
        existing_key = (case.get('target_name', ''), case.get('period', ''))
        if new_key == existing_key:
            return case

    return None
```

### 4.2 バリデーション追加

```python
def validate_v4_fields(case):
    """v4新規フィールドのバリデーション"""
    errors = []

    # subject_typeの整合性チェック
    subject_type = case.get('subject_type')
    scale = case.get('scale')

    if subject_type == 'company' and scale not in ['company', 'other']:
        errors.append(f"subject_type=company but scale={scale}")

    if subject_type == 'individual' and scale not in ['individual', 'family']:
        errors.append(f"subject_type=individual but scale={scale}")

    # event_dateの形式チェック
    event_date = case.get('event_date')
    if event_date:
        import re
        if not re.match(r'^[0-9]{4}(-[0-9]{2}(-[0-9]{2})?)?$', event_date):
            errors.append(f"Invalid event_date format: {event_date}")

    # primary_subject_idの形式チェック
    psid = case.get('primary_subject_id')
    if psid:
        import re
        if not re.match(r'^(CORP|GOV|PERS|MKT)_[A-Z]{2,}_[A-Z0-9_]+$', psid):
            errors.append(f"Invalid primary_subject_id format: {psid}")

    return errors
```

---

## 5. 設計上の考慮事項

### 5.1 後方互換性

- 全新規フィールドは `Optional` として追加
- 既存のクエリ・スクリプトは変更なしで動作
- マイグレーションは段階的に実行可能

### 5.2 将来拡張性

- `subject_type` に追加値を入れやすい構造
- `event_id` による事例間リンケージの基盤
- 将来的な「因果関係グラフ」構築への布石

### 5.3 Codex指摘への対応マッピング

| Codex指摘 | 対応 |
|-----------|------|
| 主体種別の構造化 | `subject_type` 新設 |
| 政策を外生ショックとして管理 | `subject_type=policy` / `exogenous` の分離 |
| 日付の明示 | `event_date` 新設 |
| 位相の明示 | `event_phase` 新設 |
| 混線の禁止 | 重複チェックロジック強化 |

---

## 6. 実装優先順位

1. **Phase 1（即時）**: schema_v3.py への Enum・フィールド追加
2. **Phase 2（1日以内）**: マイグレーションスクリプト実行（subject_type自動推論）
3. **Phase 3（1週間以内）**: entry_gate.py の重複チェック拡張
4. **Phase 4（後日）**: 高頻度主体へのID付与

---

## 7. 承認後の次ステップ

この設計が承認されたら：

1. `scripts/schema_v3.py` を `scripts/schema_v4.py` として更新
2. マイグレーションスクリプト `scripts/migrate_to_v4.py` を作成
3. `scripts/entry_gate.py` を v4 対応に更新
4. バッチ追加スクリプトを v4 対応に更新
5. MCPメモリに変更履歴を記録

---

*作成日: 2026-01-16*
*作成者: Claude Code*
*Codex批判対応版*
