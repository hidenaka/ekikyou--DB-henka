# 整合性制約仕様書

## 1. 参照整合性

### 1.1 entity_table参照

```
cases.jsonl.primary_subject_id → entity_table.json[entity_id]
```

| 制約 | 説明 |
|------|------|
| 存在制約 | primary_subject_idはentity_tableに存在すること |
| 型制約 | entity_tableのentity_typeはcasesのentity_typeと一致 |

### 1.2 alias_table参照

```
cases.jsonl.target_name → alias_table.json[original_name] → entity_id
```

| 制約 | 説明 |
|------|------|
| alias解決 | target_nameがalias_tableにあればprimary_subject_idに変換 |

## 2. ユニーク制約

### 2.1 事例レベル

| キー | 制約 |
|------|------|
| (event_id, event_phase, primary_subject_id) | ユニーク |
| transition_id | NOT NULL, ユニーク |

### 2.2 エンティティレベル

| キー | 制約 |
|------|------|
| entity_table.entity_id | ユニーク |
| alias_table.original_name | ユニーク（1対1マッピング） |

## 3. NOT NULL制約

### 3.1 必須フィールド

| フィールド | 制約 |
|-----------|------|
| target_name | NOT NULL |
| scale | NOT NULL |
| period | NOT NULL |
| story_summary | NOT NULL |
| before_state | NOT NULL |
| trigger_type | NOT NULL |
| action_type | NOT NULL |
| after_state | NOT NULL |
| before_hex | NOT NULL |
| trigger_hex | NOT NULL |
| action_hex | NOT NULL |
| after_hex | NOT NULL |
| pattern_type | NOT NULL |
| outcome | NOT NULL |
| source_type | NOT NULL |
| credibility_rank | NOT NULL |

### 3.2 v4.1必須フィールド

| フィールド | 制約 | デフォルト |
|-----------|------|-----------|
| entity_type | NOT NULL | company |
| event_driver_type | NOT NULL | internal |
| event_id | NOT NULL | 自動生成 |
| primary_subject_id | NOT NULL | 自動生成 |
| event_phase | NULL許容 | outcome |
| annotation_status | NOT NULL | single |

## 4. 許容値制約（Enum）

### 4.1 entity_type

```python
['company', 'individual', 'government', 'organization']
```

### 4.2 event_driver_type

```python
['internal', 'market', 'policy', 'disaster', 'pandemic', 'technology', 'competition']
```

### 4.3 event_phase

```python
['announcement', 'execution', 'completion', 'outcome']
```

### 4.4 annotation_status

```python
['single', 'double', 'verified']
```

## 5. 組み合わせ制約

### 5.1 entity_typeとscaleの整合性

| entity_type | 許容scale |
|-------------|-----------|
| company | company, other |
| individual | individual, family |
| government | country |
| organization | other |

### 5.2 event_phaseの順序制約

同一event_idの事例間:
```
announcement → execution → completion → outcome
```

event_dateがある場合、この順序で時系列が進むこと。

### 5.3 before_state/after_stateの遷移制約

| before_state | 許容after_state |
|--------------|-----------------|
| どん底・危機 | V字回復・大成功, 縮小安定・生存, 崩壊・消滅, 変質・新生 |
| 絶頂・慢心 | 崩壊・消滅, 迷走・混乱, 変質・新生, 停滞・閉塞 |
| 安定・平和 | 持続成長・大成功, 安定成長・成功, 停滞・閉塞 |

## 6. 検証スクリプト

```bash
python3 scripts/validate_integrity.py
```

### 6.1 検証項目

1. 参照整合性チェック
2. ユニーク制約チェック
3. NOT NULL制約チェック
4. Enum値チェック
5. 組み合わせ制約チェック

### 6.2 エラー報告

```json
{
  "violations": [
    {
      "type": "ref_integrity",
      "case_index": 123,
      "field": "primary_subject_id",
      "value": "CORP_JP_INVALID",
      "message": "entity_tableに存在しない"
    }
  ],
  "summary": {
    "total_violations": 0,
    "ref_integrity": 0,
    "unique": 0,
    "not_null": 0,
    "enum": 0,
    "combination": 0
  }
}
```

## 7. 回帰テスト

### 7.1 マイグレーション後チェック

全マイグレーション実行後に:
```bash
python3 scripts/validate_integrity.py --strict
```

### 7.2 CI/CD統合

新規事例追加時に自動検証:
```bash
python3 scripts/add_batch.py data/import/xxx.json --validate
```

---

*作成日: 2026-01-16*
*バージョン: 1.0*
