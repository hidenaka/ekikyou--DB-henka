# event_id仕様書

## 1. 目的

event_idは同一イベントの複数フェーズ事例を束ねるための識別子である。
これにより：
- 同一イベントの時系列分析が可能
- 重複事例の検出が容易
- event_phaseとの組み合わせでイベント進行を追跡

## 2. フォーマット

```
EVT_{YEAR}_{SUBJECT_HASH}_{EVENT_HASH}
```

| 要素 | 説明 | 例 |
|------|------|-----|
| EVT | プレフィックス（固定） | EVT |
| YEAR | イベント開始年（4桁） | 2016 |
| SUBJECT_HASH | 主体名のMD5先頭6文字 | BA5EE4 |
| EVENT_HASH | イベントキーワードのMD5先頭4文字 | AD60 |

例: `EVT_2017_BA5EE4_AD60` (WeWork関連イベント)

## 3. 同一イベント判定規則

### 3.1 グループ化条件

以下の全てを満たす事例は同一event_idを付与：

1. **主体一致**: normalize_company_name()で正規化後の名前が一致
2. **時間近接性**: period開始年の差が5年以内
3. **キーワード類似**: story_summaryから抽出したキーワードが1つ以上一致

### 3.2 正規化ルール

```python
def normalize_company_name(name: str) -> str:
    """
    1. 括弧内の説明を除去: 「任天堂（ゲーム事業）」→「任天堂」
    2. 法人形態を除去: 「株式会社」「Inc.」等
    3. 前後の空白を除去
    """
```

### 3.3 キーワード抽出ルール

| パターン | キーワード |
|---------|-----------|
| 買収\|M&A\|TOB | ACQ |
| 合併\|統合 | MERGE |
| 倒産\|破産\|清算 | BANKRUPT |
| リストラ\|人員削減 | RESTRUCT |
| 上場\|IPO | IPO |
| 撤退\|売却 | DIVEST |
| 新規事業\|参入 | ENTRY |
| 不正\|不祥事 | SCANDAL |
| 危機\|ショック | CRISIS |
| 回復\|復活\|再生 | RECOVERY |

## 4. 分割・統合ポリシー

### 4.1 分割が必要なケース

- 同一主体が完全に独立した別イベントを経験
- 例: トヨタの「リコール問題(2009)」と「EV戦略(2020)」は別event_id

**判定基準**: 時間差10年以上、またはキーワード完全不一致

### 4.2 統合が必要なケース

- 表記ゆれで別event_idが付与された同一イベント
- 例: 「東芝不正会計」「東芝の会計問題」は統合

**対処**: entity_table.jsonのaliasを参照して統合

### 4.3 再採番禁止ルール

- 一度付与したevent_idは変更しない（不変性）
- 分割・統合が必要な場合はalias関係で対応
- event_id_historyテーブルで変更履歴を管理（今後実装）

## 5. 監査ログ

### 5.1 付与時ログ

```json
{
  "timestamp": "2026-01-16T06:53:06Z",
  "action": "assign",
  "event_id": "EVT_2017_BA5EE4_AD60",
  "affected_cases": 19,
  "reason": "initial_assignment"
}
```

### 5.2 変更時ログ（今後実装）

```json
{
  "timestamp": "...",
  "action": "merge|split|reassign",
  "old_event_id": "...",
  "new_event_id": "...",
  "affected_cases": [...],
  "reason": "..."
}
```

## 6. 統計（現在値）

| 指標 | 値 |
|------|-----|
| 総event_id数 | 10,103 |
| 複数事例event_id数 | 2,112 |
| 最大束ね数 | 19件（WeWork） |
| 平均束ね数（複数のみ） | 2.3件 |

## 7. 制約

### 7.1 参照整合性

- event_idはcases.jsonlのevent_idフィールドに格納
- entity_table.jsonのエンティティと1対多関係

### 7.2 ユニーク制約

- (event_id, event_phase, primary_subject_id)の組み合わせはユニーク
- 同一イベントの同一フェーズで同一主体は1件のみ

### 7.3 NULL許容

- event_id: NOT NULL（全件必須）
- event_phase: NULL許容（デフォルト: outcome）

## 8. 再現可能性

### 8.1 再生成手順

```bash
# バックアップから復元
cp data/raw/cases_backup_eventid_*.jsonl data/raw/cases.jsonl

# event_id再付与
python3 scripts/assign_event_ids.py
```

### 8.2 決定論的生成

- 同一入力に対して常に同一event_idを生成
- ランダム要素なし
- ハッシュベースで再現可能

---

*作成日: 2026-01-16*
*バージョン: 1.0*
