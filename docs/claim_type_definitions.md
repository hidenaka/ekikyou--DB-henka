# 主張タイプ定義書 (Phase 3)

## 目的

事例の品質評価を「事例単位」から「主張（claim）単位」に移行するための定義書。
Codex批評に基づき、逆インセンティブを排除する設計。

## 主張タイプ分類

### 1. verifiable_fact（検証可能な事実）

**定義**: 客観的に検証可能な事実の記述

**特徴**:
- 日付、数値、出来事などの具体的情報
- 第三者が独立に検証可能
- 真偽が明確に判定可能

**例**:
- 「2023年3月期の売上高は1,000億円だった」
- 「2022年4月1日に新CEOが就任した」
- 「従業員数が5,000人から3,000人に削減された」

**証拠要件**: primary_source または secondary_source

---

### 2. inference（推論）

**定義**: 事実に基づく推論・分析

**特徴**:
- 「〜と考えられる」「〜の可能性がある」等の表現
- 論理的な根拠がある
- 専門家の分析を含む

**例**:
- 「売上減少は競合製品の台頭が原因と考えられる」
- 「この戦略変更は市場環境の変化に対応したものと推測される」

**証拠要件**: secondary_source（分析記事等）

---

### 3. evaluation（評価）

**定義**: 価値判断を含む評価

**特徴**:
- 「良い」「悪い」「成功」「失敗」等の判断
- 主観的要素を含む
- 評価基準が明示されない場合がある

**例**:
- 「この買収は成功だった」
- 「経営判断が優れていた」
- 「品質が業界最高水準」

**証拠要件**: context_only（評価は証拠で裏付け困難）

---

### 4. opinion（意見）

**定義**: 主観的な意見・見解

**特徴**:
- 「〜すべき」「〜が望ましい」等の規範的表現
- 個人の見解
- 検証不能

**例**:
- 「この企業は投資すべきだ」
- 「もっと早く決断すべきだった」

**証拠要件**: なし（意見は証拠対象外）

---

## Core Claims（中核主張）概念

### 定義

事例の品質を決定する上で**必須の検証対象となる主張**。

### 設計原則（逆インセンティブ排除）

**問題**: 「全主張がGoldならケースがGold」とすると、主張を減らすほど有利になる

**解決策**: Core Claimsを固定し、それ以外の主張は品質判定に影響しない

### Core Claims の条件

1. **outcome関連**: 成功/失敗の判定根拠となる主張
2. **target関連**: 対象企業・人物の特定に必要な主張
3. **timeline関連**: 時期・期間の特定に必要な主張

### Core Claims スキーマ

```python
core_claims = {
    'outcome_claim': {
        'required': True,
        'description': '成功/失敗を裏付ける主張',
        'example': '売上が3年連続で成長した',
    },
    'target_claim': {
        'required': True,
        'description': '対象を特定する主張',
        'example': 'ソニーは2020年にPS5を発売した',
    },
    'timeline_claim': {
        'required': False,
        'description': '時期を特定する主張',
        'example': '2019年から2022年にかけて',
    },
}
```

---

## 集約則（Aggregation Rules）

### Gold判定（ケースレベル）

```python
def is_gold_case(case):
    """
    ケースのGold判定
    逆インセンティブを排除する集約則
    """
    claims = case.get('claims', [])
    core_claims = [c for c in claims if c.get('is_core')]

    # 条件1: Core Claimsが存在すること
    if not core_claims:
        return False, 'no_core_claims'

    # 条件2: 全Core Claimsがverified
    unverified_core = [c for c in core_claims if not c.get('verified')]
    if unverified_core:
        return False, f'unverified_core_claims: {len(unverified_core)}'

    # 条件3: 重大な反証主張がないこと
    contradictions = [c for c in claims if c.get('contradicts_core')]
    if contradictions:
        return False, f'contradicting_claims: {len(contradictions)}'

    # 条件4: COI=selfのCore Claimsがないこと
    coi_core = [c for c in core_claims if c.get('coi') == 'self']
    if coi_core:
        return False, f'coi_self_core_claims: {len(coi_core)}'

    return True, 'all_conditions_met'
```

### 重要な設計ポイント

1. **Core Claims以外は判定に影響しない** → 主張を増やしても不利にならない
2. **Core Claimsは固定** → 必要最小限の主張のみ検証対象
3. **反証主張の検出** → 矛盾する情報がある場合は降格

---

## 証拠リンク（Evidence Linking）

### 主張と証拠の関係

```python
claim_evidence_schema = {
    'claim_id': 'c001',
    'claim_text': '2023年3月期の売上高は1,000億円',
    'claim_type': 'verifiable_fact',
    'is_core': True,  # Core Claimかどうか

    # 証拠リンク（複数可）
    'evidence': [
        {
            'source_index': 0,  # sources[0]を参照
            'evidence_type': 'primary_source',
            'excerpt': '売上高 100,000百万円',
            'verified': True,
        },
        {
            'source_index': 1,
            'evidence_type': 'secondary_source',
            'excerpt': '同社の売上は1000億円に達した',
            'verified': True,
        },
    ],

    # 矛盾検出
    'contradictions': [],  # 矛盾する主張のIDリスト

    # COI
    'coi': 'none',

    # 検証状態
    'verified': True,
}
```

---

## 移行計画

### Phase 3a: スキーマ確定（本ドキュメント）
- 主張タイプ定義
- Core Claims概念
- 集約則設計

### Phase 3b: パイロット実装（100件）
- evaluation_set_500から100件を選定
- 手動で主張タグ付け
- 集約則の検証

### Phase 3c: 自動化検討
- 主張抽出の自動化可能性評価
- LLM活用の検討
- コスト・精度のトレードオフ

---

## 検証チェックリスト

主張タグ付け時の確認事項:

- [ ] claim_typeは適切か
- [ ] is_coreの判定は正しいか
- [ ] 証拠リンクは正しいか
- [ ] 矛盾する主張はないか
- [ ] COIは適切に判定されているか

---

*作成日: 2026-01-13*
*Phase: 3 (主張単位定義)*
