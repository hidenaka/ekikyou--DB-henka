# ラベル定義書 v1

**フェーズ**: 品質改善フェーズα
**目的**: IAA（Inter-Annotator Agreement）測定の基盤となる明確な定義
**作成日**: 2026-01-13
**Codex批評対応**: ラベル間の整合性問題を解決

---

## 1. 概要

### 1.1 背景

易経変化ロジックDBの品質改善において、Codex批評で以下の問題が指摘された：

1. **correct/incorrectとminor/major_errorの関係が曖昧**
2. **ドメイン＝信頼性の前提が誤り**（同一ドメイン内で品質が大きく変動）
3. **単軸評価の限界**（信頼度と利益相反を分離すべき）

本定義書は、これらの問題を解決し、アノテーター間で一貫した判定を可能にする。

### 1.2 設計原則

| 原則 | 説明 |
|------|------|
| **直交性** | 各フィールドは独立した軸を測定する |
| **判定可能性** | 明確なフローチャートで機械的に判定可能 |
| **整合性** | フィールド間の関係を明示的に定義 |
| **廃止明示** | 使わないフィールドを明確化し混乱を防止 |

---

## 2. フィールド定義

### 2.1 outcome_status（結果検証状態）

**目的**: 事例の「結果（outcome）」が検証済みかどうかを示す

| 値 | 定義 | 判定基準 |
|----|------|----------|
| `verified_correct` | 検証済み・正しい | 一次/二次ソースで結果を確認し、事例記述と一致 |
| `verified_incorrect` | 検証済み・誤り | 一次/二次ソースで結果を確認し、事例記述と**不一致** |
| `unverified` | 未検証 | ソースで結果を確認できない |

#### unverifiedの理由分類（サブフィールド: `unverified_reason`）

| 理由コード | 説明 | 例 |
|------------|------|-----|
| `no_source` | ソースが存在しない | source_urlが空 |
| `source_unavailable` | ソースにアクセス不能 | 404、paywall、地域制限 |
| `source_insufficient` | ソースが結果を言及していない | 背景のみ記載、結果未記載 |
| `ongoing` | 進行中で結果が確定していない | 現在進行形の事例 |
| `ambiguous` | 結果の解釈が曖昧 | 成功/失敗の判断が分かれる |

#### 重要な整合性ルール

```
outcome_status = verified_correct の場合:
  → outcome は Success/Failure/Mixed/PartialSuccess のいずれか（Unknownは不可）
  → verification_confidence は high または medium

outcome_status = verified_incorrect の場合:
  → outcome を修正する必要あり
  → error_type を記録（後述）

outcome_status = unverified の場合:
  → outcome は Unknown に変更推奨
  → 統計計算から除外
```

---

### 2.2 source_role（ソース役割）

**目的**: ソースURLの情報源としての役割を分類する

| 値 | 定義 | 判定基準 | 例 |
|----|------|----------|-----|
| `primary_source` | 一次情報源 | 当事者による直接的な発信 | IR資料、官報、判決文、公式プレスリリース、有価証券報告書 |
| `secondary_source` | 二次情報源 | 一次情報を報道・分析した第三者 | 新聞記事、学術論文、調査レポート |
| `pointer_only` | ポインタのみ | 引用元への導線（記事本体は参照用） | Wikipedia、百科事典 |
| `context_only` | 文脈情報のみ | 背景説明や評価主張のみ | ブログ、意見記事、SNS |
| `rejected` | 使用不可 | 情報源として不適切 | 検索結果URL、Webキャッシュ、パラメータ付き転送URL |

#### 判定フローチャート

```
[URL入力]
    ↓
[1. 検索URL/キャッシュか？] → Yes → rejected
    ↓ No
[2. 当事者発信か？]
    ↓ Yes → primary_source
    ↓ No
[3. 報道/学術/調査機関か？] → Yes → secondary_source
    ↓ No
[4. 百科事典系か？] → Yes → pointer_only
    ↓ No
[5. 主張タイプは？]
    - 事実主張 → context_only（要確認）
    - 意見/評価主張 → context_only
```

#### 一次情報源の詳細分類（オプション: `primary_type`）

| タイプ | 説明 |
|--------|------|
| `ir_filing` | IR資料・有報・決算短信 |
| `government_gazette` | 官報・政府公報 |
| `court_document` | 判決文・訴訟記録 |
| `official_press` | 公式プレスリリース |
| `annual_report` | 年次報告書 |
| `official_video` | 公式YouTubeチャンネル等 |

---

### 2.3 verification_confidence（検証確信度）

**目的**: 検証の確実性レベルを示す

| 値 | 定義 | 判定基準 |
|----|------|----------|
| `high` | 高確信度 | 一次ソースで直接確認、複数独立ソースで裏付け |
| `medium` | 中確信度 | 二次ソースで確認、単一だが信頼性の高いソース |
| `low` | 低確信度 | 間接的な確認のみ、情報の新鮮さに懸念 |
| `none` | 確認不能 | ソースが見つからない、アクセス不能 |

#### outcome_statusとの整合性マトリクス

| outcome_status | 許容されるverification_confidence |
|----------------|-----------------------------------|
| `verified_correct` | high, medium |
| `verified_incorrect` | high, medium |
| `unverified` | low, none |

**禁止される組み合わせ**:
- `verified_correct` + `none`（検証済みなのに確認不能は矛盾）
- `verified_correct` + `low`（高確信度なしに正しいと言い切れない）

---

### 2.4 coi_status（利益相反: Conflict of Interest）

**目的**: ソースの利益相反状態を分離して評価する

| 値 | 定義 | 判定基準 | 信頼への影響 |
|----|------|----------|--------------|
| `none` | なし | 独立した第三者による情報 | 影響なし |
| `self` | 自己報告 | 当事者が自組織について発信 | **事実主張は可、評価主張は要裏取り** |
| `affiliated` | 関連組織 | 親会社・子会社・提携先による発信 | 評価主張は要裏取り |
| `unknown` | 判定不能 | 著者の所属が不明 | 慎重に扱う |

#### 重要な設計思想（Codex批評対応）

従来の設計では、企業公式サイトを「tier4_corporate → 高信頼」として単軸評価していた。
これはCodex批評で指摘された「利益相反を無視している」問題である。

**新設計**: COIを独立軸として分離

```
企業公式IR資料の例:
  source_role = primary_source（一次情報源として有用）
  coi_status = self（利益相反あり）

  → 結論: 事実主張（売上、設立日等）は信頼可
         評価主張（業界最高、革新的等）は第三者裏取り必要
```

#### 主張タイプ別の扱い

| 主張タイプ | coi_status=self の場合の扱い |
|------------|------------------------------|
| 事実主張（数値、日付、イベント） | そのまま採用可 |
| 評価主張（最高、革新的、成功） | 二次ソースで裏取り必須 |
| 比較主張（業界1位、競合より優れている） | 二次ソースで裏取り必須 |

---

## 3. フィールド間の関係図

```
┌─────────────────────────────────────────────────────────────────┐
│                     事例（Case）                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐     影響      ┌───────────────────┐           │
│   │ source_role │─────────────→│ verification_     │           │
│   │             │               │ confidence        │           │
│   └─────────────┘               └───────────────────┘           │
│         │                              │                         │
│         │ 独立                         │ 制約                    │
│         ↓                              ↓                         │
│   ┌─────────────┐               ┌───────────────────┐           │
│   │ coi_status  │               │ outcome_status    │           │
│   │ (利益相反)   │               │ (検証状態)        │           │
│   └─────────────┘               └───────────────────┘           │
│         │                              │                         │
│         │                              │ 決定                    │
│         │                              ↓                         │
│         │                       ┌───────────────────┐           │
│         └──────修飾────────────→│ outcome           │           │
│           (評価主張の             │ (結果)            │           │
│            信頼性に影響)          └───────────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

【関係の説明】
1. source_role → verification_confidence
   - primary_source → high/medium が期待される
   - context_only → low が期待される
   - rejected → none が確定

2. verification_confidence → outcome_status
   - high/medium → verified_correct/verified_incorrect が可能
   - low/none → unverified のみ

3. coi_status → outcome（評価主張の修飾）
   - self/affiliated + 評価主張 → 第三者裏取り必要
```

---

## 4. 判定フローチャート

### 4.1 完全な判定フロー

```
[事例入力]
     │
     ▼
┌────────────────────────────────────────────┐
│ STEP 1: ソース役割の判定                     │
│ 各source_urlに対してsource_roleを判定       │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ STEP 2: 利益相反の判定                       │
│ 各ソースに対してcoi_statusを判定            │
└────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ STEP 3: 検証の実施                           │
│ - primary/secondaryソースを開く             │
│ - 事例のoutcomeを確認できるか？             │
└────────────────────────────────────────────┘
     │
     ├── 確認可能 & 一致 ──────────→ [A]
     │
     ├── 確認可能 & 不一致 ─────────→ [B]
     │
     └── 確認不能 ──────────────────→ [C]

[A] verified_correct
    - verification_confidence: ソースに応じてhigh/medium
    - outcome: 既存値を維持

[B] verified_incorrect
    - verification_confidence: ソースに応じてhigh/medium
    - error_type: minor_error / major_error / critical_error を記録
    - outcome: 修正推奨

[C] unverified
    - verification_confidence: low/none
    - unverified_reason: 該当する理由コード
    - outcome: Unknown 推奨（統計から除外）
```

### 4.2 error_type（誤りの重大度）- verified_incorrect時のみ使用

| 値 | 定義 | 例 |
|----|------|-----|
| `minor_error` | 軽微な誤り | 年度が1年ずれている、金額の桁が1つ違う |
| `major_error` | 重大な誤り | outcome判定が逆（Success→Failure）、主体が別組織 |
| `critical_error` | 致命的な誤り | 事例自体が架空、ソースが完全にねつ造 |

#### Codex批評対応: correct/incorrectとerror_typeの関係整理

```
【旧設計の問題】
outcome = Success + minor_error という組み合わせが存在
→ 「成功だが軽微な誤りあり」は何を意味するのか曖昧

【新設計】
outcome_status と error_type を分離

verified_correct:
  - 誤りなし（error_type = なし）

verified_incorrect:
  - error_type = minor_error/major_error/critical_error
  - 誤りの内容を error_description に記録
  - 修正後のoutcomeを suggested_outcome に記録
```

---

## 5. 境界ケース例

### 5.1 ケース1: 企業IR + 評価主張

**事例**: 「トヨタは2023年に世界最高の自動車メーカーになった」

**ソース**: https://toyota.co.jp/ir/annual-report.pdf

**判定**:
```yaml
source_role: primary_source
coi_status: self
verification_confidence: medium

# 分解して判定
claim_1: "トヨタの2023年売上高は○○円"
  → 事実主張、IR資料で確認可 → verified_correct

claim_2: "世界最高の自動車メーカー"
  → 評価主張、自己報告 → 要第三者裏取り
  → 第三者ソースなし → この部分はunverified

# 事例全体の判定
outcome_status: unverified (評価主張部分が未検証のため)
unverified_reason: ambiguous
```

**教訓**: 事実主張と評価主張を分離して判定する

---

### 5.2 ケース2: Wikipedia経由の検証

**事例**: 「エンロン破綻（2001年）」

**ソース**: https://ja.wikipedia.org/wiki/エンロン

**判定**:
```yaml
# Wikipediaの役割
source_role: pointer_only
coi_status: none

# 引用降下（reference descent）
wikipedia_references:
  - SEC filing (primary_source)
  - WSJ article (secondary_source)
  - NYT article (secondary_source)

# 引用元を確認
descended_verification:
  SEC_filing: 破産申請の事実確認 → verified
  WSJ_article: 時系列の裏付け → verified

# 最終判定
outcome_status: verified_correct
verification_confidence: high (一次ソースで確認)
source_role: pointer_only → primary_source への降下で解決
```

**教訓**: pointer_onlyは引用元に降下して検証する

---

### 5.3 ケース3: 進行中の事例

**事例**: 「OpenAI社の企業変革（2024年〜）」

**ソース**: 複数のニュース記事

**判定**:
```yaml
source_role: secondary_source
coi_status: none (第三者報道)
verification_confidence: medium

# 結果の確認
outcome_claim: "変革成功"
actual_status: 進行中、結果未確定

# 最終判定
outcome_status: unverified
unverified_reason: ongoing
outcome: Unknown (Success/Failureの判定を保留)
```

**教訓**: 進行中の事例はunverified + ongoingとし、統計から除外

---

### 5.4 ケース4: ソース消失

**事例**: 「スタートアップX社の急成長（2020年）」

**ソース**: https://techcrunch.com/2020/xxx (404 Not Found)

**判定**:
```yaml
source_role: secondary_source (元は報道記事)
coi_status: none

# アクセス試行
access_result: 404 Not Found
archive_search: archive.org でも見つからず

# 最終判定
outcome_status: unverified
unverified_reason: source_unavailable
verification_confidence: none

# 推奨アクション
recommended_action: 代替ソースの探索、または事例削除検討
```

**教訓**: ソース消失は自動的にunverified、代替ソース探索を推奨

---

## 6. 廃止フィールド

以下のフィールドは本定義書で**廃止**とする。使用しないこと。

### 6.1 廃止: trust_level

| 旧フィールド | 廃止理由 | 代替 |
|--------------|----------|------|
| `trust_level: verified` | outcome_statusと重複 | `outcome_status` |
| `trust_level: plausible` | 曖昧な中間状態 | `verification_confidence: medium` |
| `trust_level: unverified` | outcome_statusと重複 | `outcome_status: unverified` |

### 6.2 廃止: credibility_rank (S/A/B/C)

| 旧フィールド | 廃止理由 | 代替 |
|--------------|----------|------|
| `credibility_rank: S` | ドメイン単位の評価は不適切 | `source_role` + `coi_status` |
| `credibility_rank: A/B/C` | 単軸評価の限界 | 二軸分離評価 |

### 6.3 廃止: tier (gold/silver/bronze/quarantine)

| 旧フィールド | 廃止理由 | 代替 |
|--------------|----------|------|
| `tier: gold` | 件数KPIと結合して品質低下を招く | `outcome_status: verified_correct` |
| `tier: silver` | 中間状態が曖昧 | `verification_confidence: medium` |
| `tier: bronze` | 中間状態が曖昧 | `verification_confidence: low` |
| `tier: quarantine` | 処理待ちの意味が不明確 | `source_role: rejected` または削除 |

### 6.4 移行マッピング

```python
# 旧 → 新 の変換ルール
def migrate_to_v1(case):
    # trust_level → outcome_status
    if case.get('trust_level') == 'verified':
        case['outcome_status'] = 'verified_correct'  # 要手動確認
    elif case.get('trust_level') == 'plausible':
        case['outcome_status'] = 'unverified'
        case['verification_confidence'] = 'medium'
    else:
        case['outcome_status'] = 'unverified'
        case['verification_confidence'] = 'none'

    # credibility_rank → source_role + coi_status
    # 手動判定が必要（ドメインだけでは判定不能）

    # tier → 廃止（outcome_statusに統合）

    # 廃止フィールドを削除
    for field in ['trust_level', 'credibility_rank', 'tier']:
        case.pop(field, None)

    return case
```

---

## 7. 実装チェックリスト

### 7.1 アノテーター向けチェックリスト

- [ ] source_roleを全ソースに対して判定したか
- [ ] coi_statusを全ソースに対して判定したか
- [ ] pointer_only（Wikipedia等）は引用降下を試みたか
- [ ] outcome_statusとverification_confidenceの組み合わせは整合しているか
- [ ] verified_incorrectの場合、error_typeを記録したか
- [ ] unverifiedの場合、unverified_reasonを記録したか

### 7.2 自動検証ルール

```python
def validate_case(case):
    errors = []

    # 整合性チェック1: outcome_status と verification_confidence
    if case['outcome_status'] == 'verified_correct':
        if case['verification_confidence'] not in ['high', 'medium']:
            errors.append("verified_correct requires high/medium confidence")

    if case['outcome_status'] == 'unverified':
        if case['verification_confidence'] not in ['low', 'none']:
            errors.append("unverified requires low/none confidence")

    # 整合性チェック2: unverified には理由が必要
    if case['outcome_status'] == 'unverified':
        if not case.get('unverified_reason'):
            errors.append("unverified requires unverified_reason")

    # 整合性チェック3: verified_incorrect には error_type が必要
    if case['outcome_status'] == 'verified_incorrect':
        if not case.get('error_type'):
            errors.append("verified_incorrect requires error_type")

    # 整合性チェック4: rejected ソースのみの場合
    if all(s.get('source_role') == 'rejected' for s in case.get('sources', [])):
        if case['outcome_status'] != 'unverified':
            errors.append("rejected-only sources should be unverified")

    return errors
```

---

## 8. バージョン履歴

| バージョン | 日付 | 変更内容 |
|------------|------|----------|
| v1.0 | 2026-01-13 | 初版作成、Codex批評対応 |

---

## 9. 参考文献

- Codex批評（2026-01-13）: `debate/llm-debate-20260113-品質改善アプローチの妥当性/codex-response.md`
- Phase 0実装: `scripts/quality/phase0_outcome_tristate.py`
- Phase 1実装: `scripts/quality/phase1_source_roles.py`
- Claim Schema v1: `data/schema/claim_schema_v1.json`
