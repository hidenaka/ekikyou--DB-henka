# エッジケース（自営業・家族経営等）の特徴量調査レポート

**日付**: 2026-03-04
**背景**: LLMディベート Codex指摘 — 自営業・家族経営・小規模事業は「個人/家族/企業」の境界が曖昧で、単純な離散scaleだけでは分類が破綻する

---

## 1. 既存データのフィールド構造と充填率

### 1.1 スキーマフィールド一覧（充填率付き）

| フィールド | 充填率 | 備考 |
|-----------|--------|------|
| scale | 100.0% | 5値enum: company/individual/family/country/other |
| entity_type | 100.0% | 4値: company/individual/government/organization |
| subject_type | 100.0% | 3値: company/individual/policy |
| main_domain | 79.9% | 自由テキスト（正規化不完全、50+種類） |
| story_summary | 100.0% | テキスト。エッジケース検出の主要ソース |
| free_tags | 96.7% | タグ配列。キーワード検出に利用可能 |
| period | 100.0% | YYYY-YYYY形式が89.6%。期間計算可能 |
| before_state | 100.0% | 6値enum |
| trigger_type | 100.0% | 4値enum |
| action_type | 100.0% | 8値enum |
| after_state | 100.0% | 6値enum |
| outcome | 100.0% | 4値enum |
| success_level | 97.5% | 数値 (0-100) |
| country | 86.3% | 国名テキスト |
| sources | 81.8% | URL配列 |
| trust_level | 73.4% | verified/unverified |
| credibility_rank | 100.0% | S/A/B/C |
| logic_memo | 79.7% | テキスト |
| **sub_domain** | **0.0%** | **未使用フィールド** |
| **action_detail** | **0.0%** | **未使用フィールド** |

### 1.2 現在のscale分布

| scale | 件数 | 割合 |
|-------|------|------|
| company | 5,510 | 42.2% |
| individual | 3,217 | 24.6% |
| other | 2,165 | 16.6% |
| country | 1,381 | 10.6% |
| family | 787 | 6.0% |

### 1.3 entity_type分布

| entity_type | 件数 | 割合 |
|-------------|------|------|
| company | 7,610 | 58.3% |
| individual | 3,984 | 30.5% |
| government | 1,407 | 10.8% |
| organization | 59 | 0.5% |

---

## 2. scale境界が曖昧な事例の特定

### 2.1 検出方法

story_summary, target_name, logic_memo, free_tags内の以下のキーワードでテキストマイニング:
- 日本語: 自営業, 家族経営, 個人事業, フリーランス, 小規模, 零細, 個人商店, 家業, 同族, 一人, ファミリー, オーナー, 夫婦, 親子, 兄弟, 世襲, 後継, 町工場, 老舗, 商店街, スタートアップ, 起業家
- 英語: self-employed, freelance, family business, small business, sole proprietor, entrepreneur, startup, family-owned, SME, micro

### 2.2 エッジケース件数と分布

**検出件数: 1,070件（全体の8.2%）**

#### scale別分布

| scale | エッジケース数 | そのscale内の割合 |
|-------|---------------|------------------|
| company | 515 | 9.3% |
| individual | 255 | 7.9% |
| family | 169 | 21.5% |
| other | 95 | 4.4% |
| country | 36 | 2.6% |

#### キーワード出現頻度（上位15）

| キーワード | 出現数 | 典型的なscale分類 |
|-----------|--------|-----------------|
| スタートアップ | 224 | company(66%), individual(20%) |
| 老舗 | 214 | company(多数) |
| 後継 | 111 | company/family混在 |
| 一人 | 100 | individual(多数) |
| 夫婦 | 71 | family/individual混在 |
| 起業家 | 70 | individual/company混在 |
| micro | 52 | other(多数) |
| 兄弟 | 51 | family/individual混在 |
| 家族経営 | 45 | company/family混在 |
| フリーランス | 43 | individual(多数) |
| 小規模 | 36 | company/other混在 |
| 親子 | 34 | family(多数) |
| ファミリー | 34 | company/family混在 |
| 商店街 | 22 | other(多数) |
| 家業 | 18 | company/family混在 |

### 2.3 構造的問題の発見

#### 問題1: family scale の entity_type が全件 individual
787件のfamily事例は、全てentity_type=individualに分類されている。family専用のentity_typeが存在しない。

#### 問題2: other scale の大半が company entity
2,165件のother事例のうち、2,108件(97.4%)がentity_type=company。「企業でも個人でもない」が意図なのに、実質的にcompanyの受け皿になっている。

#### 問題3: scale と entity_type の不整合（28件）
- individual として登録されているが entity_type=organization (12件)
- individual として登録されているが entity_type=government (8件)
- company として登録されているが entity_type=government (7件)

#### 問題4: 家族経営・老舗が company に分類
233件の「家族経営」「同族」「老舗」「後継」等のキーワードを含む事例がcompanyに分類されている。これらは個人・家族・企業の境界にまたがる事例。

**具体例**:
- `CORP_JP_245` 中小企業_倒産_01_製造業: 「創業50年の町工場。後継者不在と取引先の海外移転で売上激減」→ scale=company
- `CORP_JP_246` 中小企業_倒産_02_小売業: 「3代目で創業80年の歴史に幕」→ scale=company
- これらは「家族経営の企業」であり、company/family両方の性質を持つ

---

## 3. Codex提案の特徴量の実現可能性評価

### 3.1 評価サマリー

| 特徴量 | 説明 | 既存データから推定可能か | 推定方法 | 精度 |
|--------|------|----------------------|---------|------|
| 当事者数 | 意思決定に関わる人数 | 部分的 | entity_type + テキストマイニング | LOW |
| 資源制約 | 利用可能な資源の規模 | 困難 | success_level(間接) | VERY LOW |
| 時間軸 | 変化に要する期間 | **可能** | period(YYYY-YYYY) → duration算出 | HIGH |
| 可逆性 | 行動の取り消しやすさ | 部分的 | outcome + after_state | MEDIUM |
| 合意形成コスト | 意思決定の複雑さ | 部分的 | action_type分布の偏り | LOW |

### 3.2 各特徴量の詳細分析

#### 当事者数（stakeholder_count）

**現状**: 直接フィールドなし。entity_typeは4値（company/individual/government/organization）で粗すぎる。

**推定手段**:
- story_summary からのキーワード抽出（「一人」→1、「夫婦」→2、「家族」→3-5）
- scale による概算（individual→1、family→2-10、company→10+、country→1000+）

**問題点**: family scale の787件が全てentity_type=individualになっており、「家族としての複数性」が構造的に失われている。

**推奨**: `stakeholder_scale`フィールドを新設。enum: `sole(1人)`, `pair(2人)`, `small_group(3-10人)`, `organization(11-100人)`, `large(100+人)`

#### 資源制約（resource_constraint）

**現状**: 直接フィールドなし。success_level（0-100）が間接的な代理指標。

**既存データからの観察**:
| scale | 平均success_level |
|-------|------------------|
| other | 58.7 |
| individual | 57.7 |
| company | 56.7 |
| country | 48.5 |
| **family** | **47.2** |

family の成功率が最も低い。資源制約の高さを間接的に示唆。

**推奨**: 新規追加が必要。ただし事例データから推定困難。`resource_level`フィールド（enum: `micro`, `small`, `medium`, `large`, `massive`）をLLMバッチ推定で付与する方法が現実的。

#### 時間軸（time_axis / duration）

**現状**: period フィールドから算出可能。11,707件(89.6%)がYYYY-YYYY形式。

**既存データからの観察**:
| scale | 平均duration | 中央値 |
|-------|-------------|--------|
| individual | 19.3年 | — |
| company | 21.9年 | — |
| family | **33.6年** | — |
| country | 33.0年 | — |
| other | 34.4年 | — |

family と country は長期変化（30年超）、individual は短期（19年）。この差は分類の意味差を反映。

**推奨**: `duration_years`フィールドをperiodから自動算出して付与。コスト最小。

#### 可逆性（reversibility）

**現状**: outcome と after_state から間接推定可能。

**既存データからの観察**:
| scale | Failure率 | Success率 |
|-------|----------|----------|
| **family** | **46.1%** | 37.4% |
| country | 39.0% | 33.7% |
| individual | 31.9% | 52.8% |
| company | 30.4% | 48.8% |
| other | 27.3% | 52.3% |

family の Failure 率が突出して高い（46.1%）。家族・個人事業の変化は「不可逆的」になりやすい傾向を示す。

**推奨**: `reversibility_score`を outcome + after_state + action_type の組み合わせから算出するルールベーステーブルを構築。
- 「崩壊・消滅」→ 不可逆(score=1)
- 「V字回復」→ 高可逆(score=5)
- 「縮小安定」→ 中間(score=3)

#### 合意形成コスト（consensus_cost）

**現状**: action_type の分布差がscale間の合意形成コストの違いを間接的に反映。

**既存データからの観察**:
| action_type | company | individual | family | country |
|-------------|---------|-----------|--------|---------|
| 攻める・挑戦 | 25.4% | **31.7%** | 9.7% | 22.4% |
| 対話・融合 | 14.7% | 10.3% | **21.3%** | 8.2% |
| 守る・維持 | 12.0% | 7.9% | **23.5%** | 15.8% |
| 耐える・潜伏 | 12.2% | 18.7% | **23.1%** | 20.7% |
| 刷新・破壊 | 11.8% | 8.3% | 8.5% | **20.9%** |

**解釈**:
- **individual**: 「攻める・挑戦」が最多(31.7%) → 合意形成コスト最低（一人で決められる）
- **family**: 「対話・融合」「守る・維持」「耐える・潜伏」が上位 → 合意形成コスト最高（関係性維持が優先）
- **company**: バランス型。中程度の合意形成コスト
- **country**: 「刷新・破壊」が最多(20.9%) → 合意形成後の変化は大規模

**推奨**: action_type の分布エントロピーを「合意形成コストの代理指標」として使用可能。分布が均一(高エントロピー) = 選択肢が制約されていない。特定action_typeに偏る(低エントロピー) = 構造的制約あり。

---

## 4. 既存フィールドによる代替案と実装推奨

### 4.1 即時実装可能（既存データから自動算出）

| 特徴量 | 実装方法 | コスト | 精度 |
|--------|---------|------|------|
| duration_years | period(YYYY-YYYY)から差分算出 | 極低 | HIGH |
| reversibility_score | outcome + after_state のルールテーブル | 低 | MEDIUM |
| action_entropy | action_type分布のエントロピー算出（scale単位） | 低 | MEDIUM |

### 4.2 LLMバッチ推定で追加可能

| 特徴量 | 実装方法 | コスト | 精度 |
|--------|---------|------|------|
| stakeholder_scale | story_summaryからLLM分類 (5値enum) | 中 | MEDIUM |
| resource_level | story_summary + main_domain からLLM分類 (5値enum) | 中 | LOW-MEDIUM |

### 4.3 未使用フィールドの活用

- **sub_domain** (充填率0%): エッジケースの詳細分類に転用可能。例: `family_business`, `freelance`, `startup`, `sole_proprietor`
- **action_detail** (充填率0%): 合意形成プロセスの記述に転用可能

### 4.4 スキーマ拡張提案

```json
{
  "scale_features": {
    "duration_years": 5,
    "stakeholder_scale": "small_group",
    "resource_level": "small",
    "reversibility_score": 3,
    "consensus_cost_proxy": 0.72,
    "boundary_flags": ["family_business", "founder_led"]
  }
}
```

`boundary_flags`は複数のscale特性をまたぐ事例を明示的にマークするための配列フィールド。

---

## 5. 結論と推奨アクション

### 5.1 主要発見

1. **エッジケースは1,070件(8.2%)** — 無視できない規模
2. **family scale の構造的欠陥**: 全787件がentity_type=individualで、「家族としての複数性」が失われている
3. **other scale の曖昧性**: 2,165件の97.4%がentity_type=companyで、実質的にcompanyの残余カテゴリ
4. **5つの特徴量のうち、時間軸のみ既存データから高精度で算出可能**
5. **family scale は他のscaleと行動パターン・成功率が明確に異なる** — 合意形成コスト高・失敗率高・長期化傾向

### 5.2 推奨実装順序

| 優先度 | アクション | 理由 |
|--------|----------|------|
| P0 | duration_years の自動算出・付与 | コスト最低、精度最高、即実装可 |
| P1 | sub_domain フィールドにエッジケースフラグを付与 | 未使用フィールドの有効活用 |
| P2 | reversibility_score のルールテーブル構築 | outcome/after_stateから機械的に算出可能 |
| P3 | stakeholder_scale のLLMバッチ推定 | 13,060件全件のLLM処理が必要 |
| P4 | entity_type に family を追加するスキーマ改修 | 後方互換性への影響を検討要 |

---

## 付録: 統計データファイル

詳細な統計データは `analysis/phase3/edge_case_features_stats.json` に出力済み。
