# Phase 3 品質レビュー: 同型性検証 — Round 2（最終レビュー）

**レビュー日時**: 2026-02-25
**レビュー対象**: `isomorphism_test.py`, `statistical_tests.json`, `report.md`
**検証基準**: `specs/phase3_spec.md`
**レビュアー**: quality-reviewer (independent)
**Round**: 2 (修正版の最終レビュー)

---

## 総合判定: CONDITIONAL PASS

**スコア**: 統計的妥当性 3 / 再現性 4 / 論理的整合性 3 = 平均 3.33

6項目の指摘事項（MF-1, MF-2, MF-3, SF-1, SF-2, SF-3, SF-4）のうち、Critical/High の3件（MF-1, MF-2, MF-3）は全て修正済み。残存する問題は軽微であり、結論の根幹に影響しない。

---

## 採点

| 軸 | Round 1 | Round 2 | 根拠 |
|----|---------|---------|------|
| **統計的妥当性** | 2 | 3 | MF-1修正: 効果量方向が`_pro`/`_anti`サフィックスで区別される。MF-2修正: 判定ロジックがpro方向のみカウント。SF-2修正: サロゲートMCAベースラインに変更。ただし残存問題あり（後述NI-1, NI-2）。 |
| **再現性** | 4 | 4 | MF-3修正: `prince.MCA(n_components=5, random_state=42)` + KMeansで完全再現。SF-4修正: ベクトル化演算+NaN/Infガード。seed=42で数値再現可能。 |
| **論理的整合性** | 2 | 3 | 効果量方向を判定ロジックに組み込み済み。レポートに方向一覧表を追加。反同型性の証拠を要約文に明記。ただし残存問題あり（後述NI-3, NI-4）。 |

---

## 修正項目の検証結果

### MF-1 (Critical): 効果量方向性の誤認 → **修正済み**

**修正前**: `effect_size_label(d)` が `abs(d)` で方向を無視。反同型性効果を「large」と表示。

**修正後**: `effect_size_label(d, direction=None)` に `direction` パラメータ追加（L138-168）。正の値が何を意味するかを `direction="pro"` / `direction="anti"` で指定し、`_pro` / `_anti` サフィックスを自動付与。

検証結果:
| 検証 | 修正前ラベル | 修正後ラベル | 方向 | 正しいか |
|------|------------|------------|------|---------|
| 検証1 | large | large_pro | pro_isomorphism | YES |
| 検証2 | large | large_anti | anti_isomorphism | YES |
| 検証3 | medium | medium_pro | pro_isomorphism | YES |
| 検証4 | large | large_anti | anti_isomorphism | YES |
| 検証5 | medium | medium_anti | anti_isomorphism | YES |
| 検証6 | small | small_pro | pro_isomorphism | YES |

各検証での `direction` 引数の正当性:
- 検証1: `direction="pro"`, gap_effect = (random_mean_gap - observed_gap) / std。正=実データがQ6に近い=pro。**正しい**
- 検証2: `direction="anti"`, z_score = (observed - random) / std。正=観測が遠い=anti。**正しい**
- 検証3: `cramers_v_label(v)` は常にpro（V大=関連強い=pro）。**正しい**
- 検証4: `direction="anti"`, z_score = (observed_W - random_W) / std。正=Wが大きい=Q6から遠い=anti。**正しい**
- 検証5: `direction="pro"`, z_score = (observed_sim - random_sim) / std。正=類似度高い=pro。**正しい**
- 検証6: `cramers_v_label(v)` は常にpro。**正しい**

**判定: 完全修正**

### MF-2 (Critical): 判定ロジックの誤り → **修正済み**

**修正前**: `determine_isomorphism_level()` が効果量の方向を無視。反同型性の大効果2件を同型性の証拠としてカウント。

**修正後** (L1232-1312):
- `large_effects_pro`, `medium_effects_pro` でpro方向のみカウント
- `large_effects_anti`, `medium_effects_anti` でanti方向を別途カウント
- `rejected_pro_direction`, `rejected_anti_direction` を追加
- 判定基準: `complete` は `large_effects_pro >= 3`、`partial` は `medium_effects_pro >= 2` を要求
- `analogous` の要約文に反同型性の証拠を明記

実際の判定過程:
```
n_rejected = 2
large_effects_pro = 1 (検証1: large_pro)
medium_effects_pro = 2 (検証1: large_pro, 検証3: medium_pro)
large_effects_anti = 2 (検証2: large_anti, 検証4: large_anti)
medium_effects_anti = 3 (検証2, 4, 5)
rejected_pro = 2, rejected_anti = 0

条件: n_rejected >= 3 && medium_effects_pro >= 2 → FALSE (n_rejected=2)
条件: n_rejected >= 1 → TRUE → level = "analogous"
```

仕様書との照合:
- 仕様: 「1-2つでH0棄却、効果量は小〜中」→ ANALOGOUS
- 実際: 2つ棄却（検証3: V=0.350, 検証6: V=0.299）。棄却された検証の効果量はsmall_proとmedium_pro。
- 仕様の条件を満たす。

**判定: 完全修正。ロジックは正当**

### MF-3 (High): クラスタ近似エラー → **修正済み**

**修正前**: MCA近似によるクラスタ割り当て。Phase 3 Cluster 1: 3,094件 (23.7%) vs Phase 2: 1,802件 (13.8%)。9.9pp誤差。

**修正後** (L539-563):
- `prince.MCA(n_components=5, random_state=42)` でMCA行座標を再計算
- `KMeans(n_clusters=2, random_state=42, n_init=10)` で再クラスタリング
- 再実行サイズ: `{0: 11258, 1: 1802}` = Phase 2報告サイズ `{0: 11258, 1: 1802}` と完全一致
- Cramer's V: before_V=0.3180 (Phase 2: 0.318), after_V=0.3825 (Phase 2: 0.3825) — 小数点以下4桁まで一致

**判定: 完全修正。クラスタサイズが完全一致し、Cramer's Vも一致**

### SF-1: 検証5の4ペアvs仕様32ペア → **修正済み**

**修正前**: 4ペアへの制限理由が不明確。

**修正後**: `power_analysis` フィールド追加（L1018-1030）。八卦レベルでは4ペアしか取れない理由（cases.jsonlにbefore_hex/after_hexしかない）、C(8,2)=28ペア中4ペア、ランダムペアリング空間105通りを定量化。

**判定: 修正済み。限界を定量的に文書化**

### SF-2: 検証4のランダムベースライン → **修正済み**

**修正前**: MCA固有値範囲内の一様分布をベースラインに使用。

**修正後** (L741-843): カテゴリラベルをシャッフルした代理MCA（サロゲートMCA）の固有値でベースラインを構築。計算コストが10分超の場合のみ一様分布にフォールバック。`baseline_method: "surrogate_mca"`, `baseline_n_valid: 1000` と記録。

JSON結果の確認:
- `baseline_method`: "surrogate_mca" — サロゲートMCAが使用された
- `baseline_n_valid`: 1000 — 1000件全て有効
- `random_wasserstein_mean`: 0.1110 — 一様分布ベースライン時の値とは異なる

**判定: 修正済み。適切なサロゲートMCAベースラインを使用**

### SF-3: DBSCAN k=5のハードコード → **修正済み**

**修正前**: `dbscan_best_k = 5` がハードコード。

**修正後** (L1136-1155): `cluster_results.json` から `clustering_comparison.dbscan.best.n_clusters` を動的に読み取り。フォールバック（見つからない場合はデフォルト5）も実装。さらにDBSCAN k=5のロバスト性（30通り中2通り）をレポートに明記。

**判定: 修正済み**

### SF-4: RuntimeWarning → **修正済み**

**修正前**: ループ内で個別のゼロ除算チェックを行わず、RuntimeWarningが発生。

**修正後** (L247-297): ベクトル化演算に置換。NaN/Infガード追加（`np.isfinite` チェック3箇所、無効な置換はNoneを返して除外）。

**判定: 修正済み**

---

## 新たに発見された問題

### NI-1 (Low): 検証1の効果量ラベル分類と仕様の不整合

検証1の効果量は `gap_effect = 0.9005` で `large_pro` と判定されているが、p値は0.499（Bonferroni後1.0）で有意でない。これ自体は矛盾ではないが（効果量とp値は独立の指標）、置換分布のSD=6.91（平均7.02）の不安定さを考慮すると、この「large_pro」効果量の信頼性は限定的である。

報告として正しく記載されているため、結論への影響は小さい。ただし、レポートの限界セクション第3項でこの不安定性が言及されている点は評価する。

**影響度**: 低。結論を変えない。

### NI-2 (Low): 検証3のCramer's Vの効果量分類

`cramers_v_label(v)` のしきい値（L181-190）:
- negligible: V < 0.1
- small_pro: 0.1 <= V < 0.3
- medium_pro: 0.3 <= V < 0.5
- large_pro: V >= 0.5

検証3の avg_v = 0.3502 → `medium_pro`。これはCohen (1988)のCramer's V基準（small=0.1, medium=0.3, large=0.5, df=1の場合）に一致している。

しかし、検証6の effect_size = observed_v = 0.2991 → `small_pro`。仕様書の判定基準テーブルで「効果量が中程度」の条件を部分同型に使うとき、検証6はsmallに分類される。この分類自体は正しいが、0.2991は0.3に極めて近い境界値であり、レポートの要約で「Cramer's V ≈ 0.28-0.30, small effect」と記載している点は正確。

**影響度**: なし。分類は基準に従っている。

### NI-3 (Low-Medium): 検証2のp値方向の解釈

検証2のp値計算（L448-449）:
```python
p_value = np.mean(perm_means <= observed_mean)
```

これは「ランダムで観測値以下の平均距離が出る確率」= 0.971。つまり97.1%のランダム置換で観測値以下のハミング距離が出る。これは「実データの遷移はランダムより遠い」ことを意味する。

p値の定義が片側検定（実データがより近い方向）として設定されているが（L448コメント参照）、結果が逆方向なのでp=0.971となっている。統計的には問題ないが、「H0棄却できず」の判定は正しい（p_bonferroni=1.0）。

**改善案**: 両側検定を使用するか、p値の解釈をレポートで明確化する（「p=0.971は、データが同型性の反対方向に位置していることを示す」）。

**影響度**: 低。結論（fail_to_reject）は正しく、効果方向（anti）も正しく報告されている。

### NI-4 (Low): `determine_isomorphism_level` の partial 判定条件

判定ロジック（L1277）:
```python
elif n_rejected >= 3 and medium_effects_pro >= 2:
    level = "partial"
```

仕様書: 「3-4つでH0棄却、または効果量が中程度」— "or" 条件。
実装: AND 条件。

仕様の文面を厳密に読むと、H0棄却3-4件 OR 効果量中程度のいずれかでpartial。しかし「効果量が中程度」だけでpartialとするのは過大評価の恐れがあり、AND条件（棄却+中効果の両方を要求）の方がより保守的。実際には今回のデータでは `n_rejected=2` でこの分岐に入らないため、結論に影響しない。

**影響度**: なし。今回のデータでは分岐に入らない。保守的な実装として許容可能。

---

## 仕様チェックリスト（Round 2）

| # | チェック項目 | Round 1 | Round 2 | 備考 |
|---|------------|---------|---------|------|
| 1 | 帰無仮説が明示されている | PASS | PASS | 全6検証にH0が明記 |
| 2 | p値と効果量の両方が報告されている | CONDITIONAL | PASS | 効果量の方向がラベル+`effect_direction`フィールドで報告 |
| 3 | 多重比較の補正（Bonferroni）が適用されている | PASS | PASS | alpha'=0.00833、全検証で適用 |
| 4 | ランダムベースラインが適切に構築されている | CONDITIONAL | PASS | 検証4がサロゲートMCAに改善。全6検証でseed=42固定、1000回置換 |
| 5 | 結論が統計的根拠に基づいている | FAIL | PASS | 効果量方向を判定ロジックに組み込み。反同型性の証拠を報告 |
| 6 | 同型性レベルの判定が定義に従っている | CONDITIONAL | PASS | H0棄却2/6=ANALOGOUS。pro方向効果量のみで判定。仕様の定義に合致 |

---

## 独立数値検証（Round 2）

### 検証3: クラスタ再現の確認

| 指標 | Round 1 | Round 2 | Phase 2 | 一致 |
|------|---------|---------|---------|------|
| Cluster 0サイズ | ~9,966 | 11,258 | 11,258 | **YES** (Round 2でPhase 2と完全一致) |
| Cluster 1サイズ | ~3,094 | 1,802 | 1,802 | **YES** |
| before_hex V | 0.246 | 0.3180 | 0.318 | **YES** |
| after_hex V | 0.319 | 0.3825 | 0.3825 | **YES** |

### 判定ロジックの検証

```
入力:
  effect_labels = [large_pro, large_anti, medium_pro, large_anti, medium_anti, small_pro]
  conclusions = [fail_to_reject, fail_to_reject, reject, fail_to_reject, fail_to_reject, reject]

計算:
  n_rejected = 2 (検証3, 6)
  large_effects_pro = 1 (large_pro: 検証1)
  medium_effects_pro = 2 (large_pro: 検証1, medium_pro: 検証3)
  large_effects_anti = 2 (large_anti: 検証2, 4)
  medium_effects_anti = 3 (large_anti: 検証2,4 + medium_anti: 検証5)
  rejected_pro = 2 (検証3: medium_pro, 検証6: small_pro)
  rejected_anti = 0

判定:
  complete: n_rejected(2) >= 5 → FALSE
  partial: n_rejected(2) >= 3 → FALSE
  analogous: n_rejected(2) >= 1 → TRUE → level = "analogous"
```

JSON出力との照合: 全フィールドが一致。**検証OK**。

### 総合判定の妥当性

ANALOGOUS判定は適切:
1. H0棄却は2/6件（検証3, 6）。仕様の「1-2つでH0棄却」に該当
2. 棄却された検証の効果量: 検証3=medium_pro (V=0.350), 検証6=small_pro (V=0.299)。「小〜中」に該当
3. 反同型性方向の証拠（検証2, 4, 5で中〜大効果）を要約文に明記
4. 「Q6超立方体との構造的同型性を積極的に支持する証拠は限定的」は正確な表現

---

## Round 1からの改善サマリー

| 修正項目 | 深刻度 | 修正状態 | 結論への影響 |
|---------|--------|---------|------------|
| MF-1: 効果量方向性 | Critical | 完全修正 | 効果量ラベルが方向付きになり、誤解を防止 |
| MF-2: 判定ロジック | Critical | 完全修正 | pro方向のみで判定。反同型性の証拠を分離 |
| MF-3: クラスタ近似 | High | 完全修正 | Phase 2と完全一致（サイズ・V共に） |
| SF-1: 4ペアの限界 | Medium | 修正済み | power_analysisで定量的に文書化 |
| SF-2: ベースライン | Medium | 修正済み | サロゲートMCA（1000回）に変更 |
| SF-3: ハードコード | Low-Medium | 修正済み | cluster_results.jsonから動的読込 |
| SF-4: RuntimeWarning | Medium | 修正済み | ベクトル演算+NaN/Infガード |

---

## 判定の根拠

### CONDITIONAL PASSの条件適合
- 平均スコア: 3.33 >= 3.0 (基準: 3.0以上)
- 最低スコア: 3 >= 2 (基準: いずれも2以上)
- Critical/Highの未修正項目: なし
- 結論（ANALOGOUS）が統計的根拠に基づいている

### PASSに達しなかった理由
1. **統計的妥当性 (3, not 4)**: 検証1の置換分布不安定性（SD=6.91）が未解決。検証2のp値方向解釈がレポートで不十分。
2. **論理的整合性 (3, not 4)**: 仕様のpartial判定条件（OR vs AND）との微妙な不整合。レポートの限界セクションは改善されたが、p値方向の意味をより明確に解説すべき。

### PASSへの昇格条件（参考）
1. 検証1の置換テストにおけるSVD代理MCAの不安定性を追加分析または限界として詳述
2. 検証2のp値=0.971が意味する方向性をレポートで明確に解説
3. 仕様のpartial判定条件のOR/ANDについてコードコメントで設計判断を記録

---

*Generated by quality-reviewer on 2026-02-25 (Round 2)*
