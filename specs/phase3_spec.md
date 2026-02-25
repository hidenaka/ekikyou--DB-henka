# Phase 3 仕様書: 同型性検証

## 目的

Phase 1（64卦の状態空間モデル Q6）とPhase 2（13,060件の変化事例から導出された変化パターン構造）の間に、統計的に有意な構造的同型性が存在するかを検証する。

**帰無仮説 H0**: 実データの変化パターン構造と64卦のQ6超立方体構造の間に、ランダムに生成可能な構造以上の対応関係は存在しない。

## 入力データ

### Phase 1 成果物
- `analysis/phase1/graph_analysis.json` — Q6の数学的性質
- `analysis/phase1/state_space_model.py` — HexagramStateSpaceクラス（ハミング距離、構造的関係メソッド）

### Phase 2 成果物
- `analysis/phase2/mca_results.json` — MCA結果（5次元、固有値、寄与率）
- `analysis/phase2/dimension_report.json` — 次元導出の詳細
- `analysis/phase2/transition_stats.json` — 遷移パターン統計（chi2, 調整残差、Cramer's V）
- `analysis/phase2/cluster_results.json` — クラスタリング結果（k=2）
- `analysis/phase2/report.md` — Phase 2統合レポート

### 生データ（Read Only）
- `data/raw/cases.jsonl` — 13,060件の変化事例

## Phase 2からの引継ぎ事項（重要）

1. **マルコフ連鎖分析は撤回済み**: before_state(7状態)とafter_state(8状態)の状態空間不整合により無効。Phase 3では使用不可
2. **k=2クラスタリングは頻度構造の反射**: 意味的クラスタリングではない。そのまま使わず、独自の分析アプローチを設計すること
3. **MCA列座標の不安定性**: 元データ(88cat)とクリーンデータ(51cat)のDim1-2相関がr≈0.25。MCA空間の信頼性に限界あり
4. **信頼できるPhase 2結果**: カイ二乗検定(chi2=3215, p=0.0)、調整残差、確率遷移行列、Cramer's V(before_hex=0.318, after_hex=0.3825)

## 検証項目

### 検証1: 次元数の対応
- Q6の幾何学的次元: 6
- MCAのスクリー次元: 5、Benzecri累積80%次元: 6
- 問い: MCAの次元数とQ6の6次元に有意な対応があるか？
- 手法: ランダム置換テスト — カテゴリラベルをランダムに再割り当てした場合のMCA次元数の分布を生成し、実データの5次元がこの分布の下でどの位置にあるかを評価

### 検証2: 遷移パターンとQ6エッジの対応
- Q6: 64卦間のハミング距離1の遷移（192エッジ）
- 実データ: before_hex × after_hex の遷移頻度
- 問い: 実データで頻度の高い遷移は、Q6上で近いノード間（ハミング距離が小さい）の遷移に対応するか？
- 手法:
  a. 各事例のbefore_hex, trigger_hex, action_hex, after_hexから遷移の八卦ペアを特定
  b. 各ペアに対応する卦を特定し、Q6上のハミング距離を計算
  c. 実データの平均ハミング距離と、ランダム遷移モデルの平均ハミング距離を比較（置換テスト）

### 検証3: 八卦タグとクラスタの対応（Cramer's V の有意性）
- Phase 2のCramer's V: before_hex=0.318, after_hex=0.3825
- 問い: このCramer's Vの値はランダムに期待される値より有意に大きいか？
- 手法: 置換テスト — 八卦タグをランダムに再割り当てした場合のCramer's V分布を生成し、実データの値のp値を計算

### 検証4: スペクトル構造の比較
- Q6のスペクトル: 固有値 {-6,-4,-2,0,2,4,6}、多重度 C(6,k)
- MCAのスペクトル: 30固有値（寄与率 5.16%〜1.23%）
- 問い: MCAの固有値分布とQ6のスペクトルに構造的類似性はあるか？
- 手法: スペクトル正規化後の分布比較（KL divergence or Wasserstein distance）

### 検証5: 構造的関係の保存
- Q6の構造: 錯卦（cuogua, 全爻反転）、綜卦（zonggua, 上下反転）
- 実データ: 錯卦ペア間の遷移パターンに対称性があるか？
- 問い: 錯卦ペアにおけるbefore_state分布は、非錯卦ペアよりも類似しているか？
- 手法: 32個の錯卦ペアと32個のランダムペアでbefore_state分布のコサイン類似度を比較

### 検証6: 部分同型性テスト
- 完全同型（Q6と変化空間が同型）は期待できない場合
- 問い: Q6の部分構造（サブキューブ、コミュニティ）が、実データの変化パターンの部分構造と対応するか？
- 手法:
  a. Q6のLouvainコミュニティ（5個）と、実データのクラスタの対応をテスト
  b. Q6のサブキューブ構造が、特定のパターンタイプに対応するかを検証

## 統計的基準

### 有意水準
- 個別検定: α = 0.05
- 多重比較補正: Bonferroni法（6検証 → α' = 0.05/6 = 0.00833）

### 効果量の報告
- 全検定で効果量を報告: Cohen's d、Cramer's V、η²、または適切な効果量指標
- p値のみでの判断を禁止 — 効果量とのセットで解釈

### ランダムベースライン
- 置換テスト: 最低1,000回の置換（精度が必要な場合は10,000回）
- 乱数シード: 42で固定

## 同型性レベルの判定基準

| レベル | 条件 |
|--------|------|
| **完全同型** | 6検証中5つ以上でH0棄却（Bonferroni補正後）、かつ主要効果量が「大」以上 |
| **部分同型** | 6検証中3-4つでH0棄却、または効果量が「中」程度 |
| **類似構造** | 6検証中1-2つでH0棄却、効果量は「小」〜「中」 |
| **同型性なし** | H0を棄却できない、または効果量が「小」未満 |

## 成果物

### `analysis/phase3/isomorphism_test.py`
- 全6検証を実装するPythonスクリプト
- 置換テストの実装
- 結果の出力

### `analysis/phase3/statistical_tests.json`
```json
{
  "tests": [
    {
      "name": "dimension_correspondence",
      "null_hypothesis": "...",
      "test_statistic": ...,
      "p_value": ...,
      "p_value_bonferroni": ...,
      "effect_size": ...,
      "effect_size_type": "...",
      "conclusion": "reject/fail_to_reject",
      "details": {}
    }
  ],
  "overall_conclusion": {
    "isomorphism_level": "complete/partial/analogous/none",
    "significant_tests": ...,
    "total_tests": 6,
    "summary": "..."
  },
  "random_seed": 42,
  "n_permutations": 1000
}
```

### `analysis/phase3/report.md`
- 各検証の結果と解釈
- 総合的な同型性判定
- 限界と今後の課題

## NOT Allowed
- Phase 1/Phase 2の成果物を改変しない
- データを再分析しない（既存の分析結果のみ使用、ただし検証のための独自計算は可）
- 「同型性がある」という結論に誘導しない
- 撤回済みのマルコフ連鎖結果を使用しない
- 予測モデルを構築しない

## 検証基準（quality-reviewer向け）
- [ ] 帰無仮説が明示されている
- [ ] p値と効果量の両方が報告されている
- [ ] 多重比較の補正（Bonferroni）が適用されている
- [ ] ランダムベースラインが適切に構築されている
- [ ] 結論が統計的根拠に基づいている（主観的判断を排除）
- [ ] 同型性レベルの判定が定義に従っている
