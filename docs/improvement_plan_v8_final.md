# 六十四卦マッピング改善計画v8（最終版）

## 8回のCodexディベートで確立した設計原則

1. **64卦を同時スコア**（独立積ではなく）
2. **soft label**（複数の妥当卦を認める）
3. **条件付きユーザー関与**（不確実時のみ）
4. **校正された確率**（温度スケーリング+検証）
5. **統合重みは学習で決める**（固定αは禁止）
6. **閾値はrisk-coverageで決める**（恣意的数値は禁止）
7. **注釈指標はKrippendorff's α**（Fleiss kappaは不適切）

## 問題の本質的定義

> この問題は「唯一の正解を当てる分類」ではなく、
> **状況記述 → 卦の妥当度分布を返し、必要なら追加質問で分布を収束させる**問題。
> 目標は「当てる」ではなく**誤確定を抑えつつ納得度を最大化する**。

## アーキテクチャ

```
入力: ビジネス事例テキスト
    ↓
[Stage 1] 64卦同時スコアリング
    - LLM埋め込み → 64クラス分類ヘッド
    - または: 類似検索 → 再ランキング
    - 出力: 64次元ロジットベクトル
    ↓
[Stage 2] スコア統合（学習済み重み）
    - 複数シグナルを学習済みモデルで統合
    - 温度スケーリングで校正
    - 出力: 校正された64次元確率分布
    ↓
[Stage 3] 選択的予測（risk-coverage制御）
    - 目標誤確定率から閾値を決定
    - 確実 → 自動確定（Top-1を返す）
    - 不確実 → ユーザー関与へ
    ↓
[Stage 4] 条件付きユーザー関与（不確実時のみ）
    - Top-k候補提示（k=5-10）
    - 追加質問による絞り込み
    - ユーザー選択
    ↓
[Stage 5] 64卦確定 → 爻判定
    - 確定した卦の爻辞を参照
    - 卦固有の文脈で爻を判定

出力: (hexagram, yao, calibrated_prob, alternatives, reasoning)
```

## Stage 1: 64卦同時スコアリング

### v7からの修正

**v7の矛盾**: Stage 1で20-50候補に絞り、Stage 2で8×8行列（64件）を出力
**v8の修正**: 最初から64卦を同時にスコアする一貫した設計

### 実装選択肢

**Option A: 埋め込み+分類ヘッド（推奨）**
```python
def score_hexagrams_embedding(text):
    """埋め込みベースの64クラス分類"""
    # テキストを埋め込み
    embedding = embed_model.encode(text)  # 768次元等

    # 64クラス分類ヘッド（学習済み）
    logits = classification_head(embedding)  # 64次元

    return logits  # 生のロジット（校正前）
```

**Option B: 類似検索+再ランキング**
```python
def score_hexagrams_retrieval(text):
    """検索ベースのスコアリング"""
    # 類似事例を検索（全件からTop-100）
    similar_cases = search_similar(text, k=100)

    # 卦ごとの出現頻度をスコア化
    retrieval_scores = aggregate_hexagram_counts(similar_cases)

    # LLMで再ランキング（64卦を直接評価）
    rerank_scores = llm_rerank(text, top_candidates=20)

    return retrieval_scores, rerank_scores
```

### 確率の定義を厳密に

**v7の問題**: LLMの「自己申告スコア」を確率と称した
**v8の修正**: 確率はモデル内部のロジットから導出し、校正を経たもののみ

```python
# 禁止: LLMに「0-100で評価して」と聞いたスコア
# 許可: モデルのロジット出力 → softmax → 温度スケーリング
```

## Stage 2: スコア統合（学習済み重み）

### v7からの修正

**v7の問題**: `α=0.7`の固定重みで異なる尺度のスコアを線形結合
**v8の修正**: 統合重みは検証データで学習

### 統合モデル

```python
class ScoreIntegrator(nn.Module):
    """複数シグナルを統合するモデル"""
    def __init__(self, n_signals):
        super().__init__()
        # 各シグナルの重みを学習
        self.signal_weights = nn.Parameter(torch.ones(n_signals))
        # 温度パラメータも学習
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, signals):
        """
        signals: List[Tensor] - 各シグナルの64次元スコア
        """
        # 重み付き統合（softmaxで正規化した重みを使用）
        weights = F.softmax(self.signal_weights, dim=0)
        combined = sum(w * s for w, s in zip(weights, signals))

        # 温度スケーリング
        calibrated = combined / self.temperature

        return calibrated
```

### 学習方法

```python
def train_integrator(integrator, train_data, val_data):
    """統合モデルの学習"""
    optimizer = optim.Adam(integrator.parameters(), lr=0.01)

    # 損失関数: soft labelとのKLダイバージェンス
    def loss_fn(pred_logits, soft_labels):
        pred_probs = F.softmax(pred_logits, dim=-1)
        return F.kl_div(pred_probs.log(), soft_labels, reduction='batchmean')

    for epoch in range(100):
        for batch in train_data:
            signals, soft_labels = batch
            pred = integrator(signals)
            loss = loss_fn(pred, soft_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 検証セットでECEを監視
        val_ece = compute_ece(integrator, val_data)
        print(f"Epoch {epoch}: Val ECE = {val_ece:.4f}")
```

## Stage 3: 選択的予測（risk-coverage制御）

### v7からの修正

**v7の問題**: `entropy>0.7`, `margin<0.1`等の恣意的閾値
**v8の修正**: risk-coverage曲線から目標誤確定率を満たす閾値を決定

### risk-coverage曲線

```python
def compute_risk_coverage_curve(probs, labels, thresholds):
    """
    probs: 予測確率分布 (N, 64)
    labels: soft label (N, 64)
    thresholds: 試行する閾値のリスト
    """
    results = []

    for thresh in thresholds:
        # 閾値以上の確信度を持つ事例のみ自動確定
        max_probs = probs.max(dim=1).values
        confident_mask = max_probs >= thresh

        # カバレッジ: 自動確定した割合
        coverage = confident_mask.float().mean().item()

        # リスク: 自動確定した中での誤り率
        if confident_mask.sum() > 0:
            preds = probs[confident_mask].argmax(dim=1)
            true_top1 = labels[confident_mask].argmax(dim=1)
            risk = (preds != true_top1).float().mean().item()
        else:
            risk = 0.0

        results.append({'threshold': thresh, 'coverage': coverage, 'risk': risk})

    return results

def select_threshold(curve, target_risk=0.2):
    """目標リスク以下で最大カバレッジを達成する閾値を選択"""
    valid = [r for r in curve if r['risk'] <= target_risk]
    if not valid:
        return curve[-1]['threshold']  # 最も保守的な閾値
    return max(valid, key=lambda r: r['coverage'])['threshold']
```

### 運用時の判定

```python
def should_auto_confirm(probs, threshold):
    """自動確定すべきか判定"""
    max_prob = probs.max().item()
    return max_prob >= threshold

def predict_with_selective(model, text, threshold):
    """選択的予測"""
    probs = model(text)

    if should_auto_confirm(probs, threshold):
        return {
            'status': 'auto_confirmed',
            'hexagram': probs.argmax().item(),
            'confidence': probs.max().item(),
            'alternatives': get_top_k(probs, k=3)
        }
    else:
        return {
            'status': 'needs_user_input',
            'candidates': get_top_k(probs, k=10),
            'entropy': compute_entropy(probs)
        }
```

## Stage 4: 条件付きユーザー関与

### 15卦フィルタの扱い

**v7の問題**: 15卦フィルタで正解が漏れる危険
**v8の修正**: 原則禁止、使用する場合は漏れ率定量化+救済経路

```python
def filter_by_essential_trigram(candidates, essential_trigram, all_probs):
    """
    本質八卦フィルタ（救済経路付き）
    """
    filtered = []
    excluded = []

    for hex_id, prob in candidates:
        upper, lower = get_trigrams(hex_id)
        if upper == essential_trigram or lower == essential_trigram:
            filtered.append((hex_id, prob))
        else:
            excluded.append((hex_id, prob))

    # 漏れ率を計算して警告
    excluded_mass = sum(p for _, p in excluded)
    if excluded_mass > 0.3:  # 30%以上の確率質量が除外される
        return {
            'filtered': filtered,
            'warning': f'フィルタにより{excluded_mass:.1%}の候補が除外されます',
            'rescue_option': '全64卦から選択し直す',
            'excluded_top': excluded[:3]
        }

    return {'filtered': filtered, 'warning': None}
```

## ゴールドセット設計（soft label）

### 注釈設計

**v7の問題**: Fleiss kappaは単一カテゴリ名義尺度前提で不適切
**v8の修正**: Krippendorff's α（多ラベル+順序尺度に適合）

```python
def compute_krippendorff_alpha(annotations):
    """
    annotations: Dict[annotator_id, Dict[case_id, Dict[hexagram_id, score]]]
    """
    import krippendorff

    # 注釈を行列形式に変換
    # 各セル = (事例, 卦) の評価値
    matrix = convert_to_matrix(annotations)

    # 順序尺度としてαを計算
    alpha = krippendorff.alpha(matrix, level_of_measurement='ordinal')

    return alpha
```

### 注釈プロトコル

```
1事例に対して3名の注釈者が:
1. 妥当な卦を選択（複数選択可、最大5卦）
2. 各選択卦に妥当度を付与（1-5の順序尺度）
3. 選択理由を記述

soft label生成:
- 3名の評価を正規化して分布に
- 誰も選ばなかった卦は確率0
- 全員が高評価した卦は高確率
```

### 注釈者間一致の基準

| 指標 | 合格基準 | 不合格時の対応 |
|------|----------|----------------|
| Krippendorff's α | ≥ 0.4 | ルーブリック再設計 |
| Top-3一致率 | ≥ 60% | 境界事例の判定基準を追加 |
| 完全不一致率 | ≤ 10% | 問題事例を除外またはルーブリック修正 |

## 評価指標

### v7からの修正

**v7の問題**: 評価指標の選択が不適切
**v8の修正**: soft labelに適合した指標セット

| カテゴリ | 指標 | 定義 | 目標 |
|----------|------|------|------|
| **分布品質** | KL Divergence | モデル分布とsoft labelの距離 | ≤ 1.0 |
| | Brier Score | 確率予測の二乗誤差 | ≤ 0.3 |
| **ランキング品質** | nDCG@5 | soft labelを関連度として計算 | ≥ 0.6 |
| | nDCG@10 | | ≥ 0.7 |
| | MRR | soft label最上位の逆順位 | ≥ 0.4 |
| **校正品質** | ECE | 校正誤差 | ≤ 0.15 |
| **選択的予測** | Coverage@Risk=0.2 | 誤り20%以下でのカバレッジ | ≥ 60% |
| **注釈品質** | Krippendorff's α | 注釈者間一致 | ≥ 0.4 |

## 工数見積もり

### v7からの修正

**v7の問題**: 実務コスト（ルーブリック改訂、ドリフト管理等）が抜けている
**v8の修正**: 反復改訂ループを明示的に含める

| フェーズ | 内容 | 工数 |
|----------|------|------|
| **Phase 1: 設計** | | **48h** |
| - アーキテクチャ設計 | 埋め込み+分類ヘッド or 検索+再ランク | 12h |
| - ルーブリック策定 | 64卦の判定基準、境界事例の扱い | 20h |
| - 注釈ツール整備 | 複数選択+順序尺度のUI | 16h |
| **Phase 2: パイロット（2ラウンド）** | | **80h** |
| - 注釈者トレーニング | 3名×6h | 18h |
| - ラウンド1注釈 | 50件×3名 | 24h |
| - 一致度分析・ルーブリック修正 | Krippendorff's α計算、問題点特定 | 16h |
| - ラウンド2注釈 | 修正後50件×3名 | 16h |
| - 最終一致度確認 | α≥0.4の確認 | 6h |
| **Phase 3: ゴールドセット** | | **140h** |
| - 本注釈 | 300件×3名 | 90h |
| - 品質監査 | サンプル検証、ドリフトチェック | 20h |
| - 調停 | 高不一致事例の解決 | 20h |
| - soft label生成・検証 | 分布の妥当性確認 | 10h |
| **Phase 4: モデル構築** | | **52h** |
| - ベースモデル実装 | 埋め込み+分類ヘッド | 16h |
| - 統合モデル学習 | 重み最適化、温度校正 | 12h |
| - 選択的予測の閾値決定 | risk-coverage曲線 | 8h |
| - 評価・レポート | 全指標の測定 | 16h |
| **合計** | | **320h** |

## フォールバック戦略

| 条件 | 対応 |
|------|------|
| Krippendorff's α < 0.3 | 64卦→八卦（8クラス）に簡素化 |
| nDCG@10 < 0.5 | 検索ベースの重みを上げる |
| Coverage@Risk=0.2 < 40% | 目標リスクを0.3に緩和 |
| ECE > 0.25 | Platt Scaling等の別校正手法 |
| 工数超過 | ゴールドセットを200件に削減 |

## v7からの主な変更まとめ

| 項目 | v7 | v8 |
|------|-----|-----|
| Stage 1-2の整合性 | 矛盾あり | 最初から64卦同時スコア |
| スコア統合 | α=0.7固定 | 学習済み重み |
| 閾値決定 | entropy>0.7等の恣意値 | risk-coverage曲線 |
| 注釈一致指標 | Fleiss kappa | Krippendorff's α |
| 15卦フィルタ | 無条件使用 | 原則禁止+救済経路 |
| 確率の定義 | 曖昧 | モデルロジット→校正のみ |
| 工数 | 260h | 320h（反復ループ含む） |

## 成功の定義

以下を**全て**満たした場合、「六十四卦マッピングシステム」として成立:

1. **注釈品質**: Krippendorff's α ≥ 0.4
2. **分布品質**: KL Divergence ≤ 1.0, Brier ≤ 0.3
3. **ランキング品質**: nDCG@10 ≥ 0.7, MRR ≥ 0.4
4. **校正品質**: ECE ≤ 0.15
5. **選択的予測**: Coverage@Risk=0.2 ≥ 60%

1つでも満たさない場合はフォールバック戦略を適用。
