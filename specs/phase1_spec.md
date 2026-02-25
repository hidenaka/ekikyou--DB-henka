# Phase 1 仕様書: 64卦の状態空間モデル構築

## 目的
64卦を6次元二値ベクトル空間 `{0,1}^6` として数理的にモデル化し、グラフ構造の性質を分析する。

## 背景
易経の64卦は6本の爻（陰=0, 陽=1）の組み合わせで構成される。これは数学的には6次元超立方体グラフ（Q6）と同型である。このグラフの構造的性質を明らかにすることが、Phase 3の同型性検証の基盤となる。

## 成果物

### 1. 状態空間モデル (`analysis/phase1/state_space_model.py`)

```python
# 必須実装項目
class HexagramStateSpace:
    """64卦の状態空間モデル"""

    def __init__(self):
        # 64卦を6ビットベクトルとして初期化
        # 爻の順序: [初爻, 二爻, 三爻, 四爻, 五爻, 上爻]
        # 陽爻=1, 陰爻=0
        pass

    def build_graph(self) -> nx.DiGraph:
        """64ノード有向グラフを構築"""
        # ハミング距離1の全ペアにエッジを張る
        pass

    def hamming_distance(self, hex_a: int, hex_b: int) -> int:
        """2つの卦間のハミング距離を計算"""
        pass

    def get_cuogua(self, hex_id: int) -> int:
        """錯卦（全爻反転）を返す"""
        pass

    def get_zonggua(self, hex_id: int) -> int:
        """綜卦（上下反転）を返す"""
        pass

    def get_hugua(self, hex_id: int) -> int:
        """互卦（2-5爻から新しい卦を構成）を返す"""
        pass

    def get_zhigua(self, hex_id: int, changing_lines: list) -> int:
        """之卦（変爻による遷移先）を返す"""
        pass

    def transition_matrix(self) -> np.ndarray:
        """64x64の遷移行列を返す"""
        pass
```

### 2. グラフ分析 (`analysis/phase1/graph_analysis.json`)

```json
{
  "basic_properties": {
    "num_nodes": 64,
    "num_edges": "計算値",
    "diameter": "計算値",
    "average_shortest_path": "計算値",
    "clustering_coefficient": "計算値",
    "is_connected": true
  },
  "degree_distribution": {
    "min_degree": 6,
    "max_degree": 6,
    "is_regular": true
  },
  "structural_relations": {
    "cuogua_pairs": 32,
    "zonggua_pairs": "計算値（自己綜卦含む）",
    "hugua_mapping": "64卦→互卦のマッピング"
  },
  "community_structure": {
    "method": "Louvain / spectral clustering",
    "num_communities": "計算値",
    "modularity": "計算値"
  },
  "spectral_properties": {
    "eigenvalues": "隣接行列の固有値リスト",
    "spectral_gap": "計算値"
  }
}
```

### 3. 可視化 (`analysis/phase1/visualizations/`)

- `hypercube_projection.png` — 6次元超立方体の2D/3D射影
- `community_graph.png` — コミュニティ構造のグラフ
- `cuogua_pairs.png` — 錯卦ペアの可視化
- `spectral_embedding.png` — スペクトル埋め込み

### 4. レポート (`analysis/phase1/report.md`)

以下を含むこと:
- モデルの数学的定義
- グラフの構造的性質のサマリー
- コミュニティ構造の解釈
- Phase 2との接続点（change-analyzerが使うべきデータ形式）
- 既知の限界と仮定

## 卦番号とビット列の対応

King Wen配列（伝統的な1-64番号）を使用する。ビット列への変換は `data/reference/iching_texts_ctext_legge_ja.json` から取得すること。

爻の順序は下から上: `[初爻, 二爻, 三爻, 四爻, 五爻, 上爻]`
- 陽爻（━━━）= 1
- 陰爻（━ ━）= 0

例: 乾為天 (卦1) = [1,1,1,1,1,1], 坤為地 (卦2) = [0,0,0,0,0,0]

## 八卦（三爻）との関係

各卦は上卦（4-6爻）と下卦（1-3爻）の組み合わせ:
- 乾(☰)=[1,1,1], 坤(☷)=[0,0,0], 震(☳)=[0,0,1], 巽(☴)=[1,1,0]
- 坎(☵)=[0,1,0], 離(☲)=[1,0,1], 艮(☶)=[1,0,0], 兌(☱)=[0,1,1]

## 検証基準
- [ ] 64ノード全てがグラフに含まれる
- [ ] 各ノードの次数が6（6次元超立方体の性質）
- [ ] 錯卦ペアが正確に32組
- [ ] 綜卦マッピングが正しい
- [ ] 互卦の計算が`hexagram_patterns.md`と整合
- [ ] グラフが連結
- [ ] King Wen番号⇔ビット列の変換が正しい
