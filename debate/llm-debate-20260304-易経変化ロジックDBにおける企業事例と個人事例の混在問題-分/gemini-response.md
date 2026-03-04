# Gemini (Google) の見解

## 技術調査結果

Geminiはコードベースを実際に調査し、以下を確認:
- `BacktraceEngine` に `_SCALE_CATEGORIES` と `expertise_level` の定義はあるが、統計計算で活用されていない
- `CaseSearchEngine` のインデックスに `scale` が含まれていない（混在の直接原因）
- `rev_after_state.json` 等の逆引きインデックスにカテゴリ情報が完全に欠落
- `build_reverse_indices.py` が全カテゴリをフラットに合算して集計

## 議論ポイントへの回答

### 1. 論理的分離 vs 物理的分離
**論理的分離（同一DB/ファイル内での属性管理）を強く推奨。**
物理的にファイルを分けると、易経の持つ「マクロとミクロの相似性（フラクタル構造）」という強みが失われる。小規模カテゴリ（家族等）のデータ不足を他カテゴリの構造的類似性で補う（重み付き補完）ことが困難になる。

### 2. expertise_level フィールドの粒度
現在の `novice, intermediate, advanced, expert` の4段階は適切。
個人の意思決定において「初心者の売上回復」と「熟練者の市場開拓」は質的に異なる。

### 3. カテゴリ跨ぎの「参考情報」
**有用だが、明確なラベル付けが必要。**
「構造的な変化パターンはJALの事例と相似していますが、規模が異なるため参考値です」といった注釈付きで別枠表示すべき。メインの確率計算からは除外するか、極めて低い重みに設定。

### 4. 少数カテゴリ（家族：787件）の扱い
**削除せず「統合基盤からの重み付き抽出」で維持すべき。**
カテゴリ全体の平均（グローバル平均）をベースとしつつ、家族事例の偏りを反映させるベイズ的アプローチが適している。

### 5. 情報量の損失への対策
**「多次元インデックス化」を推奨。**
1. `(scale=individual, expertise=novice)` で検索
2. 件数が閾値以下なら `(scale=individual, all_expertise)` に拡大
3. さらに不足なら `(all_scale)` から構造的類似性の高いものを抽出
段階的フォールバックにより情報量を維持しつつ精度を高める。

### 6. 代替アプローチ：Analogy Scoring（類推スコアリング）
ハードなフィルタリングの代わりに `ScaleMatchWeight`（規模一致重み）を導入:
`TotalScore = BaseChangeSimilarity * ScaleMatchWeight * ExpertiseMatchWeight`
JALの事例はタクシー運転手に対して「構造は似ているが重みが極小」となり、自然に下位へ沈む。

## 提案：階層型論理分離（Hierarchical Logical Separation）

### Phase 1: データの多次元化
- `cases.jsonl` の個人事例に `expertise_level` を追加
- `CaseSearchEngine` に `scale` と `expertise_level` をインデックスとして追加

### Phase 2: 階層型統計集計
- `build_reverse_indices.py` を更新し、`rev_*.json` をカテゴリ別の階層構造に変更
  - 例: `rev_after_state["individual"]["停滞・閉塞"]`
- `prob_tables.json` の生成プロセスもカテゴリごとの出現確率を保持するよう改修

### Phase 3: スコアリングエンジンの高度化
- Soft Filtering: `ScaleSimilarityWeight` を導入（ハードフィルタではなく重み付け）
- 階層的フォールバック: カテゴリ内の事例が少ない場合、グローバル平均をブレンドする平滑化ロジック

### Phase 4: UXの改善
- カテゴリを跨いだ事例には「構造的類似事例（企業スケール）」バッジを付与

## 結論
「論理的な階層化」により、易経の汎用的な変化法則（ユニバーサル・ロジック）を維持したまま、実用上のミスマッチを解消可能。物理的分離はコードの重複とメンテナンスコスト増を招くため避けるべき。
