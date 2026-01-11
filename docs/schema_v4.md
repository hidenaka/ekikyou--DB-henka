# HaQei 変化データベース スキーマ v4

**v4 の主な変更点:**
- 爻レベル分析（`yao_analysis`）を追加
- 予測ロジックと適合性分析を導入

1レコード = 1つの「変化のストーリー（遷移）」を表す。
保存形式は JSON Lines（.jsonl）で、`data/raw/cases.jsonl` に 1行1レコードで追記していく。

---

## フィールド一覧

### メタ情報

- `transition_id` : string
  - 一意のID。例：`CORP_JP_001`, `PERS_JP_003` など。
- `target_name` : string
  - 事例の対象名（企業名・人物名・国家名など）
- `scale` : string（enum）
  - `"company" | "individual" | "family" | "country" | "other"`
- `period` : string
  - 時期。例：`1997`, `2000-2010`, `平成後期` など大まかでよい。
- `story_summary` : string
  - 変化のストーリー要約（50〜200文字程度の日本語）。

### 状態・行動（人間向けラベル）

- `before_state` : string（enum）
  - `"絶頂・慢心"`
  - `"停滞・閉塞"`
  - `"混乱・カオス"`
  - `"成長痛"`
  - `"どん底・危機"`
  - `"安定・平和"`

- `trigger_type` : string（enum）
  - `"外部ショック"`      （例：コロナ、法改正、天災）
  - `"内部崩壊"`          （例：不正発覚、ハラスメント、内紛）
  - `"意図的決断"`        （例：起業、事業転換、移住、リストラ）
  - `"偶発・出会い"`      （例：偶然の出会い、予期せぬオファー）

- `action_type` : string（enum）
  - `"攻める・挑戦"`      （投資、新規事業、拡大など）
  - `"守る・維持"`        （防衛、コストカット、現状維持）
  - `"捨てる・撤退"`      （売却、撤退、離婚、閉鎖）
  - `"耐える・潜伏"`      （下積み、準備、充電、表に出ない）
  - `"対話・融合"`        （交渉、和解、統合、M&A）
  - `"刷新・破壊"`        （大改革、ルール破壊、抜本的見直し）
  - `"逃げる・放置"`      （先送り、責任転嫁、対応しない）
  - `"分散・スピンオフ"`  （分社化、副業、ポートフォリオ化）

- `after_state` : string（enum）
  - `"V字回復・大成功"`
  - `"縮小安定・生存"`
  - `"変質・新生"`
  - `"現状維持・延命"`
  - `"迷走・混乱"`
  - `"崩壊・消滅"`

### 八卦タグ（8つの性質）

各フェーズの「性質」を、以下の 8 卦のうち 1つで表す。

- `before_hex`  : string（enum）
- `trigger_hex` : string（enum）
- `action_hex`  : string（enum）
- `after_hex`   : string（enum）

いずれも次のいずれか：
- `"乾"` ：リーダーシップ・スタート・創造 / 傲慢・強行
- `"坤"` ：受け身・基盤・支える / 迷い・依存
- `"震"` ：衝撃・スキャンダル・雷のような始動
- `"巽"` ：信用・浸透・風評・交渉・根回し
- `"坎"` ：苦難・危機・病気・問題の深堀り
- `"離"` ：知性・発見・可視化・炎上・分離
- `"艮"` ：停止・蓄積・守り・ブレーキ
- `"兌"` ：喜び・お金・対話・サービス・誘惑・享楽

### 物語パターンと結果

- `pattern_type` : string（enum）
  - `"Shock_Recovery"`   （ショック→再生）
  - `"Hubris_Collapse"`  （慢心→自滅）
  - `"Pivot_Success"`    （ピボット成功）
  - `"Endurance"`        （耐え忍び→大器晩成）
  - `"Slow_Decline"`     （決断できずじわじわ衰退）

- `outcome` : string（enum）
  - `"Success"`          （明らかな成功）
  - `"PartialSuccess"`   （成功したが代償も大きい／評価が割れる）
  - `"Failure"`          （明らかな失敗・崩壊）
  - `"Mixed"`            （評価が分かれる、まだ決着していない）

### その他メタ情報

- `free_tags` : string[]
  - 検索用・分類用の自由タグ。例：`["#不正会計", "#M&A", "#粉飾"]`
- `source_type` : string（enum）
  - `"official"`（公式資料・決算書 等）
  - `"news"`（新聞・ニュースサイト）
  - `"book"`（書籍）
  - `"blog"`（ブログ・個人サイト）
  - `"sns"`（SNS投稿 等）
- `credibility_rank` : string（enum）
  - `"S"`（公式 or 複数ソースで裏付けあり）
  - `"A"`（大手メディア・書籍等）
  - `"B"`（信頼性は中程度）
  - `"C"`（噂レベル・SNS単独 等）

### 任意フィールド（易的な深掘り用）

- `classical_before_hexagram` : string
  - 例：`"剥 (山地剥)"` のような64卦名＋彖辞。
- `classical_action_hexagram` : string
- `classical_after_hexagram` : string
- `logic_memo` : string
  - なぜそのタグや八卦を選んだか、100〜300文字程度の日本語で説明。

---

## 【v4 新規】爻レベル分析 (`yao_analysis`)

爻（こう）は卦を構成する6本の線で、それぞれの位置が「ライフサイクルの段階」を表す。
同じ卦でも、どの爻の位置にいるかで推奨される行動と予測される結果が異なる。

### 爻位の意味

| 爻位 | 名称 | 段階 | 意味 |
|-----|-----|-----|------|
| 1 | 初爻 | 発芽期 | まだ早い。力を蓄えよ |
| 2 | 二爻 | 成長期 | 着実に基盤を固めよ |
| 3 | 三爻 | 岐路 | 分岐点。慎重に選択せよ |
| 4 | 四爻 | 接近期 | 準備完了。調整を怠るな |
| 5 | 五爻 | 全盛期 | 最適なタイミング。決断せよ |
| 6 | 上爻 | 極み | 行き過ぎ。引き際を見極めよ |

### フィールド定義

```json
{
  "yao_analysis": {
    "before_hexagram_id": 1,              // 64卦の番号（1-64）
    "before_yao_position": 5,              // 診断された爻位（1-6）
    "yao_phrase_classic": "飛龍在天",      // 該当爻の古典的爻辞
    "yao_phrase_modern": "リーダーとして手腕を振るう",  // 現代語訳
    "yao_stage": "全盛期・リーダー期",     // 爻位の段階名
    "yao_basic_stance": "決断・実行",      // 基本姿勢

    "action_compatibility": {
      "score": 1,                          // 適合性スコア（1=最適〜4=不適）
      "reason": "最適なタイミング。リーダーシップを発揮せよ",
      "is_optimal": true                   // 最適な選択かどうか
    },

    "predicted_outcome": "Success",        // ロジックに基づく予測結果
    "actual_outcome": "Success",           // 実際の結果

    "prediction_analysis": {
      "match": true,                       // 予測が当たったか
      "accuracy": "exact",                 // exact/close/miss
      "note": "予測と実際が一致"
    },

    "recommended_actions": ["リーダーシップ発揮", "大きな決断", "実行"],
    "avoid_actions": ["優柔不断", "過度な委任", "現状維持"]
  }
}
```

### 適合性スコアの解釈

| スコア | 意味 | 予測結果 |
|-------|------|---------|
| 1 | 最適 | Success |
| 2 | 適切 | PartialSuccess |
| 3 | 注意 | Mixed |
| 4 | 不適 | Failure |

### 予測精度の評価

- `exact`: 予測と実際が完全一致
- `close`: 予測と実際が1段階差（Success ↔ PartialSuccess など）
- `miss`: 予測と実際が大きく乖離

---

## 爻ロジックの活用方法

### 1. 状況診断

「今、自分（自社）はどの爻の位置にいるか？」を診断する。

```
before_state + pattern_type → before_yao_position
```

### 2. 行動の適合性チェック

「この行動は、今の爻位に適しているか？」を判定する。

```
action_type + before_yao_position → action_compatibility.score
```

### 3. 結果予測

「この行動を取ると、どうなるか？」を予測する。

```
action_compatibility.score → predicted_outcome
```

### 4. 分岐シナリオの提示

「別の行動を取っていたらどうなったか？」を示す。

```
例：山一證券（上爻・逃げる放置）
- 実際: 逃げる・放置 → Failure（適合性スコア4）
- 仮定: 捨てる・撤退 → PartialSuccess（適合性スコア1）
- 仮定: 刷新・破壊 → PartialSuccess（適合性スコア2）
```

---

## 保存形式

- ファイル：`data/raw/cases.jsonl`
- 1行 = 上記スキーマに従った JSON オブジェクト 1件
- 文字コード：UTF-8

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `data/mappings/yao_transitions.json` | 64卦×6爻の遷移テーブル |
| `data/mappings/yao_recommendations.json` | 爻位別の推奨行動 |
| `data/mappings/action_yao_compatibility.json` | 行動-爻位適合性マトリクス |
| `data/yao_phrases_384.json` | 384爻の爻辞データ |
| `docs/yao_logic_design.md` | 爻ロジックの詳細設計書 |
