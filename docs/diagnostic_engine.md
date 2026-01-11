# 易経変化診断エンジン v1.0

位相（八卦）× 勢（momentum）× 時（timing）に基づく実例ベースの行動推奨システム。

## 概要

このエンジンは、ユーザーの現在の状況を8つの質問で診断し、1,300件以上の実例データに基づいて最適な行動を推奨します。

### 特徴

- **位相（八卦）判定**: 8つの基本的な状態パターン（乾・坤・震・巽・坎・離・艮・兌）を特定
- **勢（momentum）判定**: 上昇/安定/下降/混乱の4段階で状況の流れを判定
- **時（timing）判定**: 動くべき時/適応すべき時/待つべき時の3段階でタイミングを判定
- **実例ベースの推奨**: データベースの成功率・失敗率に基づいた行動推奨
- **リスク警告**: 回避すべきパターンと高リスク行動の警告

## 使い方

### コマンドライン（対話式）

```bash
python scripts/diagnostic_engine.py
```

8つの質問に順番に回答すると、診断結果が表示されます。

### プログラムからの利用

```python
from scripts.diagnostic_engine import DiagnosticEngine, format_result

# エンジン初期化
engine = DiagnosticEngine()

# 回答を記録
engine.record_answer("Q1", "active_strong")    # 激しく動いている
engine.record_answer("Q2", "outward_expand")   # 拡大・発信している
engine.record_answer("Q3", "clear_certain")    # 見通しが立っている
engine.record_answer("Q4", "intentional")      # 自分で決断した
engine.record_answer("Q5", "power_influence")  # 力・影響力が増加
engine.record_answer("Q6", "nothing")          # 減っているものは特にない
engine.record_answer("Q7", "hubris_collapse")  # 調子に乗って失敗を避けたい
engine.record_answer("Q8", "growth")           # 成長・拡大を重視

# 診断実行
result = engine.diagnose()

# 結果表示
print(format_result(result))

# または個別に結果を取得
print(f"位相: {result.primary_hex}")
print(f"勢: {result.momentum}")
print(f"時: {result.timing}")
for action, score, reason in result.recommended_actions:
    print(f"推奨: {action} (スコア: {score:.1f})")
```

## 質問と選択肢

### Q1: 動静
「今の状況は『動いている』か『止まっている』か？」

| value | label | score |
|-------|-------|-------|
| active_strong | 激しく動いている | 3 |
| active_mild | ゆるやかに動いている | 1 |
| static_stable | 安定して止まっている | -1 |
| static_stuck | 停滞して止まっている | -3 |

### Q2: 内外
「エネルギーは『外に向かう』か『内に向かう』か？」

| value | label | score |
|-------|-------|-------|
| outward_expand | 拡大・発信している | 3 |
| outward_connect | 人と繋がろうとしている | 1 |
| inward_focus | 内省・集中している | -1 |
| inward_protect | 守り・維持している | -3 |

### Q3: 明暗
「状況は『明確』か『不透明』か？」

| value | label | score |
|-------|-------|-------|
| clear_certain | 見通しが立っている | 3 |
| clear_partial | 一部は見えている | 1 |
| unclear_fog | 霧の中にいる感じ | -1 |
| unclear_danger | 危機的で見通せない | -3 |

### Q4: きっかけ
「変化のきっかけは？」

| value | label | trigger_type |
|-------|-------|-------------|
| external_shock | 外部からの衝撃 | 外部ショック |
| internal_collapse | 内部の問題 | 内部崩壊 |
| intentional | 自分で決断した | 意図的決断 |
| encounter | 偶然の出会い・発見 | 偶発・出会い |

### Q5: 増加要素
「今『増えている』ものは？」

| value | label | score |
|-------|-------|-------|
| power_influence | 力・影響力・リソース | 3 |
| clarity_insight | 理解・洞察・明確さ | 2 |
| connections | 人脈・協力者・仲間 | 2 |
| stability | 安定・基盤・信頼 | 1 |
| flexibility | 選択肢・柔軟性 | 1 |
| pressure | プレッシャー・緊張 | -2 |
| nothing | 特にない／減っている | -3 |

### Q6: 減少要素
「今『減っている』ものは？」

| value | label | score |
|-------|-------|-------|
| resources | 資金・時間・体力 | -3 |
| options | 選択肢・自由度 | -2 |
| relationships | 人間関係・信頼 | -2 |
| motivation | やる気・情熱 | -2 |
| clarity | 見通し・確信 | -1 |
| stability | 安定・平穏 | -1 |
| nothing | 特にない | 1 |

### Q7: 回避対象
「最も恐れていること・避けたいことは？」

| value | label | avoid_pattern |
|-------|-------|--------------|
| hubris_collapse | 調子に乗って失敗 | Hubris_Collapse |
| slow_decline | じわじわ衰退 | Slow_Decline |
| shock_damage | 突然の衝撃で回復を強いられる | Shock_Recovery |
| endurance_fail | 耐えきれずに崩壊 | Endurance |
| pivot_fail | 方向転換の失敗 | Pivot_Success |
| stagnation | 成長が止まること | Steady_Growth |

### Q8: 優先価値
「今の状況で最も大事にしたいことは？」

| value | label | preferred_action |
|-------|-------|-----------------|
| growth | 成長・拡大 | 攻める・挑戦 |
| stability | 安定・維持 | 守る・維持 |
| endurance | 耐え抜くこと | 耐える・潜伏 |
| renewal | リセット・刷新 | 刷新・破壊 |
| connection | 人との繋がり | 対話・融合 |
| retreat | 撤退・損切り | 捨てる・撤退 |

## スコア計算ロジック

### 1. 八卦スコア
各質問の `weights` を合算し、最大値を `primary_hex` とする。

### 2. 勢（momentum）スコア
```
momentum_score = (Q1.score + Q2.score + Q5.score + Q6.score) / 4
```

| 条件 | 判定 |
|-----|------|
| score >= 1.0 | ascending（上昇） |
| -0.5 <= score < 1.0 | stable（安定） |
| -2.0 <= score < -0.5 | descending（下降） |
| score < -2.0 | chaotic（混乱） |

### 3. 時（timing）スコア
```
timing_score = (Q3.score + Q5.score - Q6.score) / 3
```

| 条件 | 判定 |
|-----|------|
| score >= 1.0 | act_now（動くべき時） |
| -1.0 <= score < 1.0 | adapt（適応すべき時） |
| score < -1.0 | wait（待つべき時） |

### 4. 行動推奨スコア

1. 統計テーブルから基本成功率を取得（優先順位: state×trigger > state > trigger > hex）
2. 八卦の base_action に +10、risk_action に -10
3. momentum の boost/penalty を適用（×1.2 または ÷1.2）
4. timing の boost/penalty を適用（×1.3 または ÷1.3）
5. preferred_action に +15
6. avoid_pattern で高リスクな行動に -20
7. スコア順にランキング

## 出力形式

```
==================================================
【易経変化診断結果】
==================================================

◆ あなたの位相（八卦）: 乾
  (創造・主導)

◆ 勢: 上昇の勢い (スコア: 2.5)
◆ 時: 動くべき時 (スコア: 1.7)

--------------------------------------------------
【判定】好機到来。積極的に動くべき時です。
  状況は上向きで、見通しも立っています。この機会を逃さず行動を。
--------------------------------------------------

◆ 推奨される行動（上位3つ）:
  1. 刷新・破壊 (スコア: 156.0)
     理由: 勢い(ascending)に適合、時(act_now)に適合
  2. 攻める・挑戦 (スコア: 99.1)
     理由: 乾の基本行動、勢い(ascending)に適合、時(act_now)に適合
  3. 対話・融合 (スコア: 50.0)
     理由: 標準

◆ 注意事項:
  ⚠ 【絶頂からの転落パターン】成功に酔って判断を誤りやすい時期です
  ⚠   避けるべき行動: 攻める・挑戦, 逃げる・放置
==================================================
```

## ファイル構成

```
data/diagnostic/
├── question_mapping.json    # 質問と選択肢の定義
├── judgment_rules.json      # 判定ルールとモディファイア
├── statistics_table.json    # 条件別の成功率データ
├── failure_avoidance.json   # 高リスク組み合わせと警告
└── time_guidelines.json     # 時間軸ガイドライン

scripts/
└── diagnostic_engine.py     # 診断エンジン本体

tests/
└── test_diagnostic_engine.py  # ユニットテスト
```

## 今後の拡張

- Web API化（FastAPI）
- フロントエンド（React/Vue）
- 類似事例の表示機能
- 診断履歴の保存
- 経過追跡と予測更新
