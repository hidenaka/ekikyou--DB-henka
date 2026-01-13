# 六十四卦384爻マッピング実装計画 v2

## Codex批評への対応

| v1の問題 | v2での対応 |
|----------|-----------|
| pattern_typeが結果含意を持つ | **結果非依存の名称に改名** |
| change_stageが事後語 | **時系列位置（pre/during/post）に変更** |
| f()が未定義 | **決定表を完全列挙** |
| primary選択規則なし | **「時系列最初」をprimaryとする規則** |
| 384セルは過剰分割 | **8卦レベルで分析、64卦は十分なサンプル時のみ** |
| 二重アノテーションなし | **50件二重ラベリング、κ≥0.7必須** |

---

## 設計原則（v2）

### 原則1: 循環論法の完全排除
- **入力**: trigger類型, action類型, 時系列位置, 規模 のみ
- **除外**: outcome, pattern_type（結果含意あり）
- **観測値**: outcomeは卦決定後の分析対象

### 原則2: 決定表による機械的マッピング
- f()関数ではなく、入力→出力の完全列挙テーブル
- 曖昧さゼロ、誰が実装しても同じ結果

### 原則3: 階層的分析（8卦→64卦→384爻）
- まず8卦レベルで傾向を確認
- サンプル十分（≥100件/卦）なら64卦に細分化
- 爻は「解釈の補助」として限定使用

---

## Phase 0: スキーマ設計（v2）

### pattern_typeの再定義（結果非依存）
```
旧名（結果含意あり）     → 新名（結果非依存）
--------------------------------------------------
Pivot_Success            → Strategic_Pivot（戦略転換）
Failed_Attempt           → Risky_Attempt（高リスク試行）
Shock_Recovery           → External_Shock（外部衝撃）
Hubris_Collapse          → Overexpansion（過剰拡大）
Steady_Growth            → Organic_Growth（自然成長）
Slow_Decline             → Gradual_Contraction（漸進縮小）
Crisis_Pivot             → Crisis_Response（危機対応）
Breakthrough             → Innovation_Push（革新推進）
Stagnation               → Status_Quo（現状維持）
Endurance                → Persistence（持続努力）
Managed_Decline          → Controlled_Exit（計画撤退）
Quiet_Fade               → Passive_Decline（受動的衰退）
Exploration              → New_Venture（新規探索）
```

### change_stageの時系列化
```
旧定義（事後語）         → 新定義（時系列位置）
--------------------------------------------------
initiation               → T0_before（変化前の状態）
development              → T1_early（介入初期）
crisis                   → T2_middle（介入中期・転換点）
recovery                 → T3_late（介入後期）
maturation               → T4_after（変化後の状態）
```

### 新規フィールド構造（v2）
```json
{
  "case_id": "CORP_JP_001",
  "triggers": [
    {
      "trigger_id": "T_EXT_SHOCK",
      "label": "外部経済ショック",
      "span": "リーマンショックの影響で",
      "order": 1
    }
  ],
  "actions": [
    {
      "action_id": "A_RETREAT",
      "label": "事業撤退",
      "span": "不採算事業の撤退",
      "order": 1
    },
    {
      "action_id": "A_FOCUS",
      "label": "事業集中",
      "span": "社会イノベーション事業への集中",
      "order": 2
    }
  ],
  "time_position": "T3_late",
  "scale": "company",
  "hexagram": {
    "trigram_upper": "坤",
    "trigram_lower": "震",
    "number": 24,
    "name": "地雷復",
    "yao_position": 5,
    "mapping_version": "v2.0",
    "mapping_inputs": {
      "primary_trigger": "T_EXT_SHOCK",
      "primary_action": "A_RETREAT",
      "time_position": "T3_late",
      "scale": "company"
    }
  },
  "outcome": "Success",
  "extraction_meta": {
    "version": "v2.0",
    "extracted_at": "2026-01-14"
  }
}
```

---

## Phase 1: 決定表（完全列挙）

### primary選択規則
```
primary_trigger = triggers[0]  # 時系列で最初（order=1）
primary_action = actions[0]    # 時系列で最初（order=1）
```

### 下卦決定表（primary_trigger → 下卦）
| primary_trigger | 下卦 | 理由 |
|-----------------|------|------|
| T_EXT_SHOCK | 震 | 外からの衝撃・動 |
| T_INT_CRISIS | 坎 | 内なる困難・険 |
| T_REG_CHANGE | 巽 | 外からの風・変化 |
| T_MGMT_CHANGE | 艮 | 止まって転換 |
| T_MARKET_SHIFT | 兌 | 市場との交流 |
| T_OPPORTUNITY | 離 | 機会の発見・明 |
| T_STAGNATION | 坤 | 停滞・地 |

### 上卦決定表（primary_action → 上卦）
| primary_action | 上卦 | 理由 |
|----------------|------|------|
| A_RETREAT | 坤 | 退く・柔順 |
| A_FOCUS | 艮 | 止めて集中 |
| A_REORG | 震 | 動いて変える |
| A_INVEST | 乾 | 積極的拡大 |
| A_ALLIANCE | 兌 | 他者との協調 |
| A_TRANSFORM | 離 | 変容・明確化 |
| A_CULTURE | 巽 | 浸透・内部変革 |
| A_INACTION | 坎 | 困難の中の停滞 |

### 爻位決定表（time_position → 爻）
| time_position | 爻位 |
|---------------|------|
| T0_before | 初爻(1) |
| T1_early | 二爻(2) |
| T2_middle | 三爻(3)・四爻(4) |
| T3_late | 五爻(5) |
| T4_after | 上爻(6) |

### T2_middleの細分化規則
```
if scale in ['global', 'nation']:
    yao = 4  # 大規模は四爻
else:
    yao = 3  # 小規模は三爻
```

---

## Phase 2: 実装ステップ（v2）

### Step 1: 辞書構築＋二重アノテーション（100件）
1. 100件をランダム抽出
2. **50件は二重ラベリング**（2名が独立してラベル付け）
3. Cohen's κを計算、**κ≥0.7で合格**
4. 不合格なら定義を修正して再実施

### Step 2: LLM抽出パイロット（500件）
1. 選択式プロンプト（辞書から選ぶ形式）
2. 500件で抽出実行
3. 人手100件との一致率を計算
4. **一致率≥75%で合格**

### Step 3: 卦マッピング適用
1. 決定表に基づき機械的にマッピング
2. **8卦レベルで分布確認**
3. 極端な偏り（1卦に50%以上集中）があれば決定表を見直し

### Step 4: 傾向分析（8卦レベル）
1. 8卦×outcome のクロス集計
2. **各卦100件以上**のもののみ傾向を語る
3. 100件未満の卦は「サンプル不足」として保留

### Step 5: 64卦・爻への展開（条件付き）
1. 8卦内で**64卦×100件以上**のもののみ細分化
2. 爻レベルは**解釈の参考**として付与（統計検定しない）

---

## 成功指標（v2）

| 指標 | 目標値 | 測定方法 |
|------|--------|----------|
| 二重アノテーション一致度 | κ≥0.7 | trigger/action/time_positionの各フィールド |
| LLM抽出一致率 | ≥75% | 人手100件との比較 |
| 8卦分布偏り | 最大卦≤30% | 1卦への過集中を防ぐ |
| 分析可能な卦数 | ≥6/8卦 | 各100件以上 |

---

## 工数見積（v2）

| Phase | 作業 | 見積時間 |
|-------|------|---------|
| 0 | スキーマ設計・決定表作成 | 4時間 |
| 1-1 | 100件ラベリング（50件×2名） | 12時間 |
| 1-2 | κ計算・定義修正 | 4時間 |
| 2-1 | プロンプト設計 | 4時間 |
| 2-2 | 500件パイロット実行 | 2時間 |
| 2-3 | 精度評価・修正 | 4時間 |
| 3 | 全件マッピング | 4時間 |
| 4 | 8卦傾向分析 | 4時間 |
| 5 | 64卦展開（条件付き） | 4時間 |
| **合計** | | **42時間** |

---

## リスクと対策（v2）

| リスク | 対策 |
|--------|------|
| κ<0.7で定義が合意できない | フィールドを削減（例: time_positionを3段階に簡略化） |
| 特定の卦に偏りすぎる | 決定表を見直すか、8卦のうち偏った卦は統合 |
| LLM抽出精度が低い | 辞書を絞る、選択肢を減らす |
| 64卦レベルで有意義な分析ができない | 8卦レベルに留める（64卦は解釈補助） |

---

## v1からの主要変更点まとめ

1. ✅ pattern_type → 結果非依存の名称に改名
2. ✅ change_stage → 時系列位置（T0-T4）に変更
3. ✅ f()関数 → 決定表（完全列挙）に置換
4. ✅ primary選択 → 「order=1」を選ぶ明確な規則
5. ✅ 384セル分析 → 8卦レベル中心、64卦は条件付き
6. ✅ 二重アノテーション → 50件必須、κ≥0.7
7. ✅ 成功指標 → 均一性ではなく一致度・再現性重視
