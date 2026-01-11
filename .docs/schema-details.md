# スキーマ詳細

## データスキーマ（schema_v3.py）

1レコード = 1つの「変化のストーリー（遷移）」

### メタ情報
- `transition_id`: 一意識別子（自動採番）
- `target_name`: 対象名
- `scale`: individual/company/family/country/other
- `period`: 期間
- `story_summary`: ストーリー要約

### 状態・行動
- `before_state`: 絶頂・慢心/停滞・閉塞/混乱・カオス/成長痛/どん底・危機/安定・平和
- `trigger_type`: 外部ショック/内部崩壊/意図的決断/偶発・出会い
- `action_type`: 攻める・挑戦/守る・維持/捨てる・撤退/耐える・潜伏/対話・融合/刷新・破壊/逃げる・放置/分散・スピンオフ
- `after_state`: 同上

### 八卦タグ
- `before_hex`, `trigger_hex`, `action_hex`, `after_hex`: 乾/坤/震/巽/坎/離/艮/兌

### 384爻（変爻）
- `changing_lines_1`: before→trigger での変爻（1-3の配列）
- `changing_lines_2`: trigger→action での変爻
- `changing_lines_3`: action→after での変爻

### パターン・結果
- `pattern_type`: Shock_Recovery/Hubris_Collapse/Pivot_Success/Endurance/Slow_Decline/Steady_Growth
- `outcome`: Success/Failure/Mixed/Unknown

### ソース情報
- `source_type`: news/book/interview/other
- `credibility_rank`: S/A/B/C
- `free_tags`: タグ配列

## ID管理

自動採番形式: `{PREFIX}_JP_{NNN}`
- CORP: company
- PERS: individual
- FAM: family
- COUN: country
- OTHR: other
