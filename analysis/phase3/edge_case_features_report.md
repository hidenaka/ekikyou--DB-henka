# エッジケース特徴量調査レポート

**分析日**: 2026-03-04
**対象**: cases.jsonl (13,060件)
**目的**: スケール境界事例の特定とCodex提案5特徴量の実現可能性評価

## 1. 現在のスケール分布

| scale | 件数 | 割合 |
|-------|------|------|
| company | 5,510 | 42.2% |
| individual | 3,217 | 24.6% |
| other | 2,165 | 16.6% |
| country | 1,381 | 10.6% |
| family | 787 | 6.0% |
| **合計** | **13,060** | **100%** |

## 2. スケール境界事例の検出

### 2.1 キーワードマッチによる境界事例

| カテゴリ | マッチ件数 | 説明 |
|----------|-----------|------|
| micro_org | 306 | スタートアップ・零細企業 |
| self_employed | 260 | 自営業・個人事業・フリーランス |
| npo_social | 124 | NPO・社会起業 |
| family_business | 89 | 家族経営・同族経営 |
| hybrid | 7 | 第三セクター・独法等 |

#### family_business サンプル（上位5件）

- **CORP_JP_033** (scale=company, domain=金融): ビッグモーター — 中古車販売で業界トップに急成長したが、過度なノルマ主義により、保険金不正請求や街路樹への除草剤散布などのコンプライアンス無視が横行。内部告発を機に社会から断罪され、事業譲渡に追い込まれた。...
- **FAM_JP_001** (scale=family, domain=製造): 大塚家具・大塚父娘 — 父・大塚勝久が創業した大塚家具で、娘・久美子を社長に据えるも経営方針で対立。2015年の株主総会で父娘が経営権を争い、娘が勝利。しかし業績悪化が続き、最終的にヤマダ電機傘下に。家族の対立が企業の衰退を...
- **FAM_JP_002** (scale=family, domain=金融): 星野リゾート・星野親子 — 星野佳路が父が経営する星野温泉で改革を試みるも、創業家の公私混同にメスを入れたため追放される。2年後、外部株主の支持で社長に復帰したが、父は反対派。その後1年かけて父を会長に迎え和解。現在は世界的リゾ...
- **FAM_JP_003** (scale=family, domain=生活・暮らし): 旭酒造（獺祭）・桜井親子三代 — 桜井博志が家業の酒蔵に入るも父と対立し27歳で解雇。父の突然死後、倒産寸前の酒蔵を継承し「獺祭」ブランドで世界的成功。その後、息子・一宏への承継は円満に進み、三代で危機を乗り越えた。...
- **FAM_JP_004** (scale=family, domain=生活・暮らし): サントリー・鳥井佐治両家 — 創業者・鳥井信治郎が次男を親戚の佐治家に養子に出し、佐治敬三として育てる。その後、鳥井家と佐治家が共同で経営を続け、90年以上にわたり家族経営を維持。2024年には再び鳥井家から社長を輩出。...

#### micro_org サンプル（上位5件）

- **CORP_JP_040** (scale=company, domain=テクノロジー): DeNA — PCオークションサイトで創業後、「モバゲー」でガラケー市場を制覇。スマホシフトの波に乗りつつ、ゲーム依存からの脱却を目指してヘルスケア、スポーツ（横浜ベイスターズ）、オートモーティブへ事業ポートフォリ...
- **CORP_JP_146** (scale=company, domain=テクノロジー): エルピーダメモリ — 日立・NEC・三菱のDRAM事業を統合した「日の丸半導体」最後の砦だったが、リーマンショック後の円高と価格暴落に耐えきれず破綻。公的資金も投入されたが、市場変化への対応が遅れた。...
- **PERS_JP_003** (scale=individual, domain=テクノロジー): Bさん_30代半ば_ITマネージャー — プロジェクトマネージャーに昇格し、自信過剰に。「自分のやり方が絶対」とメンバーの意見を聞かず、過重労働を強いた。主要メンバーが一斉離職（クーデター）し、プロジェクトは炎上・頓挫。管理職適性なしと判断さ...
- **PERS_JP_015** (scale=individual, domain=テクノロジー): Lさん_30代_銀行員 — メガバンクで法人営業をしていたが、古い体質と将来性に限界を感じた。年収ダウンを覚悟でWeb系ベンチャーの管理部門へ転職。カルチャーショックに苦しみつつも、金融知識という武器を活かしてCFO候補となり、...
- **PERS_JP_017** (scale=individual, domain=テクノロジー): Nさん_20代後半_公務員 — 激務の霞が関で働いていたが、心身の摩耗を感じて退職。地方のベンチャー企業の「地域おこし」求人に出会い移住。年収は半減したが、裁量の大きさと生活コストの低さでQOLが劇的に向上し、人間らしい生活を取り戻...

#### self_employed サンプル（上位5件）

- **CORP_JP_053** (scale=company, domain=医療・製薬): 中外製薬 — 中堅製薬企業だったが、世界的メガファーマであるロシュと戦略的提携を結ぶ決断をした。ロシュの創薬基盤を活用しつつ経営の独立性を維持し、バイオ医薬品・抗体医薬品のリーディングカンパニーへと飛躍した。...
- **CORP_JP_173** (scale=company, domain=医療・製薬): あすか製薬 — 武田薬品工業の子会社からMBO・独立を経て、帝国臓器製薬と合併。産婦人科領域などの「スペシャリティファーマ」として生きる道を選び、大手メガファーマとは異なるニッチ市場での安定成長を実現した。...
- **PERS_JP_006** (scale=individual, domain=Lifestyle): Cさん_40代前半_広告クリエイティブ — 過去の受賞歴を鼻にかけ、クライアントや若手を軽視する態度が目立ち始めた。「自分なら独立してもやっていける」と準備不足のまま退職したが、悪評が広まっており案件が獲得できず。プライドが邪魔して再就職もでき...
- **PERS_JP_013** (scale=individual, domain=生活・暮らし): Jさん_20代後半_女性社員 — 結婚直後にパートナーの転勤で退職を余儀なくされた。キャリア断絶の危機だったが、前職のスキルを活かしてフルリモートの業務委託契約を獲得。場所に縛られない働き方を確立し、結果的に収入も増加した。...
- **PERS_JP_019** (scale=individual, domain=Lifestyle): Pさん_40代_技術職 — 会社の方針で管理職への昇進を求められたが、現場で手を動かすことにこだわった。会社と交渉し、業務委託契約に切り替えて独立。フリーランスの技術顧問として複数社と契約し、収入アップと技術探求の両立を実現した...

#### npo_social サンプル（上位5件）

- **COUN_JP_187** (scale=country, domain=テクノロジー): レバノン — 放漫財政と固定相場制の維持が限界に達し、金融システムが崩壊。デフォルト宣言、預金封鎖、通貨の90%下落が発生し、ベイルート港爆発事故が追い打ちをかけた。電力や医薬品も不足する中、国民は海外からの送金や...
- **PERS_JP_099** (scale=individual, domain=スポーツ): 浅田真央 — 15歳でグランプリファイナル優勝、世界選手権3回優勝など輝かしい成績を残すも、オリンピック金メダルには届かず。2017年に引退し、現在はアイスショーや慈善活動に注力。...
- **OTHR_JP_030** (scale=other, domain=テクノロジー): NPO法人_資金難で解散 — 障害者支援のNPO法人を設立。10年間、地道な活動を続けるも助成金が打ち切られ資金難に。クラウドファンディングも目標未達。中心メンバーの高齢化もあり、解散を決定。「思いだけでは続かない」という現実。...
- **PERS_JP_538** (scale=individual, domain=社会・コミュニティ): 転職_年収ダウン_01 — やりがいを求めてNPOに転職。年収は半減。しかし理想と現実のギャップに苦しみ、2年で退職。再び企業に戻るも、ブランクで年収は前職の7割止まり。...
- **CORP_JP_385** (scale=company, domain=エンタメ): グリー（急成長と急落） — 2004年、田中良和氏がSNS「GREE」を個人開発し会社設立。2008年東証マザーズ、2010年東証一部上場（創業者33歳で最年少）。携帯ゲーム「釣り★スタ」等で時価総額2900億円に。しかし201...

#### hybrid サンプル（上位5件）

- **PERS_JP_191** (scale=individual, domain=国家・政治): 中曽根康弘 — 群馬県高崎市出身。東京帝大卒業後、内務省入省、海軍主計少佐で終戦。1947年衆議院議員初当選、「青年将校」と呼ばれ反吉田の急先鋒に。科学技術庁長官、運輸大臣、防衛庁長官、通産大臣を歴任。1982年首相...
- **PERS_JP_406** (scale=individual, domain=エネルギー・環境): 中曽根康弘 — 群馬県高崎市出身。東京帝国大学法学部卒業。1947年初当選、科学技術庁長官で原子力政策を推進。1982年第71代首相、「戦後政治の総決算」を掲げる。国鉄・電電公社・専売公社の民営化を実現。レーガン大統...
- **CORP_JP_974** (scale=company, domain=Telecom): 日本電電公社民営化（1985） — 電電公社がNTTとして民営化され、通信自由化の幕開けとなった。...
- **OTHR_JP_803** (scale=other, domain=): 電電公社 電話網全国整備（1960-1969） — 電電公社が全国電話網を整備、高度成長を通信インフラ面から支えた。...
- **CORP_JP_3337** (scale=company, domain=IT・通信): NTT分割 - 電電公社グループ再編 — 1985年の民営化後、巨大独占企業だったNTTは1999年に持株会社制へ移行し、東西地域会社・長距離会社・移動体通信会社に分割。既存の一枚岩組織を解体し、それぞれが競争力を持つ事業体として再結集。風水...

### 2.2 スケール不整合（Mismatch）

- **合計**: 603件
  - company に個人的シグナル: 57件
  - individual に組織的シグナル: 546件

#### 不整合サンプル（上位10件）

- **CORP_JP_090** [company] signal='個人' (individual_in_company): 近畿日本ツーリスト — 団体旅行に強みを持つ旅行大手だったが、個人旅行へのシフトとネット予約（OTA）の普及に乗り遅れた。クラブツーリズムとの統合などで規模維持を図ったが、ビジネスモデルの陳腐化は止まらず存在感が薄れている。...
- **CORP_JP_100** [company] signal='個人' (individual_in_company): りそなホールディングス — 自己資本比率の低下により実質国有化（公的資金注入）される事態に。メガバンクとは一線を画し、「リテール（個人・中小企業）特化」へ戦略を転換。銀行の常識を覆す24時間営業などを導入し、公的資金を完済した。...
- **CORP_JP_149** [company] signal='個人' (individual_in_company): ベネッセコーポレーション — 教育業界の最大手として圧倒的なデータベースを持っていたが、セキュリティ管理の甘さから約3500万件の個人情報が流出。ブランドへの過信と対応の遅れが批判され、会員数が激減した。...
- **PERS_JP_008** [individual] signal='組織' (org_in_individual): Eさん_30代後半_中堅商社 — 上司の顔色ばかり伺い、部下には責任を押し付ける「事なかれ主義」を徹底。大きなミスはないが成果もなく、組織内での求心力が低下。後輩に次々と昇進を抜かれ、給与も頭打ちのまま意欲を失っている。...
- **PERS_JP_009** [individual] signal='会社' (org_in_individual): Fさん_20代後半_システムエンジニア — 客先常駐（SES）でレガシーシステムの保守のみを5年担当。モダンな技術へのキャッチアップを怠り、転職市場での価値が相対的に低下。所属元の会社も業績が悪化し、給与カットを提示されたが、他に行き場がなく受...
- **PERS_JP_010** [individual] signal='会社' (org_in_individual): Gさん_20代_大手営業 — 配属直後、重大な発注ミスを犯し会社に損害を与えた。「もう終わりだ」と絶望し退職も考えたが、上司の励ましで奮起。泥臭い謝罪行脚とリカバリー業務を完遂し、その誠実さが評価され、かえって顧客との信頼関係が深...
- **PERS_JP_011** [individual] signal='部門' (org_in_individual): Hさん_30代前半_総合職 — 激務とパワハラにより適応障害を発症し休職。キャリアのレールから外れたと感じたが、休養期間中にカウンセリングや自己分析を徹底。復職後は「頑張りすぎない」働き方を確立し、管理部門へ異動して適正な評価を得る...
- **PERS_JP_014** [individual] signal='経営' (org_in_individual): Kさん_30代_企画職 — 3年かけた新規事業案が、経営方針の変更で白紙撤回された。失意の底にいたが、その過程で築いた社外人脈から声がかかり、副業として小さく事業を開始。それが軌道に乗り、自信を取り戻して本業でも再評価された。...
- **PERS_JP_015** [individual] signal='法人' (org_in_individual): Lさん_30代_銀行員 — メガバンクで法人営業をしていたが、古い体質と将来性に限界を感じた。年収ダウンを覚悟でWeb系ベンチャーの管理部門へ転職。カルチャーショックに苦しみつつも、金融知識という武器を活かしてCFO候補となり、...
- **PERS_JP_016** [individual] signal='部門' (org_in_individual): Mさん_40代_アパレル店長 — アパレル販売員として長年働いたが、立ち仕事の限界と業界の斜陽化を感じていた。職業訓練でWebデザインを学び、ECサイトの運営担当へ社内異動を志願。販売の知見とデジタルスキルを掛け合わせ、EC部門のリー...

### 2.3 main_domain × scale クロス分析（多スケール混在ドメイン）

90%未満の支配率を持つドメイン: **74件**

| domain | 件数 | 分布 | 支配率 |
|--------|------|------|--------|
| unknown | 1449 | company:797, individual:373, other:195, country:84 | 55.0% |
| 生活・暮らし | 1204 | individual:754, family:402, company:26, other:20, country:2 | 62.6% |
| None | 1173 | company:474, individual:210, other:201, family:156, country:132 | 40.4% |
| テクノロジー | 901 | company:531, individual:175, other:102, country:85, family:8 | 58.9% |
| 金融 | 634 | company:334, individual:140, country:101, other:43, family:16 | 52.7% |
| 社会・コミュニティ | 620 | other:489, individual:54, country:50, company:20, family:7 | 78.9% |
| 国家・政治 | 612 | country:548, individual:27, other:24, family:8, company:5 | 89.5% |
| 小売・サービス | 576 | company:395, other:79, individual:76, family:23, country:3 | 68.6% |
| エンタメ | 532 | individual:197, company:183, other:103, family:33, country:16 | 37.0% |
| 製造 | 471 | company:352, individual:51, country:41, other:19, family:8 | 74.7% |
| 医療・製薬 | 318 | individual:132, other:74, company:52, family:36, country:24 | 41.5% |
| 教育 | 294 | individual:131, other:90, family:35, company:28, country:10 | 44.6% |
| エネルギー・環境 | 191 | country:55, individual:51, other:40, company:35, family:10 | 28.8% |
| スポーツ | 189 | individual:110, other:59, company:11, country:5, family:4 | 58.2% |
| 不動産・建設 | 184 | company:54, individual:47, other:36, country:30, family:17 | 29.3% |
| 農林水産 | 181 | individual:61, company:52, other:33, country:27, family:8 | 33.7% |
| 物流・交通 | 180 | company:103, other:40, individual:30, country:7 | 57.2% |
| 観光・旅行 | 118 | other:57, company:28, country:16, individual:13, family:4 | 48.3% |
| IT・通信 | 106 | company:81, individual:25 | 76.4% |
| 製造業 | 87 | other:45, company:35, individual:7 | 51.7% |

## 3. 既存フィールド一覧

**総フィールド数**: 67

```
  action
  action_hex
  action_type
  after_hex
  after_state
  alias_ids
  annotation_status
  before_hex
  before_state
  canonical_id
  changing_lines_1
  changing_lines_2
  changing_lines_3
  classical_action_hexagram
  classical_after_hexagram
  classical_before_hexagram
  coi_status
  confidence_percent
  country
  created_at
  credibility_rank
  entity_type
  event_driver_type
  event_id
  event_phase
  evidence_notes
  free_tags
  hexagram_id
  hexagram_name
  hexagram_number
  hexagram_yao_id
  id
  interpretations
  is_canonical
  life_domain
  logic_memo
  lower_trigram
  main_domain
  outcome
  outcome_status
  pattern_type
  period
  pre_outcome_text
  primary_subject_id
  scale
  source_type
  sources
  story_summary
  subject_type
  success_level
  target_name
  transition_id
  trigger
  trigger_hex
  trigger_hex_original
  trigger_hex_reassigned
  trigger_hex_rule
  trigger_type
  trust_level
  trust_reason
  updated_at
  upper_trigram
  verification_confidence
  yao
  yao_analysis
  yao_context
  yao_name
```

## 4. Codex提案5特徴量 × 既存フィールドマッピング

### 4.1 stakeholder_count — 当事者数（変化に関わる主体の数）

**推定方法**: partial_rule_plus_llm

| 既存フィールド | 関連度 | 理由 |
|---------------|--------|------|
| `scale` | high | company=多, individual=少 の粗い推定 |
| `entity_type` | medium | company/individual/family等が分布 |
| `subject_type` | medium | entity_typeと類似 |
| `free_tags` | low | #M&A等から間接的に推定可能な場合あり |
| `story_summary` | medium | LLMで人数をテキストから抽出可能 |

**メモ**: scaleで粗い区分(company→多, individual→1-2)は可能。正確な数値にはstory_summaryからのLLM抽出が必要。ルールベース: company=10+, family=2-10, individual=1-3, country=100+, other=variable

### 4.2 resource_constraint — 資源制約（人材・資金・時間の制約度）

**推定方法**: partial_rule_plus_llm

| 既存フィールド | 関連度 | 理由 |
|---------------|--------|------|
| `scale` | high | individual→高制約, company→中, country→低 |
| `main_domain` | medium | 業界によりリソース水準が異なる |
| `story_summary` | high | 資金難/人手不足等の記述からLLM抽出可能 |
| `before_state` | medium | どん底→高制約, 絶頂→低制約 |
| `action_type` | low | 耐える=高制約, 攻める=低制約の傾向 |

**メモ**: scale + before_state の組み合わせで粗い推定可能。精密な推定にはstory_summaryのLLM分析が必要。ルールベース精度は推定60-70%

### 4.3 time_horizon — 時間軸（変化プロセスの所要期間）

**推定方法**: partial_rule

| 既存フィールド | 関連度 | 理由 |
|---------------|--------|------|
| `period` | high | 'YYYY-YYYY'形式からduration算出可能。12456/13060件パース可能 |
| `story_summary` | medium | LLMで時間関連表現を抽出可能 |
| `pattern_type` | low | パターンにより典型的な時間軸が異なる |

**メモ**: periodフィールドから12456/13060件(95.4%)でduration_years を直接算出可能。推定duration分布: mean=23.8年, median=10年, range=0-928年

### 4.4 reversibility — 可逆性（変化を元に戻せる度合い）

**推定方法**: rule_based

| 既存フィールド | 関連度 | 理由 |
|---------------|--------|------|
| `after_state` | high | 崩壊・消滅→不可逆, 現状維持→可逆 |
| `outcome` | high | Failure→多くの場合不可逆 |
| `action_type` | medium | 捨てる・撤退→不可逆傾向, 守る→可逆傾向 |
| `pattern_type` | medium | Hubris_Collapse→不可逆, Slow_Decline→部分的可逆 |
| `story_summary` | medium | 破産/倒産/消滅 等のキーワードで補完可能 |

**メモ**: after_state + outcome の組み合わせでルールベース推定が高精度で可能。崩壊・消滅+Failure→irreversible(1), V字回復→partially_reversible(3), 現状維持→highly_reversible(5) のようなマッピング。推定精度80%以上。LLM不要

### 4.5 consensus_cost — 合意形成コスト（意思決定に必要な関係者調整の難易度）

**推定方法**: partial_rule

| 既存フィールド | 関連度 | 理由 |
|---------------|--------|------|
| `scale` | high | country→極高, company→高, family→中, individual→低 |
| `action_type` | medium | 対話・融合→高, 逃げる→低 |
| `entity_type` | medium | scaleと類似 |
| `main_domain` | low | 政治/行政→高, 個人→低 |
| `story_summary` | medium | 交渉/調整/対立等の記述からLLM推定可能 |

**メモ**: scale + action_type の組み合わせでルールベース推定可能。country+対話=5, individual+攻める=1 のようなマッピング。精度は推定65-75%。精密にはstory_summaryのLLM分析が望ましい

## 5. 実現可能性マトリクス

| 特徴量 | 実現性 | コスト | 価値 | 推定方法 | ルール率 | LLM率 | 優先度 |
|--------|--------|--------|------|----------|---------|-------|--------|
| stakeholder_count | 3/5 | 3/5 | 4/5 | rule_plus_llm | 100% | 40% | **medium** |
| resource_constraint | 2/5 | 4/5 | 3/5 | llm_only | 50% | 80% | **low** |
| time_horizon | 4/5 | 1/5 | 4/5 | rule_only | 95.4% | 0% | **high** |
| reversibility | 4/5 | 1/5 | 5/5 | rule_only | 100% | 0% | **high** |
| consensus_cost | 3/5 | 2/5 | 4/5 | rule_plus_llm | 100% | 30% | **medium** |

### スコア説明
- **実現性**: 1(不可能)〜5(容易)
- **コスト**: 1(低コスト)〜5(高コスト)
- **価値**: 1(低価値)〜5(高価値)
- **ルール率**: ルールベースで推定可能な事例の割合
- **LLM率**: LLMバッチ処理が必要な事例の割合

#### stakeholder_count
- scaleフィールドから全件にルールベースで粗い推定可能 (company→10+, individual→1, family→2-10, country→1000+)。境界事例(自営業=companyだが実質1人等)の精密化にLLM必要。分離ロジックにおけるスケール判定の精度向上に直結する

#### resource_constraint
- 既存フィールドからの推定精度が低い。scale+before_stateで粗い傾向は出るが、同じscale=companyでも トヨタとスタートアップでは資源制約が全く異なる。story_summaryからのLLM抽出が不可欠で、13,060件のバッチ処理コスト大

#### time_horizon
- periodフィールドから12456/13060件(95.4%)でduration_yearsを直接算出可能。LLM不要。平均23.76年, 中央値10年。残りの事例も「平成後期」等からLLMでapprox推定可能だがコスト低い

#### reversibility
- after_state + outcome + action_type の組み合わせで ルールベース推定が高精度で可能。LLM不要。崩壊・消滅+Failure→1(不可逆), V字回復+Success→4(高可逆), 現状維持→3(中可逆)。スケール分離ロジックにおいて、個人の転職(可逆)と企業の倒産(不可逆)の区別に有用

#### consensus_cost
- scale + action_type で全件にルールベース推定可能。country+対話→5(最高), individual+攻める→1(最低)。精密化にはstory_summaryのLLM分析が望ましいが、ルールベースでも実用的精度(65-75%)が期待できる

## 6. 可逆性 (reversibility) ルールベース試算

- **マッピング成功**: 13,060件 (100.0%)
- **マッピング失敗**: 0件

| ラベル | 件数 |
|--------|------|
| mostly_reversible | 3,627 |
| irreversible | 3,464 |
| mostly_irreversible | 2,642 |
| highly_reversible | 2,609 |
| partially_reversible | 718 |

## 7. 合意形成コスト (consensus_cost) ルールベース試算

| スコア | 件数 |
|--------|------|
| 1 | 3,007 |
| 2 | 2,918 |
| 3 | 3,952 |
| 4 | 2,120 |
| 5 | 1,063 |

## 8. time_horizon — period フィールド分析

- **periodフィールド保有**: 13,060件
- **YYYY-YYYY形式パース可能**: 12,456件 (95.4%)
- **推定duration**: 平均23.76年, 中央値10年, 範囲0-928年

### period サンプル
- `2015-2020`
- `令和前期`
- `2009-2015`
- `2010-2012`
- `2014-2017`
- `2012-2018`
- `1990s-2000s`
- `2000-2010`
- `2016-2020`
- `2013-2015`

## 9. 推奨アクション

### 即時実装（ルールベース、LLM不要）

1. **reversibility** (可逆性)
   - after_state × outcome のルールテーブルで全件に付与可能
   - 推定カバレッジ: ~99%
   - コスト: スクリプト1本（1時間以内）
   - 価値: スケール分離の判定精度向上に直結

2. **time_horizon** (時間軸)
   - period フィールドからduration_yearsを算出
   - 即時カバレッジ: 95.4%
   - コスト: スクリプト1本（30分以内）
   - 価値: 変化パターンの時間的特性を定量化

### 中期実装（ルール + 境界事例のLLM補完）

3. **consensus_cost** (合意形成コスト)
   - scale × action_type で全件にベースライン付与
   - 境界事例（30%）のLLM補完で精度向上
   - コスト: ルール1時間 + LLM 3-4時間

4. **stakeholder_count** (当事者数)
   - scaleで粗い区分を全件に付与
   - 境界事例（40%）のLLM補完で精密化
   - コスト: ルール1時間 + LLM 4-5時間

### 保留（コスト対効果が低い）

5. **resource_constraint** (資源制約)
   - 既存フィールドからの推定精度が低い
   - 13,060件のLLMバッチ処理コストに見合う価値が不明確
   - 他4特徴量の実装・評価後に再検討
