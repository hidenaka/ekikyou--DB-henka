# グローバル事例収集ガイドライン

## 目的

日本偏重（77.6%）を解消し、国際事例を増加させる。

## 収集基準

### 1. 必須要件

| 項目 | 要件 |
|------|------|
| ソース | tier1-4のいずれか必須（tier5のみは不可） |
| trust_level | verified または plausible |
| target_name | 実名必須（匿名・仮名は不可） |
| outcome | Success/Failure/Mixed のいずれか必須 |

### 2. 許容ソースTier

| Tier | ソース例 | Gold判定 |
|------|----------|----------|
| tier1_official | SEC, FCA, EU機関, 各国政府 | ◯ |
| tier2_major_media | Reuters, Bloomberg, FT, BBC, NYT | ◯ |
| tier3_specialist | TechCrunch, Wired, Forbes | △ (Silver) |
| tier4_corporate | 企業公式IR/プレスリリース | ◯ |
| tier5_pr | PR Newswire, BusinessWire | △ (Silver) |

### 3. 必須メタ情報

```json
{
  "target_name": "企業/組織名（英語可）",
  "country": "国名（ISO表記推奨）",
  "year": 2020,
  "pattern_type": "Steady_Growth等",
  "outcome": "Success/Failure/Mixed",
  "trust_level": "verified/plausible",
  "sources": ["https://...", "https://..."],
  "description": "事例概要（100-200字）"
}
```

## 地域別ターゲットリスト

### 米国（目標: +185件）

#### テック企業（50件）
- [ ] Palantir - Defense/AI転換
- [ ] Snowflake - クラウドデータ成長
- [ ] Stripe - フィンテック成長
- [ ] Databricks - AIプラットフォーム
- [ ] Figma - Adobe買収中止後
- [ ] Notion - プロダクティビティ
- [ ] Discord - ゲームからコミュニティへ
- [ ] Robinhood - ミーム株危機
- [ ] Coinbase - 暗号資産上場
- [ ] Block (Square) - フィンテック転換

#### 金融（30件）
- [ ] JPMorgan Chase - デジタル転換
- [ ] Goldman Sachs - Marcus撤退
- [ ] BlackRock - ESG投資拡大
- [ ] Berkshire Hathaway - 後継者問題
- [ ] Charles Schwab - TD Ameritrade統合
- [ ] SVB Financial - 銀行破綻
- [ ] First Republic - 銀行破綻

#### 小売・消費財（30件）
- [ ] Walmart - eコマース強化
- [ ] Target - オムニチャネル成功
- [ ] Costco - 会員制成長
- [ ] Starbucks - 中国展開/労働問題
- [ ] McDonald's - デジタル戦略
- [ ] Nike - D2C転換
- [ ] Lululemon - アスレジャー成長

#### ヘルスケア（25件）
- [ ] Pfizer - COVID-19ワクチン
- [ ] Moderna - mRNA革命
- [ ] Johnson & Johnson - タルク訴訟
- [ ] CVS Health - Aetna統合
- [ ] UnitedHealth - Optum成長

#### エネルギー・産業（25件）
- [ ] ExxonMobil - エネルギー転換
- [ ] Chevron - 脱炭素戦略
- [ ] Boeing - 737MAX危機
- [ ] Lockheed Martin - 防衛需要
- [ ] General Electric - 3分割

#### エンターテインメント（25件）
- [ ] Warner Bros. Discovery - 統合後混乱
- [ ] Paramount Global - 売却交渉
- [ ] Spotify - ポッドキャスト投資
- [ ] Live Nation - チケット独占批判

### 欧州（目標: +108件）

#### ドイツ（30件）
- [ ] Siemens Energy - スピンオフ後
- [ ] BASF - 中国依存
- [ ] Bayer - Monsanto訴訟
- [ ] Deutsche Bank - 長期低迷
- [ ] Allianz - 保険イノベーション
- [ ] Adidas - Kanye契約解消

#### フランス（25件）
- [ ] LVMH - ラグジュアリー成長
- [ ] TotalEnergies - エネルギー転換
- [ ] BNP Paribas - デジタルバンク
- [ ] Airbus - Boeing対抗
- [ ] Renault - 日産アライアンス

#### 英国（25件）
- [ ] BP - ネットゼロ戦略
- [ ] Shell - エネルギー転換
- [ ] HSBC - アジア再集中
- [ ] Unilever - 分割検討
- [ ] AstraZeneca - COVID-19ワクチン

#### その他欧州（28件）
- [ ] Nestlé (スイス) - 健康食品転換
- [ ] ABB (スイス) - 産業自動化
- [ ] Spotify (スウェーデン) - 音楽ストリーミング
- [ ] ASML (オランダ) - 半導体装置独占
- [ ] Shell (オランダ) - 本社移転
- [ ] Philips (オランダ) - リコール危機

### 中国（目標: +66件）

- [ ] BYD - EVリーダー
- [ ] NIO - EV新興
- [ ] Pinduoduo - Eコマース急成長
- [ ] JD.com - 物流強化
- [ ] Meituan - フードデリバリー
- [ ] ByteDance - TikTok規制対応
- [ ] Ant Group - IPO中止後
- [ ] Evergrande - 不動産危機
- [ ] Country Garden - 不動産危機

## 品質チェックリスト

新規事例追加時に以下を確認:

- [ ] target_nameが実名である
- [ ] sourcesにtier1-4が1つ以上含まれる
- [ ] sourcesのURLが実在する（404でない）
- [ ] countryが正しく設定されている
- [ ] outcomeがSuccess/Failure/Mixedのいずれか
- [ ] pattern_typeが有効なEnum値
- [ ] descriptionが100字以上

## バッチファイル命名規則

```
batch_[region]_[category]_[number].json
```

例:
- `batch_us_tech_001.json`
- `batch_eu_auto_001.json`
- `batch_cn_ev_001.json`
