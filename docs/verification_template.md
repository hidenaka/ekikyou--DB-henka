# 検証テンプレート（パイロット30件）

**目的**: IAA（Inter-Annotator Agreement）測定用の検証記録フォーマット
**対象**: `data/pilot/pilot_verification_30.jsonl`（30件）
**作成日**: 2026-01-13
**参照**: `docs/label_definitions_v1.md`（ラベル定義書）

---

## 検証者情報

| 項目 | 記入欄 |
|------|--------|
| reviewer_id | （例: R01, R02） |
| reviewer_name | （任意） |
| verification_date | YYYY-MM-DD |
| total_time_minutes | （検証にかかった時間） |

---

## 判定ガイドライン（簡潔版）

### フィールド1: outcome_status（結果検証状態）

| 値 | 判定基準 |
|----|----------|
| `verified_correct` | ソースで結果を確認し、事例記述と**一致** |
| `verified_incorrect` | ソースで結果を確認し、事例記述と**不一致** |
| `unverified` | ソースで結果を**確認できない** |

**判定フロー**:
```
ソースを開く → 結果を確認できるか？
  → YES: 事例記述と一致？
      → 一致 → verified_correct
      → 不一致 → verified_incorrect
  → NO → unverified
```

### フィールド2: verification_confidence（検証確信度）

| 値 | 判定基準 |
|----|----------|
| `high` | 一次ソースで直接確認 / 複数独立ソースで裏付け |
| `medium` | 二次ソース（ニュース等）で確認 / 単一だが信頼性の高いソース |
| `low` | 間接的な確認のみ / 情報の新鮮さに懸念 |
| `none` | ソースが見つからない / アクセス不能 |

**整合性ルール**:
- verified_correct/verified_incorrect → high または medium のみ
- unverified → low または none のみ

### フィールド3: coi_status（利益相反: Conflict of Interest）

| 値 | 判定基準 |
|----|----------|
| `none` | 独立した第三者による情報（新聞、学術機関等） |
| `self` | 当事者が自組織について発信（企業公式IR等） |
| `affiliated` | 親会社・子会社・提携先による発信 |
| `unknown` | 著者の所属が不明 |

---

## 検証フォーム

### 記入方法

1. 各事例の情報を確認
2. ソースURLにアクセスして検証
3. 3つのフィールドを記入
4. notes欄に判定理由や気づきを記録

### 出力形式

検証完了後、以下のJSONL形式で提出してください:

```json
{"pilot_id": "P001", "outcome_status": "verified_correct", "verification_confidence": "high", "coi_status": "none", "notes": "日経記事で確認"}
```

---

## パイロット事例 検証シート

### --- verified層（P001-P010）---

#### P001: 三井物産・三菱商事 再結集（1950年代）
| 項目 | 値 |
|------|-----|
| target_name | 三井物産・三菱商事 再結集（1950年代） |
| current_outcome | Success |
| sources | Google検索URL |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P002: 中国恒大集団/Evergrande（不動産バブル崩壊）
| 項目 | 値 |
|------|-----|
| target_name | 中国恒大集団/Evergrande（不動産バブル崩壊） |
| current_outcome | Failure |
| sources | https://www.mlit.go.jp/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P003: ドイツ（西ドイツ）経済の奇跡
| 項目 | 値 |
|------|-----|
| target_name | ドイツ（西ドイツ） |
| current_outcome | Mixed |
| sources | https://www.bundesregierung.de/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P004: Volkswagen・ディーゼル不正からの再建（2015-2022）
| 項目 | 値 |
|------|-----|
| target_name | Volkswagen・ディーゼル不正からの再建（2015-2022） |
| current_outcome | Success |
| sources | https://www.volkswagen.com/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P005: SpaceX・Starship開発で民間宇宙を革新（2012-2024）
| 項目 | 値 |
|------|-----|
| target_name | SpaceX・Starship開発で民間宇宙を革新（2012-2024） |
| current_outcome | Success |
| sources | https://www.spacex.com/vehicles/starship/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P006: Amazon Japan・日本EC市場で圧倒的シェア確立（2010-2024）
| 項目 | 値 |
|------|-----|
| target_name | Amazon Japan・日本EC市場で圧倒的シェア確立（2010-2024） |
| current_outcome | Success |
| sources | https://www.aboutamazon.com/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P007: フィンランド_NATO加盟と安全保障転換
| 項目 | 値 |
|------|-----|
| target_name | フィンランド_NATO加盟と安全保障転換 |
| current_outcome | Success |
| sources | https://www.mext.go.jp/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P008: 楽天モバイル（1.5兆円累積損失）
| 項目 | 値 |
|------|-----|
| target_name | 楽天モバイル（1.5兆円累積損失） |
| current_outcome | Mixed |
| sources | https://corp.rakuten.co.jp/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P009: 日産自動車_鮎川義介の設立
| 項目 | 値 |
|------|-----|
| target_name | 日産自動車_鮎川義介の設立 |
| current_outcome | Success |
| sources | https://www.nissan-global.com/JP/COMPANY/PROFILE/HERITAGE/1930/, https://www.nissan-global.com/JP/HERITAGE/LEGENDS/LEGEND_01/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P010: 碧桂園 中国不動産危機の波及（2023-2024）
| 項目 | 値 |
|------|-----|
| target_name | 碧桂園 中国不動産危機の波及（2023-2024） |
| current_outcome | Failure |
| sources | https://www.mlit.go.jp/ |
| stratum | verified |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

### --- high_quality層（P011-P020）---

#### P011: 就職失敗若者Sさん_生活保護から正社員へ
| 項目 | 値 |
|------|-----|
| target_name | 就職失敗若者Sさん_生活保護から正社員へ |
| current_outcome | Success |
| sources | https://www.meti.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P012: リーマン・ブラザーズ（金融危機での崩壊）
| 項目 | 値 |
|------|-----|
| target_name | リーマン・ブラザーズ（金融危機での崩壊） |
| current_outcome | Failure |
| sources | https://ja.wikipedia.org/wiki/リーマン・ブラザーズ, https://www.nikkei.com/article/DGXNASGM1503J_V10C08A900000/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P013: 闇バイト問題の深刻化（2023-2024）
| 項目 | 値 |
|------|-----|
| target_name | 闇バイト問題の深刻化（2023-2024） |
| current_outcome | Failure |
| sources | https://www.npa.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P014: 丹下健三
| 項目 | 値 |
|------|-----|
| target_name | 丹下健三 |
| current_outcome | Success |
| sources | https://www.mext.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P015: 桑田佳祐
| 項目 | 値 |
|------|-----|
| target_name | 桑田佳祐 |
| current_outcome | Success |
| sources | https://www.mext.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P016: 元美容師Eさん_手荒れで引退後起業
| 項目 | 値 |
|------|-----|
| target_name | 元美容師Eさん_手荒れで引退後起業 |
| current_outcome | Success |
| sources | https://www.mhlw.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P017: 自動車販売店・ディーラー変革（2018-2024）
| 項目 | 値 |
|------|-----|
| target_name | 自動車販売店・ディーラー変革（2018-2024） |
| current_outcome | Mixed |
| sources | https://www.jada.or.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P018: 人気コーヒー店・チェーン店との競争（2015-2022）
| 項目 | 値 |
|------|-----|
| target_name | 人気コーヒー店・チェーン店との競争（2015-2022） |
| current_outcome | Success |
| sources | https://www.scaj.org/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P019: 巽結果失敗事例023-3
| 項目 | 値 |
|------|-----|
| target_name | 巽結果失敗事例023-3 |
| current_outcome | Failure |
| sources | https://www.soumu.go.jp/ |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P020: WHO - COVID-19パンデミック宣言
| 項目 | 値 |
|------|-----|
| target_name | WHO - COVID-19パンデミック宣言 |
| current_outcome | Mixed |
| sources | https://www.who.int/emergencies/diseases/novel-coronavirus-2019 |
| stratum | high_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

### --- low_quality層（P021-P030）---

#### P021: Sさん_40代_元正社員
| 項目 | 値 |
|------|-----|
| target_name | Sさん_40代_元正社員 |
| current_outcome | Failure |
| sources | Google検索URL |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P022: 震結果失敗事例005-2
| 項目 | 値 |
|------|-----|
| target_name | 震結果失敗事例005-2 |
| current_outcome | Failure |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P023: 兌結果失敗事例018-1
| 項目 | 値 |
|------|-----|
| target_name | 兌結果失敗事例018-1 |
| current_outcome | Failure |
| sources | Google検索URL |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P024: 山一證券 自主廃業
| 項目 | 値 |
|------|-----|
| target_name | 山一證券 自主廃業 |
| current_outcome | Failure |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P025: 巽結果失敗事例009-1
| 項目 | 値 |
|------|-----|
| target_name | 巽結果失敗事例009-1 |
| current_outcome | Failure |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P026: 横浜赤レンガ倉庫の再生
| 項目 | 値 |
|------|-----|
| target_name | 横浜赤レンガ倉庫の再生 |
| current_outcome | Success |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P027: 艮結果失敗事例017-2
| 項目 | 値 |
|------|-----|
| target_name | 艮結果失敗事例017-2 |
| current_outcome | Failure |
| sources | Google検索URL |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P028: 中国（改革開放5年目）
| 項目 | 値 |
|------|-----|
| target_name | 中国（改革開放5年目） |
| current_outcome | Success |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P029: 離輝き事例5-1
| 項目 | 値 |
|------|-----|
| target_name | 離輝き事例5-1 |
| current_outcome | Success |
| sources | (なし) |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

#### P030: 兌交流事例12-1
| 項目 | 値 |
|------|-----|
| target_name | 兌交流事例12-1 |
| current_outcome | Success |
| sources | Google検索URL |
| stratum | low_quality |

**検証結果**:
| フィールド | 値 |
|------------|-----|
| outcome_status | ________________ |
| verification_confidence | ________________ |
| coi_status | ________________ |
| notes | |

---

## 検証完了チェックリスト

- [ ] 30件すべてに outcome_status を記入した
- [ ] 30件すべてに verification_confidence を記入した
- [ ] 30件すべてに coi_status を記入した
- [ ] 整合性ルールを確認した（verified_correct/incorrect → high/medium、unverified → low/none）
- [ ] JSONLファイルを出力した

---

## JSONL出力テンプレート

検証結果を以下の形式で出力してください。ファイル名: `reviewer_{reviewer_id}_results.jsonl`

```jsonl
{"pilot_id": "P001", "outcome_status": "verified_correct", "verification_confidence": "high", "coi_status": "none", "notes": ""}
{"pilot_id": "P002", "outcome_status": "verified_correct", "verification_confidence": "medium", "coi_status": "self", "notes": ""}
...
{"pilot_id": "P030", "outcome_status": "unverified", "verification_confidence": "none", "coi_status": "unknown", "notes": ""}
```

---

## 付録: 選択肢一覧

### outcome_status
- `verified_correct`
- `verified_incorrect`
- `unverified`

### verification_confidence
- `high`
- `medium`
- `low`
- `none`

### coi_status
- `none`
- `self`
- `affiliated`
- `unknown`
