# Label Ambiguity Analysis: Is Single-Label Hexagram Assignment Valid?

**Date**: 2026-03-09
**Method**: 50 cases sampled (first 10, middle 10, last 10, 20 random) from 11,336 total cases
**Scope**: Evaluate whether each state can be validly mapped to exactly ONE hexagram (lower + upper trigram)

---

## 1. Executive Summary

**Single-label hexagram assignment is fundamentally feasible, but the current system has a severe structural problem that must be addressed first: labels are not derived from text interpretation — they are near-deterministically derived from categorical metadata fields.**

The current trigram assignment is not a genuine "meaning-based" mapping from narrative text to I Ching symbolism. It is effectively a mechanical relabeling of `before_state` / `after_state` / `trigger_type` / `action_type` labels into trigram names. This means the ambiguity question is moot under the current system (there is almost zero ambiguity because the mapping is formulaic), but would become highly relevant if genuine text-based annotation were implemented.

---

## 2. Critical Finding: Formulaic Mapping

### 2.1 before_state → before_lower_trigram (near-deterministic)

| before_state | Dominant lower trigram | Dominance % |
|---|---|---|
| どん底・危機 | 坎 | 99.7% |
| 停滞・閉塞 | 艮 | 99.9% |
| 安定・平和 | 坤 | 97.9% |
| 絶頂・慢心 | 乾 | 98.8% |
| 混乱・カオス | 震 | 92.8% |
| 成長痛 | 巽 | 79.6% |

Five of six states map to a single trigram with >97% consistency. Even 成長痛 (the least deterministic) maps to 巽 at 79.6%. **The before_lower_trigram is essentially a synonym for before_state, not an independent assessment.**

### 2.2 after_state → after_lower_trigram (near-deterministic)

| after_state | Dominant lower trigram | Dominance % |
|---|---|---|
| V字回復・大成功 | 乾 | 99.6% |
| 崩壊・消滅 | 坎 | 99.6% |
| 現状維持・延命 | 坤 | 98.3% |
| 迷走・混乱 | 艮 | 96.0% |
| 変質・新生 | 震 | 78.5% |
| 縮小安定・生存 | 艮 | 78.6% |

### 2.3 Two trigrams are completely absent from lower positions

- **離 (Li)**: 0 cases as before_lower, only 76 as after_lower (0.7%)
- **兌 (Dui)**: 0 cases as before_lower, only 3 as after_lower (0.03%)

This means the 8-trigram system is functioning as a **6-trigram system** for lower (inner) trigrams. The annotation protocol defines 離 (vision/passion/clarity) and 兌 (joy/exchange/openness) as valid inner states, but the data contains essentially zero instances. This is a design flaw — these concepts are real and should appear, but the formulaic mapping from 6 state categories to 6 trigrams structurally excludes them.

### 2.4 Extreme concentration in hexagram combinations

- **After side**: 乾/離 = 36.4% of all cases (one combination accounts for over a third)
- **Before side**: Top 3 combinations = 41.8% (坤/巽, 艮/震, 坎/艮)
- Only 43 of 64 possible before hexagrams and 39 of 64 after hexagrams appear

---

## 3. Case-by-Case Ambiguity Assessment (50 Cases)

### 3.1 Methodology

For each case, I evaluated:
1. Does the story_summary text support the assigned trigrams?
2. Could a different trigram be equally or more justified from the text alone?
3. Is the assignment clearly driven by the state label rather than text interpretation?

### 3.2 Ambiguity Distribution

| Level | Count | Percentage | Description |
|---|---|---|---|
| **CLEAR** | 8 | 16% | Only one reasonable assignment from text |
| **MODERATE** | 18 | 36% | 2 plausible but current is defensible |
| **AMBIGUOUS** | 19 | 38% | 2-3 equally plausible assignments |
| **INTRACTABLE** | 5 | 10% | Text too sparse or multi-faceted to determine |

### 3.3 Examples by Category

#### CLEAR (8 cases, 16%)

**CORP_JP_009** (JAL + Inamori): Before: 坎/艮, After: 乾/離
> 「放漫経営と高コスト体質により戦後最大の事業会社破綻を経験。稲盛和夫氏を会長に迎え…」

- Inner state clearly 坎 (bankruptcy = deep crisis). No ambiguity.
- After: Strong leader drove success = 乾 inner. Clear.

**PERS_JP_3019** (STAP cell scandal): Before: 震/艮, After: 坎/艮
> 「STAP細胞論文…データ捏造疑惑…退職…研究者としてのキャリアは事実上終焉」

- Before: 震 (shock/chaos) is apt — sudden upheaval. However, one could argue 離 (intense public visibility/scrutiny) as outer trigram rather than 艮.
- After: 坎 (ruin) is unambiguous.

**COUN_JP_324** (Singapore): Before: 坎/坎, After: 乾/離
> 「マレーシアから追放される形で独立…リー・クアンユーの強権的統治で…」

- Before: Both inner and outer are crisis/danger — expelled, tiny state, survival at stake. Clear 坎/坎.

#### MODERATE (18 cases, 36%)

**CORP_JP_001**: Before: 艮/震, After: 乾/離
> 「停滞から外部ショックを契機に大胆な刷新を行い、V字回復」

- 艮 (stagnation) as inner is reasonable, but could also be 坤 (passive/receptive in a mature market).
- Distinction depends on whether "停滞" is read as intentional stopping (艮) or environmental passivity (坤). The text says "停滞" which maps better to 艮.
- **Verdict**: Current assignment is defensible but 坤 is a plausible alternative for lower.

**PERS_JP_109** (pitcher): Before: 乾/兌, After: 坎/震
> 「日本で無敵の投手として活躍（兌）し、メジャー挑戦を決断（乾）」

- Note: The story_summary itself contains trigram annotations in parentheses — evidence that the text was written *with* the trigram mapping in mind, not independently.
- 乾 (strong will, expansion) fits the "invincible in Japan" state. But one could argue 兌 (enjoying success, peak satisfaction) for inner.
- After: 坎 (injuries, failure) fits. But 艮 (forced stop) could also work for a player sidelined by injuries.

**CORP_JP_008** (Hitachi): Before: 艮/震, After: 乾/離
> 「リーマンショックの影響で…7873億円の赤字を計上」

- Before inner: The company was in "停滞" (stagnation) before the Lehman shock. But the text emphasizes the massive loss = crisis (坎), not mere stagnation. The 艮 assignment follows from before_state="停滞・閉塞" rather than from the text emphasis on financial devastation.
- **This is a case where the state label drives the assignment contrary to the text's emphasis.**

**CTRY_JP_1563** (Hong Kong): Before: 震/震, After: 艮/震
> 「2019年の逃亡犯条例改正反対デモで200万人…2020年…国家安全維持法を電撃施行」

- Before: 震/震 (internal chaos + external shock) is reasonable. But one could argue inner = 離 (the situation was highly visible, transparent — media attention worldwide).
- After: 艮 (stopped/suppressed) inner is arguable, but 坎 (dire situation, danger for democrats) seems equally or more fitting.

#### AMBIGUOUS (19 cases, 38%)

**CORP_JP_013** (Fujifilm): Before: 坎/乾, After: 艮/離
> 「フィルム製造で培った抗酸化技術…化粧品・医薬品・医療機器へ事業の大転換」

- Before inner: 坎 (crisis — main product disappearing) is the current assignment. But the company was *proactively* pivoting = 乾 (strong initiative). The before_state="どん底" drives the 坎 choice, but the text shows active transformation, not passive suffering.
- Before outer: 乾 (growth environment) doesn't match "film market shrinking 20%/year" — that's clearly 坎 (hostile environment) or 震 (disruptive change).
- After inner: 艮 (stopping/accumulating) is odd for a company that successfully transformed. 震 (new beginning) or 乾 (thriving) would fit better.
- **Multiple trigrams have stronger textual support than the assigned ones.**

**CORP_JP_012** (Casio): Before: 巽/艮
> 「Windows95の登場により、主力だったワープロ専用機市場が消滅するという衝撃に直面」

- Before inner: 巽 (gradual adaptation) follows from before_state="成長痛". But "市場が消滅するという衝撃" is 坎 (crisis) or 震 (shock), not gradual adaptation.
- Before outer: 艮 (stagnation/barrier) — the market *disappeared*, which is more like 震 (disruptive shock) than 艮 (mere stagnation).
- **Text clearly describes shock and crisis, but state labels give growth-adaptation.**

**PERS_JP_3013** (soccer player → cultural entrepreneur): Before: 乾/艮
> 「2006年W杯後に29歳で突如現役引退。約3年間世界放浪の旅」

- Before inner: 乾 (strong expansion/leadership) follows from before_state="絶頂・慢心". But the text describes someone who voluntarily stopped — 艮 (intentional stop/reflection) as inner state is equally valid.
- Before outer: 艮 (stagnation) — but a W杯 context is 離 (public attention) or 兌 (celebration/peak enjoyment).

**Line 4507** (Tourism association): Before: 震/離, After: 艮/坎
> 「インバウンド回復に対応する観光協会が、多言語対応と体験コンテンツを整備中」

- This is clearly a 巽 (gradual adaptation) case, not 震 (shock/chaos). The organization is methodically building infrastructure. Assigned 震 as inner because before_state="混乱・カオス" but the text shows organized preparation, not chaos.

**Line 3583** (Sumo wrestler injury): Before: 坤/巽, After: 艮/震
> 「大関昇進間近の力士が膝の大怪我で番付急落。4年かけてリハビリ」

- Before inner: 坤 (receptive/stable) from before_state="安定・平和". But "大関昇進間近" = ascending power = 乾 or at least 巽 (gradual rise).
- Before outer: 巽 (gradual change) — but knee injury is 震 (sudden shock) not gradual.
- **Both trigrams seem misaligned with the text.**

**Line 3812** (ERP consultant → DX): Before: 坤/巽, After: 乾/離
> 「ERPシステム導入で名を馳せたITコンサルがクラウド化で需要変化。5年かけてDX支援へ転換」

- Before: 坤 (stable) fits the "名を馳せた" (established) phase. But 乾 (leadership position) or 兌 (enjoying success) could equally apply.
- After: 乾/離 is reasonable but mechanical — successful transformation always gets 乾/離.

#### INTRACTABLE (5 cases, 10%)

**FAM_JP_432**: Before: 艮/震, After: 坎/艮
> 「変化を拒否して失敗したケース。」

- Story summary is one sentence with no specifics. Impossible to independently assess trigrams. The assignment is pure label-driven (停滞・閉塞 → 艮).

**OTHR_JP_575**: After: 離 as lower
> 「明確なビジョンで成功した組織。」

- One sentence. No specifics. Trigram assignment cannot be validated.

**Line 4013** (workplace bullying → job change): Before: 震/離, After: 坤/巽
> 「職場いじめに遭った30代が転職し、新しい職場で穏やかに働くようになる。」

- Before inner: 震 (chaos) from before_state="混乱・カオス" — but bullying could equally be 坎 (enduring hardship/danger).
- Before outer: 離 (public attention/visibility) — but workplace bullying is not a "public attention" phenomenon. 坎 (hostile environment) fits better.
- After: 坤/巽 (receptive/gradual) is reasonable for "穏やかに働く" but could also be 艮/坤 or even 兌 (finding peace/contentment).
- **Multiple equally valid readings with no way to decide.**

---

## 4. Trigram Definition Analysis

### 4.1 Are the 8 definitions operationalizable?

The annotation protocol definitions are **well-written and clear in isolation**, but suffer from three structural problems:

#### Problem 1: Semantic Overlaps

| Boundary | Overlap Description | Protocol Coverage |
|---|---|---|
| 坎 vs 艮 | Crisis vs. intentional stop — many real cases have forced stagnation that blends both | Covered (boundary case 5) |
| 坤 vs 艮 | Receptive stability vs. intentional maintenance — hard to distinguish without explicit intent markers | Covered (boundary case 2) |
| 巽 vs 坤 | Gradual adaptation vs. passive following — requires "strategy" keyword detection | Covered (boundary case 3) |
| 乾 vs 兌 | Aggressive expansion vs. cooperative growth | Covered (boundary case 6) |
| 離 vs 震 (outer) | Sustained attention vs. sudden shock | Covered (boundary case 4) |

The protocol addresses all major overlaps, which is good. However, the boundary rules themselves introduce new ambiguities (e.g., "is this choice intentional?" requires inference from text that is often absent).

#### Problem 2: Asymmetric Usage Creates Gaps

The system defines 8 trigrams but only uses 6 for inner positions:
- **離 as inner (0 cases)**: "Clear vision/passion" as an inner state is conceptually valid. Cases like visionary founders, passionate reformers *should* get 離 inner, but they get 乾 (expansion) instead because the decision tree prioritizes 乾 at position 1.
- **兌 as inner (0 cases)**: "Joy/exchange/openness" as an inner state should appear in cases of peak satisfaction, celebration. But these get mapped to 乾 (from "絶頂" state label) instead.

**Root cause**: The decision tree's priority ordering (乾 > 坤 > 震 > 巽 > 坎 > 離 > 艮 > 兌) means 離 and 兌 are always shadowed by higher-priority matches. Every case that could be 離 also has some element of 乾 or 巽; every case that could be 兌 also has 乾 elements.

#### Problem 3: Inner/Outer Distinction Is Often Artificial

Many narrative texts do not cleanly separate "inner state" from "outer environment." For short summaries (23% of cases have <30 characters), there is insufficient text to independently determine both inner and outer.

### 4.2 Recommendations for Definition Improvement

1. **Restructure the decision tree**: Remove strict priority ordering. Instead, use a feature-matching approach where each trigram's features are checked independently and the best match (by feature count) wins.
2. **Create explicit operationalizations for 離 and 兌 as inner trigrams**: Provide 5+ example narratives where these should be selected.
3. **Add minimum text length requirement**: Cases with story_summary < 50 characters should be flagged as `uncertain` by default.

---

## 5. The Core Question: Is Single-Label Valid?

### 5.1 Under Current (Formulaic) System: Yes, trivially

Because the current system is a near-deterministic mapping from categorical labels (before_state, after_state, trigger_type, action_type) to trigrams, there is essentially no ambiguity. Each state label produces one trigram with >95% consistency. Single-label assignment "works" because it is simply a relabeling exercise.

**But this is not meaningful**: The hexagram adds zero information beyond what the state labels already contain. The hexagram system's purpose — to provide a richer, more nuanced symbolic mapping — is completely defeated.

### 5.2 Under Genuine Text-Based Annotation: Challenging but feasible

If trigrams were assigned by reading story_summary and independently evaluating inner/outer states (as the annotation protocol intends), the analysis of 50 cases shows:

- **16% CLEAR**: Clean, unambiguous mapping. These are cases with rich narrative and a single dominant dynamic.
- **36% MODERATE**: Defensible single assignment, but with one plausible alternative. This is acceptable for a symbolic system.
- **38% AMBIGUOUS**: Two or three equally valid assignments. This is the challenge zone.
- **10% INTRACTABLE**: Text too sparse to determine. These need richer narratives or should be excluded.

**Conclusion**: ~52% of cases (CLEAR + MODERATE) can sustain single-label assignment. ~38% would benefit from a secondary/alternative label or confidence score. ~10% need better source data.

### 5.3 Recommended Approach

**Single-label with confidence metadata** (not multi-label):

```json
{
  "before_lower": "坎",
  "before_lower_confidence": 0.75,
  "before_lower_alternatives": [{"trigram": "艮", "confidence": 0.20}],
  "before_upper": "震",
  "before_upper_confidence": 0.85,
  "before_upper_alternatives": []
}
```

**Rationale against multi-label**: The I Ching hexagram system is inherently a single-state-at-a-time framework. Each hexagram represents a specific configuration. Assigning multiple hexagrams would undermine the system's interpretive logic (each hexagram has specific line meanings, nuclear hexagrams, etc.). Instead, keep single primary assignment but record uncertainty.

**Rationale against probability distributions**: A full 8-way probability distribution over trigrams would create an illusion of precision that the data cannot support. A primary + top alternative with confidence is the right level of detail.

---

## 6. Systemic Issues Beyond Ambiguity

### 6.1 The "乾/離 Attractor" Problem

36.4% of all after-states are mapped to hexagram 乾/離 (乾 lower, 離 upper). This means over a third of all transformations "end up in the same place." This extreme concentration suggests:
- The after-state label "V字回復・大成功" is overused (covers 35.3% of cases)
- The formulaic mapping funnels all "success" stories to the same hexagram
- The I Ching has multiple hexagrams for different *kinds* of success (泰 = harmonious prosperity, 大有 = great possession, 既済 = completion), but the current system collapses all success into one

### 6.2 The before_hex / after_hex Inconsistency

The `before_hex` and `after_hex` fields contain single-trigram names (坎, 乾, etc.) rather than hexagram names or numbers. These often do not match the lower/upper trigram combination. For example:
- CORP_JP_001: lower=艮, upper=震 → before_hex="艮" (ignores upper)
- CORP_JP_008: lower=艮, upper=震 → before_hex="坤" (completely unrelated)

This suggests the hex fields and the trigram fields were generated by different processes and have not been reconciled.

### 6.3 Text Contamination

Some story_summary texts contain trigram annotations embedded in the narrative:
> 「日本で無敵の投手として活躍（兌）し、メジャー挑戦を決断（乾）」

This means the text was written *after* trigrams were assigned, not independently. For genuine text-based reannotation, these embedded annotations would need to be stripped.

---

## 7. Final Verdict

| Question | Answer |
|---|---|
| Can each state map to exactly ONE hexagram? | **Yes, in principle** — but requires genuine text-based judgment, not label relay |
| Is the current mapping valid? | **No** — it is a formulaic relabeling that adds no information |
| Is the annotation protocol adequate? | **Mostly yes** — well-designed decision tree and boundary rules, but needs fixes for 離/兌 underrepresentation |
| What should be done? | 1. Break the state-label → trigram determinism. 2. Reannotate from text using the protocol. 3. Add confidence/alternative metadata. 4. Enrich short summaries before annotation. |

### Priority Actions

1. **Immediate**: Identify and flag the 2,618 cases with story_summary < 30 characters as unsuitable for text-based annotation without enrichment.
2. **Short-term**: Run a pilot reannotation of 100 cases using the annotation protocol with strict text-only judgment (ignoring state labels). Measure inter-annotator agreement.
3. **Medium-term**: Restructure the decision tree to give 離 and 兌 fair representation as inner trigrams. Consider expanding the state label vocabulary to 8 categories matching the 8 trigrams.
4. **Long-term**: Implement single-label-with-confidence as the standard format. Use the confidence data to identify cases that genuinely need richer narratives.
