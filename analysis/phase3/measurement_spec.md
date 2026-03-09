# Measurement Specification: Transition Graph G_obs

**Document**: `analysis/phase3/measurement_spec.md`
**Created**: 2026-03-09
**Purpose**: Fix the measurement instrument definition before running Phase 3 isomorphism tests (Step 0 of phase3_v4_action_plan.md)

---

## 1. Label Definition

Each case receives exactly **one** hexagram for the before-state and **one** for the after-state.

- **Hexagram** = (lower_trigram, upper_trigram), where each trigram is drawn from {乾, 坤, 震, 巽, 坎, 離, 艮, 兌}.
- **Lower trigram** = internal/foundational driver of the state.
- **Upper trigram** = external/visible manifestation of the state.
- **Encoding**: Each trigram maps to a 3-bit vector (see `TRIGRAM_BITS` in `isomorphism_test.py`). A hexagram is therefore a 6-bit vector: `lower_3bit + upper_3bit`.
- **King Wen ordering**: Hexagrams are numbered 1--64 following the traditional King Wen sequence. The mapping from (lower, upper) trigram pairs to King Wen numbers is defined by the reference file `data/reference/iching_texts_ctext_legge_ja.json`.

### Single-label assumption

The current schema assigns exactly one hexagram per state per case. Whether this single-label assumption is valid (i.e., whether the mapping from real-world state descriptions to hexagrams is sufficiently deterministic) is an open empirical question. Step 0-A of the action plan calls for validation via dual-annotation agreement analysis. If agreement is low, the model must shift to a probabilistic (multi-label or distribution-over-hexagrams) graph.

### Field resolution

The fields `classical_before_hexagram` and `classical_after_hexagram` in `cases.jsonl` use two formats:
1. `"NUMBER_NAME"` (e.g., `"52_艮"`) -- resolved by parsing the integer prefix.
2. Hexagram name (e.g., `"艮為山"`, `"地雷復"`) -- resolved by lookup against `local_name` in the reference file.
3. Variant characters (e.g., `遁→遯`, `無妄→无妄`) are handled by fallback mapping.

---

## 2. Transition Definition

- A **transition** is a directed edge from `classical_before_hexagram` to `classical_after_hexagram` within the same case.
- **Self-loops** (before == after) are included. Currently 11 unique self-loop edges exist.
- Each case contributes exactly **one** edge.
- Edges are **weighted by count**: if N cases produce the same (before, after) pair, the edge weight is N.
- Cases where either hexagram field cannot be resolved are **excluded** from graph construction.

### What a transition represents

A transition represents an observed state change in a real-world entity (company, person, nation, family, etc.). The before-hexagram captures the initial condition; the after-hexagram captures the resulting condition. The transition is not a prediction; it is a post-hoc annotation of an observed change.

---

## 3. Graph Construction Specification

**G_obs** is a directed, weighted graph on 64 nodes.

| Property | Specification |
|----------|--------------|
| Node set | Fixed at {1, 2, ..., 64} (all King Wen hexagrams), regardless of data coverage |
| Edge (u, v, w) | Exists if at least 1 case transitions from hexagram u to hexagram v; w = count of such cases |
| Edge filtering | None. All observed transitions are included (no minimum weight threshold) |
| Self-loops | Included |
| Isolated nodes | Retained in the graph but flagged as zero-frequency |
| Directionality | Preserved. Edge (u→v) and edge (v→u) are distinct |

### Derived graphs

- **G_obs_undirected**: The undirected version of G_obs, obtained by ignoring edge direction and summing weights for (u,v) and (v,u). Used for Tests A, C, E.
- **Q6**: The 6-dimensional hypercube graph on the same 64 nodes, with edges between all pairs at Hamming distance 1 in the 6-bit encoding. Q6 has exactly 192 edges and is 6-regular.

---

## 4. Acceptance Criteria

For Phase 3 isomorphism tests to produce valid results, the following minimum data quality conditions must be met:

### 4.1 Node coverage

- **Required**: At least **60 of 64** nodes must have non-zero frequency (appearing as either before or after hexagram in at least one case).
- **Rationale**: Tests C (Laplacian spectrum) and D (cuogua symmetry) are structurally degraded when many nodes are isolated. With 47/64 nodes having zero out-degree (current state), Test D could only evaluate 2/32 cuogua pairs.

### 4.2 Minimum node support

- Each active node (non-zero frequency) must have a combined (before + after) frequency of at least **5 cases**.
- **Rationale**: Nodes with 1--2 occurrences contribute noise rather than signal to the transition probability matrix.

### 4.3 Mean out-degree of active nodes

- The mean out-degree of active nodes (nodes with out-degree > 0) should be at least **3**.
- **Rationale**: Below this threshold, the transition graph is too sparse for community detection (Test E) and spectral analysis (Test C) to be meaningful.

### 4.4 Connectivity

- The largest weakly connected component should contain at least **58 of 64** nodes.
- **Rationale**: Disconnected subgraphs cannot be compared to Q6 (which is connected) in a structurally meaningful way.

---

## 5. Current Status

**Data snapshot**: 2026-03-09, 11,336 cases in `data/raw/cases.jsonl`.

### 5.1 Resolution rate

| Metric | Value |
|--------|-------|
| Total cases | 11,336 |
| Cases with resolvable before AND after | 11,336 (100%) |
| Excluded cases | 0 |

### 5.2 Node coverage

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| Unique hexagrams as before | 43/64 | -- | -- |
| Unique hexagrams as after | 39/64 | -- | -- |
| Unique hexagrams (either) | **52/64** | >=60 | FAIL |
| Isolated nodes (zero frequency) | **12** | <=4 | FAIL |

### 5.3 Hexagrams with zero occurrences (12 hexagrams)

| KW | Name | Upper Trigram | Lower Trigram |
|----|------|---------------|---------------|
| 17 | 沢雷随 | 兌 | 震 |
| 28 | 沢風大過 | 兌 | 巽 |
| 31 | 沢山咸 | 兌 | 艮 |
| 43 | 沢天夬 | 兌 | 乾 |
| 45 | 沢地萃 | 兌 | 坤 |
| 47 | 沢水困 | 兌 | 坎 |
| 50 | 火風鼎 | 離 | 巽 |
| 54 | 雷沢帰妹 | 震 | 兌 |
| 56 | 火山旅 | 離 | 艮 |
| 58 | 兌為沢 | 兌 | 兌 |
| 60 | 水沢節 | 坎 | 兌 |
| 64 | 火水未済 | 離 | 坎 |

**Pattern**: 7 of 8 hexagrams with 兌 (沢) as upper trigram are missing. 3 of 8 with 離 (火) as upper trigram are missing. The labeling pipeline has a systematic blind spot for the 兌 trigram.

### 5.4 Nodes with insufficient support (<5 cases)

24 nodes have total frequency (before + after) below 5:

| KW | Name | Before | After | Total |
|----|------|--------|-------|-------|
| 17 | 沢雷随 | 0 | 0 | 0 |
| 28 | 沢風大過 | 0 | 0 | 0 |
| 31 | 沢山咸 | 0 | 0 | 0 |
| 43 | 沢天夬 | 0 | 0 | 0 |
| 45 | 沢地萃 | 0 | 0 | 0 |
| 47 | 沢水困 | 0 | 0 | 0 |
| 50 | 火風鼎 | 0 | 0 | 0 |
| 54 | 雷沢帰妹 | 0 | 0 | 0 |
| 56 | 火山旅 | 0 | 0 | 0 |
| 58 | 兌為沢 | 0 | 0 | 0 |
| 60 | 水沢節 | 0 | 0 | 0 |
| 64 | 火水未済 | 0 | 0 | 0 |
| 14 | 火天大有 | 0 | 1 | 1 |
| 19 | 地沢臨 | 0 | 1 | 1 |
| 34 | 雷天大壮 | 0 | 1 | 1 |
| 11 | 地天泰 | 2 | 0 | 2 |
| 16 | 雷地豫 | 1 | 1 | 2 |
| 30 | 離為火 | 0 | 2 | 2 |
| 41 | 山沢損 | 1 | 1 | 2 |
| 61 | 風沢中孚 | 2 | 0 | 2 |
| 21 | 火雷噬嗑 | 0 | 3 | 3 |
| 49 | 沢火革 | 0 | 3 | 3 |
| 26 | 山天大畜 | 4 | 0 | 4 |
| 37 | 風火家人 | 1 | 3 | 4 |

**Status**: 24/64 nodes fail the >=5 minimum support criterion. **FAIL**.

### 5.5 Edge statistics

| Metric | Value |
|--------|-------|
| Unique directed edges | 391 |
| Self-loop edges | 11 |
| Edge weight min | 1 |
| Edge weight max | 834 (地風升→天火同人) |
| Edge weight mean | 28.99 |
| Edge weight median | 4 |

### 5.6 Degree distribution

| Metric | All 64 nodes | Active nodes only (52) |
|--------|-------------|----------------------|
| In-degree: nodes > 0 | 39 | 39 |
| Out-degree: nodes > 0 | 43 | 43 |
| In-degree mean | 6.11 | 7.52 |
| Out-degree mean | 6.11 | 7.52 |
| In-degree max | 38 | 38 |
| Out-degree max | 28 | 28 |

**Mean out-degree of active nodes with out-degree > 0**: 391 edges / 43 nodes = **9.09**. Criterion (>=3) is **PASS**.

However, 21 of 64 nodes have out-degree 0, meaning they never serve as a transition source.

### 5.7 Connected components

| Metric | Value |
|--------|-------|
| Weakly connected components | 13 |
| Largest WCC size | 52 nodes |
| Singleton components | 12 (the zero-frequency hexagrams) |
| Strongly connected components | 36 |
| Largest SCC size | 29 nodes |

**Criterion** (largest WCC >= 58): **FAIL** (52/64).

### 5.8 Concentration and skew

The top 5 edges by weight account for a disproportionate share of all transitions:

| Rank | Edge | Weight | Cumulative % |
|------|------|--------|-------------|
| 1 | 地風升(46)→天火同人(13) | 834 | 7.4% |
| 2 | 水山蹇(39)→天火同人(13) | 672 | 13.3% |
| 3 | 山雷頤(27)→天火同人(13) | 568 | 18.3% |
| 4 | 地風升(46)→山雷頤(27) | 364 | 21.5% |
| 5 | 山雷頤(27)→水山蹇(39) | 349 | 24.6% |

Node 13 (天火同人) receives 4,130 of 11,336 after-transitions (36.4%), indicating extreme sink concentration. The top 5 before-nodes (46, 27, 39, 55, 42) account for 5,580/11,336 = 49.2% of all transitions.

### 5.9 Impact on Phase 3 test results (current run)

| Test | Result | Root cause |
|------|--------|-----------|
| A: Edge overlap | p=0.61, anti | Sparse graph; only 110 unique undirected edges; 8 overlap Q6 |
| B: Hamming distance | p=0.72, anti | Transitions skewed to a few hub nodes, not governed by bit-distance |
| C: Laplacian spectrum | p=0.00, pro | Spectral shape partially resembles Q6 despite sparsity |
| D: Cuogua symmetry | p=0.63, anti | Only 2/32 valid cuogua pairs (47 nodes have zero out-degree) |
| E: Community NMI | p=0.08, pro | Marginal; 36 SCCs vs Q6's clean 5-community structure |
| **Combined** | **Weak** | Tests C/E show structural signal, but A/B/D are uninformative due to data sparsity |

---

## 6. Gap Analysis

### 6.1 Summary of criterion failures

| Criterion | Required | Actual | Gap |
|-----------|----------|--------|-----|
| Node coverage (non-zero) | >=60/64 | 52/64 | 12 hexagrams missing |
| Node minimum support (>=5) | 64/64 active nodes | 40/64 | 24 nodes below threshold |
| Largest WCC | >=58 nodes | 52 nodes | 6 nodes short |
| Mean out-degree (active) | >=3 | 9.09 | PASS |

### 6.2 Root causes

1. **Systematic labeling gap for 兌 (沢) trigram**: 7/8 hexagrams with 兌 as upper trigram have zero occurrences. This is a labeler deficiency, not a data deficiency -- 沢 appears in real-world cases (M&A joy/openness, diplomatic exchanges, financial liquidity events) but the classifier fails to assign it.

2. **Extreme concentration on a few hexagrams**: 5 hexagrams (天火同人, 水山蹇, 山雷頤, 地風升, 雷火豊) account for >60% of all node frequencies. This suggests either (a) the labeler has a strong bias toward these hexagrams, or (b) the case corpus is dominated by a narrow set of transformation types.

3. **Directional asymmetry**: Several hexagrams appear only as before (e.g., 天山遯: 181 before, 0 after) or only as after (e.g., 天火同人: 1 before, 4130 after). This creates structurally degenerate rows/columns in the transition matrix.

### 6.3 What must change to meet acceptance criteria

| Action | Expected impact | Dependency |
|--------|----------------|------------|
| **Fix 兌-trigram labeling** in classifier | +7--8 nodes (兌 upper hexagrams) | Step 2: Labeler rebuild |
| **Fix 離-trigram labeling** gaps | +3 nodes (火風鼎, 火山旅, 火水未済) | Step 2: Labeler rebuild |
| **Targeted case collection** for zero-occurrence hexagrams | +12 nodes if labeled correctly | Step 1: Gold set |
| **Redistribute hub concentration** via labeler calibration | Reduces sink bias on node 13 | Step 1 + Step 2 |
| **Dual-annotation validation** | Confirms whether single-label is viable | Step 1: Gold set |

### 6.4 No-Go conditions

If after Steps 1--2:
- Dual-annotation agreement (Cohen's kappa) < 0.6 across hexagram classes → single-label model is invalid; shift to probabilistic graph.
- Active nodes still < 50 after labeler rebuild → Phase 3 tests structurally cannot be valid; terminate isomorphism hypothesis.
- 兌-trigram recall < 0.3 in gold set → trigram classifier fundamentally cannot distinguish this trigram; document as negative result.

---

## Appendix A: Reference Implementation

Graph construction is implemented in `analysis/phase3/isomorphism_test.py`, function `build_g_obs()` (lines 213--251). The function:

1. Iterates over all cases
2. Resolves `classical_before_hexagram` and `classical_after_hexagram` to King Wen numbers
3. Excludes cases where resolution fails or the hexagram lacks a 6-bit mapping
4. Constructs a `nx.DiGraph` with all 64 nodes and weighted edges from transition counts

No modifications to this function are proposed. The measurement instrument (graph construction) is sound; the input data quality is the problem.

## Appendix B: Trigram-to-Bit Mapping

| Trigram | Symbol | Bits | Decimal |
|---------|--------|------|---------|
| 乾 | ☰ | (1,1,1) | 7 |
| 兌 | ☱ | (1,1,0) | 6 |
| 離 | ☲ | (1,0,1) | 5 |
| 震 | ☳ | (1,0,0) | 4 |
| 巽 | ☴ | (0,1,1) | 3 |
| 坎 | ☵ | (0,1,0) | 2 |
| 艮 | ☶ | (0,0,1) | 1 |
| 坤 | ☷ | (0,0,0) | 0 |

Hexagram 6-bit = lower_trigram_3bit + upper_trigram_3bit.
Example: 乾為天 (KW=1) = 乾 over 乾 = (1,1,1,1,1,1). 坤為地 (KW=2) = 坤 over 坤 = (0,0,0,0,0,0).
