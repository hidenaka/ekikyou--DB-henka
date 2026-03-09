# Phase 3 Measurement Specification

**Version**: 1.0
**Date**: 2026-03-09
**Status**: DRAFT — to be frozen before any Phase 3 test execution
**Prerequisite for**: Step 1 (Gold Set), Step 2 (Labeler), Step 3 (Graph), Step 4 (Tests)

---

## 1. Label Definition

### 1.1 State Encoding

Each case has two states: **before** and **after**.
Each state is encoded as a single hexagram from the 64-hexagram system.

A hexagram = (lower_trigram, upper_trigram), where each trigram is one of:

| Trigram | Binary | Meaning |
|---------|--------|---------|
| 乾 | 111 | Heaven |
| 兌 | 110 | Lake |
| 離 | 101 | Fire |
| 震 | 100 | Thunder |
| 巽 | 011 | Wind |
| 坎 | 010 | Water |
| 艮 | 001 | Mountain |
| 坤 | 000 | Earth |

A hexagram is therefore a 6-bit vector in (Z_2)^6 = {lower_bit2, lower_bit1, lower_bit0, upper_bit2, upper_bit1, upper_bit0}.

### 1.2 Assignment Rule

- **Single-label**: Each state receives exactly one hexagram assignment.
- No multi-label, no probability distributions, no "top-k" lists.
- Source fields: `before_lower_trigram` + `before_upper_trigram` -> before hexagram; `after_lower_trigram` + `after_upper_trigram` -> after hexagram.
- The legacy `before_hex` / `after_hex` fields (single trigram) are NOT used for Phase 3. These encode trigram-level labels only and were the source of the pure hexagram collapse (98.2%).

### 1.3 Hexagram Numbering

King Wen sequence (1-64). The mapping from (lower, upper) trigram pair to King Wen number is fixed and defined in `data/reference/iching_texts_ctext_legge_ja.json`.

---

## 2. Transition Definition

### 2.1 What Constitutes a Transition

A transition is a directed pair:

```
(before_hexagram, after_hexagram)
```

where both hexagrams are derived from the same case record.

### 2.2 Scope

- **Within-case only**: A transition connects the before-state and after-state of a single case.
- **Not time-series**: Transitions across different cases (e.g., "case A's after -> case B's before") are not considered.
- Each case contributes exactly 0 or 1 transition (0 if excluded per Section 5).

### 2.3 Transition Identity

Two transitions are the **same type** if and only if they share the same (before_hexagram_number, after_hexagram_number) pair. The transition (11 -> 13) from case X and the transition (11 -> 13) from case Y are counted as two instances of the same transition type.

---

## 3. Graph Construction Specification

### 3.1 Node Set

- **64 nodes**, one per hexagram (King Wen 1-64).
- All 64 nodes are always present in the graph, including those with zero frequency.
- Node attribute: `total_frequency` = count as before + count as after (across all valid cases).

### 3.2 Edge Set

- **Directed edges**: An edge from node A to node B exists if at least one case has before_hexagram=A and after_hexagram=B.
- **Edge weight** = number of cases with that specific (before, after) pair.
- Maximum possible edges: 64 x 64 = 4,096 (including self-loops).

### 3.3 Self-Loops

- **Included**. A case where before_hexagram = after_hexagram creates a self-loop.
- Self-loops are counted in degree calculations (both in-degree and out-degree).

### 3.4 Edge Weight Thresholds

Results must be reported at **three thresholds**:

| Threshold | Description | Purpose |
|-----------|-------------|---------|
| w >= 1 | All observed transitions | Maximum coverage |
| w >= 2 | Transitions observed at least twice | Remove singletons |
| w >= 3 | Transitions observed at least three times | Stability check |

The primary analysis uses **w >= 1**. Thresholds w >= 2 and w >= 3 serve as sensitivity checks. If results change qualitatively across thresholds, this indicates fragility and must be reported.

### 3.5 Missing Data Handling

A case is **excluded** from graph construction if:
- `before_lower_trigram` OR `before_upper_trigram` is missing/null/invalid
- `after_lower_trigram` OR `after_upper_trigram` is missing/null/invalid
- Either trigram value is not in {乾, 坤, 震, 巽, 坎, 離, 艮, 兌}

Excluded cases are counted and reported but do not contribute edges or node frequencies.

### 3.6 Graph Output Format

```json
{
  "metadata": {
    "n_cases_total": 11336,
    "n_cases_valid": "...",
    "n_cases_excluded": "...",
    "threshold": 1,
    "generated_at": "ISO-8601"
  },
  "nodes": [
    {"id": 1, "name": "乾為天", "lower": "乾", "upper": "乾", "freq_before": 0, "freq_after": 0, "freq_total": 0}
  ],
  "edges": [
    {"source": 11, "target": 13, "weight": 42}
  ]
}
```

---

## 4. Q6 Hypercube Reference Graph

### 4.1 Definition

Q6 is the 6-dimensional hypercube graph:
- 64 nodes (all 6-bit binary strings)
- Two nodes are adjacent if and only if they differ in exactly 1 bit (Hamming distance = 1)
- Each node has degree 6
- Total edges: 192 (undirected) or 384 (directed, both directions)

### 4.2 Mapping to Hexagrams

Node mapping: hexagram -> 6-bit vector via trigram binary encoding (Section 1.1).
The Q6 adjacency is determined by the binary encoding, not the King Wen sequence number.

### 4.3 Q6 Edge Set

An edge (A, B) is a "Q6 edge" if hamming_distance(binary(A), binary(B)) = 1.
This means exactly one trigram line (yao) changes between A and B.

---

## 5. Acceptance Criteria (Gate for Phase 3 Execution)

All four criteria must be met before running Phase 3 statistical tests. If any criterion fails, the corresponding remediation must be applied before proceeding.

### 5.1 Node Coverage

**Criterion**: >= 60 out of 64 hexagrams have non-zero frequency (across before OR after assignments in the valid case set).

**Current status (2026-03-09)**: 34/64 active (30 at zero). **NOT MET.**

**Remediation**: Gold set expansion (Step 1) targeting zero-occurrence hexagrams.

### 5.2 Minimum Node Support

**Criterion**: Each of the 60+ active nodes has a minimum support of >= 5 cases in the gold set.

**Current status (2026-03-09)**: 30/64 have >= 5 support. **NOT MET.**

**Remediation**: Active sampling to fill low-support hexagrams (Step 1).

### 5.3 Graph Connectivity

**Criterion**: Mean out-degree >= 3 at threshold w >= 1.

**Rationale**: A mean out-degree of 3 ensures the graph is sufficiently connected for spectral analysis (Test C) and community detection (Test E) to be meaningful. Below this, the graph fragments into disconnected components and these tests produce artifacts.

**Measurement**: out-degree of node i = number of distinct nodes j such that edge (i, j) exists with weight >= threshold. Mean over all 64 nodes.

### 5.4 No Dominant Hexagram

**Criterion**: No single hexagram accounts for > 15% of all assignments (before + after combined).

**Current status (2026-03-09)**: #13 天火同人 = 4,139 / 22,672 total assignments = 18.3%. **NOT MET.**

**Rationale**: A dominant hexagram distorts degree distributions and inflates edge overlap with Q6 by chance.

**Remediation**: If this persists after gold set expansion and reclassification, report results with and without the dominant hexagram as a sensitivity analysis.

---

## 6. Null Model Specification

### 6.1 Model Type

**Degree-preserving randomization** (configuration model for directed graphs).

### 6.2 Procedure

1. Take the observed directed graph G_obs (after thresholding).
2. For each permutation:
   a. Fix the in-degree sequence and out-degree sequence of G_obs.
   b. Randomly rewire edges while preserving both degree sequences.
   c. Result: a random graph G_rand with identical degree distribution but randomized edge targets.
3. Repeat 1,000 times to generate the null distribution.

### 6.3 What Is Preserved

- In-degree of each node (number of distinct sources pointing to it)
- Out-degree of each node (number of distinct targets it points to)
- Total number of edges

### 6.4 What Is NOT Preserved

- Specific edge targets (which node connects to which)
- Edge weights (randomized graphs are unweighted; thresholding is applied before randomization)
- Self-loop structure (self-loops participate in rewiring)

### 6.5 Implementation Notes

- Use `networkx.directed_configuration_model` or equivalent.
- Remove multi-edges after generation (collapse to simple directed graph).
- Seed each permutation for reproducibility.
- Store the random seed array for replication.

---

## 7. Statistical Tests (Phase 3)

Five tests, each with a pre-specified null hypothesis, test statistic, and decision rule.
All p-values are two-sided unless noted. Bonferroni correction is applied across the 5 tests (alpha = 0.05, alpha_adjusted = 0.01).

### Test A: Edge Overlap with Q6

**Question**: Do observed transitions preferentially follow Q6-adjacent hexagram pairs?

**H0**: The fraction of observed edges that are Q6 edges is equal to the expected fraction under the null model.

**Test statistic**: `overlap_ratio = |E_obs ∩ E_Q6| / |E_obs|`

**Null distribution**: Compute overlap_ratio for each of 1,000 null model graphs.

**Decision**: Reject H0 if observed overlap_ratio falls above the 99.5th percentile of the null distribution (one-sided: "more Q6-like than random", Bonferroni-adjusted).

**Effect size**: z = (observed - mean_null) / std_null.

### Test B: Hamming Distance Distribution

**Question**: Are observed transitions biased toward small Hamming distances (single-bit changes)?

**H0**: The mean Hamming distance of observed transitions equals the expected mean under the null model.

**Test statistic**: Mean Hamming distance across all observed transitions (weighted by edge weight).

**Null distribution**: Compute mean Hamming distance for each of 1,000 null model graphs, using the same Hamming metric on their edges.

**Decision**: Reject H0 if observed mean is below the 0.5th percentile of the null distribution (one-sided test for "closer than random", Bonferroni-adjusted).

**Effect size**: z-score.

### Test C: Laplacian Spectral Similarity

**Question**: Is the spectral structure of the observed graph similar to Q6?

**H0**: The Wasserstein distance between the Laplacian spectrum of G_obs and Q6 is equal to the expected distance under the null model.

**Test statistic**: `W1 = wasserstein_distance(eigenvalues(L_obs), eigenvalues(L_Q6))`
where L is the normalized Laplacian of the undirected version of the graph.

**Null distribution**: Compute W1 for each of 1,000 null model graphs vs Q6.

**Decision**: Reject H0 if observed W1 is below the 0.5th percentile (one-sided: "more similar than random", Bonferroni-adjusted).

**Critical safeguard**: Compute L_obs only on the **largest connected component**. Report the size of the LCC. If the LCC contains < 40 nodes, Test C is declared **inconclusive** (not failed, not passed).

**Effect size**: z-score.

### Test D: Cuogua (錯卦) Symmetry

**Question**: Do complementary hexagram pairs (each bit flipped) show similar transition behavior?

**H0**: The mean cosine similarity of transition probability vectors between cuogua pairs equals the expected similarity under the null model.

**Test statistic**: For each cuogua pair (h, h_complement), compute cosine similarity of their outgoing transition probability vectors. Report the mean over all 32 pairs.

**Null distribution**: Same computation on 1,000 null model graphs.

**Decision**: Reject H0 if observed mean similarity exceeds the 99.5th percentile (one-sided: "more symmetric than random", Bonferroni-adjusted).

**Note**: Cuogua pairs where both nodes have zero out-degree are excluded from the mean. Report the number of excluded pairs.

**Effect size**: z-score.

### Test E: Community Structure (NMI)

**Question**: Does the community structure of the observed graph resemble Q6's natural partition?

**H0**: The Normalized Mutual Information (NMI) between community assignments of G_obs and Q6's partition is equal to expected NMI under the null model.

**Q6 partition**: The 6-dimensional hypercube has a natural partition into 8 groups of 8 (by lower trigram). Use this 8-group partition as the reference.

**Test statistic**: Apply Louvain community detection to G_obs (undirected version). Compute NMI between detected communities and Q6's 8-group partition.

**Null distribution**: Same procedure on 1,000 null model graphs.

**Decision**: Reject H0 if observed NMI exceeds the 99.5th percentile (one-sided: "more similar to Q6 partition than random", Bonferroni-adjusted).

**Stability check**: Run Louvain 10 times on G_obs with different random seeds. Report mean and std of NMI. If std > 0.1, declare the community structure unstable and Test E inconclusive.

**Effect size**: z-score.

---

## 8. Reporting Requirements

### 8.1 Per-Test Report

Each test must report:
- Observed test statistic value
- Null distribution: mean, std, 2.5th and 97.5th percentiles
- Raw p-value
- Bonferroni-adjusted p-value
- Effect size (z-score) with interpretation: |z| < 0.5 negligible, 0.5-1.0 small, 1.0-2.0 medium, > 2.0 large
- Direction: pro-isomorphism, anti-isomorphism, or neutral

### 8.2 Summary Table

| Test | Statistic | p (raw) | p (Bonf.) | z-score | Direction | Decision |
|------|-----------|---------|-----------|---------|-----------|----------|

### 8.3 Sensitivity Analysis

Report the summary table at each of the three edge weight thresholds (w >= 1, w >= 2, w >= 3). Flag any test whose decision changes across thresholds.

---

## 9. No-Go Criteria

These are hard stops. If any is triggered, the corresponding action replaces further Phase 3 testing.

### 9.1 Single-Label Breakdown

**Trigger**: Gold set annotation reveals that independent annotators consistently assign different hexagrams to the same case (inter-annotator agreement kappa < 0.4 at the hexagram level).

**Action**: Abandon single-label graph. Switch to probabilistic transition graph where edge weights represent expected transition probabilities under the annotation distribution. Redefine all 5 tests for the probabilistic setting. Document as a methodology contribution.

### 9.2 Insufficient Node Coverage

**Trigger**: After gold set expansion and reclassification (Steps 1-2), the number of active hexagrams (non-zero frequency) remains <= 40 out of 64.

**Action**: Phase 3 input does not exist. The (Z_2)^6 space is too sparse for isomorphism testing. Document the sparsity pattern as a finding. Investigate whether a reduced model (e.g., trigram-level 8-node graph, or subset of hexagrams) is testable.

### 9.3 Classifier Failure

**Trigger**: The hexagram classifier (Step 2) achieves macro-F1 < 0.75 on the gold set.

**Action**: Full-corpus Phase 3 is unreliable because label noise dominates signal. Restrict analysis to the gold set only (subset study). Report results with explicit "gold-set-only" qualifier and note that generalization to full corpus is not supported.

---

## 10. Data Flow Summary

```
cases.jsonl (11,336 records)
    |
    v
[Exclude invalid] -- report exclusion count
    |
    v
Valid cases (with before_hexagram + after_hexagram)
    |
    v
[Build transition list: (before_hex_num, after_hex_num) x N]
    |
    v
[Construct directed graph G_obs]
    |        |
    |        v
    |   [Apply threshold w >= {1, 2, 3}]
    |        |
    |        v
    |   [G_obs_thresholded]
    |
    v
[Check Acceptance Criteria (Section 5)]
    |
    +--> FAIL --> Remediation or No-Go
    |
    +--> PASS
         |
         v
    [Generate 1,000 null model graphs]
         |
         v
    [Run Tests A-E on G_obs vs null distribution]
         |
         v
    [Report per Section 8]
```

---

## 11. Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-03-09 | Initial specification. Created per GPT-5.4 recommendation (Step 0-B of v4.0 action plan). |
