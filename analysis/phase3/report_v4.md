# Phase 3 v4.0 Results Report

**Generated**: 2026-03-09T16:32:21.929997
**Cases**: 800
**Active hexagrams**: 61/64
**Null model permutations**: 1000

## Threshold w >= 1

| Test | Statistic | Observed | Null Mean±SD | z-score | p (Bonf.) | Direction | Decision |
|------|-----------|----------|-------------|---------|-----------|-----------|----------|
| A | Edge Overlap with Q6 | 0.0659 | 0.0875±0.0132 | -1.64 | 1.0000 | anti | fail_to_reject |
| B | Hamming Distance Distribution | 3.1075 | 3.0026±0.0526 | 1.99 | 1.0000 | anti | fail_to_reject |
| C | Laplacian Spectral Similarity | 0.1359 | 0.1152±0.0048 | 4.29 | 1.0000 | anti | fail_to_reject |
| D | Cuogua Symmetry | 0.1887 | 0.1368±0.0287 | 1.81 | 0.2150 | pro | fail_to_reject |
| E | Community Structure (NMI) | 0.0857 | 0.0873±0.0076 | -0.22 | 1.0000 | anti | fail_to_reject |

**Summary**: 0/5 tests rejected H0 (1 pro-isomorphism, 4 anti-isomorphism)

## Threshold w >= 2

| Test | Statistic | Observed | Null Mean±SD | z-score | p (Bonf.) | Direction | Decision |
|------|-----------|----------|-------------|---------|-----------|-----------|----------|
| A | Edge Overlap with Q6 | 0.0476 | 0.0758±0.0197 | -1.43 | 1.0000 | anti | fail_to_reject |
| B | Hamming Distance Distribution | 3.2173 | 3.0576±0.0881 | 1.81 | 1.0000 | anti | fail_to_reject |
| C | Laplacian Spectral Similarity | 0.1408 | 0.0877±0.0072 | 7.40 | 1.0000 | anti | fail_to_reject |
| D | Cuogua Symmetry | 0.2228 | 0.1397±0.0560 | 1.49 | 0.4100 | pro | fail_to_reject |
| E | Community Structure (NMI) | 0.3242 | 0.3335±0.0136 | -0.69 | 1.0000 | anti | fail_to_reject |

**Summary**: 0/5 tests rejected H0 (1 pro-isomorphism, 4 anti-isomorphism)

## Threshold w >= 3

| Test | Statistic | Observed | Null Mean±SD | z-score | p (Bonf.) | Direction | Decision |
|------|-----------|----------|-------------|---------|-----------|-----------|----------|
| A | Edge Overlap with Q6 | 0.0282 | 0.0648±0.0281 | -1.30 | 1.0000 | anti | fail_to_reject |
| B | Hamming Distance Distribution | 3.2865 | 3.1219±0.1344 | 1.23 | 1.0000 | anti | fail_to_reject |
| C | Laplacian Spectral Similarity | 0.1325 | 0.1232±0.0060 | 1.55 | 1.0000 | anti | fail_to_reject |
| D | Cuogua Symmetry | 0.0425 | 0.1630±0.1270 | -0.95 | 1.0000 | anti | fail_to_reject |
| E | Community Structure (NMI) | 0.4723 | 0.4862±0.0238 | -0.59 | 1.0000 | anti | fail_to_reject |

**Summary**: 0/5 tests rejected H0 (0 pro-isomorphism, 5 anti-isomorphism)

## Overall Conclusion

- **Isomorphism evidence level**: NONE
- Pro-isomorphism rejections: 0/5
- Anti-isomorphism rejections: 0/5
- Effect sizes (z): A=-1.642, B=1.993, C=4.295, D=1.807, E=-0.216

### Sensitivity Analysis

| Test | w≥1 | w≥2 | w≥3 | Stable? |
|------|-----|-----|-----|---------|
| A | fail_to_reject | fail_to_reject | fail_to_reject | Yes |
| B | fail_to_reject | fail_to_reject | fail_to_reject | Yes |
| C | fail_to_reject | fail_to_reject | fail_to_reject | Yes |
| D | fail_to_reject | fail_to_reject | fail_to_reject | Yes |
| E | fail_to_reject | fail_to_reject | fail_to_reject | Yes |
