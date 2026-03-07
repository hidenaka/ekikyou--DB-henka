#!/usr/bin/env python3
"""
独立再アノテーション実験 — アノテーションバイアス検証

目的:
  cases.jsonlの classical_before_hexagram / classical_after_hexagram が
  LLMによって対角構造（Δ_lower == Δ_upper）を持つよう生成されている可能性を検証する。

方法:
  1. 層化サンプリング（source_type層化）で200件を抽出
  2. 各事例に対して独立した2つのプロンプト（before用/after用）を生成
     - before用: after情報を一切含めない
     - after用: before情報を一切含めない
  3. 64卦から1つを選ばせるJSON出力形式のプロンプト

出力:
  - analysis/phase3/reannotation_sample.json
  - analysis/phase3/reannotation_design.md
"""

import json
import sys
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

# ---------- パス設定 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
REFERENCE_FILE = BASE_DIR / "data" / "reference" / "iching_texts_ctext_legge_ja.json"
AUDIT_FILE = BASE_DIR / "analysis" / "phase3" / "annotation_bias_audit.json"
OUTPUT_SAMPLE = BASE_DIR / "analysis" / "phase3" / "reannotation_sample.json"
OUTPUT_DESIGN = BASE_DIR / "analysis" / "phase3" / "reannotation_design.md"

# isomorphism_test.py のユーティリティを再利用
sys.path.insert(0, str(BASE_DIR / "analysis" / "phase3"))
from isomorphism_test import (
    load_json,
    build_name_to_kw,
    build_kw_to_bits,
    resolve_hexagram_field,
    TRIGRAM_BITS,
)

# ---------- 定数 ----------
RANDOM_SEED = 42

# 例外事例は全件含める（211件）ため、対角事例の追加サンプル目標を別途設定
# 合計 = 例外全件 + 対角追加サンプル
TARGET_DIAGONAL_ADDITIONAL = 200  # 対角事例から追加で200件

# 対角追加サンプルの層化目標件数（source_type別）
STRATUM_TARGETS = {
    "news": 100,
    "book": 40,
    "official": 40,
    "blog": 20,
}


# ============================================================
# ユーティリティ
# ============================================================

def load_cases():
    """cases.jsonlを読み込み"""
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def compute_diagonal(bits_from, bits_to):
    """Δ_lower == Δ_upper かを判定"""
    delta = tuple(a ^ b for a, b in zip(bits_from, bits_to))
    return delta[:3] == delta[3:]


def load_exception_ids(audit_file):
    """annotation_bias_audit.jsonから例外事例のIDセットを取得"""
    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)
    records = audit.get("part2_exception_audit", {}).get("exception_records", [])
    return {r["transition_id"] for r in records}


def build_hexagram_list(reference_data):
    """64卦の選択肢リストを構築（プロンプト用）"""
    hexagram_list = []
    for kw_str, hdata in reference_data["hexagrams"].items():
        kw = int(kw_str)
        hexagram_list.append({
            "number": kw,
            "name": hdata["local_name"],
            "short": hdata.get("local_short", ""),
            "slug": hdata.get("slug", ""),
        })
    hexagram_list.sort(key=lambda x: x["number"])
    return hexagram_list


# ============================================================
# 1. 層化サンプリング
# ============================================================

def stratified_sample(cases, name_to_kw, kw_to_bits, exception_ids):
    """
    source_type層化サンプリング + 例外事例のoversampling

    戦略:
    1. 例外事例（非対角）は全件含める
    2. 残り枠を各stratum目標に応じて埋める
    """
    rng = random.Random(RANDOM_SEED)

    # 有効な遷移を持つ事例のみフィルタ
    valid_cases = []
    for case in cases:
        before_val = case.get("classical_before_hexagram", "")
        after_val = case.get("classical_after_hexagram", "")
        kw_from = resolve_hexagram_field(before_val, name_to_kw)
        kw_to = resolve_hexagram_field(after_val, name_to_kw)
        if kw_from is None or kw_to is None:
            continue
        if kw_from not in kw_to_bits or kw_to not in kw_to_bits:
            continue
        bits_from = kw_to_bits[kw_from]
        bits_to = kw_to_bits[kw_to]
        is_diagonal = compute_diagonal(bits_from, bits_to)
        tid = case.get("transition_id", "")
        is_exception = (tid != "" and tid in exception_ids)

        valid_cases.append({
            "case": case,
            "kw_from": kw_from,
            "kw_to": kw_to,
            "bits_from": bits_from,
            "bits_to": bits_to,
            "is_diagonal": is_diagonal,
            "is_exception": is_exception,
            "source_type": case.get("source_type", "unknown"),
        })

    # 1) 例外事例を全件確保
    exceptions = [v for v in valid_cases if v["is_exception"]]
    exception_by_source = Counter(e["source_type"] for e in exceptions)
    print(f"  例外事例（全件含める）: {len(exceptions)}件")
    for st, cnt in exception_by_source.most_common():
        print(f"    {st}: {cnt}")

    # 2) 対角事例から追加サンプルを抽出
    selected_ids = {e["case"].get("transition_id", f"__exc_{i}") for i, e in enumerate(exceptions)}

    # 非例外の対角事例をstratum別に分類
    non_exceptions_by_stratum = defaultdict(list)
    for v in valid_cases:
        tid = v["case"].get("transition_id", "")
        if tid and tid not in selected_ids and not v["is_exception"]:
            non_exceptions_by_stratum[v["source_type"]].append(v)

    # 3) 各stratumから目標件数をランダムサンプリング
    additional = []
    for st, target in STRATUM_TARGETS.items():
        pool = non_exceptions_by_stratum.get(st, [])
        if len(pool) <= target:
            additional.extend(pool)
        else:
            additional.extend(rng.sample(pool, target))

    # 全サンプルを統合
    all_selected = exceptions + additional
    rng.shuffle(all_selected)

    print(f"\n  サンプル総数: {len(all_selected)}件")
    source_dist = Counter(s["source_type"] for s in all_selected)
    for st, cnt in source_dist.most_common():
        print(f"    {st}: {cnt}")

    diag_count = sum(1 for s in all_selected if s["is_diagonal"])
    exc_count = sum(1 for s in all_selected if s["is_exception"])
    print(f"  対角事例: {diag_count}, 非対角例外: {exc_count}")

    return all_selected


# ============================================================
# 2. プロンプト生成
# ============================================================

def generate_prompt_a(case, hexagram_choices_text):
    """
    before_hexagram用プロンプト
    after情報は一切含めない
    """
    c = case
    title = c.get("target_name", "N/A")
    summary = c.get("story_summary", "")
    before_state = c.get("before_state", "")
    trigger_type = c.get("trigger_type", "")
    main_domain = c.get("main_domain", "")
    scale = c.get("scale", "")
    period = c.get("period", "")

    prompt = f"""あなたは易経の専門家です。以下の事例の「変化前の状態」に最もふさわしい64卦を1つ選んでください。

## 事例情報
- 対象: {title}
- 分野: {main_domain}
- 規模: {scale}
- 時期: {period}
- 変化前の状態: {before_state}
- 変化のきっかけ: {trigger_type}

## 事例の概要（変化前に着目して読んでください）
{summary}

## 指示
- 上記の「変化前の状態」を最もよく象徴する64卦を1つ選んでください
- 変化後の状態については一切考慮しないでください
- 「変化前の状態」の本質（構造、力学、エネルギーの在り方）を捉える卦を選んでください

## 64卦の選択肢
{hexagram_choices_text}

## 出力形式（JSON）
```json
{{
  "hexagram_number": <1-64の整数>,
  "hexagram_name": "<卦名>",
  "reasoning": "<選定理由を2-3文で>"
}}
```"""
    return prompt


def generate_prompt_b(case, hexagram_choices_text):
    """
    after_hexagram用プロンプト
    before情報は一切含めない
    """
    c = case
    title = c.get("target_name", "N/A")
    summary = c.get("story_summary", "")
    after_state = c.get("after_state", "")
    action_type = c.get("action_type", "")
    main_domain = c.get("main_domain", "")
    scale = c.get("scale", "")
    period = c.get("period", "")

    prompt = f"""あなたは易経の専門家です。以下の事例の「変化後の状態」に最もふさわしい64卦を1つ選んでください。

## 事例情報
- 対象: {title}
- 分野: {main_domain}
- 規模: {scale}
- 時期: {period}
- 変化後の状態: {after_state}
- 取られた行動: {action_type}

## 事例の概要（変化後の結果に着目して読んでください）
{summary}

## 指示
- 上記の「変化後の状態」を最もよく象徴する64卦を1つ選んでください
- 変化前の状態については一切考慮しないでください
- 「変化後の状態」の本質（構造、力学、エネルギーの在り方）を捉える卦を選んでください

## 64卦の選択肢
{hexagram_choices_text}

## 出力形式（JSON）
```json
{{
  "hexagram_number": <1-64の整数>,
  "hexagram_name": "<卦名>",
  "reasoning": "<選定理由を2-3文で>"
}}
```"""
    return prompt


def build_hexagram_choices_text(hexagram_list):
    """64卦選択肢のテキストを構築"""
    lines = []
    for h in hexagram_list:
        lines.append(f"{h['number']:2d}. {h['name']}（{h['short']}）")
    return "\n".join(lines)


# ============================================================
# 3. 出力
# ============================================================

def build_sample_output(selected, hexagram_choices_text, name_to_kw, reference_data):
    """サンプルJSON出力を構築"""
    hexagram_list = build_hexagram_list(reference_data)
    samples = []

    for item in selected:
        case = item["case"]
        delta = tuple(a ^ b for a, b in zip(item["bits_from"], item["bits_to"]))
        delta_lower = "".join(str(b) for b in delta[:3])
        delta_upper = "".join(str(b) for b in delta[3:])

        # 元のhexagram名を取得
        before_hex = case.get("classical_before_hexagram", "")
        after_hex = case.get("classical_after_hexagram", "")

        sample = {
            "case_id": case.get("transition_id", ""),
            "title": case.get("target_name", ""),
            "source_type": case.get("source_type", ""),
            "main_domain": case.get("main_domain", ""),
            "scale": case.get("scale", ""),
            "original_before_hex": before_hex,
            "original_after_hex": after_hex,
            "original_before_kw": item["kw_from"],
            "original_after_kw": item["kw_to"],
            "original_diagonal": item["is_diagonal"],
            "delta_lower": delta_lower,
            "delta_upper": delta_upper,
            "is_exception": item["is_exception"],
            "prompt_a": generate_prompt_a(case, hexagram_choices_text),
            "prompt_b": generate_prompt_b(case, hexagram_choices_text),
        }
        samples.append(sample)

    return samples


def generate_design_document(selected, total_cases, total_valid):
    """実験計画書を生成"""
    # 統計情報の収集
    source_dist = Counter(s["source_type"] for s in selected)
    n_diagonal = sum(1 for s in selected if s["is_diagonal"])
    n_exception = sum(1 for s in selected if s["is_exception"])
    n_total = len(selected)

    # サンプルサイズの根拠計算
    # 効果量: 95% → 80% の差 = 0.15
    # z検定で p1=0.95, p2=0.80, alpha=0.05, power=0.80
    p1, p2 = 0.95, 0.80
    alpha = 0.05
    z_alpha = 1.96  # 両側
    z_beta = 0.842  # power=0.80
    p_bar = (p1 + p2) / 2
    n_required = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                   + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
                  / (p1 - p2) ** 2)
    n_required = math.ceil(n_required)

    doc = f"""# 独立再アノテーション実験計画書

## 1. 目的

cases.jsonlの `classical_before_hexagram` と `classical_after_hexagram` に付与された64卦タグが、
LLMのアノテーションプロセスによって人工的な対角構造（Delta_lower == Delta_upper）を生成している
可能性を検証する。

### 背景
- 現行データの97.8%の遷移が対角構造（Delta_lower == Delta_upper）を示す
- ランダム期待値は12.7%であり、極端な乖離がある（enrichment = 7.68x）
- source_typeごとに例外率が劇的に異なる:
  - official: 57.4%（最も多くの例外）
  - book: 13.4%
  - news: 0.8%
  - blog: 0.0%
- この分布はLLMの生成プロセスに起因するバイアスの可能性を強く示唆する

## 2. 仮説

### H0（帰無仮説）
独立アノテーション（before/afterを互いに見せずに再アノテーション）でも、
対角構造（Delta_lower == Delta_upper）の割合は元データと同等（>95%）。

**意味**: 対角構造はデータの本質的特性であり、アノテーションプロセスのアーティファクトではない。

### H1（対立仮説）
独立アノテーションでは対角構造が有意に減少する（<80%）。

**意味**: 対角構造はLLMがbefore/afterを同時に見てアノテーションしたことによるアーティファクトである。

### 効果量
- 検出目標: 95% → 80%の差（効果量 h = 0.15）
- これは実質的に意味のある差であり、データの信頼性に直結する

## 3. 方法

### 3.1 サンプリング
- **総事例数**: {total_cases:,}件（有効遷移: {total_valid:,}件）
- **サンプル数**: {n_total}件（層化抽出 + 例外oversampling）
- **乱数シード**: 42（再現性確保）

#### 層化配分
| source_type | サンプル数 | 内例外 | 母集団 |
|-------------|-----------|--------|--------|
"""
    for st in ["news", "book", "official", "blog"]:
        cnt = source_dist.get(st, 0)
        exc = sum(1 for s in selected if s["source_type"] == st and s["is_exception"])
        doc += f"| {st} | {cnt} | {exc} | {STRATUM_TARGETS.get(st, 0)}目標 |\n"

    doc += f"""| **合計** | **{n_total}** | **{n_exception}** | |

#### Oversampling戦略
- 非対角例外事例（211件）は全件サンプルに含める
- 対角事例は各stratumから無作為抽出で残り枠を埋める
- 目的: 例外事例での再アノテーション結果を十分な統計的検出力で分析する

### 3.2 独立アノテーション手順

#### プロンプトA（before_hexagram用）
- 入力: target_name, main_domain, scale, period, before_state, trigger_type, story_summary
- **after_state, action_type, after_hexagramは一切含めない**
- 出力: 64卦から1つを選択 + 選定理由

#### プロンプトB（after_hexagram用）
- 入力: target_name, main_domain, scale, period, after_state, action_type, story_summary
- **before_state, trigger_type, before_hexagramは一切含めない**
- 出力: 64卦から1つを選択 + 選定理由

#### 情報遮断の保証
- プロンプトAとプロンプトBは完全に独立した別リクエストとして実行する
- 同一セッション内で両方を実行しない（コンテキスト汚染防止）
- story_summaryにbefore/after両方の情報が含まれることは既知のリスク
  → サブ実験として、summary除去版でも検証可能

### 3.3 再アノテーション実行（未実行・テンプレートのみ）
- 各プロンプトをLLM APIに送信し、JSON形式で回答を取得
- 同一モデル（元のアノテーションに使用したモデル）で実行することが望ましい
- 可能であれば複数モデル（GPT-4, Claude, Gemini）で比較実行

## 4. 分析計画

### 4.1 主要指標
- **対角率**: 再アノテーション結果における Delta_lower == Delta_upper の割合
- **一致率**: 元のアノテーションと再アノテーションの一致割合（before/after別）

### 4.2 統計検定

#### (a) 対角率の差の検定
- **検定**: McNemar検定（対応のあるデータ）
- **比較**: 元データの対角率 vs 再アノテーションの対角率
- **有意水準**: alpha = 0.05
- **多重比較補正**: Bonferroni（source_type別サブグループ × 4）

#### (b) 比率の差のz検定
- **検定**: 二標本比率の差の検定
- **H0**: p_original = p_reannotation
- **H1**: p_original > p_reannotation（片側）

#### (c) 一致度の評価
- **Cohen's kappa**: 元アノテーション vs 再アノテーション（before/after別）
- **完全一致率**: 同一卦が選ばれた割合
- **8卦グループ一致率**: 上卦/下卦レベルでの一致率

### 4.3 サブグループ分析
- source_type別（news / book / official / blog）
  - 各stratum内での対角率変化を比較
  - official（例外率57.4%）での再アノテーション結果が最重要
- 対角 vs 非対角サブグループ
  - 元データで対角だった事例: 再アノテーションでも対角が維持されるか
  - 元データで非対角だった事例: 再アノテーションで対角に「修正」されるか

### 4.4 サンプルサイズの根拠
- 効果量: p1 = 0.95 → p2 = 0.80（差 = 0.15）
- 有意水準: alpha = 0.05（両側）
- 検出力: 1 - beta = 0.80
- **必要サンプルサイズ**: n = {n_required}（各群）
- **本実験のサンプル**: n = {n_total}（十分な検出力を確保）

## 5. 予想される結果パターン

### パターン1: バイアス確認（H1支持）
- 再アノテーション対角率: 30-60%（大幅減少）
- 解釈: 対角構造はアノテーションプロセスのアーティファクト
- 対応: アノテーション手法の改善、独立アノテーションによるデータ修正

### パターン2: 部分的バイアス（混合結果）
- 再アノテーション対角率: 60-85%
- source_type間で大きな差
  - official/bookでは対角率が大幅に低下
  - newsでは比較的維持
- 解釈: バイアスは存在するが、一部は本質的な対角構造も含む
- 対応: source_type別の信頼度スコアの導入

### パターン3: バイアス否定（H0支持）
- 再アノテーション対角率: >90%
- 解釈: 対角構造はデータの本質的特性
- 注意: story_summaryにbefore/after両方の情報が含まれるため、
  summary経由の間接的な情報漏洩の可能性は排除できない

### パターン4: 一致率が低い場合
- 元データとの一致率（before/after別）が50%未満
- 解釈: アノテーション自体の再現性が低い（inter-annotator agreement問題）
- 対応: アノテーションガイドラインの明確化、複数アノテーターの導入

## 6. リスクと制限事項

### 6.1 story_summaryの情報漏洩
- story_summaryにはbefore→afterの変化が記述されている
- before用プロンプトでも、summaryからafterの情報を推測可能
- **緩和策**: summary除去版の追加実験を検討

### 6.2 LLMの卦選択バイアス
- LLM自体が特定の卦を好む傾向がある可能性
- 「乾為天」「坤為地」など象徴的な卦が過剰選択される可能性
- **緩和策**: 卦の出現頻度分布を元データと比較

### 6.3 再アノテーションモデルの差異
- 元のアノテーションに使用されたモデルが不明な場合、比較の妥当性に影響
- **緩和策**: 複数モデルでの再アノテーション実施

## 7. ファイル構成

| ファイル | 内容 |
|----------|------|
| `scripts/independent_reannotation.py` | サンプリング + プロンプト生成スクリプト |
| `analysis/phase3/reannotation_sample.json` | サンプル事例 + プロンプト |
| `analysis/phase3/reannotation_design.md` | 本実験計画書 |

## 8. 次のステップ

1. `reannotation_sample.json` のプロンプトをLLM APIに送信
2. 結果を `reannotation_results.json` に保存
3. `reannotation_analysis.py` で分析実行
4. 結論をPhase 3レポートに統合
"""
    return doc


# ============================================================
# メイン
# ============================================================

def main():
    print("=" * 60)
    print("独立再アノテーション実験 — サンプリング & プロンプト生成")
    print("=" * 60)

    # --- データ読み込み ---
    print("\n[1] データ読み込み...")
    reference_data = load_json(REFERENCE_FILE)
    cases = load_cases()
    print(f"  事例数: {len(cases)}")

    # --- マッピング構築 ---
    name_to_kw = build_name_to_kw(reference_data)
    kw_to_bits, bits_to_kw = build_kw_to_bits(reference_data)
    print(f"  卦名→番号: {len(name_to_kw)}件")
    print(f"  番号→6bit: {len(kw_to_bits)}件")

    # --- 例外事例ID取得 ---
    exception_ids = load_exception_ids(AUDIT_FILE)
    print(f"  例外事例ID: {len(exception_ids)}件")

    # --- 64卦選択肢テキスト構築 ---
    hexagram_list = build_hexagram_list(reference_data)
    hexagram_choices_text = build_hexagram_choices_text(hexagram_list)

    # --- 有効遷移数カウント ---
    total_valid = 0
    for case in cases:
        before_val = case.get("classical_before_hexagram", "")
        after_val = case.get("classical_after_hexagram", "")
        kw_from = resolve_hexagram_field(before_val, name_to_kw)
        kw_to = resolve_hexagram_field(after_val, name_to_kw)
        if kw_from and kw_to and kw_from in kw_to_bits and kw_to in kw_to_bits:
            total_valid += 1

    # --- 層化サンプリング ---
    print("\n[2] 層化サンプリング...")
    selected = stratified_sample(cases, name_to_kw, kw_to_bits, exception_ids)

    # --- プロンプト生成 & サンプルJSON構築 ---
    print("\n[3] プロンプト生成...")
    samples = build_sample_output(selected, hexagram_choices_text, name_to_kw, reference_data)
    print(f"  生成プロンプト数: {len(samples)} × 2 = {len(samples) * 2}")

    # --- サンプルJSON出力 ---
    output_data = {
        "metadata": {
            "description": "独立再アノテーション実験用サンプル",
            "total_samples": len(samples),
            "seed": RANDOM_SEED,
            "source_distribution": dict(Counter(s["source_type"] for s in samples)),
            "n_diagonal": sum(1 for s in samples if s["original_diagonal"]),
            "n_exception": sum(1 for s in samples if s["is_exception"]),
        },
        "samples": samples,
    }

    OUTPUT_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SAMPLE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n  サンプルJSON保存: {OUTPUT_SAMPLE}")

    # --- 実験計画書出力 ---
    print("\n[4] 実験計画書生成...")
    design_doc = generate_design_document(selected, len(cases), total_valid)
    with open(OUTPUT_DESIGN, "w", encoding="utf-8") as f:
        f.write(design_doc)
    print(f"  実験計画書保存: {OUTPUT_DESIGN}")

    # --- サマリー ---
    print(f"\n{'=' * 60}")
    print("完了サマリー")
    print(f"{'=' * 60}")
    print(f"  サンプル数:       {len(samples)}")
    print(f"  対角事例:         {sum(1 for s in samples if s['original_diagonal'])}")
    print(f"  非対角例外:       {sum(1 for s in samples if s['is_exception'])}")
    print(f"  プロンプト総数:   {len(samples) * 2}")
    print(f"  出力ファイル:")
    print(f"    {OUTPUT_SAMPLE}")
    print(f"    {OUTPUT_DESIGN}")


if __name__ == "__main__":
    main()
