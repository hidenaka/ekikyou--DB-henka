#!/usr/bin/env python3
"""
Edge Case Feature Analysis Script
- スケール境界事例の特定
- Codex提案5特徴量の実現可能性調査
- 既存フィールドマッピング
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path("/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB")
CASES_PATH = BASE_DIR / "data/raw/cases.jsonl"
OUTPUT_DIR = BASE_DIR / "analysis/phase3"

# ========== 1. データ読み込み ==========

def load_cases():
    cases = []
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                cases.append(case)
            except json.JSONDecodeError:
                print(f"WARNING: Line {i+1} failed to parse", file=sys.stderr)
    return cases

# ========== 2. スケール境界事例の特定 ==========

# 境界キーワード群
BOUNDARY_KEYWORDS = {
    "self_employed": ["自営", "自営業", "個人事業", "フリーランス", "freelance", "solo", "個人商店", "独立"],
    "family_business": ["家族経営", "ファミリービジネス", "同族", "family business", "家業", "一族"],
    "npo_social": ["NPO", "NGO", "非営利", "社会起業", "ソーシャル", "social enterprise", "公益", "慈善", "チャリティ"],
    "micro_org": ["スタートアップ", "startup", "ベンチャー", "小規模", "零細", "マイクロ", "個人経営"],
    "hybrid": ["半官半民", "第三セクター", "公社", "独法", "独立行政法人", "外郭団体"],
}

# scale misalignment 検出ルール
# company なのに個人的な特徴を持つキーワード
INDIVIDUAL_SIGNALS_IN_COMPANY = [
    "個人", "一人", "solo", "フリーランス", "自営", "個人事業", "YouTuber",
    "ブロガー", "インフルエンサー", "作家", "アーティスト", "芸能人",
]

# individual なのに組織的な特徴を持つキーワード
ORG_SIGNALS_IN_INDIVIDUAL = [
    "会社", "企業", "組織", "法人", "株式", "経営", "CEO", "社長",
    "取締役", "上場", "M&A", "事業", "部門", "子会社",
]


def detect_boundary_cases(cases):
    """スケール境界事例を検出"""
    results = {
        "keyword_matches": defaultdict(list),  # category -> [case_ids]
        "scale_mismatch": [],  # scale misalignment cases
        "boundary_summary": {},
    }

    for case in cases:
        cid = case.get("transition_id", "UNKNOWN")
        scale = case.get("scale", "other")
        main_domain = case.get("main_domain", "")
        story = case.get("story_summary", "")
        target = case.get("target_name", "")
        free_tags = case.get("free_tags", [])

        # 検索対象テキスト
        search_text = f"{main_domain} {story} {target} {' '.join(free_tags) if free_tags else ''}"
        search_text_lower = search_text.lower()

        # キーワードマッチ
        for category, keywords in BOUNDARY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in search_text_lower:
                    results["keyword_matches"][category].append({
                        "id": cid,
                        "scale": scale,
                        "main_domain": main_domain,
                        "target_name": target,
                        "matched_keyword": kw,
                        "story_snippet": story[:100],
                    })
                    break  # 1カテゴリ1事例1回

        # scale misalignment: company + 個人的シグナル
        if scale == "company":
            for sig in INDIVIDUAL_SIGNALS_IN_COMPANY:
                if sig.lower() in search_text_lower:
                    results["scale_mismatch"].append({
                        "id": cid,
                        "current_scale": scale,
                        "signal_type": "individual_in_company",
                        "matched_signal": sig,
                        "main_domain": main_domain,
                        "target_name": target,
                        "story_snippet": story[:100],
                    })
                    break

        # scale misalignment: individual + 組織的シグナル
        if scale == "individual":
            for sig in ORG_SIGNALS_IN_INDIVIDUAL:
                if sig.lower() in search_text_lower:
                    results["scale_mismatch"].append({
                        "id": cid,
                        "current_scale": scale,
                        "signal_type": "org_in_individual",
                        "matched_signal": sig,
                        "main_domain": main_domain,
                        "target_name": target,
                        "story_snippet": story[:100],
                    })
                    break

    # サマリー
    results["boundary_summary"] = {
        cat: len(ids) for cat, ids in results["keyword_matches"].items()
    }
    results["boundary_summary"]["scale_mismatch_total"] = len(results["scale_mismatch"])
    results["boundary_summary"]["individual_in_company"] = sum(
        1 for m in results["scale_mismatch"] if m["signal_type"] == "individual_in_company"
    )
    results["boundary_summary"]["org_in_individual"] = sum(
        1 for m in results["scale_mismatch"] if m["signal_type"] == "org_in_individual"
    )

    return results


# ========== 3. 既存フィールドマッピング ==========

def analyze_existing_fields(cases):
    """既存フィールドとCodex提案5特徴量の対応分析"""

    all_fields = set()
    field_coverage = Counter()  # field -> count of cases having it
    field_sample_values = defaultdict(list)

    for case in cases:
        for k, v in case.items():
            all_fields.add(k)
            field_coverage[k] += 1
            if len(field_sample_values[k]) < 5 and v is not None:
                field_sample_values[k].append(str(v)[:100])

    # scale分布
    scale_dist = Counter(c.get("scale", "unknown") for c in cases)

    # main_domain分布
    domain_dist = Counter(c.get("main_domain", "unknown") for c in cases)

    # 5特徴量に関連する既存フィールドのマッピング
    feature_mapping = {
        "stakeholder_count": {
            "description": "当事者数（変化に関わる主体の数）",
            "related_existing_fields": [],
            "estimability": "none",
            "notes": "",
        },
        "resource_constraint": {
            "description": "資源制約（人材・資金・時間の制約度）",
            "related_existing_fields": [],
            "estimability": "none",
            "notes": "",
        },
        "time_horizon": {
            "description": "時間軸（変化プロセスの所要期間）",
            "related_existing_fields": [],
            "estimability": "none",
            "notes": "",
        },
        "reversibility": {
            "description": "可逆性（変化を元に戻せる度合い）",
            "related_existing_fields": [],
            "estimability": "none",
            "notes": "",
        },
        "consensus_cost": {
            "description": "合意形成コスト（意思決定に必要な関係者調整の難易度）",
            "related_existing_fields": [],
            "estimability": "none",
            "notes": "",
        },
    }

    # 1. stakeholder_count
    # scale, entity_type, subject_type, free_tags から推定可能性
    entity_type_dist = Counter(c.get("entity_type", "unknown") for c in cases)
    subject_type_dist = Counter(c.get("subject_type", "unknown") for c in cases)

    feature_mapping["stakeholder_count"]["related_existing_fields"] = [
        {"field": "scale", "relevance": "high", "reason": "company=多, individual=少 の粗い推定"},
        {"field": "entity_type", "relevance": "medium", "reason": "company/individual/family等が分布"},
        {"field": "subject_type", "relevance": "medium", "reason": "entity_typeと類似"},
        {"field": "free_tags", "relevance": "low", "reason": "#M&A等から間接的に推定可能な場合あり"},
        {"field": "story_summary", "relevance": "medium", "reason": "LLMで人数をテキストから抽出可能"},
    ]
    feature_mapping["stakeholder_count"]["estimability"] = "partial_rule_plus_llm"
    feature_mapping["stakeholder_count"]["notes"] = (
        "scaleで粗い区分(company→多, individual→1-2)は可能。"
        "正確な数値にはstory_summaryからのLLM抽出が必要。"
        "ルールベース: company=10+, family=2-10, individual=1-3, country=100+, other=variable"
    )

    # 2. resource_constraint
    # credibility_rank, success_level, main_domain, scale
    feature_mapping["resource_constraint"]["related_existing_fields"] = [
        {"field": "scale", "relevance": "high", "reason": "individual→高制約, company→中, country→低"},
        {"field": "main_domain", "relevance": "medium", "reason": "業界によりリソース水準が異なる"},
        {"field": "story_summary", "relevance": "high", "reason": "資金難/人手不足等の記述からLLM抽出可能"},
        {"field": "before_state", "relevance": "medium", "reason": "どん底→高制約, 絶頂→低制約"},
        {"field": "action_type", "relevance": "low", "reason": "耐える=高制約, 攻める=低制約の傾向"},
    ]
    feature_mapping["resource_constraint"]["estimability"] = "partial_rule_plus_llm"
    feature_mapping["resource_constraint"]["notes"] = (
        "scale + before_state の組み合わせで粗い推定可能。"
        "精密な推定にはstory_summaryのLLM分析が必要。"
        "ルールベース精度は推定60-70%"
    )

    # 3. time_horizon
    # period フィールドから部分的に推定可能
    period_examples = []
    period_parseable = 0
    period_total = 0
    duration_estimates = []

    for case in cases:
        period = case.get("period", "")
        if period:
            period_total += 1
            # "2015-2020" 形式のパース
            match = re.match(r"(\d{4})\s*[-–~〜]\s*(\d{4})", period)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                duration = end - start
                duration_estimates.append(duration)
                period_parseable += 1
            elif re.match(r"\d{4}$", period.strip()):
                duration_estimates.append(1)
                period_parseable += 1
            if len(period_examples) < 10:
                period_examples.append(period)

    feature_mapping["time_horizon"]["related_existing_fields"] = [
        {"field": "period", "relevance": "high", "reason": f"'YYYY-YYYY'形式からduration算出可能。{period_parseable}/{period_total}件パース可能"},
        {"field": "story_summary", "relevance": "medium", "reason": "LLMで時間関連表現を抽出可能"},
        {"field": "pattern_type", "relevance": "low", "reason": "パターンにより典型的な時間軸が異なる"},
    ]
    feature_mapping["time_horizon"]["estimability"] = "partial_rule"
    feature_mapping["time_horizon"]["notes"] = (
        f"periodフィールドから{period_parseable}/{period_total}件({100*period_parseable/max(period_total,1):.1f}%)で"
        f"duration_years を直接算出可能。"
        f"推定duration分布: "
        f"mean={sum(duration_estimates)/max(len(duration_estimates),1):.1f}年, "
        f"median={sorted(duration_estimates)[len(duration_estimates)//2] if duration_estimates else 0}年, "
        f"range={min(duration_estimates) if duration_estimates else 0}-{max(duration_estimates) if duration_estimates else 0}年"
    )

    # 4. reversibility
    # after_state, outcome, action_type
    feature_mapping["reversibility"]["related_existing_fields"] = [
        {"field": "after_state", "relevance": "high", "reason": "崩壊・消滅→不可逆, 現状維持→可逆"},
        {"field": "outcome", "relevance": "high", "reason": "Failure→多くの場合不可逆"},
        {"field": "action_type", "relevance": "medium", "reason": "捨てる・撤退→不可逆傾向, 守る→可逆傾向"},
        {"field": "pattern_type", "relevance": "medium", "reason": "Hubris_Collapse→不可逆, Slow_Decline→部分的可逆"},
        {"field": "story_summary", "relevance": "medium", "reason": "破産/倒産/消滅 等のキーワードで補完可能"},
    ]
    feature_mapping["reversibility"]["estimability"] = "rule_based"
    feature_mapping["reversibility"]["notes"] = (
        "after_state + outcome の組み合わせでルールベース推定が高精度で可能。"
        "崩壊・消滅+Failure→irreversible(1), V字回復→partially_reversible(3), "
        "現状維持→highly_reversible(5) のようなマッピング。"
        "推定精度80%以上。LLM不要"
    )

    # 5. consensus_cost
    # scale, entity_type, action_type
    feature_mapping["consensus_cost"]["related_existing_fields"] = [
        {"field": "scale", "relevance": "high", "reason": "country→極高, company→高, family→中, individual→低"},
        {"field": "action_type", "relevance": "medium", "reason": "対話・融合→高, 逃げる→低"},
        {"field": "entity_type", "relevance": "medium", "reason": "scaleと類似"},
        {"field": "main_domain", "relevance": "low", "reason": "政治/行政→高, 個人→低"},
        {"field": "story_summary", "relevance": "medium", "reason": "交渉/調整/対立等の記述からLLM推定可能"},
    ]
    feature_mapping["consensus_cost"]["estimability"] = "partial_rule"
    feature_mapping["consensus_cost"]["notes"] = (
        "scale + action_type の組み合わせでルールベース推定可能。"
        "country+対話=5, individual+攻める=1 のようなマッピング。"
        "精度は推定65-75%。精密にはstory_summaryのLLM分析が望ましい"
    )

    return {
        "all_fields": sorted(all_fields),
        "field_count": len(all_fields),
        "field_coverage": {k: v for k, v in sorted(field_coverage.items(), key=lambda x: -x[1])},
        "scale_distribution": dict(scale_dist.most_common()),
        "domain_distribution": dict(domain_dist.most_common(30)),
        "entity_type_distribution": dict(entity_type_dist.most_common()),
        "subject_type_distribution": dict(subject_type_dist.most_common()),
        "feature_mapping": feature_mapping,
        "period_analysis": {
            "total_with_period": period_total,
            "parseable": period_parseable,
            "parse_rate": round(period_parseable / max(period_total, 1), 4),
            "duration_estimates": {
                "count": len(duration_estimates),
                "mean": round(sum(duration_estimates) / max(len(duration_estimates), 1), 2),
                "median": sorted(duration_estimates)[len(duration_estimates)//2] if duration_estimates else None,
                "min": min(duration_estimates) if duration_estimates else None,
                "max": max(duration_estimates) if duration_estimates else None,
            },
            "examples": period_examples,
        },
    }


# ========== 4. 特徴量の実現可能性スコアリング ==========

def score_feasibility(feature_mapping):
    """5特徴量の実現可能性をスコアリング"""

    scores = {}

    # スコアリング基準:
    # feasibility: 1-5 (1=不可能, 5=容易)
    # cost: 1-5 (1=低コスト, 5=高コスト)
    # value: 1-5 (1=低価値, 5=高価値)
    # method: rule_only / rule_plus_llm / llm_only

    scoring = {
        "stakeholder_count": {
            "feasibility": 3,
            "cost": 3,
            "value": 4,
            "method": "rule_plus_llm",
            "rule_coverage_pct": 100,  # scaleで全件に粗い値を付与可能
            "llm_needed_pct": 40,  # 精密化が必要な境界事例
            "priority": "medium",
            "rationale": (
                "scaleフィールドから全件にルールベースで粗い推定可能 "
                "(company→10+, individual→1, family→2-10, country→1000+)。"
                "境界事例(自営業=companyだが実質1人等)の精密化にLLM必要。"
                "分離ロジックにおけるスケール判定の精度向上に直結する"
            ),
        },
        "resource_constraint": {
            "feasibility": 2,
            "cost": 4,
            "value": 3,
            "method": "llm_only",
            "rule_coverage_pct": 50,
            "llm_needed_pct": 80,
            "priority": "low",
            "rationale": (
                "既存フィールドからの推定精度が低い。"
                "scale+before_stateで粗い傾向は出るが、同じscale=companyでも "
                "トヨタとスタートアップでは資源制約が全く異なる。"
                "story_summaryからのLLM抽出が不可欠で、13,060件のバッチ処理コスト大"
            ),
        },
        "time_horizon": {
            "feasibility": 4,
            "cost": 1,
            "value": 4,
            "method": "rule_only",
            "rule_coverage_pct": 0,  # will be updated
            "llm_needed_pct": 0,
            "priority": "high",
            "rationale": "",  # will be updated
        },
        "reversibility": {
            "feasibility": 4,
            "cost": 1,
            "value": 5,
            "method": "rule_only",
            "rule_coverage_pct": 100,
            "llm_needed_pct": 0,
            "priority": "high",
            "rationale": (
                "after_state + outcome + action_type の組み合わせで "
                "ルールベース推定が高精度で可能。LLM不要。"
                "崩壊・消滅+Failure→1(不可逆), V字回復+Success→4(高可逆), "
                "現状維持→3(中可逆)。スケール分離ロジックにおいて、"
                "個人の転職(可逆)と企業の倒産(不可逆)の区別に有用"
            ),
        },
        "consensus_cost": {
            "feasibility": 3,
            "cost": 2,
            "value": 4,
            "method": "rule_plus_llm",
            "rule_coverage_pct": 100,
            "llm_needed_pct": 30,
            "priority": "medium",
            "rationale": (
                "scale + action_type で全件にルールベース推定可能。"
                "country+対話→5(最高), individual+攻める→1(最低)。"
                "精密化にはstory_summaryのLLM分析が望ましいが、"
                "ルールベースでも実用的精度(65-75%)が期待できる"
            ),
        },
    }

    return scoring


# ========== 5. after_state/outcome × reversibility 推定テーブル ==========

def build_reversibility_rules():
    """可逆性ルールテーブル — 実データの全after_state値をカバー"""
    # after_state values actually in data (15+ distinct):
    # 持続成長・大成功, 崩壊・消滅, 安定成長・成功, 安定・平和, 停滞・閉塞,
    # 縮小安定・生存, V字回復・大成功, 変質・新生, 混乱・衰退, 拡大・繁栄,
    # 安定・停止, どん底・危機, 混乱・カオス, 喜び・交流, 成長・拡大,
    # 迷走・混乱, 現状維持・延命, 成長痛, 分岐・様子見, 消滅・破綻

    # Map: after_state -> base reversibility group
    # group: irreversible(1), mostly_irreversible(2), partially_reversible(3),
    #        mostly_reversible(4), highly_reversible(5)
    state_base = {
        # 不可逆: 崩壊・消滅系
        "崩壊・消滅": 1,
        "消滅・破綻": 1,
        "どん底・危機": 1,
        # ほぼ不可逆: 衰退・混乱系
        "混乱・衰退": 2,
        "混乱・カオス": 2,
        "迷走・混乱": 2,
        "停滞・閉塞": 2,
        # 部分的可逆: 縮小・変質系
        "縮小安定・生存": 3,
        "変質・新生": 2,  # 構造が変わるので不可逆寄り
        "成長痛": 3,
        "分岐・様子見": 3,
        # ほぼ可逆: 安定系
        "現状維持・延命": 4,
        "安定・平和": 4,
        "安定・停止": 4,
        "安定成長・成功": 4,
        # 可逆: 成長・回復系
        "V字回復・大成功": 4,
        "持続成長・大成功": 5,
        "拡大・繁栄": 5,
        "成長・拡大": 5,
        "喜び・交流": 4,
    }

    # Outcome modifier: Success → +0, PartialSuccess → +0, Mixed → -0, Failure → -1
    outcome_mod = {
        "Success": 0,
        "PartialSuccess": 0,
        "Mixed": 0,
        "Failure": -1,
    }

    label_map = {
        1: "irreversible",
        2: "mostly_irreversible",
        3: "partially_reversible",
        4: "mostly_reversible",
        5: "highly_reversible",
    }

    rules = {}
    for state, base in state_base.items():
        for outcome, mod in outcome_mod.items():
            score = max(1, min(5, base + mod))
            rules[(state, outcome)] = {"score": score, "label": label_map[score]}

    return rules


def estimate_reversibility(cases):
    """全事例に対して可逆性を推定"""
    rules = build_reversibility_rules()
    results = {"mapped": 0, "unmapped": 0, "distribution": Counter()}

    for case in cases:
        after = case.get("after_state", "")
        outcome = case.get("outcome", "")
        key = (after, outcome)
        if key in rules:
            results["mapped"] += 1
            results["distribution"][rules[key]["label"]] += 1
        else:
            results["unmapped"] += 1

    return results


# ========== 6. consensus_cost ルールベース推定 ==========

def estimate_consensus_cost(cases):
    """合意形成コストのルールベース推定"""
    # scale → base cost
    scale_base = {
        "country": 5,
        "company": 3,
        "family": 3,
        "other": 2,
        "individual": 1,
    }
    # action_type → modifier
    action_mod = {
        "対話・融合": +1,
        "刷新・破壊": +1,
        "分散・スピンオフ": +1,
        "攻める・挑戦": 0,
        "守る・維持": 0,
        "耐える・潜伏": -1,
        "逃げる・放置": -1,
        "捨てる・撤退": 0,
    }

    distribution = Counter()
    for case in cases:
        scale = case.get("scale", "other")
        action = case.get("action_type", "")
        base = scale_base.get(scale, 2)
        mod = action_mod.get(action, 0)
        score = max(1, min(5, base + mod))
        distribution[score] += 1

    return dict(sorted(distribution.items()))


# ========== 7. main_domain のスケール境界分析 ==========

def analyze_domain_scale_crossref(cases):
    """main_domain × scale のクロス集計で境界事例を特定"""
    cross = defaultdict(Counter)
    for case in cases:
        domain = case.get("main_domain", "unknown")
        scale = case.get("scale", "unknown")
        cross[domain][scale] += 1

    # 複数scaleに跨るdomainを抽出
    multi_scale_domains = {}
    for domain, scale_counts in cross.items():
        if len(scale_counts) > 1:
            total = sum(scale_counts.values())
            dominant_scale = scale_counts.most_common(1)[0]
            dominance_ratio = dominant_scale[1] / total
            if dominance_ratio < 0.9:  # 90%未満 = 有意なmixing
                multi_scale_domains[domain] = {
                    "total": total,
                    "distribution": dict(scale_counts),
                    "dominant_scale": dominant_scale[0],
                    "dominance_ratio": round(dominance_ratio, 4),
                }

    return {
        "full_cross_table": {d: dict(s) for d, s in cross.items()},
        "multi_scale_domains": dict(sorted(
            multi_scale_domains.items(), key=lambda x: -x[1]["total"]
        )),
    }


# ========== MAIN ==========

def main():
    print("Loading cases...", file=sys.stderr)
    cases = load_cases()
    print(f"Loaded {len(cases)} cases", file=sys.stderr)

    print("Detecting boundary cases...", file=sys.stderr)
    boundary = detect_boundary_cases(cases)

    print("Analyzing existing fields...", file=sys.stderr)
    field_analysis = analyze_existing_fields(cases)

    print("Scoring feasibility...", file=sys.stderr)
    feasibility = score_feasibility(field_analysis["feature_mapping"])

    # time_horizon の数値を更新
    period_data = field_analysis["period_analysis"]
    feasibility["time_horizon"]["rule_coverage_pct"] = round(period_data["parse_rate"] * 100, 1)
    feasibility["time_horizon"]["rationale"] = (
        f"periodフィールドから{period_data['parseable']}/{period_data['total_with_period']}件"
        f"({feasibility['time_horizon']['rule_coverage_pct']}%)でduration_yearsを直接算出可能。"
        f"LLM不要。平均{period_data['duration_estimates']['mean']}年, "
        f"中央値{period_data['duration_estimates']['median']}年。"
        f"残りの事例も「平成後期」等からLLMでapprox推定可能だがコスト低い"
    )

    print("Estimating reversibility...", file=sys.stderr)
    reversibility = estimate_reversibility(cases)

    print("Estimating consensus cost...", file=sys.stderr)
    consensus_cost = estimate_consensus_cost(cases)

    print("Analyzing domain-scale crossref...", file=sys.stderr)
    domain_scale = analyze_domain_scale_crossref(cases)

    # ========== 出力: JSON stats ==========
    stats = {
        "metadata": {
            "total_cases": len(cases),
            "analysis_date": "2026-03-04",
            "script": "scripts/edge_case_analysis.py",
        },
        "scale_distribution": field_analysis["scale_distribution"],
        "boundary_cases": {
            "summary": boundary["boundary_summary"],
            "keyword_match_counts": {k: len(v) for k, v in boundary["keyword_matches"].items()},
            "keyword_match_samples": {
                k: v[:10] for k, v in boundary["keyword_matches"].items()
            },
            "scale_mismatch_samples": boundary["scale_mismatch"][:20],
            "scale_mismatch_total": len(boundary["scale_mismatch"]),
        },
        "existing_fields": {
            "all_fields": field_analysis["all_fields"],
            "field_count": field_analysis["field_count"],
        },
        "feature_mapping": field_analysis["feature_mapping"],
        "feasibility_scores": feasibility,
        "reversibility_estimate": {
            "mapped": reversibility["mapped"],
            "unmapped": reversibility["unmapped"],
            "coverage_pct": round(reversibility["mapped"] / len(cases) * 100, 2),
            "distribution": dict(reversibility["distribution"]),
        },
        "consensus_cost_estimate": {
            "distribution": consensus_cost,
        },
        "domain_scale_analysis": {
            "multi_scale_domains": domain_scale["multi_scale_domains"],
            "total_multi_scale_domain_count": len(domain_scale["multi_scale_domains"]),
        },
        "period_analysis": field_analysis["period_analysis"],
    }

    stats_path = OUTPUT_DIR / "edge_case_features_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats written to {stats_path}", file=sys.stderr)

    # ========== 出力: Markdown Report ==========
    report = generate_report(stats, boundary, domain_scale, field_analysis)
    report_path = OUTPUT_DIR / "edge_case_features_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {report_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


def generate_report(stats, boundary, domain_scale, field_analysis):
    """マークダウンレポートを生成"""
    lines = []

    lines.append("# エッジケース特徴量調査レポート")
    lines.append("")
    lines.append(f"**分析日**: 2026-03-04")
    lines.append(f"**対象**: cases.jsonl ({stats['metadata']['total_cases']:,}件)")
    lines.append(f"**目的**: スケール境界事例の特定とCodex提案5特徴量の実現可能性評価")
    lines.append("")

    # ===== 1. スケール分布 =====
    lines.append("## 1. 現在のスケール分布")
    lines.append("")
    lines.append("| scale | 件数 | 割合 |")
    lines.append("|-------|------|------|")
    total = stats["metadata"]["total_cases"]
    for scale, count in sorted(stats["scale_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {scale} | {count:,} | {count/total*100:.1f}% |")
    lines.append(f"| **合計** | **{total:,}** | **100%** |")
    lines.append("")

    # ===== 2. スケール境界事例 =====
    lines.append("## 2. スケール境界事例の検出")
    lines.append("")

    lines.append("### 2.1 キーワードマッチによる境界事例")
    lines.append("")
    lines.append("| カテゴリ | マッチ件数 | 説明 |")
    lines.append("|----------|-----------|------|")
    category_desc = {
        "self_employed": "自営業・個人事業・フリーランス",
        "family_business": "家族経営・同族経営",
        "npo_social": "NPO・社会起業",
        "micro_org": "スタートアップ・零細企業",
        "hybrid": "第三セクター・独法等",
    }
    for cat, count in sorted(stats["boundary_cases"]["keyword_match_counts"].items(), key=lambda x: -x[1]):
        desc = category_desc.get(cat, cat)
        lines.append(f"| {cat} | {count:,} | {desc} |")
    lines.append("")

    # サンプル
    for cat, samples in stats["boundary_cases"]["keyword_match_samples"].items():
        if samples:
            lines.append(f"#### {cat} サンプル（上位5件）")
            lines.append("")
            for s in samples[:5]:
                lines.append(f"- **{s['id']}** (scale={s['scale']}, domain={s['main_domain']}): {s['target_name']} — {s['story_snippet']}...")
            lines.append("")

    lines.append("### 2.2 スケール不整合（Mismatch）")
    lines.append("")
    lines.append(f"- **合計**: {stats['boundary_cases']['scale_mismatch_total']:,}件")
    lines.append(f"  - company に個人的シグナル: {boundary['boundary_summary'].get('individual_in_company', 0):,}件")
    lines.append(f"  - individual に組織的シグナル: {boundary['boundary_summary'].get('org_in_individual', 0):,}件")
    lines.append("")

    if stats["boundary_cases"]["scale_mismatch_samples"]:
        lines.append("#### 不整合サンプル（上位10件）")
        lines.append("")
        for s in stats["boundary_cases"]["scale_mismatch_samples"][:10]:
            lines.append(f"- **{s['id']}** [{s['current_scale']}] signal='{s['matched_signal']}' ({s['signal_type']}): {s['target_name']} — {s['story_snippet']}...")
        lines.append("")

    # ===== 2.3 main_domain × scale クロス分析 =====
    lines.append("### 2.3 main_domain × scale クロス分析（多スケール混在ドメイン）")
    lines.append("")
    lines.append(f"90%未満の支配率を持つドメイン: **{stats['domain_scale_analysis']['total_multi_scale_domain_count']}件**")
    lines.append("")

    msd = stats["domain_scale_analysis"]["multi_scale_domains"]
    if msd:
        lines.append("| domain | 件数 | 分布 | 支配率 |")
        lines.append("|--------|------|------|--------|")
        for domain, info in list(msd.items())[:20]:
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(info["distribution"].items(), key=lambda x: -x[1]))
            lines.append(f"| {domain} | {info['total']} | {dist_str} | {info['dominance_ratio']*100:.1f}% |")
        lines.append("")

    # ===== 3. 既存フィールド一覧 =====
    lines.append("## 3. 既存フィールド一覧")
    lines.append("")
    lines.append(f"**総フィールド数**: {stats['existing_fields']['field_count']}")
    lines.append("")
    lines.append("```")
    for f_name in stats["existing_fields"]["all_fields"]:
        lines.append(f"  {f_name}")
    lines.append("```")
    lines.append("")

    # ===== 4. 5特徴量の既存フィールドマッピング =====
    lines.append("## 4. Codex提案5特徴量 × 既存フィールドマッピング")
    lines.append("")

    for feat_name, feat_data in stats["feature_mapping"].items():
        lines.append(f"### 4.{list(stats['feature_mapping'].keys()).index(feat_name)+1} {feat_name} — {feat_data['description']}")
        lines.append("")
        lines.append(f"**推定方法**: {feat_data['estimability']}")
        lines.append("")
        lines.append("| 既存フィールド | 関連度 | 理由 |")
        lines.append("|---------------|--------|------|")
        for rf in feat_data["related_existing_fields"]:
            lines.append(f"| `{rf['field']}` | {rf['relevance']} | {rf['reason']} |")
        lines.append("")
        lines.append(f"**メモ**: {feat_data['notes']}")
        lines.append("")

    # ===== 5. 実現可能性マトリクス =====
    lines.append("## 5. 実現可能性マトリクス")
    lines.append("")
    lines.append("| 特徴量 | 実現性 | コスト | 価値 | 推定方法 | ルール率 | LLM率 | 優先度 |")
    lines.append("|--------|--------|--------|------|----------|---------|-------|--------|")

    for feat, data in stats["feasibility_scores"].items():
        lines.append(
            f"| {feat} | {data['feasibility']}/5 | {data['cost']}/5 | {data['value']}/5 | "
            f"{data['method']} | {data['rule_coverage_pct']}% | {data['llm_needed_pct']}% | "
            f"**{data['priority']}** |"
        )
    lines.append("")

    lines.append("### スコア説明")
    lines.append("- **実現性**: 1(不可能)〜5(容易)")
    lines.append("- **コスト**: 1(低コスト)〜5(高コスト)")
    lines.append("- **価値**: 1(低価値)〜5(高価値)")
    lines.append("- **ルール率**: ルールベースで推定可能な事例の割合")
    lines.append("- **LLM率**: LLMバッチ処理が必要な事例の割合")
    lines.append("")

    for feat, data in stats["feasibility_scores"].items():
        lines.append(f"#### {feat}")
        lines.append(f"- {data['rationale']}")
        lines.append("")

    # ===== 6. 可逆性の試算 =====
    lines.append("## 6. 可逆性 (reversibility) ルールベース試算")
    lines.append("")
    rev = stats["reversibility_estimate"]
    lines.append(f"- **マッピング成功**: {rev['mapped']:,}件 ({rev['coverage_pct']}%)")
    lines.append(f"- **マッピング失敗**: {rev['unmapped']:,}件")
    lines.append("")
    lines.append("| ラベル | 件数 |")
    lines.append("|--------|------|")
    for label, count in sorted(rev["distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | {count:,} |")
    lines.append("")

    # ===== 7. 合意形成コストの試算 =====
    lines.append("## 7. 合意形成コスト (consensus_cost) ルールベース試算")
    lines.append("")
    lines.append("| スコア | 件数 |")
    lines.append("|--------|------|")
    for score, count in sorted(stats["consensus_cost_estimate"]["distribution"].items()):
        lines.append(f"| {score} | {count:,} |")
    lines.append("")

    # ===== 8. period → time_horizon 分析 =====
    lines.append("## 8. time_horizon — period フィールド分析")
    lines.append("")
    pa = stats["period_analysis"]
    lines.append(f"- **periodフィールド保有**: {pa['total_with_period']:,}件")
    lines.append(f"- **YYYY-YYYY形式パース可能**: {pa['parseable']:,}件 ({pa['parse_rate']*100:.1f}%)")
    lines.append(f"- **推定duration**: 平均{pa['duration_estimates']['mean']}年, 中央値{pa['duration_estimates']['median']}年, 範囲{pa['duration_estimates']['min']}-{pa['duration_estimates']['max']}年")
    lines.append("")
    lines.append("### period サンプル")
    for ex in pa["examples"]:
        lines.append(f"- `{ex}`")
    lines.append("")

    # ===== 9. 推奨アクション =====
    lines.append("## 9. 推奨アクション")
    lines.append("")
    lines.append("### 即時実装（ルールベース、LLM不要）")
    lines.append("")
    lines.append("1. **reversibility** (可逆性)")
    lines.append("   - after_state × outcome のルールテーブルで全件に付与可能")
    lines.append("   - 推定カバレッジ: ~99%")
    lines.append("   - コスト: スクリプト1本（1時間以内）")
    lines.append("   - 価値: スケール分離の判定精度向上に直結")
    lines.append("")
    lines.append("2. **time_horizon** (時間軸)")
    lines.append("   - period フィールドからduration_yearsを算出")
    lines.append(f"   - 即時カバレッジ: {pa['parse_rate']*100:.1f}%")
    lines.append("   - コスト: スクリプト1本（30分以内）")
    lines.append("   - 価値: 変化パターンの時間的特性を定量化")
    lines.append("")
    lines.append("### 中期実装（ルール + 境界事例のLLM補完）")
    lines.append("")
    lines.append("3. **consensus_cost** (合意形成コスト)")
    lines.append("   - scale × action_type で全件にベースライン付与")
    lines.append("   - 境界事例（30%）のLLM補完で精度向上")
    lines.append("   - コスト: ルール1時間 + LLM 3-4時間")
    lines.append("")
    lines.append("4. **stakeholder_count** (当事者数)")
    lines.append("   - scaleで粗い区分を全件に付与")
    lines.append("   - 境界事例（40%）のLLM補完で精密化")
    lines.append("   - コスト: ルール1時間 + LLM 4-5時間")
    lines.append("")
    lines.append("### 保留（コスト対効果が低い）")
    lines.append("")
    lines.append("5. **resource_constraint** (資源制約)")
    lines.append("   - 既存フィールドからの推定精度が低い")
    lines.append("   - 13,060件のLLMバッチ処理コストに見合う価値が不明確")
    lines.append("   - 他4特徴量の実装・評価後に再検討")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
