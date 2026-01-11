#!/usr/bin/env python3
"""
ギャップ分析ツール

64卦×6爻のデータカバレッジを分析し、
優先的に収集すべきデータを特定する。
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
CASES_FILE = BASE_DIR / "data" / "raw" / "cases.jsonl"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"

# 64卦の基本情報
HEXAGRAMS = {
    1: {"name": "乾為天", "keyword": "創造・天", "importance": "high"},
    2: {"name": "坤為地", "keyword": "受容・地", "importance": "high"},
    3: {"name": "水雷屯", "keyword": "困難・始動", "importance": "high"},
    4: {"name": "山水蒙", "keyword": "啓蒙・未熟", "importance": "medium"},
    5: {"name": "水天需", "keyword": "待機・需要", "importance": "medium"},
    6: {"name": "天水訟", "keyword": "争い・訴訟", "importance": "medium"},
    7: {"name": "地水師", "keyword": "軍隊・組織", "importance": "high"},
    8: {"name": "水地比", "keyword": "親和・連携", "importance": "high"},
    9: {"name": "風天小畜", "keyword": "小さな蓄積", "importance": "medium"},
    10: {"name": "天沢履", "keyword": "礼儀・実践", "importance": "medium"},
    11: {"name": "地天泰", "keyword": "泰平・繁栄", "importance": "high"},
    12: {"name": "天地否", "keyword": "閉塞・停滞", "importance": "high"},
    13: {"name": "天火同人", "keyword": "同志・協力", "importance": "high"},
    14: {"name": "火天大有", "keyword": "大いなる所有", "importance": "high"},
    15: {"name": "地山謙", "keyword": "謙虚", "importance": "high"},
    16: {"name": "雷地予", "keyword": "喜び・準備", "importance": "medium"},
    17: {"name": "沢雷随", "keyword": "随従・追従", "importance": "medium"},
    18: {"name": "山風蠱", "keyword": "腐敗・改革", "importance": "high"},
    19: {"name": "地沢臨", "keyword": "接近・臨む", "importance": "medium"},
    20: {"name": "風地観", "keyword": "観察・観想", "importance": "medium"},
    21: {"name": "火雷噬嗑", "keyword": "断行・決裂", "importance": "medium"},
    22: {"name": "山火賁", "keyword": "装飾・文化", "importance": "low"},
    23: {"name": "山地剥", "keyword": "剥落・衰退", "importance": "high"},
    24: {"name": "地雷復", "keyword": "復活・回復", "importance": "high"},
    25: {"name": "天雷无妄", "keyword": "無妄・誠実", "importance": "medium"},
    26: {"name": "山天大畜", "keyword": "大蓄積", "importance": "high"},
    27: {"name": "山雷頤", "keyword": "養い・栄養", "importance": "medium"},
    28: {"name": "沢風大過", "keyword": "過剰・危機", "importance": "high"},
    29: {"name": "坎為水", "keyword": "危険・困難", "importance": "high"},
    30: {"name": "離為火", "keyword": "明晰・離別", "importance": "high"},
    31: {"name": "沢山咸", "keyword": "感応・交流", "importance": "medium"},
    32: {"name": "雷風恒", "keyword": "恒常・持続", "importance": "medium"},
    33: {"name": "天山遯", "keyword": "退却・隠遁", "importance": "high"},
    34: {"name": "雷天大壮", "keyword": "大いなる勢い", "importance": "high"},
    35: {"name": "火地晋", "keyword": "前進・昇進", "importance": "high"},
    36: {"name": "地火明夷", "keyword": "明の傷つき", "importance": "high"},
    37: {"name": "風火家人", "keyword": "家族・組織", "importance": "medium"},
    38: {"name": "火沢睽", "keyword": "背反・対立", "importance": "medium"},
    39: {"name": "水山蹇", "keyword": "困難・蹉跌", "importance": "high"},
    40: {"name": "雷水解", "keyword": "解放・解決", "importance": "high"},
    41: {"name": "山沢損", "keyword": "損失・減少", "importance": "high"},
    42: {"name": "風雷益", "keyword": "利益・増加", "importance": "high"},
    43: {"name": "沢天夬", "keyword": "決断・決裂", "importance": "high"},
    44: {"name": "天風姤", "keyword": "遭遇・偶然", "importance": "medium"},
    45: {"name": "沢地萃", "keyword": "集合・結集", "importance": "medium"},
    46: {"name": "地風升", "keyword": "上昇・昇進", "importance": "high"},
    47: {"name": "沢水困", "keyword": "困窮・苦境", "importance": "high"},
    48: {"name": "水風井", "keyword": "井戸・源泉", "importance": "medium"},
    49: {"name": "沢火革", "keyword": "革命・変革", "importance": "high"},
    50: {"name": "火風鼎", "keyword": "鼎・刷新", "importance": "high"},
    51: {"name": "震為雷", "keyword": "震動・衝撃", "importance": "high"},
    52: {"name": "艮為山", "keyword": "停止・静止", "importance": "high"},
    53: {"name": "風山漸", "keyword": "漸進・徐々", "importance": "medium"},
    54: {"name": "雷沢帰妹", "keyword": "帰属・従属", "importance": "medium"},
    55: {"name": "雷火豊", "keyword": "豊穣・繁栄", "importance": "high"},
    56: {"name": "火山旅", "keyword": "旅・流浪", "importance": "medium"},
    57: {"name": "巽為風", "keyword": "浸透・柔順", "importance": "medium"},
    58: {"name": "兌為沢", "keyword": "喜悦・交流", "importance": "medium"},
    59: {"name": "風水渙", "keyword": "散逸・拡散", "importance": "medium"},
    60: {"name": "水沢節", "keyword": "節度・制限", "importance": "medium"},
    61: {"name": "風沢中孚", "keyword": "誠信・信頼", "importance": "high"},
    62: {"name": "雷山小過", "keyword": "小さな過ち", "importance": "medium"},
    63: {"name": "水火既済", "keyword": "完成・達成", "importance": "high"},
    64: {"name": "火水未済", "keyword": "未完成・継続", "importance": "high"},
}


def load_cases():
    """ケースデータを読み込み"""
    cases = []
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def analyze_coverage(cases):
    """カバレッジを分析"""

    # 卦別の出現数
    hex_counts = Counter()
    # 卦×爻別の出現数
    hex_yao_counts = defaultdict(int)
    # 卦×爻×行動別の出現数
    hex_yao_action_counts = defaultdict(lambda: defaultdict(int))
    # 卦×爻別の精度
    hex_yao_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for case in cases:
        yao = case.get("yao_analysis", {})
        hex_id = yao.get("before_hexagram_id")
        yao_pos = yao.get("before_yao_position")
        action = case.get("action_type", "")
        accuracy = yao.get("prediction_analysis", {}).get("accuracy", "miss")

        if hex_id:
            hex_counts[hex_id] += 1

            if yao_pos:
                key = (hex_id, yao_pos)
                hex_yao_counts[key] += 1
                hex_yao_action_counts[key][action] += 1

                hex_yao_accuracy[key]["total"] += 1
                if accuracy in ["exact", "close"]:
                    hex_yao_accuracy[key]["correct"] += 1

    return hex_counts, hex_yao_counts, hex_yao_action_counts, hex_yao_accuracy


def calculate_priority(hex_id, hex_counts, hex_yao_counts):
    """優先度を計算"""
    importance = HEXAGRAMS.get(hex_id, {}).get("importance", "low")
    importance_score = {"high": 3, "medium": 2, "low": 1}.get(importance, 1)

    # データが少ないほど優先度高
    hex_count = hex_counts.get(hex_id, 0)
    if hex_count == 0:
        data_score = 10
    elif hex_count < 10:
        data_score = 5
    elif hex_count < 50:
        data_score = 3
    else:
        data_score = 1

    return importance_score * data_score


def generate_report(cases):
    """ギャップ分析レポートを生成"""

    hex_counts, hex_yao_counts, hex_yao_action_counts, hex_yao_accuracy = analyze_coverage(cases)

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_cases": len(cases),
        "cases_with_hexagram_id": sum(hex_counts.values()),
        "summary": {},
        "missing_hexagrams": [],
        "missing_hex_yao": [],
        "low_data_hex_yao": [],
        "low_accuracy_hex_yao": [],
        "research_tasks": [],
    }

    # 1. 未出現の卦
    all_hexagrams = set(range(1, 65))
    present_hexagrams = set(hex_counts.keys())
    missing_hexagrams = all_hexagrams - present_hexagrams

    for hex_id in sorted(missing_hexagrams):
        info = HEXAGRAMS.get(hex_id, {})
        priority = calculate_priority(hex_id, hex_counts, hex_yao_counts)
        report["missing_hexagrams"].append({
            "hexagram_id": hex_id,
            "name": info.get("name", ""),
            "keyword": info.get("keyword", ""),
            "importance": info.get("importance", ""),
            "priority": priority,
        })

    # 2. データがない卦×爻
    for hex_id in range(1, 65):
        for yao_pos in range(1, 7):
            key = (hex_id, yao_pos)
            if hex_yao_counts[key] == 0:
                info = HEXAGRAMS.get(hex_id, {})
                report["missing_hex_yao"].append({
                    "hexagram_id": hex_id,
                    "yao_position": yao_pos,
                    "name": info.get("name", ""),
                    "priority": calculate_priority(hex_id, hex_counts, hex_yao_counts),
                })

    # 3. データが少ない卦×爻
    for key, count in hex_yao_counts.items():
        if 0 < count < 10:
            hex_id, yao_pos = key
            info = HEXAGRAMS.get(hex_id, {})
            report["low_data_hex_yao"].append({
                "hexagram_id": hex_id,
                "yao_position": yao_pos,
                "name": info.get("name", ""),
                "count": count,
                "priority": calculate_priority(hex_id, hex_counts, hex_yao_counts),
            })

    # 4. 精度が低い卦×爻
    for key, acc in hex_yao_accuracy.items():
        if acc["total"] >= 10:
            accuracy_rate = acc["correct"] / acc["total"]
            if accuracy_rate < 0.6:
                hex_id, yao_pos = key
                info = HEXAGRAMS.get(hex_id, {})
                report["low_accuracy_hex_yao"].append({
                    "hexagram_id": hex_id,
                    "yao_position": yao_pos,
                    "name": info.get("name", ""),
                    "accuracy": round(accuracy_rate, 2),
                    "total": acc["total"],
                    "priority": "high",  # 精度が低いものは高優先
                })

    # 5. リサーチタスクを生成
    # 高優先度の未出現卦
    high_priority_missing = sorted(
        report["missing_hexagrams"],
        key=lambda x: -x["priority"]
    )[:10]

    for item in high_priority_missing:
        report["research_tasks"].append({
            "type": "collect_hexagram_cases",
            "hexagram_id": item["hexagram_id"],
            "hexagram_name": item["name"],
            "keyword": item["keyword"],
            "target_cases": 10,
            "search_hints": generate_search_hints(item["hexagram_id"]),
        })

    # サマリー
    report["summary"] = {
        "hexagrams_present": len(present_hexagrams),
        "hexagrams_missing": len(missing_hexagrams),
        "hex_yao_combinations_with_data": len(hex_yao_counts),
        "hex_yao_combinations_missing": len(report["missing_hex_yao"]),
        "low_data_combinations": len(report["low_data_hex_yao"]),
        "low_accuracy_combinations": len(report["low_accuracy_hex_yao"]),
    }

    return report


def generate_search_hints(hex_id):
    """検索ヒントを生成"""
    hints = {
        1: ["経営者 リーダーシップ 成功", "創業者 ビジョン 事例"],
        2: ["組織 支える 縁の下", "サポート役 成功 事例"],
        3: ["スタートアップ 困難 乗り越え", "創業期 課題 解決"],
        12: ["業界 停滞 閉塞感", "市場縮小 企業 対応"],
        18: ["企業 腐敗 改革", "不祥事 立て直し 事例"],
        23: ["企業 衰退 事例", "業績悪化 転落"],
        24: ["V字回復 企業", "復活 再生 事例"],
        28: ["企業 過剰投資 失敗", "バブル 崩壊 事例"],
        29: ["経営危機 企業 事例", "倒産 回避 事例"],
        33: ["戦略的撤退 企業", "事業売却 成功"],
        39: ["困難 乗り越え 企業", "逆境 克服"],
        40: ["問題解決 企業 事例", "課題解決 成功"],
        41: ["リストラ 企業 事例", "コスト削減 成功"],
        42: ["事業拡大 成功 事例", "成長戦略 企業"],
        47: ["経営難 企業 事例", "苦境 脱出"],
        49: ["企業改革 成功", "変革 事例"],
        51: ["企業 衝撃 対応", "突然 危機 対処"],
        52: ["事業 見直し 停止", "撤退判断 企業"],
        63: ["企業 目標達成 事例", "プロジェクト 完了"],
        64: ["企業 転換期 事例", "次のステージ 移行"],
    }
    return hints.get(hex_id, ["企業 事例 " + HEXAGRAMS.get(hex_id, {}).get("keyword", "")])


def main():
    print("=== ギャップ分析 ===")
    print()

    cases = load_cases()
    report = generate_report(cases)

    # 表示
    print(f"総ケース数: {report['total_cases']}")
    print(f"卦ID付きケース: {report['cases_with_hexagram_id']}")
    print()

    print("=== サマリー ===")
    s = report["summary"]
    print(f"出現卦: {s['hexagrams_present']}/64")
    print(f"未出現卦: {s['hexagrams_missing']}")
    print(f"卦×爻データあり: {s['hex_yao_combinations_with_data']}/384")
    print(f"卦×爻データなし: {s['hex_yao_combinations_missing']}")
    print(f"低データ組み合わせ: {s['low_data_combinations']}")
    print(f"低精度組み合わせ: {s['low_accuracy_combinations']}")
    print()

    print("=== 高優先リサーチタスク（上位10）===")
    for i, task in enumerate(report["research_tasks"][:10], 1):
        print(f"{i}. {task['hexagram_name']} ({task['keyword']})")
        print(f"   目標: {task['target_cases']}件")
        print(f"   検索例: {task['search_hints'][0]}")
        print()

    # 保存
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ANALYSIS_DIR / "gap_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"詳細レポート保存: {output_file}")


if __name__ == "__main__":
    main()
