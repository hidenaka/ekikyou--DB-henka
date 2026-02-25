#!/usr/bin/env python3
"""
易経変化理解支援システム CLI インターフェース

2つの入力モード:
  A) 構造化入力（対話式）: 引数なしで実行
  B) ダイレクト入力（引数指定）: --hexagram, --yao 等を指定

Usage:
    # モードA: 対話式
    python3 scripts/iching_cli.py

    # モードB: ダイレクト入力
    python3 scripts/iching_cli.py --hexagram 12 --yao 3 --state "停滞・閉塞" --action "刷新・破壊"
    python3 scripts/iching_cli.py -H 12 -y 3 -s "停滞・閉塞" -a "刷新・破壊" --extended
    python3 scripts/iching_cli.py -H 12 -y 3 -s "停滞・閉塞" -a "刷新・破壊" --json
"""

import argparse
import json
import os
import sys
from typing import Optional

# --- パス設定 ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from feedback_engine import FeedbackEngine
from probability_tables import ProbabilityMapper
from hexagram_transformations import get_hexagram_name


# ============================================================
# 5軸カテゴリ定義（UI表示用）
# ============================================================

CURRENT_STATES = [
    "どん底・危機", "停滞・閉塞", "不安定・混乱", "安定・順調",
    "成長・発展", "頂点・過剰", "衰退・下降", "転換期",
    "萌芽・準備", "回復途上", "対立・分裂", "依存・束縛",
    "喪失・空虚", "未知・探索", "忍耐・持久",
]

ENERGY_DIRECTIONS = [
    "内向・収束", "外向・拡散", "上昇", "下降", "循環・往復", "停止",
]

INTENDED_ACTIONS = [
    "攻める・前進", "守る・維持", "待つ・忍耐", "撤退・手放す",
    "刷新・破壊", "協調・融和", "分離・独立", "育成・教育",
    "蓄積・準備", "表現・発信", "交渉・取引", "決断・断行",
    "探索・調査", "慎重・観察", "受容・適応", "統合・まとめ",
    "改善・修正", "挑戦・冒険", "奉仕・献身", "楽しむ・喜び",
    "選択・判断", "整理・清算",
]

TRIGGER_NATURES = [
    "突発的外圧", "漸進的変化", "内発的衝動", "周期的転換",
    "関係性変化", "環境変化", "制度・構造変化", "偶発的機会",
]

PHASE_STAGES = [
    "始まり", "展開初期", "展開中期", "展開後期", "頂点・転換", "終結",
]


# ============================================================
# UI表示ラベル → DB確率テーブルラベル マッピング
# ============================================================
# ProbabilityMapper のテーブルは DB の実データから生成されたカテゴリを使う。
# UI で表示するラベルとは一致しない場合があるため、最も近いラベルにマッピングする。

# before_state: UI → DB (prob_tables before_state_to_hex のキー)
STATE_TO_DB = {
    "どん底・危機":   "どん底・危機",
    "停滞・閉塞":     "停滞・閉塞",
    "不安定・混乱":   "混乱・カオス",
    "安定・順調":     "安定・平和",
    "成長・発展":     "成長・拡大",
    "頂点・過剰":     "絶頂・慢心",
    "衰退・下降":     "混乱・衰退",
    "転換期":         "成長痛",
    "萌芽・準備":     "縮小安定・生存",
    "回復途上":       "V字回復・大成功",
    "対立・分裂":     "混乱・カオス",
    "依存・束縛":     "安定・停止",
    "喪失・空虚":     "どん底・危機",
    "未知・探索":     "成長痛",
    "忍耐・持久":     "停滞・閉塞",
}

# action_type: UI → DB (prob_tables action_type_to_hex のキー)
ACTION_TO_DB = {
    "攻める・前進":   "攻める・挑戦",
    "守る・維持":     "守る・維持",
    "待つ・忍耐":     "耐える・潜伏",
    "撤退・手放す":   "撤退・縮小",
    "刷新・破壊":     "刷新・破壊",
    "協調・融和":     "対話・融合",
    "分離・独立":     "分散・独立",
    "育成・教育":     "集中・拡大",
    "蓄積・準備":     "耐える・潜伏",
    "表現・発信":     "輝く・表現",
    "交渉・取引":     "対話・融合",
    "決断・断行":     "攻める・挑戦",
    "探索・調査":     "分散・探索",
    "慎重・観察":     "守る・維持",
    "受容・適応":     "対話・融合",
    "統合・まとめ":   "集中・拡大",
    "改善・修正":     "刷新・破壊",
    "挑戦・冒険":     "攻める・挑戦",
    "奉仕・献身":     "対話・融合",
    "楽しむ・喜び":   "輝く・表現",
    "選択・判断":     "攻める・挑戦",
    "整理・清算":     "捨てる・撤退",
}

# trigger_nature: UI → DB (prob_tables trigger_type_to_hex のキー)
TRIGGER_TO_DB = {
    "突発的外圧":     "外部ショック",
    "漸進的変化":     "自然推移",
    "内発的衝動":     "意図的決断",
    "周期的転換":     "自然推移・成熟",
    "関係性変化":     "内部矛盾・自壊",
    "環境変化":       "外部ショック",
    "制度・構造変化": "内部崩壊",
    "偶発的機会":     "偶発・出会い",
}

# phase_stage: UI → DB (PHASE_TO_YAO のキー)
PHASE_TO_DB = {
    "始まり":     "潜伏・発芽",
    "展開初期":   "出現・成長",
    "展開中期":   "危機・転換点",
    "展開後期":   "選択・跳躍準備",
    "頂点・転換": "最盛・中正",
    "終結":       "過剰・衰退",
}

# energy_direction: UI → ProbabilityMapper._energy_direction_prior() のキー
ENERGY_TO_DB = {
    "内向・収束": "contracting",
    "外向・拡散": "expanding",
    "上昇":       "expanding",
    "下降":       "contracting",
    "循環・往復": None,  # 均等分布
    "停止":       "contracting",
}


# ============================================================
# 対話式ヘルパー
# ============================================================

def print_header():
    """タイトルヘッダーを表示"""
    print()
    print("\u2501" * 40)
    print("  易経変化理解支援システム")
    print("\u2501" * 40)
    print()


def prompt_choice(title: str, options: list, allow_skip: bool = False) -> Optional[int]:
    """
    番号入力で選択肢を選ばせる。

    Returns:
        選択されたインデックス (0-based)。スキップ時は None。
    """
    print(f"\u25a0 {title}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i:2d}. {opt}")

    skip_msg = "  （Enterでスキップ）" if allow_skip else ""
    while True:
        try:
            raw = input(f"> {skip_msg}").strip()
        except EOFError:
            raise KeyboardInterrupt

        if allow_skip and raw == "":
            print()
            return None

        try:
            num = int(raw)
        except ValueError:
            print(f"  ! 1-{len(options)} の番号を入力してください")
            continue

        if 1 <= num <= len(options):
            print()
            return num - 1
        else:
            print(f"  ! 1-{len(options)} の範囲で入力してください")


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    """Y/N 確認。default_yes=True → Enter で Yes。"""
    hint = "[Y/n]" if default_yes else "[y/N]"
    try:
        raw = input(f"{message} {hint} ").strip().lower()
    except EOFError:
        raise KeyboardInterrupt

    if raw == "":
        return default_yes
    return raw in ("y", "yes")


def prompt_candidate_selection(candidates: list) -> int:
    """
    候補リストから選択させる。

    Returns:
        選択された候補のインデックス (0-based)
    """
    while True:
        try:
            raw = input("  → 番号を選択してください [1]: ").strip()
        except EOFError:
            raise KeyboardInterrupt

        if raw == "":
            return 0

        # Y/y → 1番目
        if raw.lower() in ("y", "yes"):
            return 0

        # N/n → 入力やり直しではなくキャンセル的な意味はないので 1番目を返す
        try:
            num = int(raw)
        except ValueError:
            print(f"  ! 1-{len(candidates)} の番号を入力してください")
            continue

        if 1 <= num <= len(candidates):
            return num - 1
        else:
            print(f"  ! 1-{len(candidates)} の範囲で入力してください")


# ============================================================
# モードA: 構造化入力（対話式）
# ============================================================

def run_interactive():
    """対話式モードを実行"""
    print_header()

    # 1. 現在の状態
    idx_state = prompt_choice("現在の状態を選んでください", CURRENT_STATES)
    ui_state = CURRENT_STATES[idx_state]
    db_state = STATE_TO_DB[ui_state]

    # 2. エネルギーの方向
    idx_energy = prompt_choice("エネルギーの方向を選んでください", ENERGY_DIRECTIONS)
    ui_energy = ENERGY_DIRECTIONS[idx_energy]
    db_energy = ENERGY_TO_DB[ui_energy]

    # 3. 意図する行動
    idx_action = prompt_choice("意図する行動を選んでください", INTENDED_ACTIONS)
    ui_action = INTENDED_ACTIONS[idx_action]
    db_action = ACTION_TO_DB[ui_action]

    # 4. トリガーの性質
    idx_trigger = prompt_choice("トリガーの性質を選んでください", TRIGGER_NATURES)
    ui_trigger = TRIGGER_NATURES[idx_trigger]
    db_trigger = TRIGGER_TO_DB[ui_trigger]

    # 5. フェーズ段階
    idx_phase = prompt_choice("フェーズ段階を選んでください", PHASE_STAGES)
    ui_phase = PHASE_STAGES[idx_phase]
    db_phase = PHASE_TO_DB[ui_phase]

    # --- マッピング実行 ---
    print("マッピング中...")
    print()

    mapper = ProbabilityMapper()
    result = mapper.get_top_candidates(
        current_state=db_state,
        intended_action=db_action,
        trigger_nature=db_trigger,
        phase_stage=db_phase,
        energy_direction=db_energy,
        n=3,
    )

    candidates = result["candidates"]

    if not candidates:
        print("  候補が見つかりませんでした。条件を変えて再試行してください。")
        return

    # --- 候補表示 ---
    print("  本卦候補:")
    for c in candidates:
        pct = c["probability"] * 100
        hex_name = c["hexagram_name"]
        hex_num = c["hexagram_number"]
        print(f"  {c['rank']}. {hex_name}（{hex_num}）[確率: {pct:.1f}%]")

    print()
    yao = candidates[0].get("yao")
    if yao:
        phase_names = ["潜伏", "出現", "転換", "選択", "最盛", "極限"]
        phase_label = phase_names[yao - 1] if 1 <= yao <= 6 else ""
        print(f"  動爻候補: 第{yao}爻（{phase_label}）")
    print()

    # --- 候補選択 ---
    selected_idx = prompt_candidate_selection(candidates)
    selected = candidates[selected_idx]

    hex_num = selected["hexagram_number"]
    yao_pos = selected.get("yao", 3)
    confidence = selected["probability"]

    # --- フィードバック生成 ---
    print()
    print("読み解きを生成中...")
    print()

    engine = FeedbackEngine()
    text = engine.generate_text(
        hex_num, yao_pos, ui_state, ui_action,
        mapping_confidence=confidence,
        show_extended=False,
    )
    print(text)
    print()

    # --- 詳細表示の確認 ---
    if prompt_yes_no("  詳細を表示しますか？", default_yes=False):
        print()
        extended_text = engine.generate_text(
            hex_num, yao_pos, ui_state, ui_action,
            mapping_confidence=confidence,
            show_extended=True,
        )
        print(extended_text)
        print()

    # --- JSON出力の確認 ---
    if prompt_yes_no("  JSON出力しますか？", default_yes=False):
        print()
        data = engine.generate(
            hex_num, yao_pos, ui_state, ui_action,
            mapping_confidence=confidence,
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print()


# ============================================================
# モードB: ダイレクト入力（引数指定）
# ============================================================

def run_direct(args):
    """ダイレクト入力モードを実行"""
    hex_num = args.hexagram
    yao_pos = args.yao
    state = args.state or ""
    action = args.action or ""
    confidence = args.confidence
    show_extended = args.extended
    output_json = args.json

    # バリデーション
    if not (1 <= hex_num <= 64):
        print(f"エラー: 卦番号は 1-64 の範囲で指定してください（入力: {hex_num}）",
              file=sys.stderr)
        sys.exit(1)

    if not (1 <= yao_pos <= 6):
        print(f"エラー: 動爻位置は 1-6 の範囲で指定してください（入力: {yao_pos}）",
              file=sys.stderr)
        sys.exit(1)

    if not (0.0 <= confidence <= 1.0):
        print(f"エラー: 確信度は 0.0-1.0 の範囲で指定してください（入力: {confidence}）",
              file=sys.stderr)
        sys.exit(1)

    engine = FeedbackEngine()

    if output_json:
        data = engine.generate(
            hex_num, yao_pos, state, action,
            mapping_confidence=confidence,
        )
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        text = engine.generate_text(
            hex_num, yao_pos, state, action,
            mapping_confidence=confidence,
            show_extended=show_extended,
        )
        print(text)


# ============================================================
# argparse 設定
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="易経変化理解支援システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
例:
  # 対話式モード
  python3 scripts/iching_cli.py

  # ダイレクトモード（基本）
  python3 scripts/iching_cli.py --hexagram 12 --yao 3 --state "停滞・閉塞" --action "刷新・破壊"

  # ダイレクトモード（詳細展開）
  python3 scripts/iching_cli.py -H 12 -y 3 -s "停滞・閉塞" -a "刷新・破壊" --extended

  # ダイレクトモード（JSON出力）
  python3 scripts/iching_cli.py -H 12 -y 3 -s "停滞・閉塞" -a "刷新・破壊" --json
""",
    )

    parser.add_argument(
        "--hexagram", "-H", type=int,
        help="卦番号 (1-64)",
    )
    parser.add_argument(
        "--yao", "-y", type=int,
        help="動爻位置 (1-6)",
    )
    parser.add_argument(
        "--state", "-s", type=str, default="",
        help="before_state（現在の状態）",
    )
    parser.add_argument(
        "--action", "-a", type=str, default="",
        help="action_type（意図する行動）",
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.7,
        help="マッピング確信度 (0.0-1.0, デフォルト: 0.7)",
    )
    parser.add_argument(
        "--extended", "-e", action="store_true",
        help="LAYER 3-4 の詳細展開を含める",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="JSON形式で出力",
    )

    return parser


# ============================================================
# メインエントリポイント
# ============================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ダイレクトモード判定: --hexagram と --yao が両方指定されていればモードB
    if args.hexagram is not None and args.yao is not None:
        run_direct(args)
    elif args.hexagram is not None or args.yao is not None:
        # 片方だけ指定はエラー
        print("エラー: --hexagram と --yao は両方指定してください。",
              file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)
    else:
        # 引数なし → 対話式モード
        try:
            run_interactive()
        except KeyboardInterrupt:
            print("\n\n  中断しました。")
            sys.exit(0)


if __name__ == "__main__":
    main()
