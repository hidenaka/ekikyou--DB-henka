"""
スキーマ正規化スクリプト: 定義外enum値を正規値にマッピング変換する。
バックアップ作成 → 変換 → 検証 → 逆引きインデックス再構築
"""
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

DATA_FILE = Path(__file__).parent.parent / "data" / "raw" / "cases.jsonl"

# ========== マッピングテーブル ==========

BEFORE_STATE_MAP = {
    # 正規値はそのまま通す
    "絶頂・慢心": "絶頂・慢心",
    "停滞・閉塞": "停滞・閉塞",
    "混乱・カオス": "混乱・カオス",
    "成長痛": "成長痛",
    "どん底・危機": "どん底・危機",
    "安定・平和": "安定・平和",
    # 定義外 → 正規値
    "混乱・衰退": "混乱・カオス",        # 混乱が主、衰退は付随
    "安定・停止": "停滞・閉塞",           # 停止=閉塞
    "安定成長・成功": "絶頂・慢心",       # 成功の絶頂
    "拡大・繁栄": "絶頂・慢心",           # 繁栄=絶頂
    "成長・拡大": "成長痛",               # 成長過程
    "調和・繁栄": "安定・平和",           # 調和=平和
    "縮小安定・生存": "停滞・閉塞",       # 縮小安定=停滞
    "V字回復・大成功": "絶頂・慢心",      # 大成功後=絶頂
    "急成長・拡大": "成長痛",             # 急成長=成長痛
}

AFTER_STATE_MAP = {
    # 正規値はそのまま通す
    "V字回復・大成功": "V字回復・大成功",
    "縮小安定・生存": "縮小安定・生存",
    "変質・新生": "変質・新生",
    "現状維持・延命": "現状維持・延命",
    "迷走・混乱": "迷走・混乱",
    "崩壊・消滅": "崩壊・消滅",
    # 定義外 → 正規値
    "持続成長・大成功": "V字回復・大成功",  # 大成功
    "安定成長・成功": "V字回復・大成功",    # 成功
    "安定・平和": "縮小安定・生存",         # 安定的着地
    "停滞・閉塞": "現状維持・延命",         # 停滞=延命
    "混乱・衰退": "迷走・混乱",            # 混乱
    "混乱・カオス": "迷走・混乱",           # カオス=混乱
    "安定・停止": "現状維持・延命",         # 停止=延命
    "拡大・繁栄": "V字回復・大成功",        # 繁栄=成功
    "どん底・危機": "崩壊・消滅",           # 危機=崩壊
    "喜び・交流": "変質・新生",             # 交流による新生
    "成長・拡大": "V字回復・大成功",        # 成長=成功
    "成長痛": "変質・新生",                 # 変質の過程
    "分岐・様子見": "現状維持・延命",       # 様子見=延命
    "消滅・破綻": "崩壊・消滅",             # 消滅
}

ACTION_TYPE_MAP = {
    # 正規値はそのまま通す
    "攻める・挑戦": "攻める・挑戦",
    "守る・維持": "守る・維持",
    "捨てる・撤退": "捨てる・撤退",
    "耐える・潜伏": "耐える・潜伏",
    "対話・融合": "対話・融合",
    "刷新・破壊": "刷新・破壊",
    "逃げる・放置": "逃げる・放置",
    "分散・スピンオフ": "分散・スピンオフ",
    # 定義外 → 正規値
    "分散・探索": "分散・スピンオフ",        # 探索=分散
    "捨てる・転換": "捨てる・撤退",          # 転換=撤退
    "交流・発表": "対話・融合",              # 交流=対話
    "集中・拡大": "攻める・挑戦",            # 拡大=攻め
    "拡大・攻め": "攻める・挑戦",            # 攻め
    "撤退・収縮": "捨てる・撤退",            # 撤退
    "撤退・縮小": "捨てる・撤退",            # 撤退
    "撤退・逃げる": "逃げる・放置",          # 逃げる
    "逃げる・分散": "逃げる・放置",          # 逃げる
    "輝く・表現": "攻める・挑戦",            # 表現=挑戦
    "逃げる・守る": "守る・維持",            # 守る
    "分散・多角化": "分散・スピンオフ",      # 多角化=分散
    "分散・独立": "分散・スピンオフ",        # 独立=スピンオフ
    "分散する・独立する": "分散・スピンオフ", # 同上
}

PATTERN_TYPE_MAP = {
    # 正規値はそのまま通す
    "Shock_Recovery": "Shock_Recovery",
    "Hubris_Collapse": "Hubris_Collapse",
    "Pivot_Success": "Pivot_Success",
    "Endurance": "Endurance",
    "Slow_Decline": "Slow_Decline",
    # 定義外 → 正規値
    "Steady_Growth": "Endurance",          # 安定成長=忍耐の成果
    "Breakthrough": "Pivot_Success",       # 突破=ピボット成功
    "Crisis_Pivot": "Shock_Recovery",      # 危機転換=ショック回復
    "Failed_Attempt": "Hubris_Collapse",   # 失敗=崩壊
    "Stagnation": "Slow_Decline",          # 停滞=じわじわ衰退
    "Managed_Decline": "Slow_Decline",     # 管理された衰退
    "Exploration": "Endurance",            # 探索=忍耐
    "Quiet_Fade": "Slow_Decline",          # 静かに消える=衰退
    "Decline": "Slow_Decline",             # 衰退
}

ALL_MAPS = {
    "before_state": BEFORE_STATE_MAP,
    "after_state": AFTER_STATE_MAP,
    "action_type": ACTION_TYPE_MAP,
    "pattern_type": PATTERN_TYPE_MAP,
}

VALID_ENUMS = {
    "before_state": {"絶頂・慢心", "停滞・閉塞", "混乱・カオス", "成長痛", "どん底・危機", "安定・平和"},
    "after_state": {"V字回復・大成功", "縮小安定・生存", "変質・新生", "現状維持・延命", "迷走・混乱", "崩壊・消滅"},
    "action_type": {"攻める・挑戦", "守る・維持", "捨てる・撤退", "耐える・潜伏", "対話・融合", "刷新・破壊", "逃げる・放置", "分散・スピンオフ"},
    "pattern_type": {"Shock_Recovery", "Hubris_Collapse", "Pivot_Success", "Endurance", "Slow_Decline"},
}


def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.parent / f"cases_backup_{ts}.jsonl"
    shutil.copy2(path, bak)
    print(f"バックアップ作成: {bak}")
    return bak


def normalize(dry_run: bool = False):
    cases = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    total = len(cases)
    print(f"総件数: {total}")

    # 変換ログ
    changes = {field: Counter() for field in ALL_MAPS}
    unmapped = {field: Counter() for field in ALL_MAPS}
    changed_count = 0

    for case in cases:
        case_changed = False
        for field, mapping in ALL_MAPS.items():
            old_val = case.get(field, "")
            if old_val in mapping:
                new_val = mapping[old_val]
                if old_val != new_val:
                    changes[field][(old_val, new_val)] += 1
                    case[field] = new_val
                    case_changed = True
            else:
                unmapped[field][old_val] += 1
        if case_changed:
            changed_count += 1

    # レポート
    print(f"\n変換対象レコード: {changed_count}件")
    for field in ALL_MAPS:
        if changes[field]:
            print(f"\n=== {field} 変換 ===")
            for (old, new), cnt in changes[field].most_common():
                print(f"  \"{old}\" → \"{new}\": {cnt}件")

    # 未マッピング値の警告
    has_unmapped = False
    for field in ALL_MAPS:
        for val, cnt in unmapped[field].most_common():
            if val not in VALID_ENUMS[field]:
                if not has_unmapped:
                    print("\n!!! 未マッピング値（要対応）!!!")
                    has_unmapped = True
                print(f"  {field}: \"{val}\" ({cnt}件)")

    if has_unmapped:
        print("\n上記の値はマッピングテーブルに追加が必要です。")
        if not dry_run:
            print("変換を中止します。")
            return False

    # 検証
    violations = 0
    for case in cases:
        for field, valids in VALID_ENUMS.items():
            if case.get(field, "") not in valids:
                violations += 1

    if dry_run:
        if violations > 0:
            print(f"\n[DRY-RUN] 変換後もenum違反が {violations}件 残ります")
        else:
            print(f"\n[DRY-RUN] 変換後のenum違反: 0件 (100%準拠)")
        print("[DRY-RUN] ファイルは変更されていません")
        return True

    if violations > 0:
        print(f"\n変換後もenum違反が {violations}件 残ります。中止します。")
        return False

    # バックアップ & 書き込み
    backup(DATA_FILE)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n正規化完了: {total}件 書き込み済み")
    print(f"enum違反: 0件 (100%準拠)")
    return True


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY-RUN モード ===\n")
    else:
        print("=== 正規化実行モード ===\n")

    success = normalize(dry_run=dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
