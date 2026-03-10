#!/usr/bin/env python3
"""
条件付き分布計算 + 類似事例検索モジュール

LAYER 4（過去事例から見える分布）を構成するための検索エンジン。
specs/feedback_output_layer.md セクション 2.4 の仕様に基づく。

Usage:
    from case_search import CaseSearchEngine
    engine = CaseSearchEngine()
    dist = engine.get_conditional_distribution("停滞・閉塞", "刷新・破壊")
    cases = engine.search_similar_cases(before_state="どん底・危機", action_type="耐える・潜伏", limit=3)
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional


class CaseSearchEngine:
    """DB事例を使った条件付き分布計算 + 類似事例検索エンジン"""

    # 有効なscale値
    VALID_SCALES = ("company", "individual", "family", "country", "other")

    def __init__(self, cases_path: str = None):
        """
        全事例をロードしてインデックスを構築。

        Args:
            cases_path: cases.jsonlのパス。省略時はスクリプト位置から自動解決。
        """
        if cases_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cases_path = os.path.join(base_dir, "data", "raw", "cases.jsonl")

        self.cases = self._load_cases(cases_path)
        self._build_indices()

    def _load_cases(self, path: str) -> list:
        """cases.jsonlを1行ずつ読み込んでリストとして返す。"""
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return cases

    def _build_indices(self):
        """効率的な検索のためのインデックスを構築する。"""
        self._by_hexagram = {}          # hexagram_number → [事例リスト]
        self._by_hex_yao = {}           # (hexagram_number, yao) → [事例リスト]
        self._by_state_action = {}      # (before_state, action_type) → [事例リスト]
        self._by_hex_pair = {}          # (before_hex, action_hex) → [事例リスト]
        self._by_scale = {}             # scale → set(case index in self.cases)

        for idx, case in enumerate(self.cases):
            # hexagram_number インデックス
            hex_num = case.get("hexagram_number")
            if hex_num is not None:
                self._by_hexagram.setdefault(hex_num, []).append(case)

                # (hexagram_number, yao) インデックス
                yao = case.get("yao")
                if yao is not None:
                    self._by_hex_yao.setdefault((hex_num, yao), []).append(case)

            # (before_state, action_type) インデックス
            bs = case.get("before_state")
            at = case.get("action_type")
            if bs and at:
                self._by_state_action.setdefault((bs, at), []).append(case)

            # (before_hex, action_hex) インデックス
            bh = case.get("before_hex")
            ah = case.get("action_hex")
            if bh and ah:
                self._by_hex_pair.setdefault((bh, ah), []).append(case)

            # scale インデックス（scaleフィールド直接使用）
            scale = case.get("scale", "other")
            if scale not in self.VALID_SCALES:
                scale = "other"
            self._by_scale.setdefault(scale, set()).add(id(case))

    # --- scaleフィルタ ---

    def _filter_by_scale(self, cases: list, scale: Optional[str]) -> list:
        """
        事例リストをscaleでフィルタリングする。

        Args:
            cases: フィルタ対象の事例リスト
            scale: "company", "individual", "family", "country", "other" のいずれか。
                   Noneの場合はフィルタなし（全件返却）。

        Returns:
            フィルタ後の事例リスト
        """
        if scale is None:
            return cases
        scale_ids = self._by_scale.get(scale, set())
        return [c for c in cases if id(c) in scale_ids]

    # --- 条件付き分布 ---

    @staticmethod
    def _make_confidence_note(n: int, action_type: str) -> str:
        """nの範囲に応じた注意書きを生成する。"""
        if n >= 100:
            return f"同様の状況で{action_type}をとった{n}件の事例では、以下の分布が見られました。"
        elif n >= 30:
            return f"参考事例が{n}件あります。傾向の参考程度にご覧ください。"
        elif n >= 10:
            return f"該当事例は{n}件と少数です。分布は参考情報としてご覧ください。"
        else:
            return f"該当事例は{n}件のみです。統計的な傾向を読み取ることは困難です。"

    def get_conditional_distribution(
        self, before_state: str, action_type: str, scale: Optional[str] = None
    ) -> dict:
        """
        条件付き分布を算出する。

        Args:
            before_state: "停滞・閉塞" 等
            action_type: "刷新・破壊" 等
            scale: "company", "individual", "family", "country", "other" のいずれか。
                   Noneの場合は全件で集計（後方互換）。

        Returns:
            {
                "condition": {"before_state": ..., "action_type": ..., "scale": ...},
                "total_n": int,
                "confidence_note": str,
                "distribution": [{"state": str, "count": int, "percentage": float}, ...],
                "caveat": str
            }
        """
        matched = self._by_state_action.get((before_state, action_type), [])
        matched = self._filter_by_scale(matched, scale)
        total_n = len(matched)

        # after_state の分布をカウント
        counter = Counter(
            case.get("after_state", "不明") for case in matched
        )

        # percentage降順でソート
        distribution = []
        for state, count in counter.most_common():
            pct = round(count / total_n * 100, 1) if total_n > 0 else 0.0
            distribution.append({
                "state": state,
                "count": count,
                "percentage": pct,
            })

        condition = {
            "before_state": before_state,
            "action_type": action_type,
        }
        if scale is not None:
            condition["scale"] = scale

        return {
            "condition": condition,
            "total_n": total_n,
            "confidence_note": self._make_confidence_note(total_n, action_type),
            "distribution": distribution,
            "caveat": "過去事例の分布であり、あなたの結果を予測するものではありません",
        }

    # --- 類似事例検索 ---

    @staticmethod
    def _truncate_summary(text: str, max_len: int = 100) -> str:
        """story_summaryをmax_len文字に切り詰める。"""
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    @staticmethod
    def _format_case(case: dict, similarity_basis: str) -> dict:
        """事例を返却用の辞書形式に変換する。"""
        return {
            "target_name": case.get("target_name", "不明"),
            "period": case.get("period", "不明"),
            "before_state": case.get("before_state", "不明"),
            "action_type": case.get("action_type", "不明"),
            "after_state": case.get("after_state", "不明"),
            "story_summary": CaseSearchEngine._truncate_summary(
                case.get("story_summary", "")
            ),
            "similarity_basis": similarity_basis,
        }

    def search_similar_cases(
        self,
        hexagram_number: int = None,
        yao_position: int = None,
        before_state: str = None,
        action_type: str = None,
        before_hex: str = None,
        action_hex: str = None,
        limit: int = 3,
        scale: Optional[str] = None,
    ) -> list:
        """
        類似事例を検索する。

        検索優先順位（上位ほど優先）:
        1. 本卦(hexagram_number) + 動爻(yao_position)が一致
        2. 本卦(hexagram_number)が一致（動爻は異なる）
        3. before_state + action_type が一致
        4. before_hex + action_hex が一致

        各優先順位内では年度降順・名前昇順で決定的に選出（再現性保証）。
        最大 limit 件返す。

        Args:
            scale: "company", "individual", "family", "country", "other" のいずれか。
                   Noneの場合は全件から検索（後方互換）。
        """
        results = []
        used_ids = set()  # transition_id で重複排除

        def _add_cases(candidate_list: list, basis: str):
            """候補リストからまだ選ばれていないものを決定的順位で追加。"""
            filtered = self._filter_by_scale(candidate_list, scale)
            available = [
                c for c in filtered
                if c.get("transition_id", id(c)) not in used_ids
            ]
            # 決定的ランキング: year降順 → target_name昇順（再現性保証）
            available.sort(
                key=lambda c: (-(c.get("year") or 0), c.get("target_name", "")),
            )
            for case in available:
                if len(results) >= limit:
                    return
                tid = case.get("transition_id", id(case))
                used_ids.add(tid)
                results.append(self._format_case(case, basis))

        # 優先度1: 本卦 + 動爻
        if hexagram_number is not None and yao_position is not None:
            matched = self._by_hex_yao.get((hexagram_number, yao_position), [])
            _add_cases(matched, "本卦+動爻一致")

        # 優先度2: 本卦のみ（動爻一致分は既に追加済みなので重複除外される）
        if len(results) < limit and hexagram_number is not None:
            matched = self._by_hexagram.get(hexagram_number, [])
            _add_cases(matched, "本卦一致")

        # 優先度3: before_state + action_type
        if len(results) < limit and before_state and action_type:
            matched = self._by_state_action.get((before_state, action_type), [])
            _add_cases(matched, "before_state+action_type一致")

        # 優先度4: before_hex + action_hex
        if len(results) < limit and before_hex and action_hex:
            matched = self._by_hex_pair.get((before_hex, action_hex), [])
            _add_cases(matched, "before_hex+action_hex一致")

        return results

    # --- パターン分布（オプション） ---

    def get_pattern_distribution(
        self, hexagram_number: int, scale: Optional[str] = None
    ) -> dict:
        """
        指定卦のパターン分布。
        同じ本卦を持つ事例の after_state の分布を返す。

        Args:
            hexagram_number: 卦番号 (1-64)
            scale: "company", "individual", "family", "country", "other" のいずれか。
                   Noneの場合は全件で集計（後方互換）。

        Returns:
            {
                "hexagram_number": int,
                "total_n": int,
                "scale": str or None,
                "distribution": [{"state": str, "count": int, "percentage": float}, ...]
            }
        """
        matched = self._by_hexagram.get(hexagram_number, [])
        matched = self._filter_by_scale(matched, scale)
        total_n = len(matched)

        counter = Counter(
            case.get("after_state", "不明") for case in matched
        )

        distribution = []
        for state, count in counter.most_common():
            pct = round(count / total_n * 100, 1) if total_n > 0 else 0.0
            distribution.append({
                "state": state,
                "count": count,
                "percentage": pct,
            })

        result = {
            "hexagram_number": hexagram_number,
            "total_n": total_n,
            "distribution": distribution,
        }
        if scale is not None:
            result["scale"] = scale
        return result

    def get_scale_distribution(self) -> dict:
        """
        全事例のscale別件数分布を返す。

        Returns:
            {"company": int, "individual": int, "family": int, "country": int, "other": int}
        """
        return {scale: len(ids) for scale, ids in self._by_scale.items()}


# =============================================================================
# テスト実行
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CaseSearchEngine テスト実行")
    print("=" * 70)

    engine = CaseSearchEngine()
    print(f"\n  ロード完了: {len(engine.cases)} 件")

    # --- テスト1: get_conditional_distribution ---
    print("\n--- テスト1: get_conditional_distribution('停滞・閉塞', '刷新・破壊') ---")
    dist = engine.get_conditional_distribution("停滞・閉塞", "刷新・破壊")
    print(f"  total_n = {dist['total_n']}")
    print(f"  confidence_note = {dist['confidence_note']}")
    assert dist["total_n"] > 100, f"FAIL: total_n={dist['total_n']} (expected > 100)"
    assert len(dist["distribution"]) > 0, "FAIL: distribution is empty"
    print(f"  distribution ({len(dist['distribution'])} states):")
    for d in dist["distribution"][:5]:
        print(f"    {d['state']}: {d['count']}件 ({d['percentage']}%)")
    print(f"  caveat = {dist['caveat']}")
    print("  PASS")

    # --- テスト2: search_similar_cases (before_state + action_type) ---
    print("\n--- テスト2: search_similar_cases(before_state='どん底・危機', action_type='耐える・潜伏') ---")
    cases = engine.search_similar_cases(
        before_state="どん底・危機",
        action_type="耐える・潜伏",
        limit=3,
    )
    assert len(cases) == 3, f"FAIL: returned {len(cases)} cases (expected 3)"
    for c in cases:
        print(f"  ■ {c['target_name']}（{c['period']}）")
        print(f"    {c['before_state']} → {c['action_type']} → {c['after_state']}")
        print(f"    概要: {c['story_summary']}")
        print(f"    類似根拠: {c['similarity_basis']}")
    print("  PASS")

    # --- テスト3: search_similar_cases (hexagram_number + yao_position) ---
    print("\n--- テスト3: search_similar_cases(hexagram_number=12, yao_position=3) ---")
    cases = engine.search_similar_cases(
        hexagram_number=12,
        yao_position=3,
        limit=3,
    )
    assert len(cases) <= 3, f"FAIL: returned {len(cases)} cases (expected <= 3)"
    print(f"  返却件数: {len(cases)}")
    for c in cases:
        print(f"  ■ {c['target_name']}（{c['period']}）")
        print(f"    類似根拠: {c['similarity_basis']}")
        # 本卦+動爻一致 or 本卦一致 のいずれかであるべき
        assert c["similarity_basis"] in ("本卦+動爻一致", "本卦一致"), \
            f"FAIL: unexpected similarity_basis '{c['similarity_basis']}'"
    print("  PASS")

    # --- テスト4: confidence_note の分岐テスト ---
    print("\n--- テスト4: confidence_note の分岐テスト ---")

    # n >= 100
    note_large = CaseSearchEngine._make_confidence_note(442, "刷新・破壊")
    assert "442件の事例" in note_large, f"FAIL: n>=100 note: {note_large}"
    print(f"  n=442: {note_large}")

    # 30 <= n < 100
    note_mid = CaseSearchEngine._make_confidence_note(55, "守る・維持")
    assert "55件" in note_mid and "参考程度" in note_mid, f"FAIL: 30<=n<100 note: {note_mid}"
    print(f"  n=55:  {note_mid}")

    # 10 <= n < 30
    note_small = CaseSearchEngine._make_confidence_note(18, "攻める・挑戦")
    assert "18件と少数" in note_small, f"FAIL: 10<=n<30 note: {note_small}"
    print(f"  n=18:  {note_small}")

    # n < 10
    note_tiny = CaseSearchEngine._make_confidence_note(5, "撤退・手放す")
    assert "5件のみ" in note_tiny, f"FAIL: n<10 note: {note_tiny}"
    print(f"  n=5:   {note_tiny}")

    print("  PASS")

    # --- テスト5: 存在しないカテゴリの場合 ---
    print("\n--- テスト5: 存在しないカテゴリ ---")
    dist_empty = engine.get_conditional_distribution("存在しない状態", "存在しない行動")
    assert dist_empty["total_n"] == 0, f"FAIL: total_n={dist_empty['total_n']}"
    assert dist_empty["distribution"] == [], "FAIL: distribution should be empty"
    print(f"  total_n = {dist_empty['total_n']}")
    print(f"  distribution = {dist_empty['distribution']}")

    cases_empty = engine.search_similar_cases(
        before_state="存在しない状態",
        action_type="存在しない行動",
        limit=3,
    )
    assert len(cases_empty) == 0, f"FAIL: returned {len(cases_empty)} cases"
    print(f"  search result count = {len(cases_empty)}")
    print("  PASS")

    # --- テスト6: story_summary の切り詰め ---
    print("\n--- テスト6: story_summary の切り詰め ---")
    long_text = "あ" * 200
    truncated = CaseSearchEngine._truncate_summary(long_text)
    assert len(truncated) == 103, f"FAIL: truncated length={len(truncated)}"  # 100 + "..."
    assert truncated.endswith("..."), "FAIL: should end with '...'"
    short_text = "短いテキスト"
    not_truncated = CaseSearchEngine._truncate_summary(short_text)
    assert not_truncated == short_text, "FAIL: short text should not be truncated"
    print(f"  長い文 → {len(truncated)} chars (100+...)")
    print(f"  短い文 → '{not_truncated}' (そのまま)")
    print("  PASS")

    # --- テスト7: get_pattern_distribution ---
    print("\n--- テスト7: get_pattern_distribution(hexagram_number=21) ---")
    pat = engine.get_pattern_distribution(21)
    print(f"  hexagram_number = {pat['hexagram_number']}")
    print(f"  total_n = {pat['total_n']}")
    assert pat["total_n"] > 0, "FAIL: no cases for hexagram 21"
    for d in pat["distribution"][:5]:
        print(f"    {d['state']}: {d['count']}件 ({d['percentage']}%)")
    print("  PASS")

    # --- テスト8: 複合検索（優先順位の確認） ---
    print("\n--- テスト8: 複合検索 (hexagram + state + hex_pair) ---")
    cases = engine.search_similar_cases(
        hexagram_number=21,
        yao_position=4,
        before_state="停滞・閉塞",
        action_type="刷新・破壊",
        before_hex="坤",
        action_hex="震",
        limit=5,
    )
    print(f"  返却件数: {len(cases)}")
    for c in cases:
        print(f"  ■ {c['target_name']} → {c['similarity_basis']}")
    # 優先順位が上のbasisが先に来ることを確認
    bases_order = {"本卦+動爻一致": 0, "本卦一致": 1, "before_state+action_type一致": 2, "before_hex+action_hex一致": 3}
    for i in range(len(cases) - 1):
        b1 = bases_order.get(cases[i]["similarity_basis"], 99)
        b2 = bases_order.get(cases[i + 1]["similarity_basis"], 99)
        assert b1 <= b2, f"FAIL: order violation at index {i}: {cases[i]['similarity_basis']} > {cases[i+1]['similarity_basis']}"
    print("  PASS: 優先順位が正しい")

    # --- テスト9: scaleインデックスの基本確認 ---
    print("\n--- テスト9: scaleインデックス ---")
    scale_dist = engine.get_scale_distribution()
    print(f"  scale分布: {scale_dist}")
    total_scale = sum(scale_dist.values())
    assert total_scale == len(engine.cases), \
        f"FAIL: scale合計({total_scale}) != 全件数({len(engine.cases)})"
    assert "company" in scale_dist, "FAIL: 'company' scale not found"
    assert "individual" in scale_dist, "FAIL: 'individual' scale not found"
    assert "country" in scale_dist, "FAIL: 'country' scale not found"
    assert "other" in scale_dist, "FAIL: 'other' scale not found"
    print("  PASS")

    # --- テスト10: scaleフィールド直接参照の検証 ---
    print("\n--- テスト10: scaleフィールド直接参照 ---")
    scale_dist = engine.get_scale_distribution()
    # scaleフィールドの正確な件数を検証（main_domainキーワードマッチではなくscaleフィールド直接使用）
    expected_scales = {"company": 5510, "individual": 3217, "other": 2165, "country": 1381, "family": 787}
    for s, expected_n in expected_scales.items():
        actual_n = scale_dist.get(s, 0)
        assert actual_n == expected_n, \
            f"FAIL: scale '{s}' = {actual_n} (expected {expected_n})"
        print(f"  {s}: {actual_n} == {expected_n} OK")
    # 全scaleの合計が全件数と一致
    total = sum(scale_dist.values())
    assert total == len(engine.cases), \
        f"FAIL: scale合計({total}) != 全件数({len(engine.cases)})"
    print(f"  合計: {total} == {len(engine.cases)} OK")
    print("  PASS")

    # --- テスト11: get_conditional_distribution with scale ---
    print("\n--- テスト11: get_conditional_distribution(scale='company') ---")
    dist_company = engine.get_conditional_distribution("停滞・閉塞", "刷新・破壊", scale="company")
    dist_all = engine.get_conditional_distribution("停滞・閉塞", "刷新・破壊")
    print(f"  全件: {dist_all['total_n']}件, company: {dist_company['total_n']}件")
    assert dist_company["total_n"] <= dist_all["total_n"], \
        f"FAIL: company({dist_company['total_n']}) > all({dist_all['total_n']})"
    assert dist_company["total_n"] > 0, "FAIL: company distribution is empty"
    assert "scale" in dist_company["condition"], "FAIL: scale not in condition"
    assert dist_company["condition"]["scale"] == "company"
    # scale=None の場合はconditionにscaleが含まれないことを確認
    assert "scale" not in dist_all["condition"], "FAIL: scale should not be in condition when None"
    print("  PASS")

    # --- テスト12: search_similar_cases with scale ---
    print("\n--- テスト12: search_similar_cases(scale='individual') ---")
    cases_individual = engine.search_similar_cases(
        before_state="どん底・危機",
        action_type="耐える・潜伏",
        limit=3,
        scale="individual",
    )
    cases_all = engine.search_similar_cases(
        before_state="どん底・危機",
        action_type="耐える・潜伏",
        limit=100,
    )
    print(f"  全件: {len(cases_all)}件, individual: {len(cases_individual)}件")
    assert len(cases_individual) <= len(cases_all), "FAIL: individual > all"
    for c in cases_individual:
        print(f"  ■ {c['target_name']}（{c['period']}）")
    print("  PASS")

    # --- テスト13: get_pattern_distribution with scale ---
    print("\n--- テスト13: get_pattern_distribution(hexagram_number=21, scale='company') ---")
    pat_company = engine.get_pattern_distribution(21, scale="company")
    pat_all = engine.get_pattern_distribution(21)
    print(f"  全件: {pat_all['total_n']}件, company: {pat_company['total_n']}件")
    assert pat_company["total_n"] <= pat_all["total_n"], "FAIL: company > all"
    assert pat_company["total_n"] > 0, "FAIL: company pattern is empty"
    assert pat_company.get("scale") == "company"
    assert "scale" not in pat_all, "FAIL: scale should not be in result when None"
    print("  PASS")

    # --- テスト14: scale=None の後方互換性 ---
    print("\n--- テスト14: 後方互換性 (scale=None) ---")
    dist_compat = engine.get_conditional_distribution("停滞・閉塞", "刷新・破壊", scale=None)
    assert dist_compat["total_n"] == dist_all["total_n"], \
        f"FAIL: scale=None({dist_compat['total_n']}) != 引数なし({dist_all['total_n']})"
    print(f"  scale=None: {dist_compat['total_n']}件 == 引数なし: {dist_all['total_n']}件")
    print("  PASS")

    # --- インデックスサイズの確認 ---
    print("\n--- インデックスサイズ ---")
    print(f"  _by_hexagram: {len(engine._by_hexagram)} 卦")
    print(f"  _by_hex_yao: {len(engine._by_hex_yao)} 組")
    print(f"  _by_state_action: {len(engine._by_state_action)} 組")
    print(f"  _by_hex_pair: {len(engine._by_hex_pair)} 組")
    print(f"  _by_scale: {len(engine._by_scale)} スケール")

    print("\n" + "=" * 70)
    print("全テスト完了")
    print("=" * 70)
