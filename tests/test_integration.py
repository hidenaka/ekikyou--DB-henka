#!/usr/bin/env python3
"""
統合テスト -- 易経変化理解支援システム MVP

テスト対象:
1. モジュール間接続テスト
2. エンドツーエンドテスト (シナリオ A/B/C/D)
3. 品質ゲートテスト
4. 出力構造テスト
5. ProbabilityMapper -> FeedbackEngine パイプラインテスト
6. エッジケーステスト
7. CLI モードBの動作テスト
"""

import json
import os
import subprocess
import sys

import pytest

# --- パス設定 ---
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEST_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from feedback_engine import FeedbackEngine, FORBIDDEN_WORDS
from probability_tables import ProbabilityMapper
from case_search import CaseSearchEngine
from hexagram_transformations import (
    get_zhi_gua,
    get_hu_gua,
    get_cuo_gua,
    get_zong_gua,
    get_hexagram_name,
    get_trigrams,
    hexagram_to_lines,
)


# ============================================================
# セッションスコープの共有フィクスチャ
# ============================================================

@pytest.fixture(scope="session")
def feedback_engine():
    """FeedbackEngine をセッション全体で共有する (cases.jsonl のロードが重いため)"""
    return FeedbackEngine()


@pytest.fixture(scope="session")
def probability_mapper():
    """ProbabilityMapper をセッション全体で共有する"""
    return ProbabilityMapper()


@pytest.fixture(scope="session")
def case_search_engine():
    """CaseSearchEngine をセッション全体で共有する"""
    return CaseSearchEngine()


# ============================================================
# 1. モジュール間接続テスト
# ============================================================

class TestModuleConnection:
    """モジュール間の接続テスト"""

    def test_probability_mapper_to_feedback_engine(
        self, probability_mapper, feedback_engine
    ):
        """ProbabilityMapper.get_top_candidates() の出力を
        FeedbackEngine.generate() に渡せることを検証"""
        result = probability_mapper.get_top_candidates(
            "停滞・閉塞", "刷新・破壊"
        )
        candidates = result["candidates"]
        assert len(candidates) > 0, "候補が0件"

        # 最上位候補でフィードバック生成
        top = candidates[0]
        hex_num = top["hexagram_number"]
        # yao がない場合はデフォルト3
        yao = top.get("yao", 3)

        feedback = feedback_engine.generate(
            hex_num, yao, "停滞・閉塞", "刷新・破壊",
            mapping_confidence=top["probability"],
        )
        assert "layer1_current" in feedback
        assert "layer5_question" in feedback

    def test_case_search_in_feedback_layer4(self, feedback_engine):
        """CaseSearchEngine が FeedbackEngine._build_layer4() 内で
        正しく呼ばれ、結果を返すことを検証"""
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l4 = result["layer4_reference"]
        assert "conditional_distribution" in l4
        assert "similar_cases" in l4
        assert l4["conditional_distribution"]["total_n"] > 0

    def test_hexagram_transformations_in_layer2(self, feedback_engine):
        """hexagram_transformations が FeedbackEngine._build_layer2() で
        正しく使われていることを検証"""
        result = feedback_engine.generate(12, 3, "停滞・閉塞", "刷新・破壊", 0.72)
        l2 = result["layer2_direction"]
        expected_zhi = get_zhi_gua(12, 3)
        assert l2["resulting_hexagram"]["id"] == expected_zhi

    def test_hexagram_transformations_in_layer3(self, feedback_engine):
        """hexagram_transformations が FeedbackEngine._build_layer3() で
        正しく使われていることを検証"""
        result = feedback_engine.generate(12, 3, "停滞・閉塞", "刷新・破壊", 0.72)
        l3 = result["layer3_hidden"]
        assert l3["nuclear"]["hexagram_id"] == get_hu_gua(12)
        assert l3["complementary"]["hexagram_id"] == get_cuo_gua(12)
        assert l3["inverted"]["hexagram_id"] == get_zong_gua(12)


# ============================================================
# 2. エンドツーエンドテスト
# ============================================================

class TestEndToEnd:
    """複数シナリオでの FeedbackEngine.generate() / generate_text() 検証"""

    # --- シナリオA: 天地否(12) 第3爻 / 停滞・閉塞 x 刷新・破壊 ---

    def test_scenario_a_layer1(self, feedback_engine):
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l1 = result["layer1_current"]
        assert l1["hexagram"]["id"] == 12
        assert "否" in l1["hexagram"]["name"]

    def test_scenario_a_layer2(self, feedback_engine):
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l2 = result["layer2_direction"]
        expected_zhi = get_zhi_gua(12, 3)
        assert l2["resulting_hexagram"]["id"] == expected_zhi

    def test_scenario_a_layer3(self, feedback_engine):
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l3 = result["layer3_hidden"]
        assert l3["nuclear"]["hexagram_id"] is not None
        assert l3["complementary"]["hexagram_id"] is not None
        assert l3["inverted"]["hexagram_id"] is not None

    def test_scenario_a_layer4(self, feedback_engine):
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l4 = result["layer4_reference"]
        assert l4["conditional_distribution"]["total_n"] > 0
        assert isinstance(l4["similar_cases"], list)

    def test_scenario_a_layer5(self, feedback_engine):
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        l5 = result["layer5_question"]
        assert len(l5["question"]) > 0

    def test_scenario_a_text_output(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72, show_extended=True
        )
        assert isinstance(text, str)
        assert len(text) > 100
        assert "否" in text
        assert "問いかけ" in text

    # --- シナリオB: 乾為天(1) 第5爻 / 成長・拡大 x 攻める・挑戦 ---

    def test_scenario_b_qian_special(self, feedback_engine):
        """乾の特殊性: 互卦=乾(1), 綜卦=乾(1)"""
        result = feedback_engine.generate(
            1, 5, "成長・拡大", "攻める・挑戦", 0.85
        )
        l3 = result["layer3_hidden"]
        # 乾の互卦は乾
        assert l3["nuclear"]["hexagram_id"] == 1
        assert "乾" in l3["nuclear"]["hexagram_name"]
        # 乾の綜卦は乾
        assert l3["inverted"]["hexagram_id"] == 1
        assert "乾" in l3["inverted"]["hexagram_name"]

    def test_scenario_b_zhi_gua(self, feedback_engine):
        """乾為天(1) 第5爻変 → 火天大有(14)"""
        result = feedback_engine.generate(
            1, 5, "成長・拡大", "攻める・挑戦", 0.85
        )
        l2 = result["layer2_direction"]
        assert l2["resulting_hexagram"]["id"] == 14
        assert "大有" in l2["resulting_hexagram"]["name"]

    # --- シナリオC: 坤為地(2) 第1爻 / どん底・危機 x 耐える・潜伏 ---

    def test_scenario_c_kun_special(self, feedback_engine):
        """坤の特殊性: 互卦=坤(2), 錯卦=乾(1)"""
        result = feedback_engine.generate(
            2, 1, "どん底・危機", "耐える・潜伏", 0.60
        )
        l3 = result["layer3_hidden"]
        # 坤の互卦は坤
        assert l3["nuclear"]["hexagram_id"] == 2
        # 坤の錯卦は乾
        assert l3["complementary"]["hexagram_id"] == 1

    def test_scenario_c_zhi_gua(self, feedback_engine):
        """坤為地(2) 第1爻変 → 地雷復(24)"""
        result = feedback_engine.generate(
            2, 1, "どん底・危機", "耐える・潜伏", 0.60
        )
        l2 = result["layer2_direction"]
        assert l2["resulting_hexagram"]["id"] == 24
        assert "復" in l2["resulting_hexagram"]["name"]

    # --- シナリオD: 水火既済(63) 第6爻 / 絶頂・慢心 x 守る・維持 ---

    def test_scenario_d_jiji_special(self, feedback_engine):
        """既済の特殊性: 互卦=未済(64), 綜卦=未済(64), 錯卦=未済(64)"""
        result = feedback_engine.generate(
            63, 6, "絶頂・慢心", "守る・維持", 0.78
        )
        l3 = result["layer3_hidden"]
        # 水火既済の互卦 = 火水未済
        assert l3["nuclear"]["hexagram_id"] == 64
        # 水火既済の綜卦 = 火水未済
        assert l3["inverted"]["hexagram_id"] == 64
        # 水火既済の錯卦 = 火水未済
        assert l3["complementary"]["hexagram_id"] == 64

    def test_scenario_d_text_output(self, feedback_engine):
        text = feedback_engine.generate_text(
            63, 6, "絶頂・慢心", "守る・維持", 0.78, show_extended=True
        )
        assert "既済" in text
        assert "未済" in text  # 互卦/綜卦/錯卦全て未済


# ============================================================
# 3. 品質ゲートテスト
# ============================================================

class TestQualityGates:
    """品質ゲート Q1-Q6 の検証"""

    SCENARIOS = [
        (12, 3, "停滞・閉塞", "刷新・破壊", 0.72),
        (1, 5, "成長・拡大", "攻める・挑戦", 0.85),
        (2, 1, "どん底・危機", "耐える・潜伏", 0.60),
        (63, 6, "絶頂・慢心", "守る・維持", 0.78),
    ]

    @pytest.mark.parametrize(
        "hex_num,yao,state,action,conf", SCENARIOS
    )
    def test_no_critical_quality_warnings(
        self, feedback_engine, hex_num, yao, state, action, conf
    ):
        """全シナリオで品質ゲート違反(Q1,Q3,Q4,Q6)がないことを検証"""
        result = feedback_engine.generate(hex_num, yao, state, action, conf)
        warnings = result.get("quality_warnings", [])
        critical = [w for w in warnings if "違反" in w]
        assert critical == [], f"品質ゲート違反: {critical}"

    # 否定文脈で使用される禁止語を含む許容フレーズ
    ALLOWED_PHRASES = [
        "予測するものではありません",
        "「必ずこうなる」という意味ではなく",
    ]

    @pytest.mark.parametrize(
        "hex_num,yao,state,action,conf", SCENARIOS
    )
    def test_forbidden_words_not_in_text(
        self, feedback_engine, hex_num, yao, state, action, conf
    ):
        """テキスト出力に禁止語が含まれないことを検証
        (否定文脈での使用は許容: 「予測するものではありません」「必ずこうなる」という意味ではなく)"""
        text = feedback_engine.generate_text(
            hex_num, yao, state, action, conf, show_extended=True
        )
        for word in FORBIDDEN_WORDS:
            if word in text:
                # 許容フレーズを除去した上でチェック
                cleaned = text
                for phrase in self.ALLOWED_PHRASES:
                    cleaned = cleaned.replace(phrase, "")
                assert word not in cleaned, (
                    f"禁止語「{word}」がテキストに含まれています"
                )

    @pytest.mark.parametrize(
        "hex_num,yao,state,action,conf", SCENARIOS
    )
    def test_n_count_displayed(
        self, feedback_engine, hex_num, yao, state, action, conf
    ):
        """事例件数(n)がJSON出力に併記されていること"""
        result = feedback_engine.generate(hex_num, yao, state, action, conf)
        total_n = result["layer4_reference"]["conditional_distribution"]["total_n"]
        assert isinstance(total_n, int)
        assert total_n >= 0

    @pytest.mark.parametrize(
        "hex_num,yao,state,action,conf", SCENARIOS
    )
    def test_question_exists(
        self, feedback_engine, hex_num, yao, state, action, conf
    ):
        """問いかけ(LAYER 5)が必ず存在すること"""
        result = feedback_engine.generate(hex_num, yao, state, action, conf)
        question = result["layer5_question"]["question"]
        assert len(question) > 0, "LAYER 5 の問いが空です"

    @pytest.mark.parametrize(
        "hex_num,yao,state,action,conf", SCENARIOS
    )
    def test_mapping_confidence_in_output(
        self, feedback_engine, hex_num, yao, state, action, conf
    ):
        """mapping_confidence が出力に含まれること"""
        result = feedback_engine.generate(hex_num, yao, state, action, conf)
        assert "mapping_confidence" in result
        assert result["mapping_confidence"] == conf


# ============================================================
# 4. 出力構造テスト
# ============================================================

class TestOutputStructure:
    """JSON出力のスキーマ検証"""

    @pytest.fixture
    def sample_output(self, feedback_engine):
        return feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )

    def test_top_level_keys(self, sample_output):
        """トップレベルキーの存在確認"""
        required_keys = [
            "version", "generated_at", "mapping_confidence",
            "layer1_current", "layer2_direction", "layer3_hidden",
            "layer4_reference", "layer5_question",
        ]
        for key in required_keys:
            assert key in sample_output, f"トップレベルキー '{key}' が欠落"

    def test_layer1_structure(self, sample_output):
        """layer1_current の構造検証"""
        l1 = sample_output["layer1_current"]
        assert "hexagram" in l1
        assert "judgment_modern_ja" in l1
        assert "changing_line" in l1
        assert "phase_model" in l1

        hex_info = l1["hexagram"]
        assert "id" in hex_info
        assert "name" in hex_info
        assert "upper_trigram" in hex_info
        assert "lower_trigram" in hex_info
        assert "lines" in hex_info
        assert "visual" in hex_info

        cl = l1["changing_line"]
        assert "position" in cl
        assert "phase" in cl
        assert "phase_description" in cl
        assert "yao_text_modern_ja" in cl

    def test_layer2_structure(self, sample_output):
        """layer2_direction の構造検証"""
        l2 = sample_output["layer2_direction"]
        assert "resulting_hexagram" in l2
        assert "transformation" in l2
        assert "structural_reading" in l2

        rh = l2["resulting_hexagram"]
        assert "id" in rh
        assert "name" in rh
        assert "upper_trigram" in rh
        assert "lower_trigram" in rh
        assert "lines" in rh

    def test_layer3_structure(self, sample_output):
        """layer3_hidden の構造検証"""
        l3 = sample_output["layer3_hidden"]
        assert "depth" in l3
        assert "nuclear" in l3
        assert "complementary" in l3
        assert "inverted" in l3

        for section in ["nuclear", "complementary", "inverted"]:
            sub = l3[section]
            assert "hexagram_id" in sub
            assert "hexagram_name" in sub
            assert "reading" in sub
            assert "judgment_modern_ja" in sub

    def test_layer4_structure(self, sample_output):
        """layer4_reference の構造検証"""
        l4 = sample_output["layer4_reference"]
        assert "conditional_distribution" in l4
        assert "similar_cases" in l4

        dist = l4["conditional_distribution"]
        assert "condition" in dist
        assert "total_n" in dist
        assert "distribution" in dist
        assert isinstance(dist["distribution"], list)

    def test_layer5_structure(self, sample_output):
        """layer5_question の構造検証"""
        l5 = sample_output["layer5_question"]
        assert "question" in l5
        assert "generation_basis" in l5


# ============================================================
# 5. ProbabilityMapper -> FeedbackEngine パイプラインテスト
# ============================================================

class TestProbabilityMapperPipeline:
    """ProbabilityMapper で複数候補取得 → 各候補で FeedbackEngine 実行"""

    INPUT_COMBOS = [
        ("停滞・閉塞", "刷新・破壊"),
        ("どん底・危機", "耐える・潜伏"),
        ("成長・拡大", "攻める・挑戦"),
        ("安定・平和", "守る・維持"),
    ]

    @pytest.mark.parametrize("state,action", INPUT_COMBOS)
    def test_pipeline_all_candidates_succeed(
        self, probability_mapper, feedback_engine, state, action
    ):
        """ProbabilityMapper の全候補で FeedbackEngine が正常動作する"""
        result = probability_mapper.get_top_candidates(state, action, n=3)
        candidates = result["candidates"]
        assert len(candidates) > 0

        for c in candidates:
            hex_num = c["hexagram_number"]
            yao = c.get("yao", 3)
            feedback = feedback_engine.generate(
                hex_num, yao, state, action,
                mapping_confidence=c["probability"],
            )
            assert "layer1_current" in feedback
            assert "layer5_question" in feedback
            assert feedback["mapping_confidence"] == c["probability"]

    def test_pipeline_with_full_params(
        self, probability_mapper, feedback_engine
    ):
        """フルパラメータ指定時のパイプライン動作"""
        result = probability_mapper.get_top_candidates(
            current_state="どん底・危機",
            intended_action="耐える・潜伏",
            trigger_nature="外部ショック",
            phase_stage="危機・転換点",
            energy_direction="contracting",
            n=5,
        )
        candidates = result["candidates"]
        assert len(candidates) == 5

        for c in candidates:
            assert "yao" in c, "phase_stage 指定時は yao が必須"
            assert "zhi_gua_number" in c

            feedback = feedback_engine.generate(
                c["hexagram_number"],
                c["yao"],
                "どん底・危機",
                "耐える・潜伏",
                mapping_confidence=c["probability"],
            )
            assert feedback["layer2_direction"]["resulting_hexagram"]["id"] > 0


# ============================================================
# 6. エッジケーステスト
# ============================================================

class TestEdgeCases:
    """エッジケースの検証"""

    def test_rare_state_action_combo(self, feedback_engine):
        """事例数が少ない before_state x action_type の組み合わせ"""
        # 存在しない組み合わせでも動作すること
        result = feedback_engine.generate(
            12, 3, "存在しない状態", "存在しない行動", 0.50
        )
        assert result["layer4_reference"]["conditional_distribution"]["total_n"] == 0
        assert result["layer4_reference"]["similar_cases"] is not None

    @pytest.mark.parametrize("hex_num", list(range(1, 65)))
    def test_all_64_hexagrams(self, feedback_engine, hex_num):
        """全64卦に対して FeedbackEngine.generate() が正常に動作するか"""
        result = feedback_engine.generate(
            hex_num, 3, "停滞・閉塞", "刷新・破壊", 0.70
        )
        assert result["layer1_current"]["hexagram"]["id"] == hex_num
        name = result["layer1_current"]["hexagram"]["name"]
        assert len(name) > 0
        # 之卦が有効な範囲
        zhi_id = result["layer2_direction"]["resulting_hexagram"]["id"]
        assert 1 <= zhi_id <= 64

    @pytest.mark.parametrize("yao", [1, 2, 3, 4, 5, 6])
    def test_all_yao_positions(self, feedback_engine, yao):
        """yao_position 1-6 の全パターン"""
        result = feedback_engine.generate(
            12, yao, "停滞・閉塞", "刷新・破壊", 0.70
        )
        assert result["layer1_current"]["changing_line"]["position"] == yao
        # 之卦が爻ごとに異なることの確認
        zhi_id = result["layer2_direction"]["resulting_hexagram"]["id"]
        assert 1 <= zhi_id <= 64

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_mapping_confidence_boundaries(self, feedback_engine, conf):
        """mapping_confidence 0.0, 0.5, 1.0 の境界値"""
        result = feedback_engine.generate(
            12, 3, "停滞・閉塞", "刷新・破壊", conf
        )
        assert result["mapping_confidence"] == conf

    def test_text_output_basic_and_extended(self, feedback_engine):
        """show_extended=False/True の切替が正常に動作する"""
        basic = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.70, show_extended=False
        )
        extended = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.70, show_extended=True
        )
        # extended は basic より長い (互卦/錯卦/綜卦/事例の展開分)
        assert len(extended) > len(basic)
        # extended には互卦・錯卦・綜卦のセクションがある
        assert "互卦" in extended
        assert "錯卦" in extended
        assert "綜卦" in extended

    def test_zhi_gua_different_for_each_yao(self, feedback_engine):
        """同一卦(乾為天)の6爻でそれぞれ異なる之卦が生成される"""
        zhi_ids = set()
        for yao in range(1, 7):
            result = feedback_engine.generate(
                1, yao, "成長・拡大", "攻める・挑戦", 0.80
            )
            zhi_ids.add(
                result["layer2_direction"]["resulting_hexagram"]["id"]
            )
        # 乾為天の6爻は全て異なる之卦を生む
        assert len(zhi_ids) == 6


# ============================================================
# 7. CLI モードBの動作テスト
# ============================================================

class TestCLIModeB:
    """CLI の引数指定モード (モードB) テスト"""

    CLI_SCRIPT = os.path.join(_SCRIPTS_DIR, "iching_cli.py")

    def _run_cli(self, args: list, timeout: int = 60) -> subprocess.CompletedProcess:
        """CLI を実行して結果を返す"""
        cmd = [sys.executable, self.CLI_SCRIPT] + args
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=_PROJECT_ROOT,
        )

    def test_basic_invocation_exit_0(self):
        """基本的な呼び出しが exit code 0 で終了する"""
        proc = self._run_cli([
            "-H", "12", "-y", "3",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
        ])
        assert proc.returncode == 0, (
            f"exit code={proc.returncode}\nstderr={proc.stderr}"
        )
        assert len(proc.stdout) > 0

    def test_json_flag_valid_json(self):
        """--json フラグで有効な JSON が出力される"""
        proc = self._run_cli([
            "-H", "12", "-y", "3",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
            "--json",
        ])
        assert proc.returncode == 0, f"stderr={proc.stderr}"
        data = json.loads(proc.stdout)
        assert "layer1_current" in data
        assert "layer5_question" in data

    def test_extended_flag_includes_layer3_4(self):
        """--extended フラグで LAYER 3-4 が含まれる"""
        proc = self._run_cli([
            "-H", "12", "-y", "3",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
            "--extended",
        ])
        assert proc.returncode == 0
        output = proc.stdout
        assert "互卦" in output
        assert "錯卦" in output
        assert "綜卦" in output

    def test_invalid_hexagram_nonzero_exit(self):
        """不正な卦番号で exit code != 0"""
        proc = self._run_cli([
            "-H", "99", "-y", "3",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
        ])
        assert proc.returncode != 0

    def test_invalid_yao_nonzero_exit(self):
        """不正な爻位置で exit code != 0"""
        proc = self._run_cli([
            "-H", "12", "-y", "0",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
        ])
        assert proc.returncode != 0

    def test_missing_yao_nonzero_exit(self):
        """--yao なしで --hexagram のみは exit code != 0"""
        proc = self._run_cli(["-H", "12"])
        assert proc.returncode != 0

    def test_json_output_schema(self):
        """JSON出力のスキーマが仕様に合致"""
        proc = self._run_cli([
            "-H", "1", "-y", "5",
            "-s", "成長・拡大", "-a", "攻める・挑戦",
            "--json",
        ])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)

        # layer1
        assert "hexagram" in data["layer1_current"]
        assert "judgment_modern_ja" in data["layer1_current"]
        assert "changing_line" in data["layer1_current"]
        assert "phase_model" in data["layer1_current"]

        # layer2
        assert "resulting_hexagram" in data["layer2_direction"]
        assert "transformation" in data["layer2_direction"]
        assert "structural_reading" in data["layer2_direction"]

        # layer3
        assert "depth" in data["layer3_hidden"]
        assert "nuclear" in data["layer3_hidden"]
        assert "complementary" in data["layer3_hidden"]
        assert "inverted" in data["layer3_hidden"]

        # layer4
        assert "conditional_distribution" in data["layer4_reference"]
        assert "similar_cases" in data["layer4_reference"]

        # layer5
        assert "question" in data["layer5_question"]
        assert "generation_basis" in data["layer5_question"]

    def test_confidence_flag(self):
        """--confidence フラグが反映される"""
        proc = self._run_cli([
            "-H", "12", "-y", "3",
            "-s", "停滞・閉塞", "-a", "刷新・破壊",
            "--json", "-c", "0.95",
        ])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["mapping_confidence"] == 0.95

    def test_empty_state_and_action(self):
        """state/action 未指定でも動作する (空文字列)"""
        proc = self._run_cli(["-H", "12", "-y", "3", "--json"])
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        assert data["layer1_current"]["hexagram"]["id"] == 12


# ============================================================
# 8. テキスト出力内容検証 (追加)
# ============================================================

class TestTextOutputContent:
    """テキスト出力の内容面を検証"""

    def test_text_contains_hexagram_name(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        assert "天地否" in text

    def test_text_contains_confidence(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        assert "72%" in text

    def test_text_contains_question(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        # 第3爻の問い
        assert "ですか" in text or "？" in text

    def test_text_contains_phase_info(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72
        )
        assert "転換" in text  # 第3爻 = 転換フェーズ

    def test_extended_text_contains_case_count(self, feedback_engine):
        text = feedback_engine.generate_text(
            12, 3, "停滞・閉塞", "刷新・破壊", 0.72, show_extended=True
        )
        assert "件" in text  # 事例件数が表示されている


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
