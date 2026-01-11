from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Scale(str, Enum):
    company = "company"
    individual = "individual"
    family = "family"
    country = "country"
    other = "other"

class BeforeState(str, Enum):
    ZECHOUS = "絶頂・慢心"
    TEITAI = "停滞・閉塞"
    KONRAN = "混乱・カオス"
    SEICHOUTSU = "成長痛"
    DONZOKO = "どん底・危機"
    ANTEI = "安定・平和"
    SUCCESS_V = "V字回復・大成功"
    SHRINK_STABLE = "縮小安定・生存"

class TriggerType(str, Enum):
    EXTERNAL = "外部ショック"
    INTERNAL = "内部崩壊"
    INTENTIONAL = "意図的決断"
    RANDOM = "偶発・出会い"

class ActionType(str, Enum):
    ATTACK = "攻める・挑戦"
    DEFEND = "守る・維持"
    ABANDON = "捨てる・撤退"
    ENDURE = "耐える・潜伏"
    DIALOG = "対話・融合"
    RENEWAL = "刷新・破壊"
    ESCAPE = "逃げる・放置"
    SPINOFF = "分散・スピンオフ"

class AfterState(str, Enum):
    SUCCESS_V = "V字回復・大成功"
    SHRINK_STABLE = "縮小安定・生存"
    TRANSFORM = "変質・新生"
    MAINTAIN = "現状維持・延命"
    CONFUSE = "迷走・混乱"
    CHAOS = "混乱・カオス"
    COLLAPSE = "崩壊・消滅"
    TEITAI = "停滞・閉塞"
    DONZOKO = "どん底・危機"
    # 長期的な持続成長パターンの成果
    SUSTAINED_GROWTH_SUCCESS = "持続成長・大成功"
    STABLE_GROWTH_SUCCESS = "安定成長・成功"
    ANTEI = "安定・平和"

class Hex(str, Enum):
    QIAN = "乾"
    KUN = "坤"
    ZHEN = "震"
    XUN = "巽"
    KAN = "坎"
    LI = "離"
    GEN = "艮"
    DUI = "兌"

class PatternType(str, Enum):
    SHOCK_RECOVERY = "Shock_Recovery"
    HUBRIS_COLLAPSE = "Hubris_Collapse"
    PIVOT_SUCCESS = "Pivot_Success"
    ENDURANCE = "Endurance"
    SLOW_DECLINE = "Slow_Decline"
    # 長期的な持続成長パターン
    STEADY_GROWTH = "Steady_Growth"
    # 既存追加パターン
    CRISIS_PIVOT = "Crisis_Pivot"
    BREAKTHROUGH = "Breakthrough"
    EXPLORATION = "Exploration"
    MANAGED_DECLINE = "Managed_Decline"
    DECLINE = "Decline"
    # バイアス補正用の新パターン
    QUIET_FADE = "Quiet_Fade"          # 静かに消滅（話題にならず消えた）
    STAGNATION = "Stagnation"          # 停滞のまま終了（何も起きなかった）
    FAILED_ATTEMPT = "Failed_Attempt"  # 挑戦したが失敗（普通の失敗）

class Outcome(str, Enum):
    SUCCESS = "Success"
    PARTIAL = "PartialSuccess"
    FAILURE = "Failure"
    MIXED = "Mixed"

class SourceType(str, Enum):
    OFFICIAL = "official"
    NEWS = "news"
    BOOK = "book"
    BLOG = "blog"
    SNS = "sns"
    ARTICLE = "article"

class Credibility(str, Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"

class Case(BaseModel):
    transition_id: Optional[str] = None
    target_name: str
    scale: Scale
    period: str
    story_summary: str
    before_state: BeforeState
    trigger_type: TriggerType
    action_type: ActionType
    after_state: AfterState
    before_hex: Hex
    trigger_hex: Hex
    action_hex: Hex
    after_hex: Hex
    pattern_type: PatternType
    outcome: Outcome
    free_tags: List[str] = Field(default_factory=list)
    source_type: SourceType
    credibility_rank: Credibility
    classical_before_hexagram: Optional[str] = None
    classical_action_hexagram: Optional[str] = None
    classical_after_hexagram: Optional[str] = None
    logic_memo: Optional[str] = None

    # 384爻（変爻）情報 - 各変化でどの爻が変化したか (1-6)
    # 1=初爻, 2=二爻, 3=三爻, 4=四爻, 5=五爻, 6=上爻
    changing_lines_1: Optional[List[int]] = None  # before_hex → trigger_hex での変爻
    changing_lines_2: Optional[List[int]] = None  # trigger_hex → action_hex での変爻
    changing_lines_3: Optional[List[int]] = None  # action_hex → after_hex での変爻

    # 拡張フィールド
    main_domain: Optional[str] = None  # 分野（例：IT・通信、自動車、金融等）
    country: Optional[str] = None  # 国名
    sources: Optional[List[str]] = None  # ソースURL等
    hexagram_id: Optional[int] = None  # 卦番号（1-64）
    hexagram_name: Optional[str] = None  # 卦名（例：火沢睽）
    yao_context: Optional[str] = None  # 卦との関連説明
    yao_analysis: Optional[dict] = None  # 爻分析（自動生成）

    @field_validator("target_name", "period", "story_summary")
    def not_empty(cls, v: str):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("changing_lines_1", "changing_lines_2", "changing_lines_3")
    def validate_changing_lines(cls, v: Optional[List[int]]):
        """変爻は1-6の範囲内でなければならない"""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("changing_lines must be a list or None")
            for line in v:
                if not isinstance(line, int) or line < 1 or line > 6:
                    raise ValueError("changing lines must be integers between 1 and 6")
            # 重複チェック
            if len(v) != len(set(v)):
                raise ValueError("changing lines must not contain duplicates")
        return v
