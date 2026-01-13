/**
 * HaQei診断システム v5 型定義
 */

// ============================================================
// ユーザー入力
// ============================================================

export interface UserAnswers {
  changeNature: {
    expansion: number;      // 拡大 (1-5)
    contraction: number;    // 収縮 (1-5)
    maintenance: number;    // 維持 (1-5)
    transformation: number; // 転換 (1-5)
  };
  agency: number;           // 1-5 (1=全くできない, 5=完全にできる)
  timeframe: TimeframeOption;
  relationship: {
    self: boolean;
    family: boolean;
    team: boolean;
    organization: boolean;
    external: boolean;
    society: boolean;
  };
  emotionalTone: {
    excitement: number;     // ワクワク (1-5)
    caution: number;        // 慎重 (1-5)
    anxiety: number;        // 不安 (1-5)
    optimism: number;       // 楽観 (1-5)
  };
}

export type TimeframeOption =
  | 'immediate'   // 1ヶ月以内
  | 'shortTerm'   // 3ヶ月以内
  | 'midTerm'     // 半年以内
  | 'longTerm'    // 1年以上
  | 'unknown';    // わからない

// ============================================================
// 確率分布
// ============================================================

export interface AxisDistribution {
  values: Record<string, number>;  // 確率分布（合計1.0）
  entropy: number;                  // エントロピー値
  isMissing: boolean;              // 欠損フラグ
}

export interface UserProfile {
  changeNature: AxisDistribution;
  agency: AxisDistribution;
  timeframe: AxisDistribution;
  relationship: AxisDistribution;
  emotionalTone: AxisDistribution;
}

// ============================================================
// クラスプロファイル（ルーブリック）
// ============================================================

export interface ClassProfile {
  classId: number;
  hexagram: number;
  yao: number;
  name: string;
  hexagramName: string;
  yaoName: string;
  yaoStage: string;
  distributions: {
    changeNature: Record<string, number>;
    agency: Record<string, number>;
    timeframe: Record<string, number>;
    relationship: Record<string, number>;
    emotionalTone: Record<string, number>;
  };
  rubricVersion: string;
  rubricSource: string;
}

export interface Rubric {
  version: string;
  createdAt: string;
  description: string;
  axisRules: Record<string, AxisRule>;
  classProfiles: ClassProfile[];
  metadata: {
    totalClasses: number;
    hexagramCount: number;
    yaoPerHexagram: number;
    generationMethod: string;
    validationStatus: string;
  };
}

export interface AxisRule {
  description: string;
  categories: Record<string, {
    definition: string;
    iChingKeywords: string[];
    exampleHexagrams: number[];
  }>;
}

// ============================================================
// マッチング結果
// ============================================================

export interface CandidateScore {
  classId: number;
  hexagram: number;
  yao: number;
  name: string;
  hexagramName: string;
  yaoName: string;
  yaoStage: string;
  score: number;              // JS距離（0=完全一致, 1=最大乖離）
  rank: number;
  contributions: {
    changeNature: number;
    agency: number;
    timeframe: number;
    relationship: number;
    emotionalTone: number;
  };
  matchReasons: string[];
  isMixedState: boolean;
}

export interface MatchingResult {
  ranking: CandidateScore[];
  missingAxes: string[];
  overallConfidence: number;
  timestamp: string;
  version: string;
}

// ============================================================
// API レスポンス
// ============================================================

export interface DiagnoseRequest {
  answers: UserAnswers;
  sessionId?: string;
}

export interface DiagnoseResponse {
  resultId: string;
  topCandidates: CandidateScore[];
  missingAxes: string[];
  overallConfidence: number;
  version: string;
}

export interface FeedbackRequest {
  resultId: string;
  selfRating: number;         // 1-5
  selectedClassId?: number;
}

// ============================================================
// 定数
// ============================================================

export const AXIS_NAMES = [
  'changeNature',
  'agency',
  'timeframe',
  'relationship',
  'emotionalTone'
] as const;

export type AxisName = typeof AXIS_NAMES[number];

export const CHANGE_NATURE_CATEGORIES = ['拡大', '収縮', '維持', '転換'] as const;
export const AGENCY_CATEGORIES = ['自ら動く', '受け止める', '待つ'] as const;
export const TIMEFRAME_CATEGORIES = ['即時', '短期', '中期', '長期'] as const;
export const RELATIONSHIP_CATEGORIES = ['個人', '組織内', '対外'] as const;
export const EMOTIONAL_TONE_CATEGORIES = ['前向き', '慎重', '不安', '楽観'] as const;

export const VERSION = 'v5.0.0';
