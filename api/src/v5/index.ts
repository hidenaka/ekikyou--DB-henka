/**
 * HaQei診断システム v5 メインエントリポイント
 */

// 型定義
export * from './types';

// 変換関数
export {
  convertToProfile,
  convertChangeNature,
  convertAgency,
  convertTimeframe,
  convertRelationship,
  convertEmotionalTone,
  calculateEntropy,
  maxEntropy,
} from './convert';

// 距離計算
export { jsDistance, klDivergence } from './distance';

// マッチング
export {
  calculateScore,
  generateRanking,
  getTopCandidates,
} from './matching';

// 説明文生成
export {
  generateMatchReasons,
  generateConfidenceExplanation,
  generateSimilarityExplanation,
  getAxisLabel,
} from './explanation';

// ルーブリック管理
export {
  loadRubricFromFile,
  setRubric,
  getRubric,
  clearRubricCache,
  getClassProfiles,
  getClassProfileById,
  getClassProfileByHexagramYao,
  getRubricVersion,
} from './rubric';

// ============================================================
// 診断フロー統合関数
// ============================================================

import { UserAnswers, MatchingResult, CandidateScore, DiagnoseResponse } from './types';
import { convertToProfile } from './convert';
import { generateRanking, getTopCandidates } from './matching';
import { getClassProfiles } from './rubric';
import { generateConfidenceExplanation, generateSimilarityExplanation } from './explanation';

/**
 * 診断を実行
 */
export function diagnose(answers: UserAnswers): MatchingResult {
  // 1. 回答を確率分布に変換
  const userProfile = convertToProfile(answers);

  // 2. 384クラスとマッチング
  const classProfiles = getClassProfiles();
  const result = generateRanking(userProfile, classProfiles);

  return result;
}

/**
 * API用レスポンスを生成
 */
export function createDiagnoseResponse(
  result: MatchingResult,
  topN: number = 5
): DiagnoseResponse {
  const topCandidates = getTopCandidates(result, topN);

  return {
    resultId: generateResultId(),
    topCandidates,
    missingAxes: result.missingAxes,
    overallConfidence: result.overallConfidence,
    version: result.version,
  };
}

/**
 * 結果IDを生成
 */
function generateResultId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `v5-${timestamp}-${random}`;
}

/**
 * 結果の要約を生成
 */
export function createResultSummary(
  result: MatchingResult
): {
  topCandidate: CandidateScore;
  confidenceExplanation: string;
  similarityExplanation: string | null;
} {
  const topCandidates = getTopCandidates(result, 5);
  const topCandidate = topCandidates[0];

  return {
    topCandidate,
    confidenceExplanation: generateConfidenceExplanation(
      result.overallConfidence,
      result.missingAxes
    ),
    similarityExplanation: generateSimilarityExplanation(topCandidates),
  };
}
