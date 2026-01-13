/**
 * HaQei診断システム v5 マッチングエンジン
 */

import {
  UserProfile,
  ClassProfile,
  CandidateScore,
  MatchingResult,
  AXIS_NAMES,
  AxisName,
  VERSION,
} from './types';
import { maxEntropy } from './convert';
import { jsDistance } from './distance';
import { generateMatchReasons } from './explanation';

/**
 * 単一クラスのスコア計算
 */
export function calculateScore(
  userProfile: UserProfile,
  classProfile: ClassProfile
): CandidateScore {
  // 各軸の重み（欠損軸は0）
  const weights: Record<AxisName, number> = {
    changeNature: 0,
    agency: 0,
    timeframe: 0,
    relationship: 0,
    emotionalTone: 0,
  };
  let totalWeight = 0;

  for (const axis of AXIS_NAMES) {
    if (!userProfile[axis].isMissing) {
      weights[axis] = 1;
      totalWeight += 1;
    }
  }

  // 正規化
  if (totalWeight > 0) {
    for (const axis of AXIS_NAMES) {
      weights[axis] /= totalWeight;
    }
  }

  // 各軸のJS距離を計算
  const contributions: CandidateScore['contributions'] = {
    changeNature: 0,
    agency: 0,
    timeframe: 0,
    relationship: 0,
    emotionalTone: 0,
  };
  let totalScore = 0;

  for (const axis of AXIS_NAMES) {
    if (weights[axis] > 0) {
      const dist = jsDistance(
        userProfile[axis].values,
        classProfile.distributions[axis]
      );
      contributions[axis] = dist * weights[axis];
      totalScore += contributions[axis];
    }
  }

  // 混合状態の判定（高エントロピー軸が2つ以上）
  const highEntropyCount = AXIS_NAMES.filter((axis) => {
    const categoryCount = Object.keys(userProfile[axis].values).length;
    const maxEnt = maxEntropy(categoryCount);
    return userProfile[axis].entropy > maxEnt * 0.7;
  }).length;

  const isMixedState = highEntropyCount >= 2;

  return {
    classId: classProfile.classId,
    hexagram: classProfile.hexagram,
    yao: classProfile.yao,
    name: classProfile.name,
    hexagramName: classProfile.hexagramName,
    yaoName: classProfile.yaoName,
    yaoStage: classProfile.yaoStage,
    score: totalScore,
    rank: 0, // 後で設定
    contributions,
    matchReasons: [], // 後で生成
    isMixedState,
  };
}

/**
 * 384クラスのランキング生成
 */
export function generateRanking(
  userProfile: UserProfile,
  classProfiles: ClassProfile[]
): MatchingResult {
  // 全384クラスのスコア計算
  const candidates = classProfiles.map((cp) => calculateScore(userProfile, cp));

  // スコア昇順でソート（0=完全一致が最良）
  candidates.sort((a, b) => a.score - b.score);

  // 順位付け（同点処理）
  let currentRank = 1;
  let previousScore = -1;

  for (let i = 0; i < candidates.length; i++) {
    if (candidates[i].score !== previousScore) {
      currentRank = i + 1;
    }
    candidates[i].rank = currentRank;
    previousScore = candidates[i].score;
  }

  // 上位10件の説明文生成
  for (const candidate of candidates.slice(0, 10)) {
    candidate.matchReasons = generateMatchReasons(userProfile, candidate);
  }

  // 欠損軸の収集
  const missingAxes = AXIS_NAMES.filter(
    (axis) => userProfile[axis].isMissing
  );

  // 全体の測定信頼性（欠損軸があると低下）
  const overallConfidence = 1 - missingAxes.length / 5;

  return {
    ranking: candidates,
    missingAxes,
    overallConfidence,
    timestamp: new Date().toISOString(),
    version: VERSION,
  };
}

/**
 * 上位N件を取得
 */
export function getTopCandidates(
  result: MatchingResult,
  n: number = 5
): CandidateScore[] {
  return result.ranking.slice(0, n);
}
