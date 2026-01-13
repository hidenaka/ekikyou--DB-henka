/**
 * HaQei診断システム v5 回答→確率分布変換
 */

import { UserAnswers, AxisDistribution, UserProfile } from './types';

/**
 * エントロピー計算
 */
export function calculateEntropy(probabilities: number[]): number {
  let entropy = 0;
  for (const p of probabilities) {
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

/**
 * 最大エントロピー（均等分布の場合）
 */
export function maxEntropy(n: number): number {
  return Math.log2(n);
}

/**
 * changeNature（変化の性質）を確率分布に変換
 */
export function convertChangeNature(
  answers: UserAnswers['changeNature']
): AxisDistribution {
  const { expansion, contraction, maintenance, transformation } = answers;
  const total = expansion + contraction + maintenance + transformation;

  // ゼロ除算防止
  if (total === 0) {
    return {
      values: { 拡大: 0.25, 収縮: 0.25, 維持: 0.25, 転換: 0.25 },
      entropy: 2.0,
      isMissing: false,
    };
  }

  const values = {
    拡大: expansion / total,
    収縮: contraction / total,
    維持: maintenance / total,
    転換: transformation / total,
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false,
  };
}

/**
 * agency（主体性）を確率分布に変換
 */
export function convertAgency(score: number): AxisDistribution {
  const values = {
    自ら動く: 0,
    受け止める: 0,
    待つ: 0,
  };

  if (score <= 2) {
    values['待つ'] = (3 - score) / 2;
    values['受け止める'] = 1 - values['待つ'];
  } else if (score >= 4) {
    values['自ら動く'] = (score - 3) / 2;
    values['受け止める'] = 1 - values['自ら動く'];
  } else {
    // score = 3
    values['受け止める'] = 0.6;
    values['自ら動く'] = 0.2;
    values['待つ'] = 0.2;
  }

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false,
  };
}

/**
 * timeframe（時間軸）を確率分布に変換
 */
export function convertTimeframe(
  selection: UserAnswers['timeframe']
): AxisDistribution {
  // 「わからない」は欠損扱い
  if (selection === 'unknown') {
    return {
      values: { 即時: 0.25, 短期: 0.25, 中期: 0.25, 長期: 0.25 },
      entropy: 2.0,
      isMissing: true,
    };
  }

  const mapping: Record<string, Record<string, number>> = {
    immediate: { 即時: 0.7, 短期: 0.2, 中期: 0.1, 長期: 0 },
    shortTerm: { 即時: 0.2, 短期: 0.6, 中期: 0.15, 長期: 0.05 },
    midTerm: { 即時: 0.05, 短期: 0.2, 中期: 0.55, 長期: 0.2 },
    longTerm: { 即時: 0, 短期: 0.1, 中期: 0.2, 長期: 0.7 },
  };

  const values = mapping[selection];

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false,
  };
}

/**
 * relationship（関係性）を確率分布に変換
 */
export function convertRelationship(
  selections: UserAnswers['relationship']
): AxisDistribution {
  const { self, family, team, organization, external, society } = selections;

  let personal = 0;
  let organizational = 0;
  let externalScore = 0;

  if (self) personal += 1.0;
  if (family) personal += 0.8;
  if (team) organizational += 0.6;
  if (organization) organizational += 1.0;
  if (external) externalScore += 0.8;
  if (society) externalScore += 1.0;

  const total = personal + organizational + externalScore;

  // 何も選択されていない場合
  if (total === 0) {
    return {
      values: { 個人: 0.5, 組織内: 0.3, 対外: 0.2 },
      entropy: 1.5,
      isMissing: false,
    };
  }

  const values = {
    個人: personal / total,
    組織内: organizational / total,
    対外: externalScore / total,
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false,
  };
}

/**
 * emotionalTone（感情基調）を確率分布に変換
 */
export function convertEmotionalTone(
  answers: UserAnswers['emotionalTone']
): AxisDistribution {
  const { excitement, caution, anxiety, optimism } = answers;
  const total = excitement + caution + anxiety + optimism;

  if (total === 0) {
    return {
      values: { 前向き: 0.25, 慎重: 0.25, 不安: 0.25, 楽観: 0.25 },
      entropy: 2.0,
      isMissing: false,
    };
  }

  const values = {
    前向き: excitement / total,
    慎重: caution / total,
    不安: anxiety / total,
    楽観: optimism / total,
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false,
  };
}

/**
 * 全回答をUserProfileに変換
 */
export function convertToProfile(answers: UserAnswers): UserProfile {
  return {
    changeNature: convertChangeNature(answers.changeNature),
    agency: convertAgency(answers.agency),
    timeframe: convertTimeframe(answers.timeframe),
    relationship: convertRelationship(answers.relationship),
    emotionalTone: convertEmotionalTone(answers.emotionalTone),
  };
}
