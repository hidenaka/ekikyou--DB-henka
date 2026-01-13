/**
 * HaQei診断システム v5 説明文生成
 */

import { UserProfile, CandidateScore, AxisName } from './types';

/**
 * 軸名を日本語ラベルに変換
 */
export function getAxisLabel(axis: string): string {
  const labels: Record<string, string> = {
    changeNature: '変化の性質',
    agency: '主体性',
    timeframe: '時間軸',
    relationship: '関係性',
    emotionalTone: '感情基調',
  };
  return labels[axis] || axis;
}

/**
 * 一致度の表現を取得
 */
function getImpactLabel(contribution: number): string {
  if (contribution < 0.05) return '強く一致';
  if (contribution < 0.1) return '一致';
  if (contribution < 0.2) return '部分一致';
  return '参考';
}

/**
 * マッチング理由を生成
 */
export function generateMatchReasons(
  userProfile: UserProfile,
  candidate: CandidateScore
): string[] {
  const reasons: string[] = [];

  // 寄与度が大きい軸から説明（寄与度が小さい=一致している）
  const sortedContribs = (
    Object.entries(candidate.contributions) as [AxisName, number][]
  )
    .filter(([_, v]) => v > 0)
    .sort((a, b) => a[1] - b[1]); // 昇順（一致度が高い順）

  for (const [axis, contrib] of sortedContribs.slice(0, 3)) {
    const userDist = userProfile[axis];
    const topCategory = Object.entries(userDist.values).sort(
      (a, b) => b[1] - a[1]
    )[0];

    if (topCategory) {
      const percentage = Math.round(topCategory[1] * 100);
      const impact = getImpactLabel(contrib);

      reasons.push(
        `${getAxisLabel(axis)}: ${topCategory[0]}傾向(${percentage}%)が${impact}`
      );
    }
  }

  // 混合状態の説明
  if (candidate.isMixedState) {
    reasons.push('複数の傾向が混在している状態です');
  }

  return reasons;
}

/**
 * 確信度に基づく追加説明を生成
 */
export function generateConfidenceExplanation(
  overallConfidence: number,
  missingAxes: string[]
): string {
  if (overallConfidence >= 0.8) {
    return '十分な情報が得られました。結果の信頼性は高いです。';
  }

  if (overallConfidence >= 0.6) {
    const missing = missingAxes.map((a) => getAxisLabel(a)).join('、');
    return `${missing}の情報が不足していますが、参考になる結果が得られました。`;
  }

  return '情報が不足しています。結果は参考程度にご覧ください。';
}

/**
 * ランキング上位の類似性を説明
 */
export function generateSimilarityExplanation(
  candidates: CandidateScore[]
): string | null {
  if (candidates.length < 2) return null;

  const scoreDiff = candidates[1].score - candidates[0].score;

  if (scoreDiff < 0.01) {
    return '上位候補が非常に近い結果となっています。複数の可能性を検討してください。';
  }

  if (scoreDiff < 0.05) {
    return '1位と2位の差は小さいため、両方の候補を参考にしてください。';
  }

  return null;
}
