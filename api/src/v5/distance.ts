/**
 * HaQei診断システム v5 距離計算
 */

/**
 * KLダイバージェンス計算
 */
export function klDivergence(
  p: Record<string, number>,
  q: Record<string, number>
): number {
  let kl = 0;
  for (const key of Object.keys(p)) {
    if (p[key] > 0 && q[key] > 0) {
      kl += p[key] * Math.log2(p[key] / q[key]);
    }
  }
  return kl;
}

/**
 * Jensen-Shannon距離計算
 * 2つの確率分布間の類似度を測定（0=同一, 1=最大乖離）
 */
export function jsDistance(
  p: Record<string, number>,
  q: Record<string, number>
): number {
  const keys = Object.keys(p);

  // ゼロ確率の平滑化（数値安定性のため）
  const epsilon = 1e-10;
  const smoothP: Record<string, number> = {};
  const smoothQ: Record<string, number> = {};

  for (const key of keys) {
    smoothP[key] = Math.max(p[key], epsilon);
    smoothQ[key] = Math.max(q[key], epsilon);
  }

  // 正規化
  const sumP = Object.values(smoothP).reduce((a, b) => a + b, 0);
  const sumQ = Object.values(smoothQ).reduce((a, b) => a + b, 0);

  for (const key of keys) {
    smoothP[key] /= sumP;
    smoothQ[key] /= sumQ;
  }

  // 中間分布 M = (P + Q) / 2
  const m: Record<string, number> = {};
  for (const key of keys) {
    m[key] = (smoothP[key] + smoothQ[key]) / 2;
  }

  // JS距離 = sqrt((KL(P||M) + KL(Q||M)) / 2)
  const klPM = klDivergence(smoothP, m);
  const klQM = klDivergence(smoothQ, m);

  return Math.sqrt((klPM + klQM) / 2);
}
