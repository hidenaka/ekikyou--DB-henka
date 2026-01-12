/**
 * 診断エンジンモジュール
 *
 * 易経（八卦）に基づく診断ロジック
 * 10の質問から「前の状態」「後の状態」を計算
 */

// 八卦定義
const TRIGRAMS = ['乾', '兌', '離', '震', '巽', '坎', '艮', '坤'] as const;
type Trigram = typeof TRIGRAMS[number];

// 八卦シンボル
const TRIGRAM_SYMBOLS: Record<Trigram, string> = {
  '乾': '☰', '兌': '☱', '離': '☲', '震': '☳',
  '巽': '☴', '坎': '☵', '艮': '☶', '坤': '☷'
};

// パターンタイプ
const PATTERN_TYPES = [
  'Expansion_Growth',
  'Contraction_Decline',
  'Transformation_Shift',
  'Stability_Maintenance',
  'Crisis_Recovery',
  'Emergence_Innovation'
] as const;
type PatternType = typeof PATTERN_TYPES[number];

// 入力型
export interface DiagnosisInput {
  answers: number[];
}

// 診断結果型
export interface DiagnosisResult {
  hexagram: string;
  beforeTrigram: Trigram;
  afterTrigram: Trigram;
  summary: string;
  fullAnalysis: string;
  recommendedActions: string[];
  patternType: PatternType;
}

// プレビューレスポンス型
export interface PreviewResponse {
  hexagram: string;
  beforeTrigram: Trigram;
  afterTrigram: Trigram;
  summary: string;
  isPreview: true;
  fullAnalysis?: undefined;
  recommendedActions?: undefined;
}

// フルレスポンス型
export interface FullResponse extends Omit<DiagnosisResult, never> {
  isPreview: false;
}

/**
 * 診断を計算
 *
 * @param input - 10個の回答（各1-3）
 * @returns 診断結果
 */
export function computeDiagnosis(input: DiagnosisInput): DiagnosisResult {
  const { answers } = input;

  // バリデーション
  if (answers.length !== 10) {
    throw new Error('10 answers required');
  }

  if (!answers.every(a => a >= 1 && a <= 3)) {
    throw new Error('Answer values must be 1, 2, or 3');
  }

  // 前の状態（Q1-Q5）と後の状態（Q6-Q10）を計算
  const beforeAnswers = answers.slice(0, 5);
  const afterAnswers = answers.slice(5, 10);

  const beforeTrigram = calculateTrigram(beforeAnswers);
  const afterTrigram = calculateTrigram(afterAnswers);

  // 卦（hexagram）を生成
  const hexagram = TRIGRAM_SYMBOLS[beforeTrigram] + TRIGRAM_SYMBOLS[afterTrigram];

  // パターンタイプを決定
  const patternType = determinePatternType(beforeTrigram, afterTrigram);

  // 分析テキストを生成
  const summary = generateSummary(beforeTrigram, afterTrigram, patternType);
  const fullAnalysis = generateFullAnalysis(beforeTrigram, afterTrigram, patternType);
  const recommendedActions = generateRecommendations(beforeTrigram, afterTrigram, patternType);

  return {
    hexagram,
    beforeTrigram,
    afterTrigram,
    summary,
    fullAnalysis,
    recommendedActions,
    patternType
  };
}

/**
 * 5つの回答から八卦を計算
 */
function calculateTrigram(answers: number[]): Trigram {
  // 回答の合計値でインデックスを決定（5-15 → 0-7）
  const sum = answers.reduce((a, b) => a + b, 0);
  const index = Math.floor((sum - 5) / 1.375); // 5-15を0-7に正規化
  const clampedIndex = Math.max(0, Math.min(7, index));
  return TRIGRAMS[clampedIndex];
}

/**
 * パターンタイプを決定
 */
function determinePatternType(before: Trigram, after: Trigram): PatternType {
  const beforeIdx = TRIGRAMS.indexOf(before);
  const afterIdx = TRIGRAMS.indexOf(after);
  const diff = afterIdx - beforeIdx;

  // 変化の方向と大きさでパターンを分類
  if (diff === 0) return 'Stability_Maintenance';
  if (diff > 0 && diff <= 2) return 'Expansion_Growth';
  if (diff < 0 && diff >= -2) return 'Contraction_Decline';
  if (Math.abs(diff) >= 5) return 'Transformation_Shift';
  if (before === '坎' || after === '坎') return 'Crisis_Recovery';
  return 'Emergence_Innovation';
}

/**
 * サマリーを生成
 */
function generateSummary(before: Trigram, after: Trigram, pattern: PatternType): string {
  const summaries: Record<PatternType, string> = {
    'Expansion_Growth': `${before}から${after}への変化は、成長と拡大を示しています。`,
    'Contraction_Decline': `${before}から${after}への変化は、収縮と整理の時期を示しています。`,
    'Transformation_Shift': `${before}から${after}への大きな変化は、根本的な転換期を示しています。`,
    'Stability_Maintenance': `${before}の状態が維持され、安定期にあることを示しています。`,
    'Crisis_Recovery': `${before}から${after}への変化は、困難を乗り越える力を示しています。`,
    'Emergence_Innovation': `${before}から${after}への変化は、新しい可能性の出現を示しています。`
  };
  return summaries[pattern];
}

/**
 * 詳細分析を生成
 */
function generateFullAnalysis(before: Trigram, after: Trigram, pattern: PatternType): string {
  const trigramMeanings: Record<Trigram, string> = {
    '乾': '天の力、創造性、リーダーシップ',
    '兌': '喜び、コミュニケーション、交流',
    '離': '明晰さ、知恵、洞察',
    '震': '動き、始まり、衝撃',
    '巽': '柔軟性、適応、浸透',
    '坎': '困難、深み、学び',
    '艮': '静止、内省、蓄積',
    '坤': '受容、育成、基盤'
  };

  return `
【現在の状態: ${before}】
${trigramMeanings[before]}

【変化後の状態: ${after}】
${trigramMeanings[after]}

【変化のパターン: ${pattern.replace('_', ' ')}】
この変化は、${generateSummary(before, after, pattern)}

【詳細分析】
${before}の持つエネルギーから${after}のエネルギーへの移行は、
組織や個人の発展段階において重要な転換点を示しています。
  `.trim();
}

/**
 * 推奨アクションを生成
 */
function generateRecommendations(before: Trigram, after: Trigram, pattern: PatternType): string[] {
  const baseActions: Record<PatternType, string[]> = {
    'Expansion_Growth': [
      '成長の機会を積極的に追求する',
      'リソースの拡充を検討する',
      '新しいパートナーシップを模索する'
    ],
    'Contraction_Decline': [
      'コア事業に集中する',
      '無駄を見直し効率化を図る',
      '次の成長期に備えて基盤を固める'
    ],
    'Transformation_Shift': [
      '既存の枠組みを見直す',
      '抜本的な改革を検討する',
      '変化を恐れず新しい方向性を探る'
    ],
    'Stability_Maintenance': [
      '現状の強みを維持・強化する',
      '品質の向上に注力する',
      '将来に向けた準備を進める'
    ],
    'Crisis_Recovery': [
      '問題の根本原因を分析する',
      '支援を求めることを検討する',
      '回復後のビジョンを明確にする'
    ],
    'Emergence_Innovation': [
      '新しいアイデアを試す',
      '実験的なアプローチを取り入れる',
      '柔軟な姿勢で変化に対応する'
    ]
  };

  return baseActions[pattern];
}

/**
 * プレビューレスポンスを作成
 */
export function createPreviewResponse(result: DiagnosisResult): PreviewResponse {
  return {
    hexagram: result.hexagram,
    beforeTrigram: result.beforeTrigram,
    afterTrigram: result.afterTrigram,
    summary: result.summary,
    isPreview: true
  };
}

/**
 * フルレスポンスを作成
 */
export function createFullResponse(result: DiagnosisResult): FullResponse {
  return {
    ...result,
    isPreview: false
  };
}
