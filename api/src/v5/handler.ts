/**
 * HaQei v5 API ハンドラー
 * Cloudflare Workers用のAPIエンドポイント処理
 */

import {
  UserAnswers,
  DiagnoseRequest,
  DiagnoseResponse,
  CandidateScore,
  VERSION,
} from './types';
import { convertToProfile } from './convert';
import { generateRanking, getTopCandidates } from './matching';
import {
  generateConfidenceExplanation,
  generateSimilarityExplanation,
} from './explanation';
import { setRubric, getClassProfiles, getRubric } from './rubric';

// ルーブリックをインポート（ビルド時に埋め込み）
import rubricData from '../../../data/rubric_v1.json';

// 初期化フラグ
let initialized = false;

/**
 * v5エンジン初期化
 */
export function initializeV5(): void {
  if (!initialized) {
    setRubric(rubricData as any);
    initialized = true;
  }
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
 * v5診断を実行
 */
export function diagnoseV5(answers: UserAnswers): {
  response: DiagnoseResponse;
  fullRanking: CandidateScore[];
} {
  initializeV5();

  // 1. 回答を確率分布に変換
  const userProfile = convertToProfile(answers);

  // 2. 384クラスとマッチング
  const classProfiles = getClassProfiles();
  const result = generateRanking(userProfile, classProfiles);

  // 3. レスポンス生成
  const topCandidates = getTopCandidates(result, 5);

  const response: DiagnoseResponse = {
    resultId: generateResultId(),
    topCandidates,
    missingAxes: result.missingAxes,
    overallConfidence: result.overallConfidence,
    version: VERSION,
  };

  return {
    response,
    fullRanking: result.ranking,
  };
}

/**
 * プレビュー用レスポンス生成（無料版）
 */
export interface PreviewResponse {
  resultId: string;
  topCandidate: {
    name: string;
    hexagramName: string;
    yaoName: string;
    yaoStage: string;
    matchReasons: string[];
  };
  candidateCount: number;
  confidenceExplanation: string;
  similarityNote: string | null;
  version: string;
}

export function createPreviewResponseV5(
  diagnoseResult: ReturnType<typeof diagnoseV5>
): PreviewResponse {
  const { response, fullRanking } = diagnoseResult;
  const top = response.topCandidates[0];

  return {
    resultId: response.resultId,
    topCandidate: {
      name: top.name,
      hexagramName: top.hexagramName,
      yaoName: top.yaoName,
      yaoStage: top.yaoStage,
      matchReasons: top.matchReasons,
    },
    candidateCount: fullRanking.length,
    confidenceExplanation: generateConfidenceExplanation(
      response.overallConfidence,
      response.missingAxes
    ),
    similarityNote: generateSimilarityExplanation(response.topCandidates),
    version: response.version,
  };
}

/**
 * フル分析用レスポンス生成（有料版）
 */
export interface FullAnalysisResponse extends PreviewResponse {
  topCandidates: CandidateScore[];
  overallConfidence: number;
  missingAxes: string[];
  actionPlan: string[];
  failurePatterns: string[];
  isFullVersion: true;
}

export function createFullResponseV5(
  diagnoseResult: ReturnType<typeof diagnoseV5>,
  similarCases: unknown[] = []
): FullAnalysisResponse {
  const preview = createPreviewResponseV5(diagnoseResult);
  const { response } = diagnoseResult;
  const top = response.topCandidates[0];

  return {
    ...preview,
    topCandidates: response.topCandidates,
    overallConfidence: response.overallConfidence,
    missingAxes: response.missingAxes,
    actionPlan: generateActionPlanV5(top.hexagram, top.yao),
    failurePatterns: generateFailurePatternsV5(top.hexagram, top.yao),
    isFullVersion: true,
  };
}

/**
 * 90日行動計画生成（v5用）
 */
function generateActionPlanV5(hexagram: number, yao: number): string[] {
  // 爻の段階に基づく行動計画
  const yaoPlans: Record<number, string[]> = {
    1: [
      '【1-30日】潜伏期: 状況を観察し、準備を整える時期',
      '【31-60日】萌芽期: 小さな一歩を踏み出し、反応を見る',
      '【61-90日】確認期: 方向性を確認し、次の展開を計画',
    ],
    2: [
      '【1-30日】展開期: 計画を実行に移し、フィードバックを集める',
      '【31-60日】調整期: 得られた知見をもとに軌道修正',
      '【61-90日】加速期: 成功パターンを強化する',
    ],
    3: [
      '【1-30日】試練期: 困難に直面する可能性、備えを固める',
      '【31-60日】克服期: 課題に正面から取り組む',
      '【61-90日】学習期: 経験から教訓を抽出する',
    ],
    4: [
      '【1-30日】転換期: 新しい選択肢を検討する',
      '【31-60日】決断期: 方向性を定め、コミットする',
      '【61-90日】実行期: 決めたことを着実に進める',
    ],
    5: [
      '【1-30日】充実期: 成果を確認し、さらなる発展を計画',
      '【31-60日】拡大期: 影響力を広げる活動を行う',
      '【61-90日】安定期: 持続可能な仕組みを構築する',
    ],
    6: [
      '【1-30日】完成期: 現フェーズを締めくくる準備',
      '【31-60日】移行期: 次のステージへの橋渡し',
      '【61-90日】新章期: 新しいサイクルの始まり',
    ],
  };

  return yaoPlans[yao] || yaoPlans[1];
}

/**
 * 失敗パターン生成（v5用）
 */
function generateFailurePatternsV5(hexagram: number, yao: number): string[] {
  // 爻の段階に基づく失敗パターン
  const yaoPatterns: Record<number, string[]> = {
    1: [
      '準備不足のまま動き出し、基盤が揺らぐ',
      '潜伏期間を軽視し、機会を逃す',
      '焦りから時機を誤る',
    ],
    2: [
      '周囲との調和を欠き、孤立する',
      '慎重すぎて機会を逃す',
      '他者の意見を聞かず、視野が狭まる',
    ],
    3: [
      '困難を避けようとして、問題が深刻化する',
      '無理をして限界を超える',
      '中途半端な対応で事態が悪化',
    ],
    4: [
      '決断を先延ばしにし、選択肢が狭まる',
      '変化を恐れて現状に固執する',
      '周囲の期待に振り回される',
    ],
    5: [
      '成功に慢心し、備えを怠る',
      '権限を濫用し、信頼を失う',
      '過度な拡大で基盤が弱体化',
    ],
    6: [
      '終わりを認めず、しがみつく',
      '次への準備を怠る',
      '過去の成功体験に囚われる',
    ],
  };

  return yaoPatterns[yao] || yaoPatterns[1];
}

/**
 * 回答のバリデーション
 */
export function validateAnswers(answers: unknown): answers is UserAnswers {
  if (!answers || typeof answers !== 'object') return false;

  const a = answers as Record<string, unknown>;

  // changeNature
  if (!a.changeNature || typeof a.changeNature !== 'object') return false;
  const cn = a.changeNature as Record<string, unknown>;
  if (
    typeof cn.expansion !== 'number' ||
    typeof cn.contraction !== 'number' ||
    typeof cn.maintenance !== 'number' ||
    typeof cn.transformation !== 'number'
  )
    return false;

  // agency
  if (typeof a.agency !== 'number' || a.agency < 1 || a.agency > 5) return false;

  // timeframe
  const validTimeframes = ['immediate', 'shortTerm', 'midTerm', 'longTerm', 'unknown'];
  if (!validTimeframes.includes(a.timeframe as string)) return false;

  // relationship
  if (!a.relationship || typeof a.relationship !== 'object') return false;

  // emotionalTone
  if (!a.emotionalTone || typeof a.emotionalTone !== 'object') return false;

  return true;
}

/**
 * v5質問一覧を取得
 */
export function getV5Questions() {
  return {
    version: VERSION,
    questions: [
      {
        id: 'changeNature',
        label: '変化の性質',
        description:
          '今の状況において、以下の傾向はどの程度当てはまりますか？',
        type: 'multi-slider',
        items: [
          {
            key: 'expansion',
            label: '拡大',
            description: '新しいことを始める、範囲を広げる',
          },
          {
            key: 'contraction',
            label: '収縮',
            description: '規模を縮小する、絞り込む',
          },
          {
            key: 'maintenance',
            label: '維持',
            description: '現状を保つ、安定させる',
          },
          {
            key: 'transformation',
            label: '転換',
            description: '方向を変える、刷新する',
          },
        ],
        scale: { min: 1, max: 5 },
      },
      {
        id: 'agency',
        label: '主体性',
        description:
          '今の状況に対して、どの程度自分でコントロールできていると感じますか？',
        type: 'slider',
        scale: {
          min: 1,
          max: 5,
          labels: {
            1: '全くコントロールできない',
            3: 'ある程度できる',
            5: '完全にコントロールできる',
          },
        },
      },
      {
        id: 'timeframe',
        label: '時間軸',
        description: 'この状況が決着するまでの期間は？',
        type: 'select',
        options: [
          { value: 'immediate', label: '1ヶ月以内' },
          { value: 'shortTerm', label: '3ヶ月以内' },
          { value: 'midTerm', label: '半年以内' },
          { value: 'longTerm', label: '1年以上' },
          { value: 'unknown', label: 'わからない' },
        ],
      },
      {
        id: 'relationship',
        label: '関係性',
        description: 'この状況に関わっているのは誰ですか？（複数選択可）',
        type: 'multi-select',
        options: [
          { key: 'self', label: '自分自身' },
          { key: 'family', label: '家族' },
          { key: 'team', label: '同僚・チーム' },
          { key: 'organization', label: '組織全体' },
          { key: 'external', label: '顧客・取引先' },
          { key: 'society', label: '業界・社会' },
        ],
      },
      {
        id: 'emotionalTone',
        label: '感情基調',
        description: '今の状況に対する感情として、以下はどの程度当てはまりますか？',
        type: 'multi-slider',
        items: [
          { key: 'excitement', label: 'ワクワク・期待感' },
          { key: 'caution', label: '慎重さ・用心深さ' },
          { key: 'anxiety', label: '不安・心配' },
          { key: 'optimism', label: '楽観・なんとかなる感' },
        ],
        scale: { min: 1, max: 5 },
      },
    ],
  };
}
