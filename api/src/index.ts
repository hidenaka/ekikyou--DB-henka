/**
 * HaQei API - Cloudflare Workers エントリーポイント
 *
 * 易経診断APIのメインエントリーポイント
 */

import { validateLicense, extractLicenseKey } from './license';
import { handleWebhookRequest } from './webhook';
import { checkRateLimit, rateLimitResponse, DEFAULT_RATE_LIMIT } from './ratelimit';
import { computeDiagnosis, createPreviewResponse, createFullResponse, type DiagnosisInput } from './diagnose';
import { searchCases, findSimilarCases, type CaseSearchParams } from './cases';
// v2 診断エンジン
import {
  processPhase1,
  processPhase2,
  getYaoOptions,
  generatePreview,
  PHASE1_QUESTIONS,
  type Phase1Answers
} from './diagnose-v2';
// v5 診断エンジン（JS距離ベース384直接ランキング）
import {
  diagnoseV5,
  createPreviewResponseV5,
  createFullResponseV5,
  validateAnswers,
  getV5Questions,
  initializeV5,
} from './v5/handler';
import type { UserAnswers } from './v5/types';

export interface Env {
  DB: D1Database;
  ALLOWED_ORIGINS: string;
  LEMON_SQUEEZY_WEBHOOK_SECRET?: string;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS ヘッダー
    const corsHeaders = getCorsHeaders(request, env.ALLOWED_ORIGINS);

    // Preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // ルーティング
      // Health check
      if (path === '/health') {
        return json({ status: 'ok', timestamp: new Date().toISOString() }, corsHeaders);
      }

      // Webhook (Lemon Squeezy)
      if (path === '/webhook' && request.method === 'POST') {
        const secret = env.LEMON_SQUEEZY_WEBHOOK_SECRET || '';
        return handleWebhookRequest(request, env.DB, secret);
      }

      // レート制限チェック（API全体）
      const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';
      const rateLimit = await checkRateLimit(env.DB, clientIP, DEFAULT_RATE_LIMIT);
      if (!rateLimit.allowed) {
        return rateLimitResponse(rateLimit);
      }

      // 診断API（プレビュー - 認証不要）
      if (path === '/diagnose/preview' && request.method === 'POST') {
        const body = await request.json() as DiagnosisInput;
        const result = computeDiagnosis(body);
        const preview = createPreviewResponse(result);
        return json(preview, corsHeaders);
      }

      // 診断API（フル - 認証必要）
      if (path === '/diagnose/full' && request.method === 'POST') {
        const authHeader = request.headers.get('Authorization');
        const licenseKey = extractLicenseKey(authHeader);

        if (!licenseKey) {
          return json({ error: 'Unauthorized: License key required' }, corsHeaders, 401);
        }

        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: 'Unauthorized: Invalid license key' }, corsHeaders, 401);
        }

        const body = await request.json() as DiagnosisInput;
        const result = computeDiagnosis(body);

        // 類似事例を取得
        const similarCases = await findSimilarCases(
          env.DB,
          result.beforeTrigram as '乾' | '坤' | '震' | '巽' | '坎' | '離' | '艮' | '兌',
          result.afterTrigram as '乾' | '坤' | '震' | '巽' | '坎' | '離' | '艮' | '兌',
          5
        );

        const fullResponse = createFullResponse(result);
        return json({ ...fullResponse, similarCases }, corsHeaders);
      }

      // 事例検索API（認証必要）
      if (path === '/cases/search' && request.method === 'GET') {
        const authHeader = request.headers.get('Authorization');
        const licenseKey = extractLicenseKey(authHeader);

        if (!licenseKey) {
          return json({ error: 'Unauthorized: License key required' }, corsHeaders, 401);
        }

        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: 'Unauthorized: Invalid license key' }, corsHeaders, 401);
        }

        // クエリパラメータから検索条件を取得
        const params: CaseSearchParams = {};
        const patternType = url.searchParams.get('pattern_type');
        const beforeHex = url.searchParams.get('before_hex');
        const afterHex = url.searchParams.get('after_hex');
        const scale = url.searchParams.get('scale');

        if (patternType) params.pattern_type = patternType as CaseSearchParams['pattern_type'];
        if (beforeHex) params.before_trigram = beforeHex as CaseSearchParams['before_trigram'];
        if (afterHex) params.after_trigram = afterHex as CaseSearchParams['after_trigram'];
        if (scale) params.scale = scale as CaseSearchParams['scale'];

        const result = await searchCases(env.DB, params);
        return json(result, corsHeaders);
      }

      // ============================================================
      // v2 診断API（新設計: Top-k候補 + ユーザー選択方式）
      // ============================================================

      // v2: 質問一覧取得
      if (path === '/v2/diagnose/questions' && request.method === 'GET') {
        return json({ questions: PHASE1_QUESTIONS }, corsHeaders);
      }

      // v2: Phase 1 - 初期絞り込み
      if (path === '/v2/diagnose/phase1' && request.method === 'POST') {
        const body = await request.json() as { answers: Phase1Answers };
        const result = processPhase1(body.answers);
        return json(result, corsHeaders);
      }

      // v2: Phase 2 - 追加質問処理
      if (path === '/v2/diagnose/phase2' && request.method === 'POST') {
        const body = await request.json() as {
          phase1Answers: Phase1Answers;
          phase2Answers: Record<string, number>;
        };
        const result = processPhase2(body.phase1Answers, body.phase2Answers);
        return json(result, corsHeaders);
      }

      // v2: Phase 3 - ユーザー選択後の爻オプション取得
      if (path === '/v2/diagnose/select' && request.method === 'POST') {
        const body = await request.json() as { hexagramNumber: number };
        const result = getYaoOptions(body.hexagramNumber);
        return json(result, corsHeaders);
      }

      // v2: Phase 4 - プレビュー（無料）
      if (path === '/v2/diagnose/preview' && request.method === 'POST') {
        const body = await request.json() as { hexagramNumber: number; yao: number };

        // DBから類似事例数を取得
        let caseCount = 0;
        try {
          const countResult = await env.DB.prepare(
            `SELECT COUNT(*) as count FROM cases
             WHERE trigger_hex_number = ? OR result_hex_number = ?`
          ).bind(body.hexagramNumber, body.hexagramNumber).first<{ count: number }>();
          caseCount = countResult?.count || 0;
        } catch {
          // DBエラー時は0件として処理
          caseCount = 0;
        }

        const result = generatePreview(body.hexagramNumber, body.yao, caseCount);
        return json(result, corsHeaders);
      }

      // v2: Phase 4 - フル分析（有料・認証必要）
      if (path === '/v2/diagnose/full' && request.method === 'POST') {
        const authHeader = request.headers.get('Authorization');
        const licenseKey = extractLicenseKey(authHeader);

        if (!licenseKey) {
          return json({ error: 'Unauthorized: License key required' }, corsHeaders, 401);
        }

        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: 'Unauthorized: Invalid license key' }, corsHeaders, 401);
        }

        const body = await request.json() as { hexagramNumber: number; yao: number };

        // 類似事例を取得
        let similarCases: unknown[] = [];
        try {
          const casesResult = await env.DB.prepare(
            `SELECT entity_name, domain, description, outcome, source_url
             FROM cases
             WHERE (trigger_hex_number = ? OR result_hex_number = ?)
               AND yao_position = ?
             LIMIT 5`
          ).bind(body.hexagramNumber, body.hexagramNumber, body.yao).all();
          similarCases = casesResult.results || [];
        } catch {
          similarCases = [];
        }

        const preview = generatePreview(body.hexagramNumber, body.yao, similarCases.length);

        return json({
          ...preview,
          similarCases,
          actionPlan: generateActionPlan(body.hexagramNumber, body.yao),
          failurePatterns: generateFailurePatterns(body.hexagramNumber),
          isFullVersion: true
        }, corsHeaders);
      }

      // ============================================================
      // v5 診断API（限定ベータ: JS距離 + 384直接ランキング）
      // ============================================================

      // v5: 質問一覧取得
      if (path === '/v5/diagnose/questions' && request.method === 'GET') {
        return json(getV5Questions(), corsHeaders);
      }

      // v5: 診断実行（プレビュー - 無料版）
      if (path === '/v5/diagnose/preview' && request.method === 'POST') {
        const body = await request.json() as { answers: unknown };

        if (!validateAnswers(body.answers)) {
          return json({ error: 'Invalid answers format' }, corsHeaders, 400);
        }

        const result = diagnoseV5(body.answers as UserAnswers);
        const preview = createPreviewResponseV5(result);
        return json(preview, corsHeaders);
      }

      // v5: 診断実行（フル分析 - 有料版・認証必要）
      if (path === '/v5/diagnose/full' && request.method === 'POST') {
        const authHeader = request.headers.get('Authorization');
        const licenseKey = extractLicenseKey(authHeader);

        if (!licenseKey) {
          return json({ error: 'Unauthorized: License key required' }, corsHeaders, 401);
        }

        const validation = await validateLicense(licenseKey, env.DB);
        if (!validation.valid) {
          return json({ error: 'Unauthorized: Invalid license key' }, corsHeaders, 401);
        }

        const body = await request.json() as { answers: unknown };

        if (!validateAnswers(body.answers)) {
          return json({ error: 'Invalid answers format' }, corsHeaders, 400);
        }

        const result = diagnoseV5(body.answers as UserAnswers);

        // 類似事例を取得
        const top = result.response.topCandidates[0];
        let similarCases: unknown[] = [];
        try {
          const casesResult = await env.DB.prepare(
            `SELECT entity_name, domain, description, outcome, source_url
             FROM cases
             WHERE (trigger_hex_number = ? OR result_hex_number = ?)
               AND yao_position = ?
             LIMIT 5`
          ).bind(top.hexagram, top.hexagram, top.yao).all();
          similarCases = casesResult.results || [];
        } catch {
          similarCases = [];
        }

        const fullResponse = createFullResponseV5(result, similarCases);
        return json({ ...fullResponse, similarCases }, corsHeaders);
      }

      // v5: デバッグ用 - 全ランキング取得（開発環境のみ）
      if (path === '/v5/diagnose/debug/ranking' && request.method === 'POST') {
        // 本番では無効化
        const isDev = url.hostname === 'localhost' || url.hostname === '127.0.0.1';
        if (!isDev) {
          return json({ error: 'Debug endpoint not available in production' }, corsHeaders, 403);
        }

        const body = await request.json() as { answers: unknown };
        if (!validateAnswers(body.answers)) {
          return json({ error: 'Invalid answers format' }, corsHeaders, 400);
        }

        const result = diagnoseV5(body.answers as UserAnswers);
        return json({
          resultId: result.response.resultId,
          ranking: result.fullRanking.slice(0, 50), // 上位50件のみ
          version: result.response.version,
        }, corsHeaders);
      }

      // 404
      return json({ error: 'Not Found' }, corsHeaders, 404);

    } catch (error) {
      console.error('API Error:', error);
      const message = error instanceof Error ? error.message : 'Internal Server Error';
      return json(
        { error: message },
        corsHeaders,
        error instanceof Error && error.message.includes('required') ? 400 : 500
      );
    }
  },
};

// JSON レスポンスヘルパー
function json(data: unknown, headers: Headers, status = 200): Response {
  const responseHeaders = new Headers(headers);
  responseHeaders.set('Content-Type', 'application/json');
  return new Response(JSON.stringify(data), { status, headers: responseHeaders });
}

// CORS ヘッダー生成
function getCorsHeaders(request: Request, allowedOrigins: string): Headers {
  const origin = request.headers.get('Origin') || '';
  const allowed = allowedOrigins.split(',').map(o => o.trim());
  const headers = new Headers();

  if (allowed.includes(origin) || allowed.includes('*')) {
    headers.set('Access-Control-Allow-Origin', origin);
  }

  headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  headers.set('Access-Control-Max-Age', '86400');

  return headers;
}

// 90日行動計画生成（有料版用）
function generateActionPlan(hexagramNumber: number, yao: number): string[] {
  const yaoPlans: Record<number, string[]> = {
    1: [
      '【1-30日】準備期間: 情報収集と計画立案に集中する',
      '【31-60日】小さな一歩: 低リスクな実験を開始する',
      '【61-90日】振り返り: 結果を分析し、次のステップを決定'
    ],
    2: [
      '【1-30日】展開期: 計画を実行に移し、フィードバックを集める',
      '【31-60日】調整期: 得られた知見をもとに軌道修正',
      '【61-90日】加速期: 成功パターンを拡大する'
    ],
    3: [
      '【1-30日】課題直視: 問題の根本原因を特定する',
      '【31-60日】対策実行: 優先順位をつけて一つずつ解決',
      '【61-90日】予防策: 同じ問題が起きない仕組みを構築'
    ],
    4: [
      '【1-30日】選択肢整理: 可能な選択肢を洗い出す',
      '【31-60日】決断実行: 最善の選択を行い、コミットする',
      '【61-90日】結果検証: 決断の結果を評価し、必要なら修正'
    ],
    5: [
      '【1-30日】成果確認: 達成したことを整理し、次の目標を設定',
      '【31-60日】影響拡大: 成功を他の領域にも展開',
      '【61-90日】持続化: 成果を維持する仕組みを作る'
    ],
    6: [
      '【1-30日】収束準備: 現フェーズの締めくくりを計画',
      '【31-60日】引き継ぎ: 次のステージへの移行準備',
      '【61-90日】新章開始: 新しいサイクルの第一歩を踏み出す'
    ]
  };

  return yaoPlans[yao] || yaoPlans[1];
}

// 失敗パターン生成（有料版用）
function generateFailurePatterns(hexagramNumber: number): string[] {
  // 卦の特性に基づく失敗パターン
  const patterns: string[] = [
    '焦って動きすぎる: 時機を待たずに行動し、機会を逃す',
    '変化を恐れすぎる: 必要な変化を先延ばしにして状況が悪化',
    '一人で抱え込む: 協力を求めず、限界を超えて疲弊',
    '過去に固執する: 古い方法に執着し、新しい可能性を見逃す',
    '楽観しすぎる: リスクを過小評価し、備えを怠る'
  ];

  // 卦番号に基づいて順序を変える（多様性のため）
  const offset = hexagramNumber % patterns.length;
  return [...patterns.slice(offset), ...patterns.slice(0, offset)].slice(0, 3);
}
