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
