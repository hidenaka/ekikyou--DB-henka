/**
 * HaQei API - Cloudflare Workers エントリーポイント
 *
 * 易経診断APIのメインエントリーポイント
 */

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
      if (path === '/health') {
        return json({ status: 'ok', timestamp: new Date().toISOString() }, corsHeaders);
      }

      // 404
      return json({ error: 'Not Found' }, corsHeaders, 404);

    } catch (error) {
      console.error('API Error:', error);
      return json(
        { error: 'Internal Server Error' },
        corsHeaders,
        500
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
