/**
 * レート制限モジュール
 *
 * スライディングウィンドウ方式でリクエスト数を制限
 * D1をバックエンドストレージとして使用
 */

// レート制限設定
export interface RateLimitConfig {
  maxRequests: number;
  windowSeconds: number;
}

// レート制限結果
export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  retryAfter?: number;
}

// データベースのレコード型
interface RateLimitRecord {
  identifier: string;
  request_count: number;
  window_start: string;
}

// デフォルト設定
export const DEFAULT_RATE_LIMIT: RateLimitConfig = {
  maxRequests: 100,
  windowSeconds: 60
};

/**
 * レート制限をチェック
 *
 * @param db - D1データベース
 * @param identifier - 識別子（IP or ライセンスキーハッシュ）
 * @param config - レート制限設定
 * @returns レート制限結果
 */
export async function checkRateLimit(
  db: D1Database,
  identifier: string,
  config: RateLimitConfig = DEFAULT_RATE_LIMIT
): Promise<RateLimitResult> {
  const now = Date.now();

  // 現在のレコードを取得
  const record = await db.prepare(`
    SELECT identifier, request_count, window_start
    FROM rate_limits
    WHERE identifier = ?
  `).bind(identifier).first<RateLimitRecord>();

  // レコードが存在しない場合 - 新規作成
  if (!record) {
    await db.prepare(`
      INSERT INTO rate_limits (identifier, request_count, window_start)
      VALUES (?, 1, datetime('now'))
    `).bind(identifier).run();

    return {
      allowed: true,
      remaining: config.maxRequests - 1
    };
  }

  // ウィンドウの開始時刻を解析
  const windowStart = new Date(record.window_start).getTime();
  const windowEnd = windowStart + config.windowSeconds * 1000;

  // ウィンドウが期限切れの場合 - リセット
  if (now >= windowEnd) {
    await db.prepare(`
      UPDATE rate_limits
      SET request_count = 1, window_start = datetime('now')
      WHERE identifier = ?
    `).bind(identifier).run();

    return {
      allowed: true,
      remaining: config.maxRequests - 1
    };
  }

  // 制限超過チェック
  if (record.request_count >= config.maxRequests) {
    const retryAfter = Math.ceil((windowEnd - now) / 1000);
    return {
      allowed: false,
      remaining: 0,
      retryAfter
    };
  }

  // カウント増加
  await db.prepare(`
    UPDATE rate_limits
    SET request_count = request_count + 1
    WHERE identifier = ?
  `).bind(identifier).run();

  return {
    allowed: true,
    remaining: config.maxRequests - record.request_count - 1
  };
}

/**
 * レート制限ヘッダーを生成
 *
 * @param result - レート制限結果
 * @param config - レート制限設定
 * @returns HTTPヘッダーオブジェクト
 */
export function getRateLimitHeaders(
  result: RateLimitResult,
  config: RateLimitConfig = DEFAULT_RATE_LIMIT
): Record<string, string> {
  const headers: Record<string, string> = {
    'X-RateLimit-Limit': config.maxRequests.toString(),
    'X-RateLimit-Remaining': result.remaining.toString()
  };

  if (!result.allowed && result.retryAfter) {
    headers['Retry-After'] = result.retryAfter.toString();
  }

  return headers;
}

/**
 * レート制限エラーレスポンスを生成
 *
 * @param result - レート制限結果
 * @param config - レート制限設定
 * @returns HTTPレスポンス
 */
export function rateLimitResponse(
  result: RateLimitResult,
  config: RateLimitConfig = DEFAULT_RATE_LIMIT
): Response {
  const headers = getRateLimitHeaders(result, config);

  return new Response(
    JSON.stringify({
      error: 'Too Many Requests',
      retryAfter: result.retryAfter
    }),
    {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        ...headers
      }
    }
  );
}
