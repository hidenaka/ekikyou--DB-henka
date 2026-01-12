/**
 * ライセンス検証モジュール
 *
 * Lemon Squeezy License API との連携
 * D1キャッシュによる検証結果の保持
 */

import { sha256 } from './utils';

// ライセンス検証結果の型
export interface LicenseValidation {
  valid: boolean;
  status: 'active' | 'inactive' | 'expired';
  plan?: string;
  expiresAt?: string | null;
}

// Lemon Squeezy APIレスポンスの型
export interface LemonSqueezyResponse {
  valid: boolean;
  license_key?: {
    status: string;
    expires_at?: string | null;
  };
  meta?: {
    variant_name?: string;
  };
  error?: string;
}

// Lemon Squeezy API クライアントの型（DI用）
export type LemonSqueezyApiClient = (licenseKey: string) => Promise<LemonSqueezyResponse>;

/**
 * ライセンスキーを検証
 *
 * 1. D1キャッシュを確認
 * 2. キャッシュミス or 期限切れの場合、Lemon Squeezy APIを呼び出し
 * 3. 結果をキャッシュに保存
 *
 * @param licenseKey - 検証するライセンスキー
 * @param db - D1データベース
 * @param lsApiClient - Lemon Squeezy APIクライアント（テスト用にDI可能）
 */
export async function validateLicense(
  licenseKey: string,
  db: D1Database,
  lsApiClient?: LemonSqueezyApiClient
): Promise<LicenseValidation> {
  const keyHash = await sha256(licenseKey);

  // 1. キャッシュ確認
  const cached = await db.prepare(`
    SELECT * FROM license_cache
    WHERE key_hash = ?
    AND status = 'active'
    AND datetime(last_verified_at, '+' || cache_ttl_seconds || ' seconds') > datetime('now')
  `).bind(keyHash).first<{
    status: string;
    plan: string;
    expires_at: string | null;
  }>();

  if (cached) {
    return {
      valid: true,
      status: cached.status as 'active',
      plan: cached.plan,
      expiresAt: cached.expires_at
    };
  }

  // 2. Lemon Squeezy API検証
  const apiClient = lsApiClient || callLemonSqueezyValidate;
  const lsResult = await apiClient(licenseKey);

  // 3. キャッシュ更新
  const status = lsResult.valid ? 'active' : 'inactive';
  const plan = lsResult.meta?.variant_name || 'basic';
  const expiresAt = lsResult.license_key?.expires_at || null;

  await db.prepare(`
    INSERT OR REPLACE INTO license_cache
    (key_hash, status, plan, expires_at, last_verified_at, updated_at)
    VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
  `).bind(keyHash, status, plan, expiresAt).run();

  return {
    valid: lsResult.valid,
    status: status as 'active' | 'inactive',
    plan,
    expiresAt
  };
}

/**
 * Lemon Squeezy License Validation API を呼び出し
 * https://docs.lemonsqueezy.com/api/license-api
 */
async function callLemonSqueezyValidate(licenseKey: string): Promise<LemonSqueezyResponse> {
  const response = await fetch('https://api.lemonsqueezy.com/v1/licenses/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ license_key: licenseKey })
  });

  if (!response.ok) {
    throw new Error(`Lemon Squeezy API error: ${response.status}`);
  }

  return response.json();
}

/**
 * 認証ヘッダーからライセンスキーを抽出
 * @param authHeader - Authorization ヘッダーの値 (Bearer <key>)
 */
export function extractLicenseKey(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const match = authHeader.match(/^Bearer\s+(.+)$/i);
  return match ? match[1] : null;
}
