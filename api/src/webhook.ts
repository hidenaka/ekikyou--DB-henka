/**
 * Webhook処理モジュール
 *
 * Lemon Squeezy Webhookの受信と処理
 * - 署名検証
 * - イベント別ハンドリング
 */

import { sha256 } from './utils';

// Webhookペイロードの型定義
export interface WebhookPayload {
  meta: {
    event_name: string;
  };
  data: {
    attributes: {
      license_key: string;
      status?: string;
      expires_at?: string | null;
    };
  };
}

// 失効イベント一覧
const REVOCATION_EVENTS = [
  'order_refunded',
  'subscription_cancelled',
  'subscription_expired',
  'license_key_revoked'
];

// 有効化イベント一覧
const ACTIVATION_EVENTS = [
  'license_key_created',
  'order_created',
  'subscription_created'
];

/**
 * Webhookイベントを処理
 *
 * @param payload - Webhookペイロード
 * @param db - D1データベース
 */
export async function handleWebhook(
  payload: WebhookPayload,
  db: D1Database
): Promise<void> {
  const eventName = payload.meta.event_name;
  const licenseKey = payload.data.attributes.license_key;

  if (!licenseKey) {
    console.warn('Webhook payload missing license_key');
    return;
  }

  const keyHash = await sha256(licenseKey);

  // 失効イベント
  if (REVOCATION_EVENTS.includes(eventName)) {
    await db.prepare(`
      UPDATE license_cache
      SET status = ?, updated_at = datetime('now')
      WHERE key_hash = ?
    `).bind('inactive', keyHash).run();

    console.log(`License revoked: ${eventName}`);
    return;
  }

  // 有効化イベント
  if (ACTIVATION_EVENTS.includes(eventName)) {
    const status = payload.data.attributes.status || 'active';
    const expiresAt = payload.data.attributes.expires_at || null;

    await db.prepare(`
      INSERT OR REPLACE INTO license_cache
      (key_hash, status, plan, expires_at, last_verified_at, updated_at)
      VALUES (?, ?, 'basic', ?, datetime('now'), datetime('now'))
    `).bind(keyHash, status, expiresAt).run();

    console.log(`License activated: ${eventName}`);
    return;
  }

  // 未知のイベントは無視
  console.log(`Ignoring unknown event: ${eventName}`);
}

/**
 * Webhook署名を検証（HMAC-SHA256）
 *
 * @param payload - 生のリクエストボディ
 * @param signature - X-Signatureヘッダーの値
 * @param secret - Webhookシークレット
 * @returns 署名が有効ならtrue
 */
export async function verifyWebhookSignature(
  payload: string,
  signature: string,
  secret: string
): Promise<boolean> {
  if (!signature || !secret) {
    return false;
  }

  try {
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    );

    const signatureBuffer = await crypto.subtle.sign('HMAC', key, encoder.encode(payload));
    const expectedSignature = Array.from(new Uint8Array(signatureBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');

    // タイミング攻撃を防ぐための定数時間比較
    if (signature.length !== expectedSignature.length) {
      return false;
    }

    let result = 0;
    for (let i = 0; i < signature.length; i++) {
      result |= signature.charCodeAt(i) ^ expectedSignature.charCodeAt(i);
    }

    return result === 0;
  } catch {
    return false;
  }
}

/**
 * WebhookリクエストをハンドリングするHTTPエンドポイント用関数
 *
 * @param request - HTTPリクエスト
 * @param db - D1データベース
 * @param webhookSecret - Webhookシークレット
 * @returns HTTPレスポンス
 */
export async function handleWebhookRequest(
  request: Request,
  db: D1Database,
  webhookSecret: string
): Promise<Response> {
  // POSTメソッドのみ許可
  if (request.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  const signature = request.headers.get('X-Signature') || '';
  const body = await request.text();

  // 署名検証
  const isValid = await verifyWebhookSignature(body, signature, webhookSecret);
  if (!isValid) {
    return new Response('Unauthorized', { status: 401 });
  }

  try {
    const payload = JSON.parse(body) as WebhookPayload;
    await handleWebhook(payload, db);
    return new Response('OK', { status: 200 });
  } catch (error) {
    console.error('Webhook processing error:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}
