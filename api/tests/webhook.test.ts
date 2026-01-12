/**
 * Phase 2: Webhook処理のテスト（TDD）
 *
 * テスト対象: handleWebhook, verifyWebhookSignature
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect, vi } from 'vitest';
import { sha256 } from '../src/utils';
import {
  handleWebhook,
  verifyWebhookSignature,
  type WebhookPayload
} from '../src/webhook';

// Mock D1Database型
interface MockStatement {
  bind: (...args: unknown[]) => MockStatement;
  first: <T>() => Promise<T | null>;
  run: () => Promise<{ success: boolean }>;
}

interface MockD1Database {
  prepare: (sql: string) => MockStatement;
}

// テスト用のモックD1データベースを作成
function createMockDb(): { db: MockD1Database; cache: Map<string, unknown> } {
  const cache = new Map<string, unknown>();

  const db: MockD1Database = {
    prepare: (sql: string) => {
      const statement: MockStatement = {
        bind: (...args: unknown[]) => {
          (statement as unknown as { boundArgs: unknown[] }).boundArgs = args;
          return statement;
        },
        first: async <T>() => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];
          const keyHash = args[0] as string;

          if (sql.includes('SELECT') && sql.includes('license_cache')) {
            return cache.get(keyHash) as T | null;
          }
          return null;
        },
        run: async () => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];

          // UPDATE クエリ - ステータスを更新
          if (sql.includes('UPDATE') && sql.includes('license_cache')) {
            const status = args[0] as string;
            const keyHash = args[1] as string;
            const existing = cache.get(keyHash);
            if (existing) {
              cache.set(keyHash, { ...(existing as object), status });
            }
          }

          // INSERT クエリ
          if (sql.includes('INSERT') && sql.includes('license_cache')) {
            const [keyHash, status, plan] = args as [string, string, string];
            cache.set(keyHash, { key_hash: keyHash, status, plan });
          }

          return { success: true };
        }
      };
      return statement;
    }
  };

  return { db, cache };
}

describe('handleWebhook - order_refunded', () => {
  it('返金時にライセンスを即時失効', async () => {
    // Arrange
    const licenseKey = 'REFUNDED-KEY';
    const keyHash = await sha256(licenseKey);
    const { db, cache } = createMockDb();

    // 有効なキャッシュを事前登録
    cache.set(keyHash, {
      key_hash: keyHash,
      status: 'active',
      plan: 'pro'
    });

    const webhookPayload: WebhookPayload = {
      meta: { event_name: 'order_refunded' },
      data: { attributes: { license_key: licenseKey } }
    };

    // Act
    await handleWebhook(webhookPayload, db as unknown as D1Database);

    // Assert
    const cached = cache.get(keyHash) as { status: string } | undefined;
    expect(cached?.status).toBe('inactive');
  });
});

describe('handleWebhook - subscription_cancelled', () => {
  it('解約時にライセンスを失効', async () => {
    // Arrange
    const licenseKey = 'CANCELLED-KEY';
    const keyHash = await sha256(licenseKey);
    const { db, cache } = createMockDb();

    // 有効なキャッシュを事前登録
    cache.set(keyHash, {
      key_hash: keyHash,
      status: 'active',
      plan: 'pro'
    });

    const webhookPayload: WebhookPayload = {
      meta: { event_name: 'subscription_cancelled' },
      data: { attributes: { license_key: licenseKey } }
    };

    // Act
    await handleWebhook(webhookPayload, db as unknown as D1Database);

    // Assert
    const cached = cache.get(keyHash) as { status: string } | undefined;
    expect(cached?.status).toBe('inactive');
  });
});

describe('handleWebhook - license_key_created', () => {
  it('新規ライセンス作成時にキャッシュに登録', async () => {
    // Arrange
    const licenseKey = 'NEW-LICENSE-KEY';
    const keyHash = await sha256(licenseKey);
    const { db, cache } = createMockDb();

    const webhookPayload: WebhookPayload = {
      meta: { event_name: 'license_key_created' },
      data: {
        attributes: {
          license_key: licenseKey,
          status: 'active',
          expires_at: '2027-01-01T00:00:00Z'
        }
      }
    };

    // Act
    await handleWebhook(webhookPayload, db as unknown as D1Database);

    // Assert
    const cached = cache.get(keyHash) as { status: string } | undefined;
    expect(cached).toBeDefined();
    expect(cached?.status).toBe('active');
  });
});

describe('verifyWebhookSignature', () => {
  it('有効な署名の場合trueを返す', async () => {
    const payload = '{"test": "data"}';
    const secret = 'test-secret';

    // HMAC-SHA256署名を計算
    const encoder = new TextEncoder();
    const key = await crypto.subtle.importKey(
      'raw',
      encoder.encode(secret),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    );
    const signatureBuffer = await crypto.subtle.sign('HMAC', key, encoder.encode(payload));
    const signature = Array.from(new Uint8Array(signatureBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');

    // Act
    const result = await verifyWebhookSignature(payload, signature, secret);

    // Assert
    expect(result).toBe(true);
  });

  it('無効な署名の場合falseを返す', async () => {
    const payload = '{"test": "data"}';
    const secret = 'test-secret';
    const invalidSignature = 'invalid-signature-here';

    // Act
    const result = await verifyWebhookSignature(payload, invalidSignature, secret);

    // Assert
    expect(result).toBe(false);
  });

  it('空の署名の場合falseを返す', async () => {
    const payload = '{"test": "data"}';
    const secret = 'test-secret';

    // Act
    const result = await verifyWebhookSignature(payload, '', secret);

    // Assert
    expect(result).toBe(false);
  });
});

describe('handleWebhook - 未知のイベント', () => {
  it('未知のイベントは無視する', async () => {
    // Arrange
    const { db, cache } = createMockDb();

    const webhookPayload: WebhookPayload = {
      meta: { event_name: 'unknown_event' },
      data: { attributes: { license_key: 'SOME-KEY' } }
    };

    // Act - エラーなく完了する
    await expect(handleWebhook(webhookPayload, db as unknown as D1Database))
      .resolves.not.toThrow();
  });
});
