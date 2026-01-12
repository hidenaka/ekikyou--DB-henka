/**
 * Phase 1: ライセンス検証のテスト（TDD）
 *
 * テスト対象: validateLicense関数
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { sha256 } from '../src/utils';
import { validateLicense, extractLicenseKey, type LemonSqueezyApiClient } from '../src/license';

// Mock D1Database型
interface MockStatement {
  bind: (...args: unknown[]) => MockStatement;
  first: <T>() => Promise<T | null>;
  run: () => Promise<{ success: boolean }>;
}

interface MockD1Database {
  prepare: (sql: string) => MockStatement;
  exec: (sql: string) => Promise<void>;
}

// テスト用のモックD1データベースを作成
function createMockDb(): { db: MockD1Database; cache: Map<string, unknown> } {
  const cache = new Map<string, unknown>();

  const db: MockD1Database = {
    prepare: (sql: string) => {
      const statement: MockStatement = {
        bind: (...args: unknown[]) => {
          // bind時にキーハッシュを保存
          (statement as unknown as { boundArgs: unknown[] }).boundArgs = args;
          return statement;
        },
        first: async <T>() => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];
          const keyHash = args[0] as string;

          // SELECT クエリ - キャッシュから取得
          if (sql.includes('SELECT') && sql.includes('license_cache')) {
            return cache.get(keyHash) as T | null;
          }
          return null;
        },
        run: async () => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];

          // INSERT クエリ - キャッシュに保存
          if (sql.includes('INSERT') && sql.includes('license_cache')) {
            const [keyHash, status, plan, expiresAt] = args as [string, string, string, string | null];
            cache.set(keyHash, {
              key_hash: keyHash,
              status,
              plan,
              expires_at: expiresAt,
              last_verified_at: new Date().toISOString(),
              cache_ttl_seconds: 3600
            });
          }
          return { success: true };
        }
      };
      return statement;
    },
    exec: async () => {}
  };

  return { db, cache };
}

describe('validateLicense - キャッシュヒット', () => {
  it('有効なキャッシュが存在する場合、MoR APIを呼ばずに結果を返す', async () => {
    // Arrange
    const licenseKey = 'VALID-KEY-123';
    const keyHash = await sha256(licenseKey);
    const { db, cache } = createMockDb();

    // キャッシュにデータを設定（有効期限内）
    cache.set(keyHash, {
      key_hash: keyHash,
      status: 'active',
      plan: 'pro',
      expires_at: null,
      last_verified_at: new Date().toISOString(),
      cache_ttl_seconds: 3600
    });

    const lsApiMock = vi.fn();

    // Act
    const result = await validateLicense(licenseKey, db as unknown as D1Database, lsApiMock);

    // Assert
    expect(result.valid).toBe(true);
    expect(result.status).toBe('active');
    expect(result.plan).toBe('pro');
    expect(lsApiMock).not.toHaveBeenCalled(); // APIは呼ばれない
  });
});

describe('validateLicense - キャッシュミス', () => {
  it('キャッシュ未登録の場合、MoR APIを呼んでキャッシュに保存', async () => {
    // Arrange
    const licenseKey = 'NEW-KEY-456';
    const { db, cache } = createMockDb();

    const lsApiMock = vi.fn<LemonSqueezyApiClient>().mockResolvedValue({
      valid: true,
      license_key: { status: 'active', expires_at: '2026-12-31' },
      meta: { variant_name: 'basic' }
    });

    // Act
    const result = await validateLicense(licenseKey, db as unknown as D1Database, lsApiMock);

    // Assert
    expect(result.valid).toBe(true);
    expect(result.plan).toBe('basic');
    expect(lsApiMock).toHaveBeenCalledWith(licenseKey);

    // キャッシュに保存されている
    const keyHash = await sha256(licenseKey);
    expect(cache.has(keyHash)).toBe(true);
  });

  it('無効なキーの場合、inactiveとしてキャッシュ', async () => {
    // Arrange
    const licenseKey = 'INVALID-KEY';
    const { db } = createMockDb();

    const lsApiMock = vi.fn<LemonSqueezyApiClient>().mockResolvedValue({
      valid: false,
      error: 'license_key_not_found'
    });

    // Act
    const result = await validateLicense(licenseKey, db as unknown as D1Database, lsApiMock);

    // Assert
    expect(result.valid).toBe(false);
    expect(result.status).toBe('inactive');
  });
});

describe('validateLicense - エラーハンドリング', () => {
  it('MoR APIがエラーを返した場合、適切に処理する', async () => {
    // Arrange
    const licenseKey = 'ERROR-KEY';
    const { db } = createMockDb();

    const lsApiMock = vi.fn<LemonSqueezyApiClient>().mockRejectedValue(
      new Error('Network error')
    );

    // Act & Assert
    await expect(validateLicense(licenseKey, db as unknown as D1Database, lsApiMock))
      .rejects.toThrow('Network error');
  });
});

describe('extractLicenseKey', () => {
  it('Bearer トークンからライセンスキーを抽出', () => {
    expect(extractLicenseKey('Bearer MY-LICENSE-KEY')).toBe('MY-LICENSE-KEY');
  });

  it('大文字小文字を区別せずにBearerを認識', () => {
    expect(extractLicenseKey('bearer my-key')).toBe('my-key');
    expect(extractLicenseKey('BEARER MY-KEY')).toBe('MY-KEY');
  });

  it('無効なヘッダーの場合nullを返す', () => {
    expect(extractLicenseKey(null)).toBeNull();
    expect(extractLicenseKey('')).toBeNull();
    expect(extractLicenseKey('Basic base64string')).toBeNull();
  });
});
