/**
 * Phase 3: レート制限のテスト（TDD）
 *
 * テスト対象: checkRateLimit関数
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect } from 'vitest';
import {
  checkRateLimit,
  type RateLimitConfig,
  type RateLimitResult
} from '../src/ratelimit';

// Mock D1Database型
interface MockStatement {
  bind: (...args: unknown[]) => MockStatement;
  first: <T>() => Promise<T | null>;
  run: () => Promise<{ success: boolean }>;
}

interface MockD1Database {
  prepare: (sql: string) => MockStatement;
}

interface RateLimitRecord {
  identifier: string;
  request_count: number;
  window_start: string;
}

// テスト用のモックD1データベースを作成
function createMockDb(): { db: MockD1Database; records: Map<string, RateLimitRecord> } {
  const records = new Map<string, RateLimitRecord>();

  const db: MockD1Database = {
    prepare: (sql: string) => {
      const statement: MockStatement = {
        bind: (...args: unknown[]) => {
          (statement as unknown as { boundArgs: unknown[] }).boundArgs = args;
          return statement;
        },
        first: async <T>() => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];

          if (sql.includes('SELECT') && sql.includes('rate_limits')) {
            const identifier = args[0] as string;
            const record = records.get(identifier);
            if (record) {
              // コピーを返す（オブジェクト参照の問題を回避）
              return { ...record } as T;
            }
          }
          return null;
        },
        run: async () => {
          const args = (statement as unknown as { boundArgs: unknown[] }).boundArgs || [];

          // INSERT クエリ
          if (sql.includes('INSERT') && sql.includes('rate_limits')) {
            const identifier = args[0] as string;
            records.set(identifier, {
              identifier,
              request_count: 1,
              window_start: new Date().toISOString()
            });
          }

          // UPDATE クエリ - カウント増加
          if (sql.includes('UPDATE') && sql.includes('request_count')) {
            const identifier = args[0] as string;
            const existing = records.get(identifier);
            if (existing) {
              existing.request_count += 1;
            }
          }

          return { success: true };
        }
      };
      return statement;
    }
  };

  return { db, records };
}

describe('checkRateLimit', () => {
  it('制限内のリクエストは許可', async () => {
    // Arrange
    const { db } = createMockDb();
    const ip = '192.168.1.1';
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // Act
    const result = await checkRateLimit(db as unknown as D1Database, ip, config);

    // Assert
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(99);
  });

  it('制限超過時はブロック', async () => {
    // Arrange
    const { db, records } = createMockDb();
    const ip = '192.168.1.1';
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // 100回リクエスト済みを模擬（現在時刻のウィンドウ）
    records.set(ip, {
      identifier: ip,
      request_count: 100,
      window_start: new Date().toISOString()
    });

    // Act
    const result = await checkRateLimit(db as unknown as D1Database, ip, config);

    // Assert
    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeGreaterThan(0);
  });

  it('ウィンドウ期限切れ後はカウントリセット', async () => {
    // Arrange
    const { db, records } = createMockDb();
    const ip = '192.168.1.1';
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // 2分前のウィンドウで100回（1分ウィンドウなら期限切れ）
    const twoMinutesAgo = new Date(Date.now() - 2 * 60 * 1000).toISOString();
    records.set(ip, {
      identifier: ip,
      request_count: 100,
      window_start: twoMinutesAgo
    });

    // Act
    const result = await checkRateLimit(db as unknown as D1Database, ip, config);

    // Assert
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(99);
  });

  it('異なるIDは独立してカウント', async () => {
    // Arrange
    const { db, records } = createMockDb();
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // IP1は制限到達
    records.set('192.168.1.1', {
      identifier: '192.168.1.1',
      request_count: 100,
      window_start: new Date().toISOString()
    });

    // Act - 別のIPでリクエスト
    const result = await checkRateLimit(db as unknown as D1Database, '192.168.1.2', config);

    // Assert
    expect(result.allowed).toBe(true);
  });

  it('残り回数を正確に返す', async () => {
    // Arrange
    const { db, records } = createMockDb();
    const ip = '192.168.1.1';
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // 50回リクエスト済み
    records.set(ip, {
      identifier: ip,
      request_count: 50,
      window_start: new Date().toISOString()
    });

    // Act
    const result = await checkRateLimit(db as unknown as D1Database, ip, config);

    // Assert
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(49); // 50 + 1(今回) = 51, 100 - 51 = 49
  });

  it('制限超過時のretryAfterはウィンドウ終了までの秒数', async () => {
    // Arrange
    const { db, records } = createMockDb();
    const ip = '192.168.1.1';
    const config: RateLimitConfig = { maxRequests: 100, windowSeconds: 60 };

    // 30秒前に開始したウィンドウで制限到達
    const thirtySecondsAgo = new Date(Date.now() - 30 * 1000).toISOString();
    records.set(ip, {
      identifier: ip,
      request_count: 100,
      window_start: thirtySecondsAgo
    });

    // Act
    const result = await checkRateLimit(db as unknown as D1Database, ip, config);

    // Assert
    expect(result.allowed).toBe(false);
    // retryAfterは約30秒（ウィンドウ60秒 - 経過30秒）
    expect(result.retryAfter).toBeGreaterThanOrEqual(25);
    expect(result.retryAfter).toBeLessThanOrEqual(35);
  });
});
