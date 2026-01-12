/**
 * Phase 1: ユーティリティ関数のテスト（TDD）
 *
 * テスト対象: sha256ハッシュ関数
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect } from 'vitest';
import { sha256 } from '../src/utils';

describe('sha256', () => {
  it('同じ入力に対して同じハッシュを返す', async () => {
    const key = 'TEST-LICENSE-KEY-12345';
    const hash1 = await sha256(key);
    const hash2 = await sha256(key);
    expect(hash1).toBe(hash2);
  });

  it('異なる入力に対して異なるハッシュを返す', async () => {
    const hash1 = await sha256('key-1');
    const hash2 = await sha256('key-2');
    expect(hash1).not.toBe(hash2);
  });

  it('64文字の16進数文字列を返す', async () => {
    const hash = await sha256('any-key');
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('空文字列でもハッシュを返す', async () => {
    const hash = await sha256('');
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('日本語を含む文字列でもハッシュを返す', async () => {
    const hash = await sha256('テスト-ライセンス-キー');
    expect(hash).toMatch(/^[a-f0-9]{64}$/);
  });
});
