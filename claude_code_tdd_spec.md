# HaQei 有料API 実装仕様書（TDD方式）

## 開発方針

**テストファースト**で実装する。各機能について：
1. テストケースを先に書く
2. テストが失敗することを確認
3. 最小限の実装でテストを通す
4. リファクタリング

---

## 前提条件

### 技術スタック
- Cloudflare Workers + Pages
- D1 (SQLite)
- Vitest（テストフレームワーク）
- Lemon Squeezy License API

### 制約
- D1: 500MB/DB
- Workers: 100,000リクエスト/日
- ライセンスキーは生で保存しない（SHA-256ハッシュ）

---

# Phase 1: ライセンス検証

## テストケース

### 1.1 ライセンスキーのハッシュ化

```typescript
// tests/utils.test.ts
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
});
```

### 1.2 キャッシュヒット時の検証

```typescript
// tests/license.test.ts
describe('validateLicense - キャッシュヒット', () => {
  it('有効なキャッシュが存在する場合、MoR APIを呼ばずに結果を返す', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'VALID-KEY-123';
    const keyHash = await sha256(licenseKey);
    
    // 30分前に検証済み、TTL=1時間
    await db.prepare(`
      INSERT INTO license_cache (key_hash, status, plan, last_verified_at, cache_ttl_seconds)
      VALUES (?, 'active', 'pro', datetime('now', '-30 minutes'), 3600)
    `).bind(keyHash).run();
    
    const lsApiMock = vi.fn();
    
    // Act
    const result = await validateLicense(licenseKey, db, lsApiMock);
    
    // Assert
    expect(result.valid).toBe(true);
    expect(result.status).toBe('active');
    expect(result.plan).toBe('pro');
    expect(lsApiMock).not.toHaveBeenCalled(); // APIは呼ばれない
  });

  it('キャッシュが期限切れの場合、MoR APIを呼ぶ', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'VALID-KEY-123';
    const keyHash = await sha256(licenseKey);
    
    // 2時間前に検証済み、TTL=1時間 → 期限切れ
    await db.prepare(`
      INSERT INTO license_cache (key_hash, status, plan, last_verified_at, cache_ttl_seconds)
      VALUES (?, 'active', 'pro', datetime('now', '-2 hours'), 3600)
    `).bind(keyHash).run();
    
    const lsApiMock = vi.fn().mockResolvedValue({
      valid: true,
      license_key: { status: 'active' },
      meta: { variant_name: 'pro' }
    });
    
    // Act
    const result = await validateLicense(licenseKey, db, lsApiMock);
    
    // Assert
    expect(lsApiMock).toHaveBeenCalled();
  });
});
```

### 1.3 キャッシュミス時の検証

```typescript
describe('validateLicense - キャッシュミス', () => {
  it('キャッシュ未登録の場合、MoR APIを呼んでキャッシュに保存', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'NEW-KEY-456';
    
    const lsApiMock = vi.fn().mockResolvedValue({
      valid: true,
      license_key: { status: 'active', expires_at: '2026-12-31' },
      meta: { variant_name: 'basic' }
    });
    
    // Act
    const result = await validateLicense(licenseKey, db, lsApiMock);
    
    // Assert
    expect(result.valid).toBe(true);
    expect(result.plan).toBe('basic');
    
    // キャッシュに保存されている
    const cached = await db.prepare(
      `SELECT * FROM license_cache WHERE key_hash = ?`
    ).bind(await sha256(licenseKey)).first();
    expect(cached).not.toBeNull();
    expect(cached.status).toBe('active');
  });

  it('無効なキーの場合、inactiveとしてキャッシュ', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'INVALID-KEY';
    
    const lsApiMock = vi.fn().mockResolvedValue({
      valid: false,
      error: 'license_key_not_found'
    });
    
    // Act
    const result = await validateLicense(licenseKey, db, lsApiMock);
    
    // Assert
    expect(result.valid).toBe(false);
    expect(result.status).toBe('inactive');
  });
});
```

---

# Phase 2: Webhook処理

## テストケース

### 2.1 返金イベント

```typescript
// tests/webhook.test.ts
describe('handleWebhook - order_refunded', () => {
  it('返金時にライセンスを即時失効', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'REFUNDED-KEY';
    const keyHash = await sha256(licenseKey);
    
    // 有効なキャッシュを事前登録
    await db.prepare(`
      INSERT INTO license_cache (key_hash, status, plan)
      VALUES (?, 'active', 'pro')
    `).bind(keyHash).run();
    
    const webhookPayload = {
      meta: { event_name: 'order_refunded' },
      data: { attributes: { license_key: licenseKey } }
    };
    
    // Act
    await handleWebhook(webhookPayload, db);
    
    // Assert
    const cached = await db.prepare(
      `SELECT status FROM license_cache WHERE key_hash = ?`
    ).bind(keyHash).first();
    expect(cached.status).toBe('inactive');
  });
});
```

### 2.2 サブスク解約イベント

```typescript
describe('handleWebhook - subscription_cancelled', () => {
  it('解約時にライセンスを失効', async () => {
    // Arrange
    const db = createMockD1();
    const licenseKey = 'CANCELLED-KEY';
    const keyHash = await sha256(licenseKey);
    
    await db.prepare(`
      INSERT INTO license_cache (key_hash, status, plan)
      VALUES (?, 'active', 'pro')
    `).bind(keyHash).run();
    
    const webhookPayload = {
      meta: { event_name: 'subscription_cancelled' },
      data: { attributes: { license_key: licenseKey } }
    };
    
    // Act
    await handleWebhook(webhookPayload, db);
    
    // Assert
    const cached = await db.prepare(
      `SELECT status FROM license_cache WHERE key_hash = ?`
    ).bind(keyHash).first();
    expect(cached.status).toBe('inactive');
  });
});
```

### 2.3 署名検証

```typescript
describe('handleWebhook - 署名検証', () => {
  it('無効な署名の場合は401を返す', async () => {
    // Arrange
    const request = new Request('https://api.example.com/webhook', {
      method: 'POST',
      headers: { 'X-Signature': 'invalid-signature' },
      body: JSON.stringify({ meta: { event_name: 'test' } })
    });
    
    // Act
    const response = await handleWebhookRequest(request, db, 'correct-secret');
    
    // Assert
    expect(response.status).toBe(401);
  });
});
```

---

# Phase 3: レート制限

## テストケース

```typescript
// tests/ratelimit.test.ts
describe('checkRateLimit', () => {
  it('制限内のリクエストは許可', async () => {
    // Arrange
    const db = createMockD1();
    const ip = '192.168.1.1';
    
    // Act
    const result = await checkRateLimit(db, ip, { maxRequests: 100, windowSeconds: 60 });
    
    // Assert
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(99);
  });

  it('制限超過時はブロック', async () => {
    // Arrange
    const db = createMockD1();
    const ip = '192.168.1.1';
    
    // 100回リクエスト済みを模擬
    await db.prepare(`
      INSERT INTO rate_limits (identifier, request_count, window_start)
      VALUES (?, 100, datetime('now'))
    `).bind(ip).run();
    
    // Act
    const result = await checkRateLimit(db, ip, { maxRequests: 100, windowSeconds: 60 });
    
    // Assert
    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeGreaterThan(0);
  });

  it('ウィンドウ期限切れ後はカウントリセット', async () => {
    // Arrange
    const db = createMockD1();
    const ip = '192.168.1.1';
    
    // 2分前のウィンドウで100回（1分ウィンドウなら期限切れ）
    await db.prepare(`
      INSERT INTO rate_limits (identifier, request_count, window_start)
      VALUES (?, 100, datetime('now', '-2 minutes'))
    `).bind(ip).run();
    
    // Act
    const result = await checkRateLimit(db, ip, { maxRequests: 100, windowSeconds: 60 });
    
    // Assert
    expect(result.allowed).toBe(true);
  });
});
```

---

# Phase 4: 診断API

## テストケース

```typescript
// tests/diagnose.test.ts
describe('POST /diagnose/preview', () => {
  it('認証なしでアクセス可能', async () => {
    // Act
    const response = await app.fetch(new Request('http://localhost/diagnose/preview', {
      method: 'POST',
      body: JSON.stringify({ answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] })
    }));
    
    // Assert
    expect(response.status).toBe(200);
  });

  it('結果は一部のみ返す（詳細はマスク）', async () => {
    // Act
    const response = await app.fetch(new Request('http://localhost/diagnose/preview', {
      method: 'POST',
      body: JSON.stringify({ answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] })
    }));
    const data = await response.json();
    
    // Assert
    expect(data.hexagram).toBeDefined();
    expect(data.summary).toBeDefined();
    expect(data.full_analysis).toBeUndefined(); // プレビューでは非公開
    expect(data.recommended_actions).toBeUndefined();
  });
});

describe('POST /diagnose/full', () => {
  it('有効なライセンスキーで全結果を返す', async () => {
    // Arrange
    // ... キャッシュにactiveキーを設定
    
    // Act
    const response = await app.fetch(new Request('http://localhost/diagnose/full', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer VALID-LICENSE-KEY' },
      body: JSON.stringify({ answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] })
    }));
    const data = await response.json();
    
    // Assert
    expect(response.status).toBe(200);
    expect(data.full_analysis).toBeDefined();
    expect(data.recommended_actions).toBeDefined();
    expect(data.similar_cases).toBeDefined();
  });

  it('無効なライセンスキーで401', async () => {
    // Act
    const response = await app.fetch(new Request('http://localhost/diagnose/full', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer INVALID-KEY' },
      body: JSON.stringify({ answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] })
    }));
    
    // Assert
    expect(response.status).toBe(401);
  });

  it('ライセンスキーなしで401', async () => {
    // Act
    const response = await app.fetch(new Request('http://localhost/diagnose/full', {
      method: 'POST',
      body: JSON.stringify({ answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] })
    }));
    
    // Assert
    expect(response.status).toBe(401);
  });
});
```

---

# Phase 5: 事例検索API

## テストケース

```typescript
// tests/cases.test.ts
describe('GET /cases/search', () => {
  it('パターンタイプで検索できる', async () => {
    // Act
    const response = await app.fetch(new Request(
      'http://localhost/cases/search?pattern_type=Shock_Recovery',
      { headers: { 'Authorization': 'Bearer VALID-KEY' } }
    ));
    const data = await response.json();
    
    // Assert
    expect(response.status).toBe(200);
    expect(data.cases).toBeInstanceOf(Array);
    expect(data.cases.every(c => c.pattern_type === 'Shock_Recovery')).toBe(true);
  });

  it('八卦の組み合わせで検索できる', async () => {
    // Act
    const response = await app.fetch(new Request(
      'http://localhost/cases/search?before_hex=坎&after_hex=乾',
      { headers: { 'Authorization': 'Bearer VALID-KEY' } }
    ));
    const data = await response.json();
    
    // Assert
    expect(data.cases.every(c => c.before_hex === '坎' && c.after_hex === '乾')).toBe(true);
  });

  it('結果は最大20件まで', async () => {
    // Act
    const response = await app.fetch(new Request(
      'http://localhost/cases/search?scale=company',
      { headers: { 'Authorization': 'Bearer VALID-KEY' } }
    ));
    const data = await response.json();
    
    // Assert
    expect(data.cases.length).toBeLessThanOrEqual(20);
  });
});
```

---

# 実装順序

```
1. プロジェクト初期化 + Vitest設定
   └── npm create cloudflare + vitest

2. Phase 1: ライセンス検証
   ├── tests/utils.test.ts → src/utils.ts (sha256)
   ├── tests/license.test.ts → src/license.ts
   └── D1スキーマ (license_cache)

3. Phase 2: Webhook処理
   ├── tests/webhook.test.ts → src/webhook.ts
   └── 署名検証ロジック

4. Phase 3: レート制限
   ├── tests/ratelimit.test.ts → src/ratelimit.ts
   └── D1スキーマ (rate_limits)

5. Phase 4: 診断API
   ├── tests/diagnose.test.ts → src/diagnose.ts
   ├── diagnostic_engine.py → TypeScript移植
   └── 質問マッピングJSON読み込み

6. Phase 5: 事例検索
   ├── tests/cases.test.ts → src/cases.ts
   ├── D1スキーマ (cases)
   └── cases.jsonlインポート

7. 統合 + デプロイ
   ├── wrangler.toml設定
   ├── wrangler dev でローカルテスト
   └── wrangler deploy
```

---

# 成功基準

- [ ] 全テストがGreen
- [ ] `wrangler dev` でローカル動作確認
- [ ] Lemon Squeezy Webhook受信テスト（ngrok等）
- [ ] 本番デプロイ + 疎通確認
