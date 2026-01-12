# HaQei 有料API実装指示書（Claude Code向け）

## 実装目標

Cloudflare Workers + D1 を使用した **有料診断API** の構築。
**MoRのライセンスキー機能**を活用し、自前でのキー発行管理を避ける。

---

## ⚠️ 重要: 設計方針

### ❌ やらないこと（落とし穴回避）
- 自前でAPIキーを生成・メール送付（サポート負荷が高い）
- クライアントからMoR検証APIを直叩き（キー露出リスク）
- 毎リクエストでMoR検証（レート制限・遅延）

### ✅ やること（推奨方式）
- **MoR（Lemon Squeezy推奨）のライセンスキー機能**を使う
- Workers経由でMoR検証APIを叩く
- D1でキー検証結果を**キャッシュ**
- Webhookで返金/解約を**即時失効**

---

## アーキテクチャ

```
ユーザー → Lemon Squeezy購入 → ライセンスキー自動発行
                ↓
         Webフロント（キー入力フォーム）
                ↓ Authorization: Bearer <license_key>
         Cloudflare Workers
                ↓ D1キャッシュ確認
           ┌───┴───┐
      キャッシュHit  Miss→Lemon Squeezy検証API
           ↓             ↓
        応答         D1に結果保存→応答
```

---

## プラットフォーム制限（正確な値）

### Cloudflare

| リソース | 無料枠制限 |
|----------|-----------|
| Workers | **100,000リクエスト/日**（UTC基準、全Worker合算） |
| D1 | **500MB/DB**、アカウント合計5GB |
| Pages Functions | Workers枠に含まれる（静的は別） |

### 課金プラットフォーム

| サービス | 手数料 | 備考 |
|----------|--------|------|
| Lemon Squeezy | **5% + $0.50** | 追加機能で別料金あり |
| Gumroad | **10% + $0.50 + 決済処理手数料** | 例: +2.9%+$0.30 |

**推奨: Lemon Squeezy**（手数料低、Webhook/License API整備済）

---

## D1 スキーマ設計

```sql
-- ライセンスキーのキャッシュ（検証結果を保持）
CREATE TABLE license_cache (
  id INTEGER PRIMARY KEY,
  key_hash TEXT UNIQUE NOT NULL,  -- SHA-256ハッシュ（生キーは保存しない）
  status TEXT NOT NULL,           -- 'active' | 'inactive' | 'expired'
  plan TEXT DEFAULT 'basic',      -- 'basic' | 'pro' など
  expires_at TEXT,                -- ISO8601 or NULL（無期限）
  usage_count INTEGER DEFAULT 0,
  last_verified_at TEXT,          -- 最終検証日時
  cache_ttl_seconds INTEGER DEFAULT 3600,  -- キャッシュ有効期間
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 事例データ（cases.jsonlインポート）
CREATE TABLE cases (
  id INTEGER PRIMARY KEY,
  transition_id TEXT UNIQUE,
  target_name TEXT,
  scale TEXT,
  before_state TEXT,
  trigger_type TEXT,
  action_type TEXT,
  after_state TEXT,
  before_hex TEXT,
  trigger_hex TEXT,
  action_hex TEXT,
  after_hex TEXT,
  pattern_type TEXT,
  outcome TEXT,
  story_summary TEXT,
  yao_analysis TEXT,  -- JSON文字列
  full_data TEXT      -- 元のJSON全体
);

-- レート制限用（IP/キー単位）
CREATE TABLE rate_limits (
  id INTEGER PRIMARY KEY,
  identifier TEXT UNIQUE,  -- IP or key_hash
  request_count INTEGER DEFAULT 0,
  window_start TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_license_key_hash ON license_cache(key_hash);
CREATE INDEX idx_license_status ON license_cache(status);
CREATE INDEX idx_cases_pattern ON cases(pattern_type);
CREATE INDEX idx_cases_hex ON cases(before_hex, after_hex);
```

---

## Lemon Squeezy License API 連携

### 検証フロー（Workers側）

```typescript
// src/license.ts
import { sha256 } from './utils';

interface LicenseValidation {
  valid: boolean;
  status: 'active' | 'inactive' | 'expired';
  plan?: string;
  expiresAt?: string;
}

export async function validateLicense(
  licenseKey: string,
  db: D1Database
): Promise<LicenseValidation> {
  const keyHash = await sha256(licenseKey);
  
  // 1. キャッシュ確認
  const cached = await db.prepare(
    `SELECT * FROM license_cache 
     WHERE key_hash = ? 
     AND status = 'active'
     AND datetime(last_verified_at, '+' || cache_ttl_seconds || ' seconds') > datetime('now')`
  ).bind(keyHash).first();
  
  if (cached) {
    return {
      valid: true,
      status: cached.status as 'active',
      plan: cached.plan as string,
      expiresAt: cached.expires_at as string
    };
  }
  
  // 2. Lemon Squeezy API検証
  const lsResult = await callLemonSqueezyValidate(licenseKey);
  
  // 3. キャッシュ更新
  await db.prepare(
    `INSERT OR REPLACE INTO license_cache 
     (key_hash, status, plan, expires_at, last_verified_at, updated_at)
     VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))`
  ).bind(
    keyHash,
    lsResult.valid ? 'active' : 'inactive',
    lsResult.meta?.variant_name || 'basic',
    lsResult.license_key?.expires_at || null
  ).run();
  
  return {
    valid: lsResult.valid,
    status: lsResult.valid ? 'active' : 'inactive',
    plan: lsResult.meta?.variant_name,
    expiresAt: lsResult.license_key?.expires_at
  };
}

async function callLemonSqueezyValidate(key: string) {
  // https://docs.lemonsqueezy.com/api/license-api
  const response = await fetch('https://api.lemonsqueezy.com/v1/licenses/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ license_key: key })
  });
  return response.json();
}
```

### Webhook受信（返金/解約の即時失効）

```typescript
// src/webhook.ts
export async function handleLemonSqueezyWebhook(
  request: Request,
  db: D1Database,
  webhookSecret: string
): Promise<Response> {
  // 署名検証
  const signature = request.headers.get('X-Signature');
  // ... 署名検証ロジック（docs.lemonsqueezy.com参照）
  
  const body = await request.json();
  const eventType = body.meta.event_name;
  
  // 失効イベント
  if (['subscription_cancelled', 'order_refunded', 'license_key_deactivated'].includes(eventType)) {
    const licenseKey = body.data.attributes.license_key;
    const keyHash = await sha256(licenseKey);
    
    await db.prepare(
      `UPDATE license_cache SET status = 'inactive', updated_at = datetime('now') WHERE key_hash = ?`
    ).bind(keyHash).run();
  }
  
  return new Response('OK', { status: 200 });
}
```

---

## API エンドポイント設計

| エンドポイント | 認証 | 機能 |
|--------------|------|------|
| `GET /health` | 不要 | ヘルスチェック |
| `POST /diagnose/preview` | 不要 | 簡易診断（結果一部、無料） |
| `POST /diagnose/full` | ライセンスキー | 詳細診断 |
| `POST /analyze/path` | ライセンスキー | 経路分析 |
| `GET /cases/search` | ライセンスキー | 類似事例検索 |
| `POST /webhook/lemonsqueezy` | 署名検証 | 返金/解約の即時失効 |

---

## セキュリティ ガードレール（必須）

### 1. レート制限
```typescript
// IP単位: 100リクエスト/分
// キー単位: 1000リクエスト/日
```

### 2. CORS設定
```typescript
const ALLOWED_ORIGINS = [
  'https://haqei.com',
  'http://localhost:3000' // 開発用
];
```

### 3. キー総当たり対策
- 無効キーが5回連続 → 15分ブロック
- ログに警告出力

---

## データソース

### 変化ロジックDB
- **場所**: `/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/`
- **主データ**: `data/raw/cases.jsonl`（約12,400件、42MB → **500MB制限内OK**）
- **スキーマ**: `docs/schema_v3.md`
- **診断ロジック**: 
  - `data/diagnostic/question_mapping.json`
  - `data/diagnostic/yao_384.json`
  - `data/diagnostic/hexagram_64.json`

### 移植対象
- `scripts/diagnostic_engine.py` → `src/diagnosis.ts`

---

## 成果物チェックリスト

- [ ] Cloudflare Pages + Workers プロジェクト初期化
- [ ] D1スキーマ作成（license_cache, cases, rate_limits）
- [ ] cases.jsonl → D1 インポートスクリプト
- [ ] 診断エンジン TypeScript版
- [ ] ライセンス検証ミドルウェア（Lemon Squeezy API連携）
- [ ] 検証結果キャッシュ（D1）
- [ ] Webhook受信（返金/解約→即時失効）
- [ ] レート制限実装
- [ ] CORS設定
- [ ] ローカルテスト（wrangler dev）
- [ ] 本番デプロイ

---

## Option B へ移行するトリガー（将来用メモ）

以下が発生したらSupabase + Magic Link認証へ移行検討：
- 1ユーザーが複数端末で使う要件
- "キー入力"のUXが離脱要因に
- キー紛失サポートが週次で発生

---

## 参照ドキュメント

- [Lemon Squeezy License API](https://docs.lemonsqueezy.com/api/license-api)
- [Lemon Squeezy Webhooks](https://docs.lemonsqueezy.com/help/webhooks)
- [Cloudflare D1 Limits](https://developers.cloudflare.com/d1/platform/limits/)
- [Cloudflare Workers Pricing](https://www.cloudflare.com/plans/developer-platform/)
