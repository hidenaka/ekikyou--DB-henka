-- HaQei API D1 Schema
-- ライセンスキーのキャッシュ（検証結果を保持）
CREATE TABLE IF NOT EXISTS license_cache (
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
CREATE TABLE IF NOT EXISTS cases (
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
CREATE TABLE IF NOT EXISTS rate_limits (
  id INTEGER PRIMARY KEY,
  identifier TEXT UNIQUE,  -- IP or key_hash
  request_count INTEGER DEFAULT 0,
  window_start TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_license_key_hash ON license_cache(key_hash);
CREATE INDEX IF NOT EXISTS idx_license_status ON license_cache(status);
CREATE INDEX IF NOT EXISTS idx_cases_pattern ON cases(pattern_type);
CREATE INDEX IF NOT EXISTS idx_cases_hex ON cases(before_hex, after_hex);
CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier);
