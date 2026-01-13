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

-- 事例データ
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY,
  entity_name TEXT NOT NULL,
  before_trigram TEXT NOT NULL,   -- 八卦（乾/坤/震/巽/坎/離/艮/兌）
  after_trigram TEXT NOT NULL,    -- 八卦
  pattern_type TEXT NOT NULL,     -- Expansion_Growth, Crisis_Recovery, etc.
  scale TEXT NOT NULL,            -- individual/family/company/nation/global
  main_domain TEXT,               -- technology, retail, finance, etc.
  year INTEGER,
  summary TEXT,
  source_url TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- レート制限用（IP/キー単位）
CREATE TABLE IF NOT EXISTS rate_limits (
  id INTEGER PRIMARY KEY,
  identifier TEXT UNIQUE NOT NULL,  -- IP or key_hash
  request_count INTEGER DEFAULT 0,
  window_start TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- インデックス
CREATE INDEX IF NOT EXISTS idx_license_key_hash ON license_cache(key_hash);
CREATE INDEX IF NOT EXISTS idx_license_status ON license_cache(status);
CREATE INDEX IF NOT EXISTS idx_cases_pattern ON cases(pattern_type);
CREATE INDEX IF NOT EXISTS idx_cases_trigram ON cases(before_trigram, after_trigram);
CREATE INDEX IF NOT EXISTS idx_cases_scale ON cases(scale);
CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier);
