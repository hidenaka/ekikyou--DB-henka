/**
 * 事例検索モジュール
 *
 * D1データベースから事例を検索
 */

// 八卦型
type Trigram = '乾' | '坤' | '震' | '巽' | '坎' | '離' | '艮' | '兌';

// スケール型
type Scale = 'individual' | 'family' | 'company' | 'nation' | 'global';

// パターンタイプ型
type PatternType =
  | 'Expansion_Growth'
  | 'Contraction_Decline'
  | 'Transformation_Shift'
  | 'Stability_Maintenance'
  | 'Crisis_Recovery'
  | 'Shock_Recovery'
  | 'Emergence_Innovation';

// 事例型
export interface Case {
  id: string;
  entity_name: string;
  before_trigram: Trigram;
  after_trigram: Trigram;
  pattern_type: PatternType;
  scale: Scale;
  main_domain: string;
  year: number;
  summary: string;
  source_url?: string;
}

// 検索パラメータ型
export interface CaseSearchParams {
  pattern_type?: PatternType;
  before_trigram?: Trigram;
  after_trigram?: Trigram;
  scale?: Scale;
  main_domain?: string;
  year_from?: number;
  year_to?: number;
  limit?: number;
  offset?: number;
}

// 検索結果型
export interface CaseSearchResult {
  cases: Case[];
  total: number;
  limit: number;
  offset: number;
}

// デフォルト値
const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 20;

/**
 * 事例を検索
 *
 * @param db - D1データベース
 * @param params - 検索パラメータ
 * @returns 検索結果
 */
export async function searchCases(
  db: D1Database,
  params: CaseSearchParams
): Promise<CaseSearchResult> {
  const conditions: string[] = [];
  const bindings: unknown[] = [];

  // パターンタイプ
  if (params.pattern_type) {
    conditions.push('pattern_type = ?');
    bindings.push(params.pattern_type);
  }

  // 八卦（before）
  if (params.before_trigram) {
    conditions.push('before_trigram = ?');
    bindings.push(params.before_trigram);
  }

  // 八卦（after）
  if (params.after_trigram) {
    conditions.push('after_trigram = ?');
    bindings.push(params.after_trigram);
  }

  // スケール
  if (params.scale) {
    conditions.push('scale = ?');
    bindings.push(params.scale);
  }

  // メインドメイン
  if (params.main_domain) {
    conditions.push('main_domain = ?');
    bindings.push(params.main_domain);
  }

  // 年（from）
  if (params.year_from) {
    conditions.push('year >= ?');
    bindings.push(params.year_from);
  }

  // 年（to）
  if (params.year_to) {
    conditions.push('year <= ?');
    bindings.push(params.year_to);
  }

  // SQLクエリ構築
  const whereClause = conditions.length > 0
    ? `WHERE ${conditions.join(' AND ')}`
    : '';

  const limit = Math.min(params.limit || DEFAULT_LIMIT, MAX_LIMIT);
  const offset = params.offset || 0;

  const sql = `
    SELECT
      id,
      entity_name,
      before_trigram,
      after_trigram,
      pattern_type,
      scale,
      main_domain,
      year,
      summary,
      source_url
    FROM cases
    ${whereClause}
    ORDER BY year DESC, id
    LIMIT ?
    OFFSET ?
  `;

  bindings.push(limit, offset);

  const result = await db.prepare(sql).bind(...bindings).all<Case>();

  return {
    cases: result.results || [],
    total: result.results?.length || 0,
    limit,
    offset
  };
}

/**
 * 類似事例を検索
 *
 * 同じ八卦の組み合わせを持つ事例を検索
 *
 * @param db - D1データベース
 * @param beforeTrigram - 前の八卦
 * @param afterTrigram - 後の八卦
 * @param limit - 最大件数
 * @returns 類似事例リスト
 */
export async function findSimilarCases(
  db: D1Database,
  beforeTrigram: Trigram,
  afterTrigram: Trigram,
  limit: number = 5
): Promise<Case[]> {
  const result = await searchCases(db, {
    before_trigram: beforeTrigram,
    after_trigram: afterTrigram,
    limit: Math.min(limit, MAX_LIMIT)
  });

  return result.cases;
}

/**
 * パターンタイプ別の統計を取得
 *
 * @param db - D1データベース
 * @returns パターンタイプごとの件数
 */
export async function getPatternTypeStats(
  db: D1Database
): Promise<Record<string, number>> {
  const sql = `
    SELECT pattern_type, COUNT(*) as count
    FROM cases
    GROUP BY pattern_type
    ORDER BY count DESC
  `;

  const result = await db.prepare(sql).all<{ pattern_type: string; count: number }>();

  const stats: Record<string, number> = {};
  for (const row of result.results || []) {
    stats[row.pattern_type] = row.count;
  }

  return stats;
}
