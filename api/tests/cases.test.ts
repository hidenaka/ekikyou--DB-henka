/**
 * Phase 5: 事例検索APIのテスト（TDD）
 *
 * テスト対象: searchCases関数
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect } from 'vitest';
import {
  searchCases,
  type CaseSearchParams,
  type Case
} from '../src/cases';

// Mock D1Database型
interface MockStatement {
  bind: (...args: unknown[]) => MockStatement;
  all: <T>() => Promise<{ results: T[] }>;
}

interface MockD1Database {
  prepare: (sql: string) => MockStatement;
}

// テスト用のモック事例データ
const mockCases: Case[] = [
  {
    id: '1',
    entity_name: 'テスト企業A',
    before_trigram: '坎',
    after_trigram: '乾',
    pattern_type: 'Shock_Recovery',
    scale: 'company',
    main_domain: 'technology',
    year: 2020,
    summary: 'V字回復の事例'
  },
  {
    id: '2',
    entity_name: 'テスト企業B',
    before_trigram: '乾',
    after_trigram: '兌',
    pattern_type: 'Expansion_Growth',
    scale: 'company',
    main_domain: 'retail',
    year: 2021,
    summary: '急成長の事例'
  },
  {
    id: '3',
    entity_name: 'テスト個人C',
    before_trigram: '坤',
    after_trigram: '震',
    pattern_type: 'Emergence_Innovation',
    scale: 'individual',
    main_domain: 'entertainment',
    year: 2022,
    summary: 'ブレイクスルーの事例'
  },
  {
    id: '4',
    entity_name: 'テスト企業D',
    before_trigram: '坎',
    after_trigram: '乾',
    pattern_type: 'Shock_Recovery',
    scale: 'company',
    main_domain: 'finance',
    year: 2019,
    summary: '危機からの復活'
  }
];

// テスト用のモックD1データベースを作成
function createMockDb(cases: Case[]): MockD1Database {
  const db: MockD1Database = {
    prepare: (sql: string) => {
      const statement: MockStatement = {
        bind: (...args: unknown[]) => {
          (statement as unknown as { boundArgs: unknown[]; sql: string }).boundArgs = args;
          (statement as unknown as { boundArgs: unknown[]; sql: string }).sql = sql;
          return statement;
        },
        all: async <T>() => {
          const ctx = statement as unknown as { boundArgs: unknown[]; sql: string };
          const args = ctx.boundArgs || [];
          const sqlLower = ctx.sql.toLowerCase();

          let filtered = [...cases];

          // パターンタイプでフィルター
          if (sqlLower.includes('pattern_type')) {
            const patternArg = args.find(a => typeof a === 'string' && mockCases.some(c => c.pattern_type === a));
            if (patternArg) {
              filtered = filtered.filter(c => c.pattern_type === patternArg);
            }
          }

          // 八卦でフィルター
          if (sqlLower.includes('before_trigram') && !sqlLower.includes('?')) {
            // SQL内に直接値が含まれている場合の処理（簡易版）
          }
          const beforeArg = args.find((a, i) => {
            const validTrigrams = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌'];
            return typeof a === 'string' && validTrigrams.includes(a) && i === 0;
          });
          const afterArg = args.find((a, i) => {
            const validTrigrams = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌'];
            return typeof a === 'string' && validTrigrams.includes(a) && i === 1;
          });

          if (beforeArg && afterArg) {
            filtered = filtered.filter(c => c.before_trigram === beforeArg && c.after_trigram === afterArg);
          }

          // scaleでフィルター
          const scaleArg = args.find(a => a === 'company' || a === 'individual' || a === 'family' || a === 'nation');
          if (scaleArg) {
            filtered = filtered.filter(c => c.scale === scaleArg);
          }

          // 最大20件
          filtered = filtered.slice(0, 20);

          return { results: filtered as T[] };
        }
      };
      return statement;
    }
  };

  return db;
}

describe('searchCases - パターンタイプ検索', () => {
  it('パターンタイプで検索できる', async () => {
    // Arrange
    const db = createMockDb(mockCases);
    const params: CaseSearchParams = { pattern_type: 'Shock_Recovery' };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases).toBeInstanceOf(Array);
    expect(result.cases.every(c => c.pattern_type === 'Shock_Recovery')).toBe(true);
    expect(result.cases.length).toBe(2);
  });
});

describe('searchCases - 八卦検索', () => {
  it('八卦の組み合わせで検索できる', async () => {
    // Arrange
    const db = createMockDb(mockCases);
    const params: CaseSearchParams = { before_trigram: '坎', after_trigram: '乾' };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases.every(c => c.before_trigram === '坎' && c.after_trigram === '乾')).toBe(true);
  });
});

describe('searchCases - スケール検索', () => {
  it('スケールで検索できる', async () => {
    // Arrange
    const db = createMockDb(mockCases);
    const params: CaseSearchParams = { scale: 'company' };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases.every(c => c.scale === 'company')).toBe(true);
  });
});

describe('searchCases - 結果制限', () => {
  it('結果は最大20件まで', async () => {
    // Arrange - 30件のダミーデータ
    const manyCases: Case[] = Array(30).fill(null).map((_, i) => ({
      id: String(i),
      entity_name: `Entity ${i}`,
      before_trigram: '乾' as const,
      after_trigram: '坤' as const,
      pattern_type: 'Expansion_Growth' as const,
      scale: 'company' as const,
      main_domain: 'technology',
      year: 2020,
      summary: `Summary ${i}`
    }));
    const db = createMockDb(manyCases);
    const params: CaseSearchParams = { scale: 'company' };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases.length).toBeLessThanOrEqual(20);
  });
});

describe('searchCases - 空の結果', () => {
  it('該当なしの場合は空配列を返す', async () => {
    // Arrange
    const db = createMockDb(mockCases);
    const params: CaseSearchParams = { pattern_type: 'NonExistent' as 'Shock_Recovery' };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases).toEqual([]);
    expect(result.total).toBe(0);
  });
});

describe('searchCases - 複合検索', () => {
  it('複数条件の組み合わせで検索できる', async () => {
    // Arrange
    const db = createMockDb(mockCases);
    const params: CaseSearchParams = {
      pattern_type: 'Shock_Recovery',
      scale: 'company'
    };

    // Act
    const result = await searchCases(db as unknown as D1Database, params);

    // Assert
    expect(result.cases.every(c =>
      c.pattern_type === 'Shock_Recovery' && c.scale === 'company'
    )).toBe(true);
  });
});
