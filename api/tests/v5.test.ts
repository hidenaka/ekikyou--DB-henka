/**
 * HaQei診断システム v5 テスト
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  convertToProfile,
  convertChangeNature,
  convertAgency,
  convertTimeframe,
  convertRelationship,
  convertEmotionalTone,
  calculateEntropy,
  maxEntropy,
} from '../src/v5/convert';
import { jsDistance, klDivergence } from '../src/v5/distance';
import { calculateScore, generateRanking } from '../src/v5/matching';
import { setRubric, getClassProfiles, clearRubricCache } from '../src/v5/rubric';
import { UserAnswers, Rubric, ClassProfile } from '../src/v5/types';

// テスト用のミニルーブリック
const createTestRubric = (): Rubric => {
  const classProfiles: ClassProfile[] = [];

  // 3つのテストクラスを作成
  const testClasses = [
    {
      classId: 1,
      hexagram: 1,
      yao: 1,
      name: '乾為天 初九',
      hexagramName: '乾為天',
      yaoName: '初九',
      yaoStage: '潜龍',
      distributions: {
        changeNature: { 拡大: 0.7, 収縮: 0.1, 維持: 0.1, 転換: 0.1 },
        agency: { 自ら動く: 0.2, 受け止める: 0.3, 待つ: 0.5 },
        timeframe: { 即時: 0.1, 短期: 0.2, 中期: 0.3, 長期: 0.4 },
        relationship: { 個人: 0.7, 組織内: 0.2, 対外: 0.1 },
        emotionalTone: { 前向き: 0.5, 慎重: 0.3, 不安: 0.1, 楽観: 0.1 },
      },
    },
    {
      classId: 2,
      hexagram: 2,
      yao: 1,
      name: '坤為地 初六',
      hexagramName: '坤為地',
      yaoName: '初六',
      yaoStage: '履霜',
      distributions: {
        changeNature: { 拡大: 0.1, 収縮: 0.2, 維持: 0.6, 転換: 0.1 },
        agency: { 自ら動く: 0.1, 受け止める: 0.6, 待つ: 0.3 },
        timeframe: { 即時: 0.3, 短期: 0.4, 中期: 0.2, 長期: 0.1 },
        relationship: { 個人: 0.3, 組織内: 0.5, 対外: 0.2 },
        emotionalTone: { 前向き: 0.2, 慎重: 0.5, 不安: 0.2, 楽観: 0.1 },
      },
    },
    {
      classId: 3,
      hexagram: 3,
      yao: 1,
      name: '水雷屯 初九',
      hexagramName: '水雷屯',
      yaoName: '初九',
      yaoStage: '磐桓',
      distributions: {
        changeNature: { 拡大: 0.3, 収縮: 0.1, 維持: 0.3, 転換: 0.3 },
        agency: { 自ら動く: 0.4, 受け止める: 0.4, 待つ: 0.2 },
        timeframe: { 即時: 0.5, 短期: 0.3, 中期: 0.15, 長期: 0.05 },
        relationship: { 個人: 0.4, 組織内: 0.3, 対外: 0.3 },
        emotionalTone: { 前向き: 0.3, 慎重: 0.3, 不安: 0.3, 楽観: 0.1 },
      },
    },
  ];

  for (const tc of testClasses) {
    classProfiles.push({
      ...tc,
      rubricVersion: 'test-v1',
      rubricSource: 'test',
    });
  }

  return {
    version: 'test-v1',
    createdAt: '2026-01-14',
    description: 'Test rubric',
    axisRules: {},
    classProfiles,
    metadata: {
      totalClasses: 3,
      hexagramCount: 3,
      yaoPerHexagram: 1,
      generationMethod: 'test',
      validationStatus: 'test',
    },
  };
};

describe('変換関数テスト', () => {
  describe('calculateEntropy', () => {
    it('均等分布で最大エントロピー', () => {
      const entropy = calculateEntropy([0.25, 0.25, 0.25, 0.25]);
      expect(entropy).toBeCloseTo(2.0, 5);
    });

    it('集中分布でゼロエントロピー', () => {
      const entropy = calculateEntropy([1.0, 0, 0, 0]);
      expect(entropy).toBe(0);
    });

    it('偏った分布で中間エントロピー', () => {
      const entropy = calculateEntropy([0.7, 0.1, 0.1, 0.1]);
      expect(entropy).toBeGreaterThan(0);
      expect(entropy).toBeLessThan(2.0);
    });
  });

  describe('maxEntropy', () => {
    it('4カテゴリで2.0', () => {
      expect(maxEntropy(4)).toBeCloseTo(2.0, 5);
    });

    it('3カテゴリで約1.58', () => {
      expect(maxEntropy(3)).toBeCloseTo(1.585, 2);
    });
  });

  describe('convertChangeNature', () => {
    it('均等入力で均等分布', () => {
      const result = convertChangeNature({
        expansion: 3,
        contraction: 3,
        maintenance: 3,
        transformation: 3,
      });
      expect(result.values['拡大']).toBeCloseTo(0.25, 5);
      expect(result.values['収縮']).toBeCloseTo(0.25, 5);
      expect(result.isMissing).toBe(false);
    });

    it('偏った入力で偏った分布', () => {
      const result = convertChangeNature({
        expansion: 5,
        contraction: 1,
        maintenance: 1,
        transformation: 1,
      });
      expect(result.values['拡大']).toBeCloseTo(0.625, 3);
      expect(result.values['収縮']).toBeCloseTo(0.125, 3);
    });

    it('ゼロ入力でデフォルト分布', () => {
      const result = convertChangeNature({
        expansion: 0,
        contraction: 0,
        maintenance: 0,
        transformation: 0,
      });
      expect(result.values['拡大']).toBe(0.25);
      expect(result.entropy).toBe(2.0);
    });
  });

  describe('convertAgency', () => {
    it('score=1で「待つ」優勢', () => {
      const result = convertAgency(1);
      expect(result.values['待つ']).toBe(1.0);
      expect(result.values['自ら動く']).toBe(0);
    });

    it('score=5で「自ら動く」優勢', () => {
      const result = convertAgency(5);
      expect(result.values['自ら動く']).toBe(1.0);
      expect(result.values['待つ']).toBe(0);
    });

    it('score=3で「受け止める」中心', () => {
      const result = convertAgency(3);
      expect(result.values['受け止める']).toBe(0.6);
      expect(result.values['自ら動く']).toBe(0.2);
      expect(result.values['待つ']).toBe(0.2);
    });
  });

  describe('convertTimeframe', () => {
    it('immediateで即時優勢', () => {
      const result = convertTimeframe('immediate');
      expect(result.values['即時']).toBe(0.7);
      expect(result.isMissing).toBe(false);
    });

    it('unknownで欠損フラグ', () => {
      const result = convertTimeframe('unknown');
      expect(result.isMissing).toBe(true);
      expect(result.values['即時']).toBe(0.25);
    });
  });

  describe('convertRelationship', () => {
    it('self選択で個人優勢', () => {
      const result = convertRelationship({
        self: true,
        family: false,
        team: false,
        organization: false,
        external: false,
        society: false,
      });
      expect(result.values['個人']).toBe(1.0);
    });

    it('複数選択で混合', () => {
      const result = convertRelationship({
        self: true,
        family: false,
        team: true,
        organization: false,
        external: false,
        society: false,
      });
      expect(result.values['個人']).toBeGreaterThan(0);
      expect(result.values['組織内']).toBeGreaterThan(0);
    });

    it('何も選択しないでデフォルト', () => {
      const result = convertRelationship({
        self: false,
        family: false,
        team: false,
        organization: false,
        external: false,
        society: false,
      });
      expect(result.values['個人']).toBe(0.5);
    });
  });

  describe('convertEmotionalTone', () => {
    it('均等入力で均等分布', () => {
      const result = convertEmotionalTone({
        excitement: 3,
        caution: 3,
        anxiety: 3,
        optimism: 3,
      });
      expect(result.values['前向き']).toBeCloseTo(0.25, 5);
    });
  });
});

describe('距離計算テスト', () => {
  describe('jsDistance', () => {
    it('同一分布で距離0', () => {
      const p = { a: 0.5, b: 0.3, c: 0.2 };
      const dist = jsDistance(p, p);
      expect(dist).toBeCloseTo(0, 5);
    });

    it('異なる分布で距離>0', () => {
      const p = { a: 0.7, b: 0.2, c: 0.1 };
      const q = { a: 0.1, b: 0.2, c: 0.7 };
      const dist = jsDistance(p, q);
      expect(dist).toBeGreaterThan(0);
      expect(dist).toBeLessThanOrEqual(1);
    });

    it('対称性: JS(P,Q) = JS(Q,P)', () => {
      const p = { a: 0.6, b: 0.3, c: 0.1 };
      const q = { a: 0.2, b: 0.5, c: 0.3 };
      expect(jsDistance(p, q)).toBeCloseTo(jsDistance(q, p), 10);
    });
  });

  describe('klDivergence', () => {
    it('同一分布でKL=0', () => {
      const p = { a: 0.5, b: 0.3, c: 0.2 };
      expect(klDivergence(p, p)).toBeCloseTo(0, 5);
    });
  });
});

describe('マッチングテスト', () => {
  beforeAll(() => {
    clearRubricCache();
    setRubric(createTestRubric());
  });

  describe('calculateScore', () => {
    it('スコア計算が動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 5, contraction: 1, maintenance: 1, transformation: 1 },
        agency: 4,
        timeframe: 'immediate',
        relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 4, caution: 2, anxiety: 1, optimism: 3 },
      };

      const userProfile = convertToProfile(answers);
      const classProfiles = getClassProfiles();
      const score = calculateScore(userProfile, classProfiles[0]);

      expect(score.classId).toBe(1);
      expect(score.score).toBeGreaterThanOrEqual(0);
      expect(score.score).toBeLessThanOrEqual(1);
      expect(score.contributions).toBeDefined();
    });
  });

  describe('generateRanking', () => {
    it('ランキング生成が動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
        agency: 3,
        timeframe: 'midTerm',
        relationship: { self: true, family: true, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
      };

      const userProfile = convertToProfile(answers);
      const classProfiles = getClassProfiles();
      const result = generateRanking(userProfile, classProfiles);

      expect(result.ranking.length).toBe(3);
      expect(result.ranking[0].rank).toBe(1);
      expect(result.version).toBe('v5.0.0');
    });

    it('欠損軸の処理', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
        agency: 3,
        timeframe: 'unknown', // 欠損
        relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
      };

      const userProfile = convertToProfile(answers);
      const classProfiles = getClassProfiles();
      const result = generateRanking(userProfile, classProfiles);

      expect(result.missingAxes).toContain('timeframe');
      expect(result.overallConfidence).toBe(0.8); // 1 - 1/5
    });

    it('スコア順にソートされる', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 5, contraction: 1, maintenance: 1, transformation: 1 },
        agency: 5,
        timeframe: 'longTerm',
        relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 5, caution: 1, anxiety: 1, optimism: 3 },
      };

      const userProfile = convertToProfile(answers);
      const classProfiles = getClassProfiles();
      const result = generateRanking(userProfile, classProfiles);

      for (let i = 1; i < result.ranking.length; i++) {
        expect(result.ranking[i].score).toBeGreaterThanOrEqual(
          result.ranking[i - 1].score
        );
      }
    });
  });
});

describe('統合テスト', () => {
  beforeAll(() => {
    clearRubricCache();
    setRubric(createTestRubric());
  });

  it('convertToProfileが全軸を変換する', () => {
    const answers: UserAnswers = {
      changeNature: { expansion: 4, contraction: 2, maintenance: 3, transformation: 1 },
      agency: 4,
      timeframe: 'shortTerm',
      relationship: { self: true, family: true, team: true, organization: false, external: false, society: false },
      emotionalTone: { excitement: 4, caution: 2, anxiety: 1, optimism: 3 },
    };

    const profile = convertToProfile(answers);

    expect(profile.changeNature).toBeDefined();
    expect(profile.agency).toBeDefined();
    expect(profile.timeframe).toBeDefined();
    expect(profile.relationship).toBeDefined();
    expect(profile.emotionalTone).toBeDefined();

    // 全ての確率の合計が1.0
    const sumChangeNature = Object.values(profile.changeNature.values).reduce((a, b) => a + b, 0);
    expect(sumChangeNature).toBeCloseTo(1.0, 5);
  });
});
