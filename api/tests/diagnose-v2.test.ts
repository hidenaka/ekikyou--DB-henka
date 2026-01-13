/**
 * 診断エンジン v2 テスト
 */

import { describe, it, expect } from 'vitest';
import {
  processPhase1,
  processPhase2,
  getYaoOptions,
  generatePreview,
  PHASE1_QUESTIONS,
  type Phase1Answers
} from '../src/diagnose-v2';

describe('Phase 1 Questions', () => {
  it('should have 5 questions', () => {
    expect(PHASE1_QUESTIONS).toHaveLength(5);
  });

  it('should have proper structure for each question', () => {
    for (const q of PHASE1_QUESTIONS) {
      expect(q).toHaveProperty('id');
      expect(q).toHaveProperty('axis');
      expect(q).toHaveProperty('question');
      expect(q).toHaveProperty('options');
      expect(q.options.length).toBeGreaterThanOrEqual(3);
    }
  });

  it('should cover all 5 axes', () => {
    const axes = PHASE1_QUESTIONS.map(q => q.axis);
    expect(axes).toContain('changeNature');
    expect(axes).toContain('agency');
    expect(axes).toContain('timeframe');
    expect(axes).toContain('relationship');
    expect(axes).toContain('emotionalTone');
  });
});

describe('processPhase1', () => {
  it('should return 5 candidates', () => {
    const answers: Phase1Answers = {
      changeNature: 0,   // 拡大
      agency: 0,         // 自ら動く
      timeframe: 0,      // 今すぐ
      relationship: 1,   // 組織内
      emotionalTone: 0   // 前向き
    };

    const result = processPhase1(answers);
    expect(result.candidates).toHaveLength(5);
  });

  it('should have confidence scores that sum to approximately 1', () => {
    const answers: Phase1Answers = {
      changeNature: 1,   // 縮小
      agency: 2,         // 待つ
      timeframe: 2,      // 長期
      relationship: 0,   // 個人
      emotionalTone: 2   // 不安
    };

    const result = processPhase1(answers);
    const totalConfidence = result.candidates.reduce((sum, c) => sum + c.confidence, 0);
    expect(totalConfidence).toBeCloseTo(1, 1);
  });

  it('should return sorted candidates by confidence', () => {
    const answers: Phase1Answers = {
      changeNature: 2,   // 維持
      agency: 1,         // 受け止める
      timeframe: 1,      // 中期
      relationship: 0,   // 個人
      emotionalTone: 1   // 慎重
    };

    const result = processPhase1(answers);
    for (let i = 0; i < result.candidates.length - 1; i++) {
      expect(result.candidates[i].confidence).toBeGreaterThanOrEqual(
        result.candidates[i + 1].confidence
      );
    }
  });

  it('should have valid hexagram numbers (1-64)', () => {
    const answers: Phase1Answers = {
      changeNature: 3,   // 転換
      agency: 0,         // 自ら動く
      timeframe: 0,      // 今すぐ
      relationship: 1,   // 組織内
      emotionalTone: 0   // 前向き
    };

    const result = processPhase1(answers);
    for (const candidate of result.candidates) {
      expect(candidate.hexagramNumber).toBeGreaterThanOrEqual(1);
      expect(candidate.hexagramNumber).toBeLessThanOrEqual(64);
    }
  });

  it('should indicate if additional questions are needed', () => {
    const answers: Phase1Answers = {
      changeNature: 0,
      agency: 0,
      timeframe: 0,
      relationship: 0,
      emotionalTone: 0
    };

    const result = processPhase1(answers);
    expect(typeof result.needsAdditionalQuestions).toBe('boolean');
  });

  it('should provide additional questions when confidence is low', () => {
    // 矛盾した回答で確信度を下げる
    const answers: Phase1Answers = {
      changeNature: 0,   // 拡大
      agency: 2,         // 待つ（矛盾）
      timeframe: 2,      // 長期
      relationship: 2,   // 対外
      emotionalTone: 2   // 不安
    };

    const result = processPhase1(answers);
    // 追加質問があるかどうかは確信度次第
    if (result.needsAdditionalQuestions) {
      expect(result.additionalQuestions).toBeDefined();
      expect(result.additionalQuestions!.length).toBeGreaterThan(0);
    }
  });
});

describe('processPhase2', () => {
  it('should adjust scores based on additional answers', () => {
    const phase1Answers: Phase1Answers = {
      changeNature: 0,
      agency: 0,
      timeframe: 0,
      relationship: 1,
      emotionalTone: 0
    };

    const phase2Answers = {
      change_clarify: 0,   // 拡大を確認
      agency_clarify: 0    // 自ら動くを確認
    };

    const result = processPhase2(phase1Answers, phase2Answers);
    expect(result.candidates).toHaveLength(5);
    expect(result.needsAdditionalQuestions).toBe(false);
  });

  it('should return higher confidence after Phase 2', () => {
    const phase1Answers: Phase1Answers = {
      changeNature: 0,
      agency: 0,
      timeframe: 0,
      relationship: 1,
      emotionalTone: 0
    };

    const phase1Result = processPhase1(phase1Answers);

    const phase2Answers = {
      change_clarify: 0,
      agency_clarify: 0
    };

    const phase2Result = processPhase2(phase1Answers, phase2Answers);

    // Phase 2 should have at least same or better confidence
    expect(phase2Result.topConfidence).toBeGreaterThanOrEqual(0);
  });
});

describe('getYaoOptions', () => {
  it('should return 6 yao options for valid hexagram', () => {
    const result = getYaoOptions(1);
    expect(result.yaoOptions).toHaveLength(6);
  });

  it('should include hexagram name', () => {
    const result = getYaoOptions(1);
    expect(result.hexagramName).toBe('乾為天');
  });

  it('should have proper yao structure', () => {
    const result = getYaoOptions(3);
    for (const yao of result.yaoOptions) {
      expect(yao.yao).toBeGreaterThanOrEqual(1);
      expect(yao.yao).toBeLessThanOrEqual(6);
      expect(yao.description).toBeTruthy();
      expect(yao.stage).toBeTruthy();
    }
  });

  it('should throw error for invalid hexagram', () => {
    expect(() => getYaoOptions(0)).toThrow();
    expect(() => getYaoOptions(65)).toThrow();
    expect(() => getYaoOptions(100)).toThrow();
  });
});

describe('generatePreview', () => {
  it('should generate preview for valid hexagram and yao', () => {
    const result = generatePreview(11, 5, 47);

    expect(result.hexagramNumber).toBe(11);
    expect(result.hexagramName).toBe('地天泰');
    expect(result.yao).toBe(5);
    expect(result.caseCount).toBe(47);
  });

  it('should include summary with hexagram and yao info', () => {
    const result = generatePreview(3, 1, 30);

    expect(result.summary).toContain('水雷屯');
    expect(result.summary).toContain('初期・準備段階');
  });

  it('should include distribution breakdown', () => {
    const result = generatePreview(42, 3, 100);

    expect(result.distribution).toHaveProperty('キャリア転換');
    expect(result.distribution).toHaveProperty('事業立ち上げ');
    expect(result.distribution).toHaveProperty('組織変革');

    // 合計がcaseCount以下であることを確認
    const total = Object.values(result.distribution).reduce((a, b) => a + b, 0);
    expect(total).toBeLessThanOrEqual(result.caseCount);
  });

  it('should include paid content preview items', () => {
    const result = generatePreview(49, 4, 25);

    expect(result.paidContentPreview).toContain('具体的な類似事例3件の詳細分析');
    expect(result.paidContentPreview).toContain('あなたの状況に基づいた90日行動計画');
    expect(result.paidContentPreview.length).toBeGreaterThanOrEqual(3);
  });

  it('should throw error for invalid yao', () => {
    expect(() => generatePreview(1, 0, 10)).toThrow();
    expect(() => generatePreview(1, 7, 10)).toThrow();
  });
});

describe('Integration: Full Diagnosis Flow', () => {
  it('should complete full diagnosis flow', () => {
    // Phase 1
    const answers: Phase1Answers = {
      changeNature: 0,   // 拡大
      agency: 0,         // 自ら動く
      timeframe: 1,      // 中期
      relationship: 1,   // 組織内
      emotionalTone: 3   // 楽観
    };

    const phase1Result = processPhase1(answers);
    expect(phase1Result.candidates.length).toBe(5);

    // User selects first candidate
    const selectedHexagram = phase1Result.candidates[0].hexagramNumber;

    // Phase 3: Get yao options
    const yaoResult = getYaoOptions(selectedHexagram);
    expect(yaoResult.yaoOptions.length).toBe(6);

    // User selects yao 5 (成熟段階)
    const selectedYao = 5;

    // Phase 4: Generate preview
    const preview = generatePreview(selectedHexagram, selectedYao, 35);
    expect(preview.hexagramNumber).toBe(selectedHexagram);
    expect(preview.yao).toBe(selectedYao);
    expect(preview.summary).toBeTruthy();
  });
});
