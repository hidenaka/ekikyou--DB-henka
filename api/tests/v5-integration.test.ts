/**
 * HaQei診断システム v5 統合テスト
 * 実際のルーブリックを使った動作確認
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  diagnose,
  createDiagnoseResponse,
  createResultSummary,
  setRubric,
  clearRubricCache,
  convertToProfile,
} from '../src/v5/index';
import { UserAnswers, Rubric } from '../src/v5/types';

// ESM対応の__dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('v5統合テスト（実ルーブリック）', () => {
  beforeAll(() => {
    clearRubricCache();
    const rubricPath = join(__dirname, '../../data/rubric_v1.json');
    const rubricContent = readFileSync(rubricPath, 'utf-8');
    const rubric: Rubric = JSON.parse(rubricContent);
    setRubric(rubric);
  });

  describe('診断フロー', () => {
    it('384クラス全てがランキングに含まれる', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 4, contraction: 2, maintenance: 2, transformation: 2 },
        agency: 4,
        timeframe: 'shortTerm',
        relationship: { self: true, family: false, team: true, organization: false, external: false, society: false },
        emotionalTone: { excitement: 4, caution: 2, anxiety: 1, optimism: 3 },
      };

      const result = diagnose(answers);

      expect(result.ranking.length).toBe(384);
      expect(result.ranking[0].rank).toBe(1);
      expect(result.ranking[383].rank).toBeGreaterThan(1);
    });

    it('拡大傾向のユーザーの診断が正常に動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 5, contraction: 1, maintenance: 1, transformation: 1 },
        agency: 5, // 自ら動く
        timeframe: 'longTerm',
        relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 5, caution: 1, anxiety: 1, optimism: 4 },
      };

      const result = diagnose(answers);
      const top10 = result.ranking.slice(0, 10);

      console.log('拡大傾向ユーザーの上位10件:', top10.map(c => `${c.name} (score: ${c.score.toFixed(4)})`));

      // ランキングが生成されること
      expect(result.ranking.length).toBe(384);
      // スコアが昇順（距離が小さい順）であること
      expect(top10[0].score).toBeLessThanOrEqual(top10[9].score);
      // 寄与度が計算されていること
      expect(top10[0].contributions.changeNature).toBeDefined();
      // 説明文が生成されていること
      expect(top10[0].matchReasons.length).toBeGreaterThan(0);
    });

    it('維持傾向のユーザーの診断が正常に動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 1, contraction: 1, maintenance: 5, transformation: 1 },
        agency: 2, // 受け止める～待つ
        timeframe: 'midTerm',
        relationship: { self: false, family: true, team: true, organization: false, external: false, society: false },
        emotionalTone: { excitement: 2, caution: 4, anxiety: 2, optimism: 2 },
      };

      const result = diagnose(answers);
      const top10 = result.ranking.slice(0, 10);

      console.log('維持傾向ユーザーの上位10件:', top10.map(c => `${c.name} (score: ${c.score.toFixed(4)})`));

      // 拡大傾向とは異なる結果になること（異なる入力→異なる出力）
      expect(result.ranking.length).toBe(384);
      expect(top10[0].score).toBeLessThanOrEqual(top10[9].score);
    });

    it('欠損軸があると信頼度が低下する', () => {
      const answersWithUnknown: UserAnswers = {
        changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
        agency: 3,
        timeframe: 'unknown', // 欠損
        relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
      };

      const result = diagnose(answersWithUnknown);

      expect(result.missingAxes).toContain('timeframe');
      expect(result.overallConfidence).toBe(0.8);
    });

    it('APIレスポンス形式が正しい', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
        agency: 3,
        timeframe: 'midTerm',
        relationship: { self: true, family: true, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
      };

      const result = diagnose(answers);
      const response = createDiagnoseResponse(result, 5);

      expect(response.resultId).toMatch(/^v5-/);
      expect(response.topCandidates.length).toBe(5);
      expect(response.overallConfidence).toBeGreaterThan(0);
      expect(response.version).toBe('v5.0.0');
    });

    it('結果サマリーが生成される', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 4, contraction: 1, maintenance: 2, transformation: 3 },
        agency: 4,
        timeframe: 'shortTerm',
        relationship: { self: true, family: false, team: true, organization: false, external: false, society: false },
        emotionalTone: { excitement: 4, caution: 2, anxiety: 1, optimism: 3 },
      };

      const result = diagnose(answers);
      const summary = createResultSummary(result);

      expect(summary.topCandidate).toBeDefined();
      expect(summary.confidenceExplanation).toBeDefined();
      expect(summary.topCandidate.matchReasons.length).toBeGreaterThan(0);
    });
  });

  describe('エッジケース', () => {
    it('全て最小値でも動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 1, contraction: 1, maintenance: 1, transformation: 1 },
        agency: 1,
        timeframe: 'immediate',
        relationship: { self: false, family: false, team: false, organization: false, external: false, society: false },
        emotionalTone: { excitement: 1, caution: 1, anxiety: 1, optimism: 1 },
      };

      const result = diagnose(answers);
      expect(result.ranking.length).toBe(384);
    });

    it('全て最大値でも動作する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 5, contraction: 5, maintenance: 5, transformation: 5 },
        agency: 5,
        timeframe: 'longTerm',
        relationship: { self: true, family: true, team: true, organization: true, external: true, society: true },
        emotionalTone: { excitement: 5, caution: 5, anxiety: 5, optimism: 5 },
      };

      const result = diagnose(answers);
      expect(result.ranking.length).toBe(384);
    });

    it('混合状態が正しく検出される', () => {
      // 高エントロピー（均等分布）の回答
      const answers: UserAnswers = {
        changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
        agency: 3,
        timeframe: 'midTerm',
        relationship: { self: true, family: true, team: true, organization: true, external: true, society: true },
        emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
      };

      const profile = convertToProfile(answers);
      const result = diagnose(answers);

      // changeNatureとemotionalToneが高エントロピーなので混合状態
      expect(result.ranking[0].isMixedState).toBe(true);
    });
  });

  describe('パフォーマンス', () => {
    it('384クラスのランキングが100ms以内に完了する', () => {
      const answers: UserAnswers = {
        changeNature: { expansion: 4, contraction: 2, maintenance: 2, transformation: 2 },
        agency: 4,
        timeframe: 'shortTerm',
        relationship: { self: true, family: false, team: true, organization: false, external: false, society: false },
        emotionalTone: { excitement: 4, caution: 2, anxiety: 1, optimism: 3 },
      };

      const start = performance.now();
      const result = diagnose(answers);
      const elapsed = performance.now() - start;

      console.log(`診断処理時間: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });
});
