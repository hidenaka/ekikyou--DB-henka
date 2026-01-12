/**
 * Phase 4: 診断APIのテスト（TDD）
 *
 * テスト対象: computeDiagnosis, DiagnoseHandler
 *
 * テストファースト: このテストが先に書かれ、実装はテストが通るように後から書く
 */

import { describe, it, expect, vi } from 'vitest';
import {
  computeDiagnosis,
  createPreviewResponse,
  createFullResponse,
  type DiagnosisInput,
  type DiagnosisResult
} from '../src/diagnose';

describe('computeDiagnosis', () => {
  it('10個の回答から診断結果を計算', () => {
    // Arrange
    const input: DiagnosisInput = {
      answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    };

    // Act
    const result = computeDiagnosis(input);

    // Assert
    expect(result.hexagram).toBeDefined();
    expect(result.beforeTrigram).toBeDefined();
    expect(result.afterTrigram).toBeDefined();
    expect(result.summary).toBeDefined();
    expect(result.fullAnalysis).toBeDefined();
    expect(result.recommendedActions).toBeDefined();
    expect(result.recommendedActions.length).toBeGreaterThan(0);
  });

  it('回答パターンにより異なる結果を返す', () => {
    // Arrange
    const input1: DiagnosisInput = { answers: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] };
    const input2: DiagnosisInput = { answers: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] };

    // Act
    const result1 = computeDiagnosis(input1);
    const result2 = computeDiagnosis(input2);

    // Assert
    expect(result1.hexagram).not.toBe(result2.hexagram);
  });

  it('八卦名は正しい漢字を返す', () => {
    // Arrange
    const input: DiagnosisInput = { answers: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] };
    const validTrigrams = ['乾', '坤', '震', '巽', '坎', '離', '艮', '兌'];

    // Act
    const result = computeDiagnosis(input);

    // Assert
    expect(validTrigrams).toContain(result.beforeTrigram);
    expect(validTrigrams).toContain(result.afterTrigram);
  });
});

describe('createPreviewResponse', () => {
  it('プレビューは制限された情報のみを含む', () => {
    // Arrange
    const fullResult: DiagnosisResult = {
      hexagram: '☰☷',
      beforeTrigram: '乾',
      afterTrigram: '坤',
      summary: 'テスト概要',
      fullAnalysis: '詳細な分析...',
      recommendedActions: ['アクション1', 'アクション2'],
      patternType: 'Expansion_Growth'
    };

    // Act
    const preview = createPreviewResponse(fullResult);

    // Assert
    expect(preview.hexagram).toBe('☰☷');
    expect(preview.summary).toBe('テスト概要');
    expect(preview.fullAnalysis).toBeUndefined();
    expect(preview.recommendedActions).toBeUndefined();
    expect(preview.isPreview).toBe(true);
  });
});

describe('createFullResponse', () => {
  it('フルレスポンスは全情報を含む', () => {
    // Arrange
    const fullResult: DiagnosisResult = {
      hexagram: '☰☷',
      beforeTrigram: '乾',
      afterTrigram: '坤',
      summary: 'テスト概要',
      fullAnalysis: '詳細な分析...',
      recommendedActions: ['アクション1', 'アクション2'],
      patternType: 'Expansion_Growth'
    };

    // Act
    const response = createFullResponse(fullResult);

    // Assert
    expect(response.hexagram).toBe('☰☷');
    expect(response.summary).toBe('テスト概要');
    expect(response.fullAnalysis).toBe('詳細な分析...');
    expect(response.recommendedActions).toEqual(['アクション1', 'アクション2']);
    expect(response.patternType).toBe('Expansion_Growth');
    expect(response.isPreview).toBe(false);
  });
});

describe('入力バリデーション', () => {
  it('回答が10個未満の場合はエラー', () => {
    // Arrange
    const input: DiagnosisInput = { answers: [1, 2, 3] };

    // Act & Assert
    expect(() => computeDiagnosis(input)).toThrow('10 answers required');
  });

  it('回答値が1-3範囲外の場合はエラー', () => {
    // Arrange
    const input: DiagnosisInput = { answers: [1, 2, 3, 1, 2, 3, 1, 2, 3, 5] };

    // Act & Assert
    expect(() => computeDiagnosis(input)).toThrow('Answer values must be 1, 2, or 3');
  });
});

describe('パターンタイプ計算', () => {
  it('変化パターンを正しく分類', () => {
    // Arrange - 乾→坤 (最大変化) はTransformation系
    const input: DiagnosisInput = { answers: [1, 1, 1, 3, 3, 3, 1, 1, 3, 3] };

    // Act
    const result = computeDiagnosis(input);

    // Assert
    expect(result.patternType).toBeDefined();
    expect(typeof result.patternType).toBe('string');
  });
});
