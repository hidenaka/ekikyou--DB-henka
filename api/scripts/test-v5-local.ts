/**
 * v5 ローカルテストスクリプト
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// ESM対応
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// v5モジュールをインポート
import {
  setRubric,
  convertToProfile,
  generateRanking,
  getTopCandidates,
  getClassProfiles,
} from '../src/v5/index.js';
import type { UserAnswers, Rubric } from '../src/v5/types.js';

// ルーブリック読み込み
const rubricPath = join(__dirname, '../../data/rubric_v1.json');
const rubricContent = readFileSync(rubricPath, 'utf-8');
const rubric: Rubric = JSON.parse(rubricContent);
setRubric(rubric);

console.log('=== HaQei v5 ローカルテスト ===\n');
console.log(`ルーブリック: ${rubric.version}`);
console.log(`クラス数: ${rubric.classProfiles.length}`);
console.log('');

// テスト1: 拡大傾向
console.log('--- テスト1: 拡大傾向 ---');
const test1: UserAnswers = {
  changeNature: { expansion: 5, contraction: 1, maintenance: 1, transformation: 1 },
  agency: 5,
  timeframe: 'longTerm',
  relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
  emotionalTone: { excitement: 5, caution: 1, anxiety: 1, optimism: 4 },
};

const profile1 = convertToProfile(test1);
const result1 = generateRanking(profile1, getClassProfiles());
const top5_1 = getTopCandidates(result1, 5);

console.log('Top 5:');
top5_1.forEach((c, i) => {
  console.log(`  ${i + 1}. ${c.name} (score: ${c.score.toFixed(4)})`);
});
console.log(`信頼度: ${result1.overallConfidence}`);
console.log('');

// テスト2: 維持傾向
console.log('--- テスト2: 維持傾向 ---');
const test2: UserAnswers = {
  changeNature: { expansion: 1, contraction: 1, maintenance: 5, transformation: 1 },
  agency: 2,
  timeframe: 'midTerm',
  relationship: { self: false, family: true, team: true, organization: false, external: false, society: false },
  emotionalTone: { excitement: 2, caution: 4, anxiety: 2, optimism: 2 },
};

const profile2 = convertToProfile(test2);
const result2 = generateRanking(profile2, getClassProfiles());
const top5_2 = getTopCandidates(result2, 5);

console.log('Top 5:');
top5_2.forEach((c, i) => {
  console.log(`  ${i + 1}. ${c.name} (score: ${c.score.toFixed(4)})`);
});
console.log(`信頼度: ${result2.overallConfidence}`);
console.log('');

// テスト3: 欠損あり
console.log('--- テスト3: 欠損あり（timeframe=unknown） ---');
const test3: UserAnswers = {
  changeNature: { expansion: 3, contraction: 3, maintenance: 3, transformation: 3 },
  agency: 3,
  timeframe: 'unknown',
  relationship: { self: true, family: false, team: false, organization: false, external: false, society: false },
  emotionalTone: { excitement: 3, caution: 3, anxiety: 3, optimism: 3 },
};

const profile3 = convertToProfile(test3);
const result3 = generateRanking(profile3, getClassProfiles());

console.log(`欠損軸: ${result3.missingAxes.join(', ')}`);
console.log(`信頼度: ${result3.overallConfidence}`);
console.log('Top 3:');
getTopCandidates(result3, 3).forEach((c, i) => {
  console.log(`  ${i + 1}. ${c.name} (score: ${c.score.toFixed(4)})`);
});
console.log('');

console.log('=== テスト完了 ===');
