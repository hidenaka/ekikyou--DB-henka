/**
 * HaQei診断システム v5 ルーブリック管理
 */

import { Rubric, ClassProfile } from './types';

// ルーブリックキャッシュ
let cachedRubric: Rubric | null = null;

/**
 * ルーブリックを直接設定（Cloudflare Workers等用）
 */
export function setRubric(rubric: Rubric): void {
  cachedRubric = rubric;
}

/**
 * キャッシュされたルーブリックを取得
 */
export function getRubric(): Rubric | null {
  return cachedRubric;
}

/**
 * キャッシュをクリア
 */
export function clearRubricCache(): void {
  cachedRubric = null;
}

/**
 * クラスプロファイル一覧を取得
 */
export function getClassProfiles(): ClassProfile[] {
  if (!cachedRubric) {
    throw new Error('Rubric not loaded. Call loadRubricFromFile or setRubric first.');
  }
  return cachedRubric.classProfiles;
}

/**
 * 特定のクラスプロファイルを取得
 */
export function getClassProfileById(classId: number): ClassProfile | undefined {
  if (!cachedRubric) {
    throw new Error('Rubric not loaded.');
  }
  return cachedRubric.classProfiles.find((cp) => cp.classId === classId);
}

/**
 * 卦番号と爻番号からクラスプロファイルを取得
 */
export function getClassProfileByHexagramYao(
  hexagram: number,
  yao: number
): ClassProfile | undefined {
  if (!cachedRubric) {
    throw new Error('Rubric not loaded.');
  }
  return cachedRubric.classProfiles.find(
    (cp) => cp.hexagram === hexagram && cp.yao === yao
  );
}

/**
 * ルーブリックのバージョン情報を取得
 */
export function getRubricVersion(): string {
  if (!cachedRubric) {
    throw new Error('Rubric not loaded.');
  }
  return cachedRubric.version;
}
