/**
 * ユーティリティ関数
 */

/**
 * SHA-256ハッシュを計算
 * @param input - ハッシュ化する文字列
 * @returns 64文字の16進数文字列
 */
export async function sha256(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  return hashHex;
}
