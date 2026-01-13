#!/usr/bin/env node
/**
 * LLM Debate Skill - Codex (GPT-5.2) 批評スクリプト
 *
 * Claudeの意見に対してCodexが厳格な批評を行う
 *
 * 使用方法:
 *   node llm-debate.js --topic <議題> --claude-opinion <Claudeの意見> [--output <出力先>]
 */

import { parseArgs } from "node:util";
import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
  unlinkSync,
} from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));

// タイムアウト設定（ミリ秒）
const CODEX_TIMEOUT = 300000; // 5分

const { values } = parseArgs({
  options: {
    topic: { type: "string", short: "t" },
    "claude-opinion": { type: "string", short: "c" },
    context: { type: "string", short: "x" },
    "context-file": { type: "string" },
    output: { type: "string", short: "o" },
  },
  strict: true,
});

// === 事前条件チェック: gitリポジトリか確認 ===
function checkGitRepository() {
  try {
    execSync("git rev-parse --git-dir", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    return true;
  } catch {
    return false;
  }
}

if (!checkGitRepository()) {
  console.error("========================================");
  console.error("エラー: このスキルはgitリポジトリ内でのみ使用できます");
  console.error("");
  console.error("Codex CLIはセキュリティ上の理由から、");
  console.error("gitリポジトリ外での実行を拒否します。");
  console.error("");
  console.error("解決方法:");
  console.error("  git init");
  console.error("");
  console.error("を実行してリポジトリを初期化してください。");
  console.error("========================================");
  process.exit(1);
}

if (!values.topic) {
  console.error("エラー: --topic オプションは必須です");
  process.exit(1);
}

if (!values["claude-opinion"]) {
  console.error("エラー: --claude-opinion オプションは必須です");
  process.exit(1);
}

const topic = values.topic;
const claudeOpinion = values["claude-opinion"];

// コンテキスト情報の取得
let sharedContext = "";
if (values["context-file"] && existsSync(values["context-file"])) {
  sharedContext = readFileSync(values["context-file"], "utf-8");
} else if (values.context) {
  sharedContext = values.context;
}

// 出力先ディレクトリの決定
const today = new Date().toISOString().slice(0, 10).replace(/-/g, "");
const sanitizedTopic = topic
  .replace(/[^a-zA-Z0-9\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/g, "-")
  .slice(0, 30);
const defaultOutputDir = resolve(
  __dirname,
  `../../../../debate/llm-debate-${today}-${sanitizedTopic}`
);
const outputDir = values.output ? resolve(values.output) : defaultOutputDir;

if (!existsSync(outputDir)) {
  mkdirSync(outputDir, { recursive: true });
}

/**
 * プロンプトを生成
 */
function buildPrompt() {
  const contextSection = sharedContext
    ? `
## 共有コンテキスト（Claudeが収集した情報）
以下はClaudeが検索やファイル読み込みで収集した情報です。この情報を前提に議論してください。

${sharedContext}
`
    : "";

  return `あなたはOpenAI GPT-5.2として、以下の議題について**プロフェッショナルとして忖度なく厳密に**批評してください。

## あなたの役割
あなたは該当分野の第一人者であり、査読者・批評家です。
- **甘い評価は害悪**：実装不可能なアイデアを「良い」と言うのは相手のためにならない
- **本質を突く**：表面的な賛同ではなく、論理の穴・前提の誤り・実現可能性を厳しく検証
- **具体的に指摘**：「〜かもしれない」ではなく「〜は誤りである。なぜなら〜」と断言
- **建設的批判**：ただ否定するのではなく、より良い代替案や修正点を提示

## 重要な注意事項
あなたは検索ツールを持っていないため、最新情報を取得できません。
ユーザーやClaudeから未知の技術・フレームワーク・将来の情報が伝えられた場合は、**存在するものとして**議論を進めてください。
「知らない」「情報がない」とは言わず、与えられた情報を前提に論理的に考察してください。
${contextSection}
## 議題
${topic}

## Claudeの意見（参考）
${claudeOpinion}

## 評価の観点（必ずすべて検討すること）
1. **論理的整合性**: 主張と根拠に矛盾はないか？前提は妥当か？
2. **実現可能性**: 技術的・実務的に本当に実装できるか？隠れたコストは？
3. **新規性の真偽**: 本当に新しいのか？既存手法との差分は明確か？
4. **スケーラビリティ**: 規模が大きくなっても成立するか？
5. **反例・エッジケース**: この主張が破綻するケースは？
6. **代替案との比較**: より優れたアプローチは存在しないか？

## 指示
- **忖度禁止**: Claudeの意見が間違っていると思えば遠慮なく否定せよ
- **曖昧な表現禁止**: 「〜と思われる」「〜の可能性がある」は使わない
- **断言せよ**: 自分の見解を明確に述べる。根拠を示した上で強く主張する
- **日本語で回答**

## 出力形式
マークダウン形式で、以下の構成で回答してください：

### 結論（最初に明言）
（この議題に対するあなたの明確な判定。「〇〇である」と断言）

### 批判的分析
（論理の穴、前提の誤り、実現可能性の問題点を具体的に指摘）

### Claudeの意見への反論/修正
（Claudeの意見の誤り・甘さを指摘。正しい部分があれば認める）

### より正確な見解
（あなたが考える本質的な評価と、その根拠）
`;
}

/**
 * Codex呼び出し
 */
async function callCodex() {
  console.error("[Codex] GPT-5.2 にリクエスト中...");
  const start = Date.now();

  const prompt = buildPrompt();
  const tempPromptPath = join(outputDir, ".codex-prompt-temp.md");

  writeFileSync(tempPromptPath, prompt, "utf-8");

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error(`Codex タイムアウト (${CODEX_TIMEOUT / 1000}秒)`));
    }, CODEX_TIMEOUT);

    try {
      const command = `codex exec --dangerously-bypass-approvals-and-sandbox --model "gpt-5.2" "$(cat '${tempPromptPath}')"`;

      const result = execSync(command, {
        encoding: "utf-8",
        maxBuffer: 50 * 1024 * 1024,
        timeout: CODEX_TIMEOUT,
        stdio: ["pipe", "pipe", "pipe"],
      });

      clearTimeout(timeout);
      console.error(`[Codex] 成功 (${Date.now() - start}ms)`);

      if (existsSync(tempPromptPath)) {
        unlinkSync(tempPromptPath);
      }

      resolve(result);
    } catch (error) {
      clearTimeout(timeout);
      if (existsSync(tempPromptPath)) {
        unlinkSync(tempPromptPath);
      }
      reject(error);
    }
  });
}

/**
 * 結果をMarkdownで整形
 */
function formatResult(codexResult) {
  const codexContent =
    codexResult.status === "fulfilled"
      ? codexResult.value
      : `（エラー: ${codexResult.reason?.message || "不明なエラー"}）`;

  return `# LLM Debate: ${topic}

## 議題
${topic}

## Claude (Anthropic) の見解
${claudeOpinion}

## Codex (OpenAI GPT-5.2) の批評
${codexContent}

## 統合分析
### 共通点
（Claudeによる分析が必要）

### 相違点
（Claudeによる分析が必要）

### 結論・推奨
（Claudeによる最終まとめが必要）

---
*生成日時: ${new Date().toISOString()}*
`;
}

/**
 * メイン処理
 */
async function main() {
  console.error("=== LLM Debate Skill (Claude + Codex) ===");
  console.error(`議題: ${topic}`);
  console.error(`出力先: ${outputDir}`);
  console.error("");

  // Claudeの意見を保存
  const claudeOpinionPath = join(outputDir, "claude-opinion.md");
  writeFileSync(
    claudeOpinionPath,
    `# Claude (Anthropic) の見解\n\n${claudeOpinion}`,
    "utf-8"
  );
  console.error(`[保存] ${claudeOpinionPath}`);

  // Codexを呼び出し
  console.error("\n[実行] Codex を呼び出し中...\n");

  const codexResult = await callCodex()
    .then((value) => ({ status: "fulfilled", value }))
    .catch((reason) => ({ status: "rejected", reason }));

  // 結果を保存
  const codexPath = join(outputDir, "codex-response.md");
  if (codexResult.status === "fulfilled") {
    writeFileSync(
      codexPath,
      `# Codex (OpenAI GPT-5.2) の批評\n\n${codexResult.value}`,
      "utf-8"
    );
    console.error(`[保存] ${codexPath}`);
  } else {
    writeFileSync(
      codexPath,
      `# Codex (OpenAI GPT-5.2) の批評\n\n（エラー: ${codexResult.reason?.message || "不明なエラー"}）`,
      "utf-8"
    );
    console.error(`[エラー] Codex: ${codexResult.reason?.message}`);
  }

  // 統合結果を生成・保存
  const resultContent = formatResult(codexResult);
  const resultPath = join(outputDir, "debate-result.md");
  writeFileSync(resultPath, resultContent, "utf-8");
  console.error(`\n[保存] ${resultPath}`);

  // 結果をJSON形式でも出力
  const jsonResult = {
    topic,
    claudeOpinion,
    sharedContext: sharedContext || null,
    codex: {
      status: codexResult.status,
      content:
        codexResult.status === "fulfilled"
          ? codexResult.value
          : codexResult.reason?.message,
    },
    outputDir,
    timestamp: new Date().toISOString(),
  };

  const jsonPath = join(outputDir, "debate-result.json");
  writeFileSync(jsonPath, JSON.stringify(jsonResult, null, 2), "utf-8");
  console.error(`[保存] ${jsonPath}`);

  // 標準出力にJSONを出力
  console.log(JSON.stringify(jsonResult, null, 2));

  console.error("\n=== 完了 ===");
}

main().catch((error) => {
  console.error("エラーが発生しました:", error?.message ?? error);
  process.exit(1);
});
