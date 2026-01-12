#!/bin/bash
# オートセーブ: 応答完了時・Task完了時・セッション終了時に自動コミット＆プッシュ
# 日本語で詳細なコミットメッセージを生成

cd "$CLAUDE_PROJECT_DIR" || exit 1

# 変更が存在するか確認
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "[AutoSave] No changes to commit"
    exit 0
fi

# ステージングに追加
git add -A

# 変更ファイルをカテゴリ別に分類
DATA_FILES=$(git diff --cached --name-only | grep -E '^data/' | wc -l | tr -d ' ')
SCRIPT_FILES=$(git diff --cached --name-only | grep -E '^scripts/' | wc -l | tr -d ' ')
DOC_FILES=$(git diff --cached --name-only | grep -E '\.(md|txt)$' | wc -l | tr -d ' ')
CONFIG_FILES=$(git diff --cached --name-only | grep -E '\.(json|yaml|yml|sh)$' | wc -l | tr -d ' ')
TOTAL_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')

# 主要な変更ファイル（最大5件）
MAIN_FILES=$(git diff --cached --name-only | head -5 | xargs -I {} basename {} | paste -sd ', ' -)

# 変更タイプを判定
CHANGE_TYPE=""
if [ "$DATA_FILES" -gt 0 ]; then
    CHANGE_TYPE="${CHANGE_TYPE}データ更新 "
fi
if [ "$SCRIPT_FILES" -gt 0 ]; then
    CHANGE_TYPE="${CHANGE_TYPE}スクリプト変更 "
fi
if [ "$DOC_FILES" -gt 0 ]; then
    CHANGE_TYPE="${CHANGE_TYPE}ドキュメント更新 "
fi
if [ "$CONFIG_FILES" -gt 0 ]; then
    CHANGE_TYPE="${CHANGE_TYPE}設定変更 "
fi

# 変更タイプがなければデフォルト
if [ -z "$CHANGE_TYPE" ]; then
    CHANGE_TYPE="ファイル更新"
fi

# 日時
NOW=$(date '+%Y-%m-%d %H:%M')

# コミットメッセージ生成
git commit -m "$(cat <<EOF
[AutoSave] ${CHANGE_TYPE}(${TOTAL_FILES}件)

📅 ${NOW}
📁 主な変更: ${MAIN_FILES}

詳細:
- データファイル: ${DATA_FILES}件
- スクリプト: ${SCRIPT_FILES}件
- ドキュメント: ${DOC_FILES}件
- 設定ファイル: ${CONFIG_FILES}件

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# プッシュ実行
git push origin HEAD 2>/dev/null || echo "[AutoSave] Push skipped (no remote or offline)"

echo "[AutoSave] Completed: ${NOW}"
exit 0
