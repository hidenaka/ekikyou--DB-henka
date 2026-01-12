#!/bin/bash
# オートセーブ: 応答完了時・Task完了時・セッション終了時に自動コミット＆プッシュ

cd "$CLAUDE_PROJECT_DIR" || exit 1

# 変更が存在するか確認
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "[AutoSave] No changes to commit"
    exit 0
fi

# ステージングに追加
git add -A

# 変更ファイル数をカウント
CHANGED_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')
SUMMARY=$(git diff --cached --stat | tail -1)

# コミット作成（変更内容サマリ付き）
git commit -m "$(cat <<EOF
AutoSave: ${CHANGED_FILES}ファイル変更

${SUMMARY}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# プッシュ実行
git push origin HEAD 2>/dev/null || echo "[AutoSave] Push skipped (no remote or offline)"

echo "[AutoSave] Completed: $(date '+%Y-%m-%d %H:%M:%S')"
exit 0
