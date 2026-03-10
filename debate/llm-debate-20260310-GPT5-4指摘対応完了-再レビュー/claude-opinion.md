# Claude (Anthropic) の見解

GPT-5.4の前回指摘に対して3つの修正を実施した。

【修正1: MEDIUMカテゴリ追加】
- depression: 布団から出られない/ベッドから出られない/食欲がない/食べられない/眠れない日が続
- economic_crisis: 借金が返せ/破産/闇金/取り立て/返済できない
- severity=mediumはサービス遮断せず注意フラグ付き継続（feedbackレスポンスにsafety_flag含む）
- severity=criticalは従来通りサービス遮断

【修正2: 類似事例3件未満ルール】
- matched_n < 3: 警告+evidence_label='事例数不足: 仮説としてのみ提示'
- フロントエンドにもevidence_labelバナー表示追加

【修正3: DV検出パターン拡充】
- 物を投げ/壁を殴/監禁 を追加。REG_018「物を投げるようになりました」を正しく検出

【E2Eテスト結果】
- REG_015(抑うつ): depression MEDIUM検出→サービス継続→feedbackにsafety_flag付き ✅
- REG_016(経済危機): economic_crisis MEDIUM検出→サービス継続→feedbackにsafety_flag付き ✅
- REG_018(DV): dv_abuse CRITICAL検出→サービス遮断 ✅
- 自傷(死にたい): self_harm CRITICAL→遮断 ✅（既存維持）
- 否定(死にたいわけじゃない): 非検出 ✅（既存維持）
- MEDIUM継続: extract→confirm→feedback全フロー完走 ✅
- pytest 515テスト全パス ✅

【未対応項目（ベータ中に対応可能）】
- Layer 2 LLM文脈分類: 将来的にLLMで偽陽性除外（現在はキーワードのみ）
- Layer 3 人間判断: 運用プロセスとして構築
- セッション永続化: クローズドベータは少人数のため当面インメモリで十分
- 監査ログ: ベータ中に実装予定
- 10ケースドライラン: ベータ準備作業として次に実施