# Codex (OpenAI GPT-5.4) の批評

### 結論（最初に明言）
この再修正は**不合格**である。前回の露骨な穴は一部塞がったが、安全設計・品質保証・検索順位づけのどれも「動く」段階にとどまり、「信用して出せる」段階に達していない。

### 批判的分析
1. **安全設計と実装が矛盾している。**
設計書では Layer 1 はフラグ付与のみで、遮断は Layer 2 以降で行うと定義しているが、実装はキーワード一致だけで即遮断している。これは改善ではなく設計逸脱である。
[docs/safety_design.md#L22](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L22)
[app.py#L273](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L273)

2. **複合危機で重大度を誤る。**
危機判定はカテゴリ順に走査し、`critical/high` を見つけた瞬間に return している。したがって「DV + 自傷」の複合入力でも、先に DV を拾えば `HIGH` で終了し、`CRITICAL` な自傷を見落とす。これは安全機能として致命的な欠陥である。
[app.py#L134](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L134)
[app.py#L264](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L264)

3. **MEDIUM 案内は実際には届かない。**
標準フローでは `extract/followup` で `safety_flag` を保存するだけで、その場では返していない。日記フローでは保存すら返却経路に乗っていない。つまり「カテゴリ別の専門機関案内を実装した」という主張は、少なくとも MEDIUM については実装実態と一致しない。
[app.py#L405](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L405)
[app.py#L450](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L450)
[app.py#L653](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L653)
[app.py#L697](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L697)

4. **否定文脈除外は文脈理解ではなく、脆弱な後置否定パッチにすぎない。**
実装は「キーワードの直後 5 文字以内」に6語の否定句があるかしか見ていない。これは「死にたい気持ちはない」「自殺なんてしない」「友人が『死にたい』と言った」などで簡単に破綻する。自然言語の否定スコープを扱えていない。
[app.py#L204](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L204)
[app.py#L244](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L244)

5. **品質ゲートは「ゲート」ではない。**
実装は warning を配列に積むだけで、生成停止も再生成も fail-close もしていない。これは lint であって gate ではない。「384ポジションで警告ゼロ」は、軽い構文検査を通っただけという意味しか持たない。
[scripts/feedback_engine.py#L349](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/feedback_engine.py#L349)
[scripts/feedback_engine.py#L658](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/feedback_engine.py#L658)

6. **決定的ランキングは再現性を得ただけで、妥当性を得ていない。**
年度降順・名前昇順は deterministic だが relevance-aware ではない。古くても本質的に近い事例を落とし、名前順という意味のないタイブレークに依存している。これは検索品質の改善ではなく、並び順の固定化である。
[scripts/case_search.py#L231](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/case_search.py#L231)

7. **検証規模が小さすぎる。**
安全テストは標準フローに9件あるだけで、日記モードには危機検出の検証がない。これで「全4エンドポイントに適用して確認済み」と言い切るのは不正確である。
[tests/test_web_app.py#L540](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/test_web_app.py#L540)
[tests/test_diary_mode.py#L525](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/test_diary_mode.py#L525)

8. **運用前提が未整備で、クローズドベータ移行判断は早すぎる。**
セッションはインメモリで、設計書が要求する Layer 2、Layer 3、ログ運用も未実装である。これは「残タスクは10件ドライランだけ」という段階ではない。
[app.py#L92](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L92)
[docs/safety_design.md#L26](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L26)
[debate/llm-debate-20260310-GPT5-4指摘対応完了-再レビュー/claude-opinion.md#L27](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/debate/llm-debate-20260310-GPT5-4指摘対応完了-再レビュー/claude-opinion.md#L27)

### Claudeの意見への反論/修正
Claudeが正しいのは、`format` バリデーション追加とランダムシャッフル除去が**局所的な衛生改善**として有効だという点だけである。

Claudeが誤っている点は3つある。
- 「危機検出改善」は過大評価である。実態はキーワード即時遮断であり、文脈理解でも多段防御でもない。
- 「品質ゲート実装」は誤記である。実装は warning 出力であって gate ではない。
- 「残タスクは少ない」は誤りである。安全系で未実装の Layer 2、Layer 3、監査ログ、永続化が残っている以上、ベータ前提条件が未充足である。

加えて、DV の重大度は説明文の間で揺れている。安全ポリシーの分類基準が説明ごとに変わる状態そのものが管理不全である。

### より正確な見解
本件の本質は「修正が入ったか」ではない。**安全判定の優先順位、ユーザーへの即時案内、fail-closeな品質統制、再現性と関連性を両立した検索順位、そして運用監査**が成立しているかである。現状はどれも未達である。

より正しい次の一手は明確である。
- 危機判定は「最初の一致」ではなく「全一致収集 → 最大重大度選択 → 多ラベル保持」に直す。
- MEDIUM/HIGH/CRITICAL の案内は、検出したそのエンドポイントで即時返す。日記系にも同じ契約を適用する。
- 否定処理は局所語句一致ではなく、少なくとも正規化・引用/第三者言及除外・否定スコープ拡張を入れる。
- 品質ゲートは warning ではなく、失敗時に生成停止または再生成する本当の gate にする。
- ランキングは `priority tier → 類似度 → 根拠品質 → 年度 → 安定ID` の順にする。
- ベータ前に、複合危機・否定・言い換え・日記系を含む対向テスト群を作る。10件ドライランでは足りない。少なくとも危機カテゴリ × 文脈種別 × endpoint の直交表が必要である。

判定を一文で言い切る。**この修正は「前回よりまし」ではあるが、「ベータ投入可」ではない。**
