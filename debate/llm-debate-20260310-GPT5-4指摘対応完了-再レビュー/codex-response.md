# Codex (OpenAI GPT-5.4) の批評

### 結論（最初に明言）
**再レビューは不合格である。**  
3つの修正のうち、`DV` パターン拡充と `format` 周りの不具合修正は成立している。だが、**「対応完了」は誤り**である。安全仕様、根拠仕様、テスト主張の3点でまだ破綻している。

### 批判的分析
- **論理的整合性が崩れている。**  
  `REG_018` は仕様でも回帰ケースでも `HIGH` 期待なのに、Claudeは `CRITICAL` 検出を成功扱いしている。これは合格ではない。`HIGH` 経路を実装せず `critical` に潰しているだけである。[regression_cases.json](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/regression_cases.json#L479) [safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L17) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L136)

- **修正2は実装されているが、仕様違反のままである。**  
  `matched_n < 3` で警告と `evidence_label` を出す処理は入った。フロント表示も入っている。ここは事実である。[feedback_engine.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/feedback_engine.py#L365) [static/index.html](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/static/index.html#L2280)  
  しかし根拠仕様は「3件未満ならレポート生成を中止し、返金対応」である。警告付き継続は仕様違反である。[evidence_specification.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/evidence_specification.md#L58)

- **修正1は半分しか成功していない。**  
  実確認では `REG_015` は `MEDIUM`、`REG_018` は検出された。だが `REG_016` は現在の `_check_crisis_input()` で **`None`** だった。原因は `economic_crisis` の語彙が狭すぎるからである。`借金が返せない` は拾うが、実際の回帰ケース本文である「借金は残り800万円」「返済が重い」は拾えない。[app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L189) [regression_cases.json](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/regression_cases.json#L423)

- **MEDIUM判定の質も低い。**  
  仕様は重度抑うつを「2週間以上」で扱うのに、実装は期間条件を見ていない。`今日は食欲がない。` や `一日だけ布団から出られない。` でも `MEDIUM` になる。これは偽陽性である。[safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L19) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L175)

- **安全設計は依然として未完成である。**  
  仕様は Layer 1 / Layer 2 / Layer 3 の3層、人間通知、ログ記録を要求している。実装はキーワード一致と `phase` 更新しかしていない。`HIGH` も `FALSE_POSITIVE` も存在しない。依存症も未実装である。[safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L7) [safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L44) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L253)

- **スケーラビリティがない。**  
  これは安全機能ではなく、語彙のモグラ叩きである。実確認でも `復讐したい。` `ギャンブルがやめられない。` `薬物をやめられない。` はすべて非検出だった。ルール追加を続けても破綻する。

- **テスト主張が事実ではない。**  
  `python3 -m pytest -q tests/test_web_app.py tests/test_diary_mode.py tests/test_integration.py` は `200 passed, 2 warnings`。  
  `python3 -m pytest -q` は **`520 passed, 5 errors`** である。したがって「515テスト全パス」は誤りである。  
  さらに `REG_015`〜`REG_018` は `tests/regression_cases.json` にあるだけで、テストコードから参照されていない。E2E成功の主張は自動保証ではない。

- **新規性はない。**  
  今回の差分は「カテゴリ追加」「警告ラベル」「バナー表示」である。新しい安全推論でも、新しい検証枠組みでもない。これは補修であって完成ではない。

### Claudeの意見への反論/修正
- 正しい部分はある。`evidence_label` のバックエンド付与とフロント表示は実装されている。`DV` の追加パターンも `REG_018` の本文を拾う。
- `format` の400化も前回の欠陥は直っている。無効 `format` の後に正しい `format` で再送しても、セッションは壊れない。
- だが `REG_016` 成功は誤りである。現行コードでは落ちる。
- `REG_018 CRITICAL成功` という評価も誤りである。仕様と回帰ケースの期待値は `HIGH` であり、そこを `CRITICAL` に潰して通したことは改善ではない。
- 「未対応項目はベータ中に対応可能」という判断は甘い。Layer 2、Layer 3、監査ログ、返金導線、10ケース検証は**ベータ中にやる改善項目ではなく、ベータ前提条件**である。

### より正確な見解
**正しい評価はこうである。**  
修正3は成功。修正2は実装成功だが仕様違反。修正1は部分成功で、経済危機と `HIGH` 設計が未達である。したがって全体判定は不合格である。

次にやるべきことは明確である。
1. `HIGH` を復活させ、DVを `HIGH` として正しく扱う。依存症・他害も仕様どおり入れる。
2. `economic_crisis` と `depression` を表現拡張し、期間条件や文脈条件を入れる。regexの足し算だけでは足りない。
3. `matched_n < 3` は仕様どおり停止に戻す。継続したいなら、先に仕様を正式改訂しろ。
4. `tests/regression_cases.json` を pytest に組み込み、`REG_015`〜`REG_018` を強制しろ。
5. `validation/runs/` と `validation/summary.md` を実際に作り、10ケースの Go/No Go を埋めろ。
6. 監査ログ、人間通知、永続セッションを実装しろ。

**要するに、今回の状態は「部分修正済み・再レビュー要」であって、「指摘対応完了」ではない。**
