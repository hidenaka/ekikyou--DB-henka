# Codex (OpenAI GPT-5.4) の批評

### 結論（最初に明言）
この実装評価は**不合格**である。  
`/api/feedback` の 5point 追加は「一時的なアダプタ」としては成立しているが、**危機ケース拒否は安全機能として成立していない**。したがって「2機能とも実装完了」と言うのは誤りである。

### 批判的分析
1. **論理的整合性が崩れている。**  
   「LLM呼び出し前の入口で検査しているから安全」という説明は誤りである。危機検査は [`/app.py:283`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L283) と [`/app.py:550`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L550) にしかなく、自由記述を受ける `/api/followup` と `/api/diary/ideal-followup` には存在しない [`/app.py:345`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L345) [`/app.py:576`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L576)。中立入力で開始し、追加入力で危機内容を書くとそのまま通る。入口防御ではない。  
   さらに安全仕様は Layer 1 キーワード、Layer 2 LLM分類、Layer 3 人間判断の3層である [`/docs/safety_design.md:7`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L7) [`/docs/safety_design.md:44`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L44)。実装はそれを1本の正規表現拒否に潰している。仕様違反である。

2. **実現可能性はあるが、運用実装が欠落している。**  
   危機検知後にやっていることは `phase = crisis_rejected` のセットだけである [`/app.py:285`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L285)。仕様にある人間通知、ログ、返金対応、カテゴリ別導線は存在しない [`/docs/safety_design.md:51`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L51) [`/validation/personas/V10_persona.md:35`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/validation/personas/V10_persona.md#L35)。しかもセッションはインメモリで、監査証跡にならない [`/app.py:92`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L92)。  
   5point もバックエンド追加だけで、フロントは依然として旧 `feedback` しか読んでいない [`/static/index.html:2110`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/static/index.html#L2110) [`/static/index.html:2135`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/static/index.html#L2135)。これを「実装完了」と言うのは甘い。

3. **新規性は誇張である。**  
   5point は新しい推論ではない。既存 5layer を `build_5point_view()` で写像しているだけである [`/scripts/feedback_engine.py:250`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/feedback_engine.py#L250)。新しいのは表示形式だけだ。  
   しかも「今やるな/今やれ」は 384爻ルールを使っている点は正しいが、APIは `source_text` を捨て、`action/reason/strength` だけ返している [`/scripts/feedback_engine.py:291`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/feedback_engine.py#L291) [`/data/diagnostic/yao_action_rules.json`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/data/diagnostic/yao_action_rules.json)。根拠を落とした時点で、実質は「行動カード風の薄い要約」である。

4. **スケーラビリティがない。**  
   危機検知は 15 パターンの literal match しかない [`/app.py:133`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L133)。安全仕様が定義している HIGH/MEDIUM 群をそもそも扱えていない [`/docs/safety_design.md:15`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L15)。実際、回帰ケース `REG_015`、`REG_016`、`REG_018` は期待上 Safety 対象だが、現行正規表現では落ちる [`/tests/regression_cases.json:395`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/regression_cases.json#L395) [`/tests/regression_cases.json:423`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/regression_cases.json#L423) [`/tests/regression_cases.json:479`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/regression_cases.json#L479)。  
   5point 側も類似事例がランダム抽出で、再現性がない [`/scripts/case_search.py:228`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/scripts/case_search.py#L228)。カードUIで毎回事例が揺れる設計は説明責任に耐えない。

5. **反例とエッジケースに弱い。**  
   危機拒否は否定文脈を捌けない。「死にたいわけじゃない」が入っても literal match で即遮断する。仕様が想定している FALSE_POSITIVE/HIGH/MEDIUM 分岐を捨てた副作用である。  
   5point の行動カードも条件付き爻を潰している。たとえば `64_3` の元ルールは「進めば凶」と「大川を渡るに利あり」を同居させるが、APIでは一般化された `do_not/do` に圧縮される。これは条件依存性の喪失である。さらに `format` の不正値を 400 にせず旧形式へ黙ってフォールバックするのも契約として弱い [`/app.py:515`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/app.py#L515)。

6. **より良い代替案が明確に存在する。**  
   危機対応は「regexで全部拒否」ではなく、`Layer1: 広いフラグ` → `Layer2: 文脈分類` → `Layer3: カテゴリ別テンプレート + 監査ログ` に戻すべきである。DVにはDV窓口、借金には法テラスを出すべきで、現行の一律3窓口は誤ったトリアージである [`/docs/safety_design.md:100`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/docs/safety_design.md#L100)。  
   5point は `format=5point` 時に本当に 5point だけ返すか、`include_legacy=true` を別フラグに分けるべきである。今の形は「切替」ではなく「二重返却」である。

### Claudeの意見への反論/修正
- 正しい部分はある。既存 `feedback` を残して新形式を足した判断自体は、移行期の互換策として妥当である。
- 「危機ケース拒否はシンプルで良い」は誤りである。安全機能で単純化は美徳ではない。仕様の3層判定を潰し、しかも後続入力の抜け道まで作っている以上、不良実装である。
- 「LLMコスト節約になる」は副次効果にすぎない。安全性を下げてまで優先する論点ではない。
- 「専門機関の連絡先を案内して処理中断」は不十分である。DV被害者に自殺ホットラインだけ返す設計は雑である。カテゴリ別導線がない。
- 「テスト7件全パス」は根拠にならない。少なくともリポジトリ上の主要テストは `format=5point` も `crisis_rejected` も検証していない [`/tests/test_web_app.py:355`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/test_web_app.py#L355) [`/tests/test_diary_mode.py:525`](/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB/tests/test_diary_mode.py#L525)。この状態で「品質ゲート前にクローズドベータ」は危険である。

### より正確な見解
5point APIは**ベータ前段の内部アダプタとしては可**、危機ケース拒否は**出荷不可**。本質的評価はこれである。

修正優先順位は明確である。  
1. すべての自由記述入口に安全判定を適用する。最低でも `/api/followup` と `/api/diary/ideal-followup` を塞ぐ。  
2. 安全判定を仕様どおり `CRITICAL / HIGH / MEDIUM / FALSE_POSITIVE` に戻す。`REG_015`〜`REG_018` を必須テスト化する。  
3. 危機応答をカテゴリ別テンプレートに分け、ログ・人間通知・監査項目を実装する。  
4. 5point はスキーマを固定し、不正 `format` を 400 にする。必要なら `include_legacy` を別指定にする。  
5. 類似事例のランダム抽出をやめ、決定的順位付けにする。

要するに、**5point は「まだ未完成」だが前に進める。危機ケース拒否は「今のまま進めてはいけない」**。この差を曖昧にしてはいけない。
