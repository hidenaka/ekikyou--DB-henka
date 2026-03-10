# Codex (OpenAI GPT-5.4) の批評

### 結論（最初に明言）
全修正完了という判定は誤りである。現状は再レビュー不合格であり、フロント更新とクローズドベータへ進める段階ではない。

### 批判的分析
1. 危機検出は4入口に追加されたが、安全設計としては未達である。`_check_crisis_input()` は否定表現をテキスト全体で一度でも見つけると全カテゴリを無効化するため、「死にたいわけじゃない。でも首を吊って消えたい。」「自殺するつもりはないが、DVされて殴られている。」がどちらも非検出になる。これは危険な偽陰性である。[app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L184) [docs/safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/docs/safety_design.md#L7)

2. カテゴリ分離も不完全である。仕様は「他害」「依存」「重度抑うつ」「経済的危機」を含むが、実装は自傷とDVの2系統しか持たない。しかも `殺したい` は自傷カテゴリに入っており、他害を自傷ホットラインへ誤誘導する。さらに「借金と取り立てで限界です」は未検出だった。これはトリアージとして誤りである。[app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L133) [docs/safety_design.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/docs/safety_design.md#L15)

3. `format` の400化は実装されたが、実装順序が壊れている。`/api/feedback` は `format` を検証する前に `s["phase"] = "result"` を書き込むため、無効 `format` で一度400を返した後、正しい `format` で再送しても「現在のフェーズでは実行できません」で詰む。入力バリデーションがセッション破壊を起こしている。[app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L513) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L553)

4. 決定的ランキングの説明は事実に反する。実装は `year` 降順で並べているが、`cases.jsonl` の11,336件を確認すると `year` が埋まっている件数は0件だった。したがって実際の並びは「年度降順」ではなく、ほぼ `target_name` 昇順である。決定性はあるが、主張された順位根拠は虚偽である。[case_search.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/scripts/case_search.py#L244) [cases.jsonl](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/data/raw/cases.jsonl)

5. 品質ゲートは「ある」だけで、「効いている」ではない。5point側のゲートは「禁止と推奨が同一」「空欄」「母集団0」しか見ておらず、`matched_n == 0` ですら警告しない。実際に存在しない状態・行動を入れると `matched_n=0` なのに `quality_warnings=[]` だった。「384ポジション警告ゼロ」は、ゲートが甘すぎるだけで品質証明ではない。[feedback_engine.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/scripts/feedback_engine.py#L250) [feedback_engine.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/scripts/feedback_engine.py#L349) [docs/quality_gates_report.md](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/docs/quality_gates_report.md#L19)

6. テスト通過は根拠として弱い。私は `python3 -m pytest -q tests/test_web_app.py tests/test_diary_mode.py tests/test_integration.py` を実行し、`200 passed, 2 warnings` を確認した。しかしそのテスト群は危機検出の反例、`feedback_5point`、無効 `format` 後の再送、決定的ランキングをほぼ検証していない。既存テストは主に正常系である。[test_web_app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/tests/test_web_app.py#L127) [test_web_app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/tests/test_web_app.py#L354) [test_diary_mode.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/tests/test_diary_mode.py#L525) [test_integration.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/tests/test_integration.py#L260)

### Claudeの意見への反論/修正
- 正しい部分はある。危機検出の呼び出し自体は4エンドポイントに入った。[app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L318) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L387) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L594) [app.py](/Users/hideakimacbookair/Library/Mobile%20Documents/com~apple~CloudDocs/%E6%98%93%E7%B5%8C%E5%A4%89%E5%8C%96%E3%83%AD%E3%82%B7%E3%82%99%E3%83%83%E3%82%AFDB/app.py#L624)

- 「否定文脈除外を修正済み」は誤りである。修正ではなく、全文否定フラグという危険な近道である。見逃しを増やしている。

- 「DV/虐待と自傷を分離したので十分」は誤りである。他害と経済危機が落ちており、カテゴリ設計は未完成である。

- 「無効format値で400エラー」は部分的に正しい。しかし400の副作用でセッションが壊れるので、完成扱いはできない。

- 「決定的事例ランキング（年度降順→名前昇順）」は誤りである。実装はそのつもりだが、データ側に `year` が無いので年度降順は実現していない。

- 「384ポジション全て品質警告ゼロ」は誤導である。ゼロ警告は品質の証拠ではなく、ゲートの弱さの証拠に近い。

### より正確な見解
正確な評価はこうである。前回の露骨な欠陥は一部塞がった。しかし「全修正完了確認」は成立していない。現状はベータ前の修正継続フェーズである。

優先順位は明確である。  
1. 否定文脈を全文判定から局所判定へ変更し、他害・依存・重度抑うつ・経済危機を仕様どおり実装する。  
2. `/api/feedback` は `format` 検証を `phase` 更新より前に移し、無効入力でセッションを壊さないようにする。  
3. 類似事例順位は `period` から年を正規化するか、データに `year` を実格納する。  
4. 5point品質ゲートに `matched_n==0`、禁止語、根拠欠落、空の反対視点を追加する。  
5. 今回露呈した反例を pytest に固定し、`tests/regression_cases.json` を未使用ファイルのまま放置しない。  

判定は一言で十分である。**Claudeの「残はフロントエンド更新とクローズドベータのみ」という結論は甘い。残っているのはUIではなく、バックエンドの安全性と検証の本体である。**
