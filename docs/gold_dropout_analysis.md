# Gold未達原因分析レポート

## サマリー

| 指標 | 値 |
|------|-----|
| verified事例総数 | 1,481件 |
| Gold到達 | 930件 (62.8%) |
| Gold未達 | 551件 (37.2%) |
| 目標 | 1,100件 |
| 不足 | 170件 |

## Gold未達の内訳

### 1. Quarantine送り: 295件 (19.9%)

| 理由 | 件数 | 説明 |
|------|------|------|
| only_rejected_urls | 292件 | Google検索URLのみ |
| anonymous | 3件 | 匿名事例（「XXさん」等） |

**改善方法**:
- 292件は信頼できるソースURLを追加すればGold昇格可能
- 3件は匿名のため構造的に救済不可

### 2. Silver送り: 146件 (9.9%)

| ソースTier | 件数 | 説明 |
|------------|------|------|
| tier3_specialist | 145件 | Wikipedia等の二次情報源 |
| tier5_pr | 1件 | PRサイトのみ |

**改善方法**:
- tier1/tier2ソースを追加発見すればGold昇格可能
- Wikipedia参照事例は一次情報源への差し替えが必要

### 3. Bronze送り: 110件 (7.4%)

ソースドメインがホワイトリスト未登録:

| ドメイン | 件数 | 追加検討 |
|----------|------|----------|
| fundbook.co.jp | 6件 | M&A仲介（検討余地） |
| pro-d-use.jp | 5件 | 経営コンサル |
| samsung.com | 1件 | **追加推奨** |
| tsmc.com | 1件 | **追加推奨** |
| その他中小サイト | 97件 | 個別判断 |

**改善方法**:
- 大手企業ドメイン（samsung, tsmc等）をtier4_corporateに追加
- 業界団体サイトをtier1_officialに追加

## 目標達成へのロードマップ

### 即座に改善可能（+30〜50件見込み）

1. **tier4_corporate追加**:
   - samsung.com, tsmc.com, kyocera.co.jp, fujitsu.com
   - nec.com, jal.com, keyence.co.jp
   - berkshirehathaway.com

2. **tier1_official追加**（業界団体）:
   - riaj.or.jp（日本レコード協会）
   - eiren.org（映倫）
   - joc.or.jp（日本オリンピック委員会）
   - whc.unesco.org（UNESCO世界遺産）

### 中期的改善（+100〜150件見込み）

3. **Quarantine救済**: only_rejected_urls 292件に対し、
   信頼ソースを手動/自動で補完

4. **Silver昇格**: tier3ソースのみ事例に、tier1-2ソースを追加

## 構造的制約

以下は改善困難:

| 原因 | 件数 | 理由 |
|------|------|------|
| 匿名事例 | 3件 | 実名化不可 |
| ソース消失 | 不明 | リンク切れは検出困難 |

## 次のアクション

1. [ ] tier4_corporateに大手企業8社追加
2. [ ] tier1_officialに業界団体4組織追加
3. [ ] Quarantine 292件のソース補完タスク作成
4. [ ] Silver 145件の一次情報源調査
