# 動画自動生成システム × 易経DB 統合戦略

## 現状分析

### 既存の動画生成システム
**場所**: `/Users/hideakimacbookair/ローカル動画環境`

**特徴**:
- タイムライン型ストーリー（昭和→平成→令和→2035年→2040年）
- 抽象的な概念の変化を追跡（例: 「趣味の価値観の変遷」）
- production_prompts.md による構造化されたスクリプト
- Cut単位での動画生成（YAML仕様）
- ナレーション、視覚演出、情報源の明記

**例**: `2025-12-29_hobby_productivity_zero`
```
Hook（3秒）→ 昭和（5秒）→ 平成（5秒）→ 令和（5秒）
→ 2035年（5秒）→ 2040年（5秒）→ Outro（3秒）
```

### 易経DB
**場所**: `/Users/hideakimacbookair/Library/Mobile Documents/com~apple~CloudDocs/易経変化ロジックDB`

**特徴**:
- 673件の実例データ（人物・企業の変化）
- 八卦による状態分類（乾/坤/震/巽/坎/離/艮/兌）
- 変化のロジック（before_hex → trigger_hex → action_hex → after_hex）
- 成功・失敗パターンの知見
- 診断アルゴリズム

---

## 統合の方向性

### コンセプト: 「抽象概念の変化」を「易経ロジック」で解釈

**Before（現状）**:
- 抽象的な社会変化を時系列で描写
- 「なぜそうなったか」の因果関係が曖昧

**After（統合後）**:
- 各時代の状態を八卦で表現
- 変化の理由を易経ロジックで説明
- データに基づいた説得力のある解釈

---

## 統合アーキテクチャ

### 1. プロジェクト構造の拡張

```
2025-XX-XX_[テーマ名]/
├── production_prompts.md  # 従来通り
├── hexagram_mapping.yaml  # NEW: 易経マッピング
├── case_references.json   # NEW: 参照事例リスト
├── 00_写真/
├── 01_動画/
├── 02_字幕/
├── 03_BGM/
├── 04_ナレーション/
└── output/
```

### 2. hexagram_mapping.yaml の仕様

```yaml
project:
  theme: "趣味という生産性ゼロの行為"
  theme_id: "009"
  scale: "other"  # 社会・文化の変化

timeline:
  # Hook - 現在の状態
  - era: "Hook"
    year: "2025"
    state: "混乱・カオス"
    hexagram: "坎"  # 水・危険・困難
    description: "趣味がないことへの不安、社会的圧力"

  # 昭和 - 義務としての趣味
  - era: "昭和"
    year: "1970-1989"
    state: "停滞・閉塞"
    hexagram: "艮"  # 山・止まる・義務
    description: "上司の趣味に合わせる、出世のための義務的趣味"

  # 平成 - 個の解放
  - era: "平成"
    year: "1989-2019"
    state: "変質・新生"
    hexagram: "兌"  # 沢・喜び・自由
    description: "おひとりさま文化、自己実現としての趣味"

  # 令和 - 個性の証明
  - era: "令和"
    year: "2019-現在"
    state: "成長痛"
    hexagram: "離"  # 火・分離・個性
    description: "SNSでの自己表現、趣味=個性の証明"

  # 2035年 - 効率化の極致
  - era: "2035年"
    year: "2035"
    state: "絶頂・慢心"
    hexagram: "乾"  # 天・強さ・効率
    description: "AI最適化、時間の完全管理、趣味も生産性重視"

  # 2040年 - 崩壊と再発見
  - era: "2040年"
    year: "2040"
    state: "変質・新生"
    hexagram: "坤"  # 地・受容・原点回帰
    description: "効率化の限界、無駄の再評価、趣味の本質回帰"

transitions:
  # Hook → 昭和
  - from_era: "Hook"
    to_era: "昭和"
    from_hex: "坎"
    to_hex: "艮"
    trigger_type: "外部ショック"
    changing_lines: [2, 3]
    logic: "不安（坎）から義務的停滞（艮）へ"

  # 昭和 → 平成
  - from_era: "昭和"
    to_era: "平成"
    from_hex: "艮"
    to_hex: "兌"
    trigger_type: "意図的決断"
    changing_lines: [1, 2]
    logic: "義務（艮）から解放・喜び（兌）へ、おひとりさま革命"
    reference_cases:
      - "OTHR_JP_XXX"  # 平成おひとりさま文化の台頭（仮想事例ID）

  # 平成 → 令和
  - from_era: "平成"
    to_era: "令和"
    from_hex: "兌"
    to_hex: "離"
    trigger_type: "偶発・出会い"
    changing_lines: [2]
    logic: "喜び（兌）から個性表現（離）へ、SNS時代の変容"

  # 令和 → 2035年
  - from_era: "令和"
    to_era: "2035年"
    from_hex: "離"
    to_hex: "乾"
    trigger_type: "外部ショック"
    changing_lines: [1, 3]
    logic: "個性（離）から効率追求（乾）へ、AI最適化の波"
    pattern_type: "Hubris_Collapse"  # 注意: 失敗パターンの兆候

  # 2035年 → 2040年
  - from_era: "2035年"
    to_era: "2040年"
    from_hex: "乾"
    to_hex: "坤"
    trigger_type: "内部崩壊"
    changing_lines: [1, 2, 3]
    logic: "効率の極致（乾）から原点回帰（坤）へ、天地否の克服"
    pattern_type: "Shock_Recovery"
    reference_cases:
      - "CORP_JP_123"  # トヨタのリコール問題（品質第一への回帰）
```

---

## 3. production_prompts.md の拡張

### Before（従来）
```markdown
**Cut 02 (Past - 昭和 - 5秒)**
ナレーション:
昭和
上司の趣味に合わせる
ゴルフも釣りも出世のため
```

### After（易経統合版）
```markdown
**Cut 02 (Past - 昭和 - 5秒)**

易経解釈:
- 卦: 艮（☶ 山）- 止まる、義務、閉塞
- 状態: 停滞・閉塞
- ロジック: 個人の意思が抑圧され、組織のルールに従う時代

ナレーション（Ver.A 従来版）:
昭和
上司の趣味に合わせる
ゴルフも釣りも出世のため

ナレーション（Ver.B 易経版）:
昭和・艮の時代
止まった個性
義務としての趣味
上司に合わせる山のように

視覚的演出:
- 主人公がゴルフウェアでクラブを磨いている
- 【NEW】背景に艮（☶）のシンボルが薄く浮かぶ
- 【NEW】画面下部に「艮 - 義務の時代」のテキスト（オプション）
- 上司が満足げに見下ろす
- タバコの煙、強制された笑顔

易経的解説（YouTube説明欄用）:
昭和時代を易経の「艮（山）」で表現しました。艮は「止まる」「蓄積」を意味し、個人の自由が抑圧され、組織の一員として義務を果たす時代を象徴しています。データベース内の同様のパターン（艮の状態）を持つ事例では、外部からの変化がなければ停滞が続く傾向があります。
```

---

## 4. 動画生成への組み込み

### 4-1. ナレーション生成の強化

**現状**: 手動でナレーション文を作成

**改善**: 易経DBから自動生成

```python
# scripts/generate_narration_with_hexagram.py

from pathlib import Path
import yaml
import json
from schema_v3 import Hex

def load_hexagram_mapping(project_path: Path) -> dict:
    """hexagram_mapping.yamlを読み込み"""
    mapping_file = project_path / "hexagram_mapping.yaml"
    with open(mapping_file) as f:
        return yaml.safe_load(f)

def get_hexagram_description(hex_name: str) -> dict:
    """八卦の意味を取得"""
    descriptions = {
        "乾": {"symbol": "☰", "meaning": "天・創造・剛健", "keyword": "強さ"},
        "坤": {"symbol": "☷", "meaning": "地・受容・柔順", "keyword": "基盤"},
        "震": {"symbol": "☳", "meaning": "雷・動き・奮起", "keyword": "衝撃"},
        "巽": {"symbol": "☴", "meaning": "風・浸透・柔軟", "keyword": "対話"},
        "坎": {"symbol": "☵", "meaning": "水・危険・困難", "keyword": "試練"},
        "離": {"symbol": "☲", "meaning": "火・明知・分離", "keyword": "才能"},
        "艮": {"symbol": "☶", "meaning": "山・止まる・蓄積", "keyword": "義務"},
        "兌": {"symbol": "☱", "meaning": "沢・喜び・和悦", "keyword": "自由"},
    }
    return descriptions.get(hex_name, {})

def generate_narration_script(era_data: dict, version: str = "standard") -> str:
    """
    ナレーション台本を生成

    Args:
        era_data: timeline中の1つの時代データ
        version: "standard" or "hexagram"
    """
    hex_name = era_data.get("hexagram")
    hex_info = get_hexagram_description(hex_name)

    if version == "standard":
        # 従来版（易経要素なし）
        return era_data.get("description", "")

    elif version == "hexagram":
        # 易経版
        era = era_data.get("era")
        keyword = hex_info.get("keyword", "")
        symbol = hex_info.get("symbol", "")

        template = f"""
{era}・{hex_name}の時代
{symbol}
{keyword}
{era_data.get('description', '')}
        """
        return template.strip()

def generate_full_script(project_path: Path, version: str = "hexagram"):
    """プロジェクト全体のナレーション台本を生成"""
    mapping = load_hexagram_mapping(project_path)

    script = []
    for i, era in enumerate(mapping['timeline'], 1):
        narration = generate_narration_script(era, version)

        script.append(f"""
**Cut {i:02d} ({era['era']} - {era['year']})**

易経解釈:
- 卦: {era['hexagram']} ({get_hexagram_description(era['hexagram'])['symbol']})
- 意味: {get_hexagram_description(era['hexagram'])['meaning']}
- 状態: {era['state']}

ナレーション:
{narration}

---
        """)

    return "\n".join(script)

# 使用例
if __name__ == "__main__":
    project_path = Path("/Users/hideakimacbookair/ローカル動画環境/2025-12-29_hobby_productivity_zero")
    script = generate_full_script(project_path, version="hexagram")

    output_file = project_path / "narration_script_hexagram.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"ナレーション台本を生成: {output_file}")
```

---

### 4-2. 視覚演出への組み込み

**八卦シンボルのオーバーレイ**

```python
# scripts/add_hexagram_overlay.py

from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_hexagram_symbol(hex_name: str, size: int = 200, opacity: float = 0.3):
    """
    八卦のシンボル画像を生成

    Args:
        hex_name: 八卦の名前（"乾", "坤"など）
        size: 画像サイズ（px）
        opacity: 不透明度（0.0-1.0）
    """
    symbols = {
        "乾": "☰", "坤": "☷", "震": "☳", "巽": "☴",
        "坎": "☵", "離": "☲", "艮": "☶", "兌": "☱",
    }

    # 透明な背景画像を作成
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # フォント設定（システムフォント使用）
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Hiragino Sans GB.ttc", size=int(size * 0.6))
    except:
        font = ImageFont.load_default()

    # シンボルを描画
    symbol = symbols.get(hex_name, "")
    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size - text_width) // 2
    y = (size - text_height) // 2

    # 白色、指定された不透明度
    draw.text((x, y), symbol, fill=(255, 255, 255, int(255 * opacity)), font=font)

    return np.array(img)

def add_hexagram_to_video(video_path: str, hex_name: str, output_path: str,
                          position: str = "top-right", opacity: float = 0.3):
    """
    動画に八卦シンボルをオーバーレイ

    Args:
        video_path: 元動画のパス
        hex_name: 八卦の名前
        output_path: 出力動画のパス
        position: 配置位置（"top-right", "center"など）
        opacity: シンボルの不透明度
    """
    # 動画を読み込み
    video = VideoFileClip(video_path)

    # 八卦シンボルを生成
    symbol_img = create_hexagram_symbol(hex_name, size=200, opacity=opacity)
    symbol_clip = ImageClip(symbol_img, duration=video.duration)

    # 配置位置を計算
    if position == "top-right":
        symbol_clip = symbol_clip.set_position((video.w - 250, 50))
    elif position == "center":
        symbol_clip = symbol_clip.set_position("center")

    # 動画とシンボルを合成
    final = CompositeVideoClip([video, symbol_clip])

    # 出力
    final.write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f"シンボル付き動画を生成: {output_path}")

# 使用例
if __name__ == "__main__":
    add_hexagram_to_video(
        video_path="01_動画/cut_02_昭和.mp4",
        hex_name="艮",
        output_path="01_動画/cut_02_昭和_with_hexagram.mp4",
        position="top-right",
        opacity=0.25
    )
```

---

### 4-3. YouTube説明欄の自動生成

```python
# scripts/generate_youtube_description.py

def generate_youtube_description(project_path: Path) -> str:
    """
    易経解説を含むYouTube説明欄を生成
    """
    mapping = load_hexagram_mapping(project_path)

    description = f"""
{mapping['project']['theme']}

このショート動画は、易経（八卦）の変化ロジックに基づいて構成されています。
実例データベース673件の分析から導き出されたパターンを活用しています。

【各時代の易経解釈】

"""

    for era in mapping['timeline']:
        hex_info = get_hexagram_description(era['hexagram'])
        description += f"""
■ {era['era']}（{era['year']}）- {era['hexagram']} {hex_info['symbol']}
　意味: {hex_info['meaning']}
　状態: {era['state']}
　解説: {era['description']}

"""

    description += """

【変化のロジック】

"""

    for transition in mapping.get('transitions', []):
        description += f"""
{transition['from_era']} → {transition['to_era']}
　{transition['from_hex']} → {transition['to_hex']}
　トリガー: {transition['trigger_type']}
　ロジック: {transition['logic']}

"""

    description += """

【易経データベースについて】
このチャンネルでは、673件の実例（企業・個人・社会の変化）を易経の八卦で分析したデータベースを活用しています。

詳しくはこちら: [診断ツールURL]

#易経 #八卦 #変化のロジック #データサイエンス
    """

    return description
```

---

## 5. ワークフロー統合

### 新しい動画制作フロー

```
【Step 1】テーマ決定
└─ 例: 「趣味という生産性ゼロの行為」

【Step 2】易経マッピング設計
├─ hexagram_mapping.yaml 作成
├─ 各時代を八卦で表現
├─ 変化のロジックを定義
└─ 類似事例を易経DBから検索

【Step 3】ナレーション生成
├─ generate_narration_with_hexagram.py 実行
├─ 標準版と易経版の2バージョン生成
└─ production_prompts.md に統合

【Step 4】動画生成
├─ 従来の動画生成フロー
└─ add_hexagram_overlay.py でシンボル追加（オプション）

【Step 5】YouTube公開
├─ generate_youtube_description.py で説明欄生成
└─ 易経解説を含む説明文で公開
```

---

## 6. A/Bテスト戦略

### パターン1: 易経要素なし（従来版）
- ナレーション: 標準版
- 視覚: シンボルなし
- 説明欄: 簡易版

### パターン2: 易経要素あり（統合版）
- ナレーション: 易経版
- 視覚: 八卦シンボルオーバーレイ
- 説明欄: 易経解説フル

### パターン3: ハイブリッド
- ナレーション: 標準版
- 視覚: シンボルのみ追加
- 説明欄: 易経解説フル

**測定指標**:
- 視聴維持率
- エンゲージメント率（いいね・コメント）
- 診断ツールへのクリック率

---

## 7. 実装優先順位

### Phase 1（即実装可能）
1. ✅ hexagram_mapping.yaml のテンプレート作成
2. ✅ generate_narration_with_hexagram.py 実装
3. ✅ generate_youtube_description.py 実装

### Phase 2（1週間以内）
4. ⏸️ add_hexagram_overlay.py 実装
5. ⏸️ 既存プロジェクトへの適用（hobby_productivity_zero）
6. ⏸️ A/Bテスト実施

### Phase 3（2週間以内）
7. ⏸️ 易経DB参照機能（類似事例の自動検索）
8. ⏸️ ワークフロー自動化スクリプト
9. ⏸️ テンプレート化・ドキュメント整備

---

## 8. 期待される効果

### コンテンツ品質
- ✅ 変化の理由が明確に（易経ロジックで説明）
- ✅ データに基づいた説得力
- ✅ 差別化された独自性

### 集客・エンゲージメント
- ✅ 易経に興味がある層を獲得
- ✅ 説明欄からの診断ツール誘導
- ✅ シリーズ化しやすい（他のテーマも同じフレームワーク）

### 制作効率
- ✅ ナレーション自動生成
- ✅ 説明欄自動生成
- ✅ 易経マッピングの再利用

---

## 9. サンプル: hobby_productivity_zero の易経版

次のメッセージで、実際の `hexagram_mapping.yaml` と生成されたナレーション台本のサンプルを提供します。

---

これで動画生成システムと易経DBの統合戦略が完成です。
統合により、単なる時系列の変化ではなく、「なぜそうなったか」を易経のロジックで説得力を持って説明できるようになります！
