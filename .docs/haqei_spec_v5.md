# HaQei診断システム v5 仕様書

**ステータス**: 凍結（実装可能）
**作成日**: 2026-01-13
**根拠**: LLMディベート4回の結果を統合

---

## 1. 概要

### 1.1 目的
ユーザーの現在の状況を5軸で測定し、易経384爻（64卦×6爻）の中から最適な候補をランキング形式で提示する診断システム。

### 1.2 設計原則（ディベート1-4で確定）

| 原則 | 内容 | 根拠 |
|------|------|------|
| 内部軸維持 | Q1-Q5の5軸分類ロジックは有効 | ディベート1 |
| 行動指標翻訳 | 抽象概念を具体的行動・状況で測定 | ディベート1 |
| 混合状態保持 | 確率分布で表現、max方式禁止 | ディベート2 |
| 5問固定 | Phase 1は各軸1問、計5問必須 | ディベート2 |
| JS距離 | コサイン類似度ではなくJensen-Shannon距離 | ディベート3 |
| 384直接ランキング | 卦→爻の二段階ではなく384を直接評価 | ディベート3 |
| ルーブリック必須 | 参照分布は検証可能なプロセスで作成 | ディベート3 |
| 限定ベータ | 公開デプロイではなく限定環境で学習 | ディベート4 |

---

## 2. データモデル

### 2.1 ユーザー回答（UserAnswers）

```typescript
interface UserAnswers {
  // Q1: changeNature（変化の性質）- 4項目それぞれ1-5段階
  changeNature: {
    expansion: number;   // 拡大 (1-5)
    contraction: number; // 収縮 (1-5)
    maintenance: number; // 維持 (1-5)
    transformation: number; // 転換 (1-5)
  };

  // Q2: agency（主体性）- 1-5段階
  agency: number;  // 1=全くコントロールできない, 5=完全にコントロールできる

  // Q3: timeframe（時間軸）- 選択式
  timeframe: 'immediate' | 'shortTerm' | 'midTerm' | 'longTerm' | 'unknown';
  // immediate=1ヶ月以内, shortTerm=3ヶ月以内, midTerm=半年以内, longTerm=1年以上, unknown=わからない

  // Q4: relationship（関係性）- 複数選択
  relationship: {
    self: boolean;      // 自分自身
    family: boolean;    // 家族
    team: boolean;      // 同僚・チーム
    organization: boolean; // 組織全体
    external: boolean;  // 顧客・取引先
    society: boolean;   // 業界・社会
  };

  // Q5: emotionalTone（感情基調）- 4項目それぞれ1-5段階
  emotionalTone: {
    excitement: number;  // ワクワク・期待感 (1-5)
    caution: number;     // 慎重さ・用心深さ (1-5)
    anxiety: number;     // 不安・心配 (1-5)
    optimism: number;    // 楽観・なんとかなる感 (1-5)
  };
}
```

### 2.2 軸の確率分布（AxisDistribution）

```typescript
interface AxisDistribution {
  values: Record<string, number>;  // 各カテゴリの確率（合計1.0）
  entropy: number;                  // エントロピー値（混合度）
  isMissing: boolean;              // 欠損フラグ（「わからない」等）
}

interface UserProfile {
  changeNature: AxisDistribution;  // {拡大, 収縮, 維持, 転換}
  agency: AxisDistribution;        // {自ら動く, 受け止める, 待つ}
  timeframe: AxisDistribution;     // {即時, 短期, 中期, 長期}
  relationship: AxisDistribution;  // {個人, 組織内, 対外}
  emotionalTone: AxisDistribution; // {前向き, 慎重, 不安, 楽観}
}
```

### 2.3 クラスプロファイル（ClassProfile）

```typescript
interface ClassProfile {
  classId: number;       // 1-384
  hexagram: number;      // 1-64
  yao: number;           // 1-6
  name: string;          // 例: "乾為天 初九"

  // 参照分布（ルーブリックから生成）
  distributions: {
    changeNature: Record<string, number>;
    agency: Record<string, number>;
    timeframe: Record<string, number>;
    relationship: Record<string, number>;
    emotionalTone: Record<string, number>;
  };

  // メタデータ
  rubricVersion: string;  // ルーブリックのバージョン
  rubricSource: string;   // 根拠となる易経記述
}
```

### 2.4 マッチング結果（MatchingResult）

```typescript
interface MatchingResult {
  ranking: CandidateScore[];
  missingAxes: string[];      // 欠損扱いにした軸
  overallConfidence: number;  // 全体の測定信頼性 (0-1)
  timestamp: string;
  version: string;            // アルゴリズムバージョン
}

interface CandidateScore {
  classId: number;
  hexagram: number;
  yao: number;
  name: string;

  score: number;              // JS距離ベース（0=完全一致, 1=最大乖離）
  rank: number;               // 順位

  // 寄与度分解
  contributions: {
    changeNature: number;
    agency: number;
    timeframe: number;
    relationship: number;
    emotionalTone: number;
  };

  // 説明用
  matchReasons: string[];     // 「なぜこの結果か」の根拠
  isMixedState: boolean;      // 混合状態フラグ
}
```

---

## 3. アルゴリズム仕様

### 3.1 回答→確率分布変換

#### 3.1.1 changeNature（4カテゴリ）

```typescript
function convertChangeNature(answers: UserAnswers['changeNature']): AxisDistribution {
  const { expansion, contraction, maintenance, transformation } = answers;
  const total = expansion + contraction + maintenance + transformation;

  // ゼロ除算防止
  if (total === 0) {
    return { values: { 拡大: 0.25, 収縮: 0.25, 維持: 0.25, 転換: 0.25 }, entropy: 2.0, isMissing: false };
  }

  const values = {
    拡大: expansion / total,
    収縮: contraction / total,
    維持: maintenance / total,
    転換: transformation / total
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
```

#### 3.1.2 agency（3カテゴリ）

```typescript
function convertAgency(score: number): AxisDistribution {
  // 1-5段階を確率分布に変換
  // ソフトマックス的な変換（隣接カテゴリにも確率を分配）

  const values = {
    自ら動く: 0,
    受け止める: 0,
    待つ: 0
  };

  if (score <= 2) {
    values['待つ'] = (3 - score) / 2;
    values['受け止める'] = 1 - values['待つ'];
  } else if (score >= 4) {
    values['自ら動く'] = (score - 3) / 2;
    values['受け止める'] = 1 - values['自ら動く'];
  } else {
    // score = 3
    values['受け止める'] = 0.6;
    values['自ら動く'] = 0.2;
    values['待つ'] = 0.2;
  }

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
```

#### 3.1.3 timeframe（4カテゴリ + 欠損）

```typescript
function convertTimeframe(selection: UserAnswers['timeframe']): AxisDistribution {
  // 「わからない」は欠損扱い
  if (selection === 'unknown') {
    return {
      values: { 即時: 0.25, 短期: 0.25, 中期: 0.25, 長期: 0.25 },
      entropy: 2.0,
      isMissing: true  // 重み=0でスコア計算から除外
    };
  }

  // one-hotではなく、隣接カテゴリにも確率を分配
  const mapping: Record<string, Record<string, number>> = {
    immediate: { 即時: 0.7, 短期: 0.2, 中期: 0.1, 長期: 0 },
    shortTerm: { 即時: 0.2, 短期: 0.6, 中期: 0.15, 長期: 0.05 },
    midTerm: { 即時: 0.05, 短期: 0.2, 中期: 0.55, 長期: 0.2 },
    longTerm: { 即時: 0, 短期: 0.1, 中期: 0.2, 長期: 0.7 }
  };

  const values = mapping[selection];

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
```

#### 3.1.4 relationship（3カテゴリ）

```typescript
function convertRelationship(selections: UserAnswers['relationship']): AxisDistribution {
  const { self, family, team, organization, external, society } = selections;

  // カテゴリへの寄与度
  let personal = 0, organizational = 0, external_ = 0;

  if (self) personal += 1.0;
  if (family) personal += 0.8;
  if (team) organizational += 0.6;
  if (organization) organizational += 1.0;
  if (external) external_ += 0.8;
  if (society) external_ += 1.0;

  const total = personal + organizational + external_;

  // 何も選択されていない場合
  if (total === 0) {
    return {
      values: { 個人: 0.5, 組織内: 0.3, 対外: 0.2 },
      entropy: 1.5,
      isMissing: false
    };
  }

  const values = {
    個人: personal / total,
    組織内: organizational / total,
    対外: external_ / total
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
```

#### 3.1.5 emotionalTone（4カテゴリ）

```typescript
function convertEmotionalTone(answers: UserAnswers['emotionalTone']): AxisDistribution {
  const { excitement, caution, anxiety, optimism } = answers;
  const total = excitement + caution + anxiety + optimism;

  if (total === 0) {
    return { values: { 前向き: 0.25, 慎重: 0.25, 不安: 0.25, 楽観: 0.25 }, entropy: 2.0, isMissing: false };
  }

  const values = {
    前向き: excitement / total,
    慎重: caution / total,
    不安: anxiety / total,
    楽観: optimism / total
  };

  return {
    values,
    entropy: calculateEntropy(Object.values(values)),
    isMissing: false
  };
}
```

### 3.2 エントロピー計算

```typescript
function calculateEntropy(probabilities: number[]): number {
  let entropy = 0;
  for (const p of probabilities) {
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

// 最大エントロピー（均等分布の場合）
function maxEntropy(n: number): number {
  return Math.log2(n);
}
```

### 3.3 Jensen-Shannon距離

```typescript
function jsDistance(p: Record<string, number>, q: Record<string, number>): number {
  const keys = Object.keys(p);

  // 中間分布 M = (P + Q) / 2
  const m: Record<string, number> = {};
  for (const key of keys) {
    m[key] = (p[key] + q[key]) / 2;
  }

  // JS距離 = sqrt((KL(P||M) + KL(Q||M)) / 2)
  const klPM = klDivergence(p, m);
  const klQM = klDivergence(q, m);

  return Math.sqrt((klPM + klQM) / 2);
}

function klDivergence(p: Record<string, number>, q: Record<string, number>): number {
  let kl = 0;
  for (const key of Object.keys(p)) {
    if (p[key] > 0 && q[key] > 0) {
      kl += p[key] * Math.log2(p[key] / q[key]);
    }
  }
  return kl;
}
```

### 3.4 384クラスのスコア計算

```typescript
function calculateScore(
  userProfile: UserProfile,
  classProfile: ClassProfile
): CandidateScore {
  const axes = ['changeNature', 'agency', 'timeframe', 'relationship', 'emotionalTone'] as const;

  // 各軸の重み（欠損軸は0）
  const weights: Record<string, number> = {};
  let totalWeight = 0;

  for (const axis of axes) {
    if (userProfile[axis].isMissing) {
      weights[axis] = 0;
    } else {
      weights[axis] = 1;  // 初期値は均等、将来的に調整可能
      totalWeight += 1;
    }
  }

  // 正規化
  if (totalWeight > 0) {
    for (const axis of axes) {
      weights[axis] /= totalWeight;
    }
  }

  // 各軸のJS距離を計算
  const contributions: Record<string, number> = {};
  let totalScore = 0;

  for (const axis of axes) {
    if (weights[axis] > 0) {
      const dist = jsDistance(
        userProfile[axis].values,
        classProfile.distributions[axis]
      );
      contributions[axis] = dist * weights[axis];
      totalScore += contributions[axis];
    } else {
      contributions[axis] = 0;
    }
  }

  // 混合状態の判定（高エントロピー軸が2つ以上）
  const highEntropyCount = axes.filter(axis => {
    const maxEnt = maxEntropy(Object.keys(userProfile[axis].values).length);
    return userProfile[axis].entropy > maxEnt * 0.7;
  }).length;

  const isMixedState = highEntropyCount >= 2;

  return {
    classId: classProfile.classId,
    hexagram: classProfile.hexagram,
    yao: classProfile.yao,
    name: classProfile.name,
    score: totalScore,
    rank: 0,  // 後で設定
    contributions: contributions as CandidateScore['contributions'],
    matchReasons: [],  // 後で生成
    isMixedState
  };
}
```

### 3.5 ランキング生成

```typescript
function generateRanking(
  userProfile: UserProfile,
  classProfiles: ClassProfile[]
): MatchingResult {
  // 全384クラスのスコア計算
  const candidates = classProfiles.map(cp => calculateScore(userProfile, cp));

  // スコア昇順でソート（0=完全一致が最良）
  candidates.sort((a, b) => a.score - b.score);

  // 順位付け（同点処理）
  let currentRank = 1;
  let previousScore = -1;

  for (let i = 0; i < candidates.length; i++) {
    if (candidates[i].score !== previousScore) {
      currentRank = i + 1;
    }
    candidates[i].rank = currentRank;
    previousScore = candidates[i].score;
  }

  // 説明文生成
  for (const candidate of candidates.slice(0, 10)) {
    candidate.matchReasons = generateMatchReasons(userProfile, candidate);
  }

  // 欠損軸の収集
  const missingAxes = Object.entries(userProfile)
    .filter(([_, dist]) => dist.isMissing)
    .map(([axis, _]) => axis);

  // 全体の測定信頼性（欠損軸があると低下）
  const overallConfidence = 1 - (missingAxes.length / 5);

  return {
    ranking: candidates,
    missingAxes,
    overallConfidence,
    timestamp: new Date().toISOString(),
    version: 'v5.0.0'
  };
}
```

### 3.6 説明文生成

```typescript
function generateMatchReasons(
  userProfile: UserProfile,
  candidate: CandidateScore
): string[] {
  const reasons: string[] = [];

  // 寄与度が大きい軸から説明
  const sortedContribs = Object.entries(candidate.contributions)
    .filter(([_, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);

  for (const [axis, contrib] of sortedContribs.slice(0, 3)) {
    const userDist = userProfile[axis as keyof UserProfile];
    const topCategory = Object.entries(userDist.values)
      .sort((a, b) => b[1] - a[1])[0];

    const percentage = Math.round(topCategory[1] * 100);
    const impact = contrib < 0.1 ? '強く一致' : contrib < 0.2 ? '一致' : '部分一致';

    reasons.push(`${getAxisLabel(axis)}: ${topCategory[0]}傾向(${percentage}%)が${impact}`);
  }

  // 混合状態の場合は追記
  if (candidate.isMixedState) {
    reasons.push('※ 複数の傾向が混在する状態です');
  }

  return reasons;
}

function getAxisLabel(axis: string): string {
  const labels: Record<string, string> = {
    changeNature: '変化の性質',
    agency: '主体性',
    timeframe: '時間軸',
    relationship: '関係性',
    emotionalTone: '感情基調'
  };
  return labels[axis] || axis;
}
```

---

## 4. API仕様

### 4.1 エンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| GET | `/v5/questions` | 質問一覧取得 |
| POST | `/v5/diagnose` | 診断実行 |
| GET | `/v5/result/:id` | 結果取得 |
| POST | `/v5/feedback` | 納得度フィードバック |

### 4.2 POST /v5/diagnose

**リクエスト**:
```json
{
  "answers": {
    "changeNature": { "expansion": 4, "contraction": 2, "maintenance": 3, "transformation": 1 },
    "agency": 4,
    "timeframe": "shortTerm",
    "relationship": { "self": true, "family": false, "team": true, "organization": false, "external": true, "society": false },
    "emotionalTone": { "excitement": 4, "caution": 3, "anxiety": 2, "optimism": 4 }
  },
  "sessionId": "uuid-xxx"
}
```

**レスポンス**:
```json
{
  "resultId": "result-xxx",
  "topCandidates": [
    {
      "rank": 1,
      "classId": 1,
      "hexagram": 1,
      "yao": 1,
      "name": "乾為天 初九",
      "score": 0.12,
      "contributions": {
        "changeNature": 0.03,
        "agency": 0.02,
        "timeframe": 0.04,
        "relationship": 0.02,
        "emotionalTone": 0.01
      },
      "matchReasons": [
        "変化の性質: 拡大傾向(40%)が強く一致",
        "主体性: 自ら動く傾向が一致",
        "感情基調: 前向き・楽観傾向が一致"
      ],
      "isMixedState": false
    }
    // ... Top 5まで
  ],
  "missingAxes": [],
  "overallConfidence": 1.0,
  "version": "v5.0.0"
}
```

---

## 5. UI仕様

### 5.1 質問画面フロー

```
画面1: changeNature
┌─────────────────────────────────────────────────┐
│ Q1. 今のあなたの状況について、                    │
│     それぞれどの程度当てはまりますか？            │
│                                                  │
│ 新しいことを始める・広げる段階                    │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 手放す・整理する段階                              │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 今の状態を守る・安定させる段階                    │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 方向転換・大きく変える段階                        │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│                              [次へ →]            │
└─────────────────────────────────────────────────┘

画面2: agency
┌─────────────────────────────────────────────────┐
│ Q2. この変化について、                           │
│     自分でどの程度コントロールできると感じますか？ │
│                                                  │
│ 全くできない [1] [2] [3] [4] [5] 完全にできる    │
│                                                  │
│                              [次へ →]            │
└─────────────────────────────────────────────────┘

画面3: timeframe
┌─────────────────────────────────────────────────┐
│ Q3. この変化はいつ頃起きる予定ですか？            │
│                                                  │
│ ○ 1ヶ月以内                                      │
│ ○ 3ヶ月以内                                      │
│ ○ 半年以内                                       │
│ ○ 1年以内                                        │
│ ○ 1年以上先                                      │
│ ○ わからない                                     │
│                                                  │
│                              [次へ →]            │
└─────────────────────────────────────────────────┘

画面4: relationship
┌─────────────────────────────────────────────────┐
│ Q4. この変化は主に誰に影響しますか？（複数選択可）│
│                                                  │
│ ☐ 自分自身                                       │
│ ☐ 家族                                           │
│ ☐ 同僚・チーム                                   │
│ ☐ 組織全体                                       │
│ ☐ 顧客・取引先                                   │
│ ☐ 業界・社会                                     │
│                                                  │
│                              [次へ →]            │
└─────────────────────────────────────────────────┘

画面5: emotionalTone
┌─────────────────────────────────────────────────┐
│ Q5. 今、以下の感情をどの程度感じていますか？      │
│                                                  │
│ ワクワク・期待感                                  │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 慎重さ・用心深さ                                  │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 不安・心配                                        │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│ 楽観・なんとかなる感                              │
│ [1] [2] [3] [4] [5]                              │
│                                                  │
│                              [診断する →]        │
└─────────────────────────────────────────────────┘
```

### 5.2 結果画面

```
┌─────────────────────────────────────────────────┐
│ あなたの診断結果                                  │
│                                                  │
│ ┌─────────────────────────────────────────────┐ │
│ │ 1位: 乾為天 初九（けんいてん しょきゅう）    │ │
│ │ マッチ度: 88%                                │ │
│ │                                              │ │
│ │ 【判定理由】                                  │ │
│ │ ・変化の性質: 拡大傾向(40%)が強く一致        │ │
│ │ ・主体性: 自ら動く傾向が一致                 │ │
│ │ ・感情基調: 前向き・楽観傾向が一致           │ │
│ │                                              │ │
│ │ [詳しく見る（無料プレビュー）]               │ │
│ └─────────────────────────────────────────────┘ │
│                                                  │
│ 他の候補:                                        │
│ 2位: 乾為天 九二 (85%)                           │
│ 3位: 大有 初九 (82%)                             │
│ 4位: 同人 初九 (80%)                             │
│ 5位: 姤 初六 (78%)                               │
│                                                  │
│ ────────────────────────────────────────────── │
│                                                  │
│ この結果は当てはまりますか？                      │
│ [1] [2] [3] [4] [5]                              │
│ 全く当てはまらない  とても当てはまる              │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 6. ログ仕様

### 6.1 診断ログ

```typescript
interface DiagnosisLog {
  // 識別情報
  logId: string;
  sessionId: string;
  userId?: string;       // 匿名可
  timestamp: string;

  // バージョン情報
  algorithmVersion: string;  // "v5.0.0"
  rubricVersion: string;     // "rubric_v1"

  // 入力
  rawAnswers: UserAnswers;

  // 中間結果
  userProfile: UserProfile;

  // 出力
  topCandidates: CandidateScore[];  // Top 10
  missingAxes: string[];
  overallConfidence: number;

  // フィードバック（後から追加）
  selfRating?: number;       // 1-5
  selectedCandidate?: number; // 最終選択したclassId
}
```

### 6.2 シャドー評価ログ（v2比較用）

```typescript
interface ShadowEvaluationLog {
  logId: string;
  sessionId: string;
  timestamp: string;

  // 同一入力に対する両方の結果
  v2Result: {
    topHexagram: number;
    topYao: number;
    score: number;
  };

  v5Result: {
    topClassId: number;
    topHexagram: number;
    topYao: number;
    score: number;
  };

  // 差分
  isMatch: boolean;         // Top1が一致するか
  rankDifference: number;   // v2のTop1がv5で何位か
}
```

---

## 7. ルーブリック仕様

### 7.1 ルーブリックの構造

```typescript
interface Rubric {
  version: string;        // "v1"
  createdAt: string;

  // 384クラスの参照分布
  classProfiles: ClassProfile[];

  // 軸→易経記述の対応規則
  axisRules: {
    changeNature: AxisRule;
    agency: AxisRule;
    timeframe: AxisRule;
    relationship: AxisRule;
    emotionalTone: AxisRule;
  };
}

interface AxisRule {
  description: string;     // 軸の定義
  categories: {
    [category: string]: {
      definition: string;  // カテゴリの定義
      iChingKeywords: string[];  // 対応する易経キーワード
      exampleHexagrams: number[]; // 代表的な卦
    };
  };
}
```

### 7.2 ルーブリック作成プロセス

```
Step 1: 軸→易経対応規則の定義
├── 各軸の各カテゴリに対応する易経キーワードを特定
├── 専門家レビュー（または既存文献参照）
└── 成果物: axisRules

Step 2: 384クラスの参照分布生成
├── 各クラス（卦×爻）の説明文を入力
├── LLMで5軸の分布を推定
├── axisRulesとの整合性チェック
└── 成果物: classProfiles（初期版）

Step 3: パイロットテストでの調整
├── 限定ユーザーでの診断実行
├── 納得度が低いケースの分析
├── 参照分布の微調整
└── 成果物: classProfiles（調整版）
```

---

## 8. 実装ファイル構成

```
api/
├── src/
│   ├── v5/
│   │   ├── index.ts           # v5 APIルーティング
│   │   ├── types.ts           # 型定義
│   │   ├── convert.ts         # 回答→分布変換
│   │   ├── distance.ts        # JS距離計算
│   │   ├── matching.ts        # ランキング生成
│   │   ├── explanation.ts     # 説明文生成
│   │   └── logging.ts         # ログ出力
│   └── rubric/
│       └── loader.ts          # ルーブリック読み込み
├── tests/
│   └── v5/
│       ├── convert.test.ts
│       ├── distance.test.ts
│       ├── matching.test.ts
│       └── integration.test.ts
└── data/
    └── rubric_v1.json         # ルーブリック本体

lp/
└── v5/
    └── index.html             # v5用LP
```

---

## 9. テスト仕様

### 9.1 単体テスト

| テスト対象 | ケース |
|-----------|--------|
| convertChangeNature | 均等入力→均等分布、偏り入力→偏り分布 |
| convertAgency | 1→待つ優位、3→受け止める優位、5→自ら動く優位 |
| convertTimeframe | unknown→isMissing=true |
| jsDistance | 同一分布→0、対極分布→最大値 |
| calculateScore | 欠損軸の重み=0を確認 |
| generateRanking | 同点処理、順位付けの正確性 |

### 9.2 統合テスト

| シナリオ | 期待結果 |
|---------|----------|
| 拡大傾向+高主体性+短期 | 乾系の卦がTop5に入る |
| 収縮傾向+低主体性+長期 | 坤系の卦がTop5に入る |
| 全て中央値 | 確信度が低く表示される |
| timeframe=unknown | missingAxesに含まれる |

### 9.3 エッジケーステスト

| ケース | 期待動作 |
|-------|----------|
| 全項目0入力 | デフォルト均等分布で計算続行 |
| 極端な偏り（5,1,1,1） | 正常に計算、混合フラグ=false |
| 高エントロピー（3,3,3,3） | 混合フラグ=true |
| Top5が同点 | 全て同順位で表示 |

---

## 10. 変更可能点（将来の調整用）

以下は初期値を設定するが、運用データで調整可能:

| 項目 | 初期値 | 調整方法 |
|------|--------|----------|
| 軸の重み | 均等（各0.2） | 納得度との相関分析 |
| timeframe隣接確率 | 0.7/0.2/0.1/0 | 分布の検証 |
| 混合状態閾値 | エントロピー>70%×最大 | ユーザーフィードバック |
| Top-k表示数 | 5 | UI評価 |

---

*仕様凍結日: 2026-01-13*
*次回レビュー: 限定ベータ後*
