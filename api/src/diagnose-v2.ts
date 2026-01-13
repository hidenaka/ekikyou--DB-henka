/**
 * 診断エンジン v2
 *
 * LLM Debate結果に基づく新設計:
 * - 10問で「正確な分類」は不可能 → Top-k候補 + ユーザー選択方式
 * - Phase 1: 初期絞り込み（5問固定）
 * - Phase 2: 適応型質問（確信度<70%時）
 * - Phase 3: ユーザー選択 → 爻レベル深掘り
 * - Phase 4: 事例検索
 */

// ============================================================
// 型定義
// ============================================================

export interface Phase1Question {
  id: number;
  axis: string;
  question: string;
  options: { value: number; label: string; description: string }[];
}

export interface Phase1Answers {
  changeNature: number;      // 0-3: 拡大/収縮/維持/転換
  agency: number;            // 0-2: 自ら動く/受け止める/待つ
  timeframe: number;         // 0-2: 今すぐ/数ヶ月/1年以上
  relationship: number;      // 0-2: 個人/組織内/対外
  emotionalTone: number;     // 0-3: 前向き/慎重/不安/楽観
}

export interface HexagramCandidate {
  hexagramNumber: number;
  name: string;
  confidence: number;        // 0-1
  description: string;       // 卦名は見せず状況説明
  keywords: string[];
}

export interface Phase1Result {
  candidates: HexagramCandidate[];
  needsAdditionalQuestions: boolean;
  additionalQuestions?: Phase2Question[];
  topConfidence: number;
}

export interface Phase2Question {
  id: string;
  question: string;
  targetHexagrams: number[]; // この質問で弁別する卦
  options: { value: number; label: string }[];
}

export interface YaoOption {
  yao: number;               // 1-6
  description: string;
  stage: string;             // 段階の名前
}

export interface SelectionResult {
  hexagramNumber: number;
  hexagramName: string;
  yaoOptions: YaoOption[];
}

export interface PreviewResult {
  hexagramNumber: number;
  hexagramName: string;
  yao: number;
  summary: string;
  caseCount: number;
  distribution: Record<string, number>;
  paidContentPreview: string[];
}

// ============================================================
// 定数: 64卦マスターデータ（簡易版）
// ============================================================

interface HexagramData {
  number: number;
  name: string;
  upper: string;
  lower: string;
  meaning: string;
  situation: string;
  keywords: string[];
  // スコアリング用の属性
  changeType: 'expansion' | 'contraction' | 'stability' | 'transformation';
  agencyType: 'active' | 'receptive' | 'waiting';
  timeHorizon: 'immediate' | 'medium' | 'long';
  relationScope: 'personal' | 'organizational' | 'external';
  emotionalQuality: 'positive' | 'cautious' | 'anxious' | 'optimistic';
}

const HEXAGRAM_DATA: Record<number, HexagramData> = {
  1: { number: 1, name: '乾為天', upper: '乾', lower: '乾', meaning: '創造・剛健', situation: 'すべてが満ちている状態。力があり、積極的に動ける時期', keywords: ['創造', 'リーダーシップ', '剛健', '前進'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'positive' },
  2: { number: 2, name: '坤為地', upper: '坤', lower: '坤', meaning: '受容・柔順', situation: '受け入れ、育てる時期。自ら主導するより、流れに従う', keywords: ['受容', '柔順', '育成', '従う'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  3: { number: 3, name: '水雷屯', upper: '坎', lower: '震', meaning: '困難の始まり', situation: '物事の始まりで困難が多い。焦らず基盤を固める時期', keywords: ['困難', '始まり', '忍耐', '基盤づくり'], changeType: 'expansion', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'anxious' },
  4: { number: 4, name: '山水蒙', upper: '艮', lower: '坎', meaning: '未熟・啓蒙', situation: 'まだ未熟で学ぶべき時期。指導者や師を求めるべき', keywords: ['未熟', '学び', '教育', '指導'], changeType: 'expansion', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  5: { number: 5, name: '水天需', upper: '坎', lower: '乾', meaning: '待つ・養う', situation: '時機を待つべき時期。焦らず力を蓄える', keywords: ['待機', '忍耐', '準備', '時機'], changeType: 'stability', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'cautious' },
  6: { number: 6, name: '天水訟', upper: '乾', lower: '坎', meaning: '争い・訴訟', situation: '対立・紛争が起きやすい。争いは避け、妥協点を探る', keywords: ['争い', '対立', '訴訟', '仲裁'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'anxious' },
  7: { number: 7, name: '地水師', upper: '坤', lower: '坎', meaning: '軍隊・統率', situation: '組織を率いる時期。規律と統率が重要', keywords: ['統率', '組織', '規律', 'リーダー'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  8: { number: 8, name: '水地比', upper: '坎', lower: '坤', meaning: '親しむ・団結', situation: '協力・連携の時期。仲間を集め、力を合わせる', keywords: ['団結', '協力', '親睦', '連携'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  9: { number: 9, name: '風天小畜', upper: '巽', lower: '乾', meaning: '小さく蓄える', situation: '大きな蓄積はできないが、少しずつ積み上げる時期', keywords: ['蓄積', '抑制', '小さな成功'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'cautious' },
  10: { number: 10, name: '天沢履', upper: '乾', lower: '兌', meaning: '慎重に進む', situation: '危険を認識しながら慎重に進む時期。礼節を守る', keywords: ['慎重', '礼節', '危険回避'], changeType: 'stability', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'cautious' },
  11: { number: 11, name: '地天泰', upper: '坤', lower: '乾', meaning: '安泰・繁栄', situation: '天地が交わり万物が通じる。最も良い時期', keywords: ['繁栄', '安泰', '調和', '成功'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  12: { number: 12, name: '天地否', upper: '乾', lower: '坤', meaning: '閉塞・停滞', situation: '天地が交わらず停滞する。困難な時期', keywords: ['停滞', '閉塞', '困難', '忍耐'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'organizational', emotionalQuality: 'anxious' },
  // 13-64は省略形式で追加（実際の運用では完全版を使用）
  13: { number: 13, name: '天火同人', upper: '乾', lower: '離', meaning: '同志との協力', situation: '志を同じくする仲間と力を合わせる時期', keywords: ['同志', '協力', '連帯'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  14: { number: 14, name: '火天大有', upper: '離', lower: '乾', meaning: '大いに所有する', situation: '大きな成果を得られる時期。謙虚さを忘れずに', keywords: ['繁栄', '成功', '所有'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  15: { number: 15, name: '地山謙', upper: '坤', lower: '艮', meaning: '謙虚・謙遜', situation: '謙虚な姿勢が成功をもたらす時期', keywords: ['謙虚', '謙遜', '控えめ'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  16: { number: 16, name: '雷地予', upper: '震', lower: '坤', meaning: '喜び・楽観', situation: '喜びと楽観に満ちた時期。しかし慢心に注意', keywords: ['喜び', '楽観', '準備'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  17: { number: 17, name: '沢雷随', upper: '兌', lower: '震', meaning: '従う・随う', situation: '時勢に従い、流れに乗る時期', keywords: ['随順', '従う', '適応'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'positive' },
  18: { number: 18, name: '山風蠱', upper: '艮', lower: '巽', meaning: '腐敗を正す', situation: '過去の問題を正し、改革する時期', keywords: ['改革', '修正', '再建'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'cautious' },
  19: { number: 19, name: '地沢臨', upper: '坤', lower: '兌', meaning: '臨む・接近', situation: '好機が近づいている。積極的に臨む時期', keywords: ['接近', '好機', '積極'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'positive' },
  20: { number: 20, name: '風地観', upper: '巽', lower: '坤', meaning: '観察・洞察', situation: '全体を見渡し、状況を観察する時期', keywords: ['観察', '洞察', '理解'], changeType: 'stability', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'external', emotionalQuality: 'cautious' },
  // 21-40
  21: { number: 21, name: '火雷噬嗑', upper: '離', lower: '震', meaning: '障害を噛み砕く', situation: '障害を取り除き、問題を解決する時期', keywords: ['解決', '断固', '克服'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'positive' },
  22: { number: 22, name: '山火賁', upper: '艮', lower: '離', meaning: '飾り・美', situation: '外見を整え、形式を重視する時期', keywords: ['美', '形式', '装飾'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'positive' },
  23: { number: 23, name: '山地剥', upper: '艮', lower: '坤', meaning: '剥落・衰退', situation: '物事が剥げ落ちる時期。無理に動かない', keywords: ['衰退', '剥落', '忍耐'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'anxious' },
  24: { number: 24, name: '地雷復', upper: '坤', lower: '震', meaning: '復帰・再生', situation: '回復と再生の時期。新たな始まり', keywords: ['復帰', '再生', '回復'], changeType: 'expansion', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'optimistic' },
  25: { number: 25, name: '天雷无妄', upper: '乾', lower: '震', meaning: '無妄・自然', situation: '作為なく自然に任せる時期', keywords: ['自然', '無作為', '誠実'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'immediate', relationScope: 'personal', emotionalQuality: 'positive' },
  26: { number: 26, name: '山天大畜', upper: '艮', lower: '乾', meaning: '大いに蓄える', situation: '力を大いに蓄える時期。大きな器が必要', keywords: ['蓄積', '忍耐', '大器'], changeType: 'expansion', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'organizational', emotionalQuality: 'cautious' },
  27: { number: 27, name: '山雷頤', upper: '艮', lower: '震', meaning: '養う・育てる', situation: '自分や他者を養い育てる時期', keywords: ['養育', '栄養', '育成'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  28: { number: 28, name: '沢風大過', upper: '兌', lower: '巽', meaning: '過剰・極限', situation: '限界を超えた状態。慎重な対応が必要', keywords: ['過剰', '極限', '転換'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'anxious' },
  29: { number: 29, name: '坎為水', upper: '坎', lower: '坎', meaning: '険難・困難', situation: '困難が重なる時期。誠意を持って対処', keywords: ['困難', '険難', '誠意'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'anxious' },
  30: { number: 30, name: '離為火', upper: '離', lower: '離', meaning: '明るさ・付着', situation: '明るく輝く時期。しかし依存に注意', keywords: ['明晰', '輝き', '付着'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'positive' },
  31: { number: 31, name: '沢山咸', upper: '兌', lower: '艮', meaning: '感応・交流', situation: '心が通じ合う時期。感受性を大切に', keywords: ['感応', '交流', '共感'], changeType: 'expansion', agencyType: 'receptive', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'positive' },
  32: { number: 32, name: '雷風恒', upper: '震', lower: '巽', meaning: '持続・恒常', situation: '継続と持続の時期。変わらぬ努力', keywords: ['持続', '恒常', '継続'], changeType: 'stability', agencyType: 'active', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  33: { number: 33, name: '天山遯', upper: '乾', lower: '艮', meaning: '退く・隠れる', situation: '退くべき時期。戦略的撤退', keywords: ['退避', '隠遁', '撤退'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'immediate', relationScope: 'personal', emotionalQuality: 'cautious' },
  34: { number: 34, name: '雷天大壮', upper: '震', lower: '乾', meaning: '大いに盛ん', situation: '力が満ちている時期。しかし暴走に注意', keywords: ['壮大', '勢い', '力'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  35: { number: 35, name: '火地晋', upper: '離', lower: '坤', meaning: '進む・昇進', situation: '順調に進む時期。昇進や発展', keywords: ['進歩', '昇進', '発展'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  36: { number: 36, name: '地火明夷', upper: '坤', lower: '離', meaning: '明かりが傷つく', situation: '才能を隠す時期。忍耐が必要', keywords: ['隠蔽', '忍耐', '保身'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'anxious' },
  37: { number: 37, name: '風火家人', upper: '巽', lower: '離', meaning: '家庭・内部', situation: '内部を固める時期。家庭や組織内の調和', keywords: ['家庭', '内部', '調和'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'positive' },
  38: { number: 38, name: '火沢睽', upper: '離', lower: '兌', meaning: '背反・対立', situation: '意見が分かれる時期。小事から始める', keywords: ['対立', '背反', '小事'], changeType: 'transformation', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'external', emotionalQuality: 'cautious' },
  39: { number: 39, name: '水山蹇', upper: '坎', lower: '艮', meaning: '足踏み・困難', situation: '進めない時期。立ち止まって考える', keywords: ['困難', '足踏み', '停滞'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'anxious' },
  40: { number: 40, name: '雷水解', upper: '震', lower: '坎', meaning: '解放・緩和', situation: '困難が解ける時期。速やかに行動', keywords: ['解放', '緩和', '解決'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  // 41-64
  41: { number: 41, name: '山沢損', upper: '艮', lower: '兌', meaning: '減らす・損', situation: '減らすことで得る時期。自己犠牲', keywords: ['損失', '減少', '犠牲'], changeType: 'contraction', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'cautious' },
  42: { number: 42, name: '風雷益', upper: '巽', lower: '震', meaning: '増やす・益', situation: '増やす時期。積極的な行動が実る', keywords: ['利益', '増加', '発展'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  43: { number: 43, name: '沢天夬', upper: '兌', lower: '乾', meaning: '決断・決行', situation: '決断の時期。断固として行動', keywords: ['決断', '決行', '断固'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'positive' },
  44: { number: 44, name: '天風姤', upper: '乾', lower: '巽', meaning: '出会い・遭遇', situation: '予期せぬ出会いの時期。慎重に対処', keywords: ['出会い', '遭遇', '慎重'], changeType: 'transformation', agencyType: 'receptive', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'cautious' },
  45: { number: 45, name: '沢地萃', upper: '兌', lower: '坤', meaning: '集まる・結集', situation: '人が集まる時期。協力体制を築く', keywords: ['結集', '集合', '協力'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  46: { number: 46, name: '地風升', upper: '坤', lower: '巽', meaning: '昇る・上昇', situation: '着実に上昇する時期。努力が実る', keywords: ['上昇', '昇進', '発展'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  47: { number: 47, name: '沢水困', upper: '兌', lower: '坎', meaning: '困窮・苦境', situation: '困難な状況。言葉より行動で示す', keywords: ['困窮', '苦境', '忍耐'], changeType: 'contraction', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'anxious' },
  48: { number: 48, name: '水風井', upper: '坎', lower: '巽', meaning: '井戸・資源', situation: '変わらぬ価値を守る時期。基盤を大切に', keywords: ['資源', '基盤', '安定'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'organizational', emotionalQuality: 'cautious' },
  49: { number: 49, name: '沢火革', upper: '兌', lower: '離', meaning: '革命・改革', situation: '大きな変革の時期。旧を捨て新を取る', keywords: ['革命', '改革', '変革'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'positive' },
  50: { number: 50, name: '火風鼎', upper: '離', lower: '巽', meaning: '鼎・安定', situation: '新しい秩序を確立する時期', keywords: ['安定', '秩序', '確立'], changeType: 'stability', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  51: { number: 51, name: '震為雷', upper: '震', lower: '震', meaning: '雷・震動', situation: '衝撃的な出来事。驚きを乗り越える', keywords: ['衝撃', '震動', '覚醒'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'personal', emotionalQuality: 'anxious' },
  52: { number: 52, name: '艮為山', upper: '艮', lower: '艮', meaning: '止まる・静止', situation: '動きを止める時期。内省と瞑想', keywords: ['静止', '内省', '瞑想'], changeType: 'stability', agencyType: 'waiting', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  53: { number: 53, name: '風山漸', upper: '巽', lower: '艮', meaning: '漸進・段階', situation: '段階的に進む時期。焦らない', keywords: ['漸進', '段階', '着実'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'long', relationScope: 'personal', emotionalQuality: 'cautious' },
  54: { number: 54, name: '雷沢帰妹', upper: '震', lower: '兌', meaning: '嫁ぐ・従属', situation: '主体性を持ちにくい時期。慎重に', keywords: ['従属', '慎重', '忍耐'], changeType: 'stability', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'external', emotionalQuality: 'cautious' },
  55: { number: 55, name: '雷火豊', upper: '震', lower: '離', meaning: '豊か・繁栄', situation: '最も豊かな時期。しかし衰退に備える', keywords: ['豊穣', '繁栄', '絶頂'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'organizational', emotionalQuality: 'optimistic' },
  56: { number: 56, name: '火山旅', upper: '離', lower: '艮', meaning: '旅・移動', situation: '移動や変化の時期。柔軟に対応', keywords: ['旅', '移動', '変化'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'medium', relationScope: 'external', emotionalQuality: 'cautious' },
  57: { number: 57, name: '巽為風', upper: '巽', lower: '巽', meaning: '風・浸透', situation: '柔軟に浸透する時期。穏やかな影響力', keywords: ['浸透', '柔軟', '適応'], changeType: 'expansion', agencyType: 'receptive', timeHorizon: 'long', relationScope: 'external', emotionalQuality: 'positive' },
  58: { number: 58, name: '兌為沢', upper: '兌', lower: '兌', meaning: '喜び・悦び', situation: '喜びと交流の時期。コミュニケーション', keywords: ['喜び', '交流', '和楽'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'immediate', relationScope: 'external', emotionalQuality: 'optimistic' },
  59: { number: 59, name: '風水渙', upper: '巽', lower: '坎', meaning: '散る・拡散', situation: '固まったものが散る時期。柔軟に対応', keywords: ['拡散', '分散', '解消'], changeType: 'transformation', agencyType: 'active', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  60: { number: 60, name: '水沢節', upper: '坎', lower: '兌', meaning: '節度・制限', situation: '節度を守る時期。自制が大切', keywords: ['節度', '制限', '自制'], changeType: 'contraction', agencyType: 'receptive', timeHorizon: 'medium', relationScope: 'personal', emotionalQuality: 'cautious' },
  61: { number: 61, name: '風沢中孚', upper: '巽', lower: '兌', meaning: '誠実・信頼', situation: '誠意を持って対応する時期。信頼を築く', keywords: ['誠実', '信頼', '真心'], changeType: 'stability', agencyType: 'active', timeHorizon: 'long', relationScope: 'external', emotionalQuality: 'positive' },
  62: { number: 62, name: '雷山小過', upper: '震', lower: '艮', meaning: '小さく過ぎる', situation: '小さなことに注意する時期。謙虚に', keywords: ['小事', '謙虚', '注意'], changeType: 'contraction', agencyType: 'receptive', timeHorizon: 'immediate', relationScope: 'personal', emotionalQuality: 'cautious' },
  63: { number: 63, name: '水火既済', upper: '坎', lower: '離', meaning: '完成・成就', situation: '物事が完成した時期。しかし維持に注意', keywords: ['完成', '成就', '維持'], changeType: 'stability', agencyType: 'waiting', timeHorizon: 'medium', relationScope: 'organizational', emotionalQuality: 'positive' },
  64: { number: 64, name: '火水未済', upper: '離', lower: '坎', meaning: '未完成・未達', situation: 'まだ完成していない時期。慎重に進める', keywords: ['未完成', '継続', '慎重'], changeType: 'expansion', agencyType: 'active', timeHorizon: 'long', relationScope: 'organizational', emotionalQuality: 'cautious' }
};

// ============================================================
// Phase 1: 質問定義
// ============================================================

export const PHASE1_QUESTIONS: Phase1Question[] = [
  {
    id: 1,
    axis: 'changeNature',
    question: '今、あなたが直面している状況の変化はどのようなものですか？',
    options: [
      { value: 0, label: '拡大・成長', description: '新しいことを始める、規模を大きくする、可能性を広げる' },
      { value: 1, label: '縮小・整理', description: '手放す、減らす、集中する、終わらせる' },
      { value: 2, label: '現状維持', description: '今の状態を保つ、安定させる、守る' },
      { value: 3, label: '方向転換', description: '根本的に変える、別の道を選ぶ、リセットする' }
    ]
  },
  {
    id: 2,
    axis: 'agency',
    question: 'この状況に対して、あなたはどのような姿勢を取っていますか？',
    options: [
      { value: 0, label: '自ら動く', description: '主導権を握り、積極的に行動している' },
      { value: 1, label: '受け止める', description: '流れに従い、柔軟に対応している' },
      { value: 2, label: '待つ', description: '時機を見計らい、様子を見ている' }
    ]
  },
  {
    id: 3,
    axis: 'timeframe',
    question: 'この状況は、どのくらいの時間軸で考えていますか？',
    options: [
      { value: 0, label: '今すぐ', description: '緊急性が高い、すぐに結果が必要' },
      { value: 1, label: '数ヶ月', description: '中期的な視点で取り組む' },
      { value: 2, label: '1年以上', description: '長期的な視点で腰を据えて取り組む' }
    ]
  },
  {
    id: 4,
    axis: 'relationship',
    question: 'この状況は主にどの範囲に関わるものですか？',
    options: [
      { value: 0, label: '個人の問題', description: '自分自身のキャリア、生き方、内面の問題' },
      { value: 1, label: '組織内', description: '会社、チーム、家族など所属する組織の中の問題' },
      { value: 2, label: '対外関係', description: '外部との関係、交渉、新しい出会い' }
    ]
  },
  {
    id: 5,
    axis: 'emotionalTone',
    question: '今のあなたの気持ちに最も近いものはどれですか？',
    options: [
      { value: 0, label: '前向き・積極的', description: 'やる気がある、エネルギーを感じる' },
      { value: 1, label: '慎重・用心深い', description: '注意深く、リスクを考えている' },
      { value: 2, label: '不安・心配', description: '先行きが不透明で、心配がある' },
      { value: 3, label: '楽観・期待', description: 'うまくいく予感がある、希望を感じる' }
    ]
  }
];

// ============================================================
// スコアリングロジック
// ============================================================

function calculateHexagramScores(answers: Phase1Answers): Map<number, number> {
  const scores = new Map<number, number>();

  // 変換マップ
  const changeMap: Record<number, HexagramData['changeType']> = {
    0: 'expansion', 1: 'contraction', 2: 'stability', 3: 'transformation'
  };
  const agencyMap: Record<number, HexagramData['agencyType']> = {
    0: 'active', 1: 'receptive', 2: 'waiting'
  };
  const timeMap: Record<number, HexagramData['timeHorizon']> = {
    0: 'immediate', 1: 'medium', 2: 'long'
  };
  const relationMap: Record<number, HexagramData['relationScope']> = {
    0: 'personal', 1: 'organizational', 2: 'external'
  };
  const emotionMap: Record<number, HexagramData['emotionalQuality']> = {
    0: 'positive', 1: 'cautious', 2: 'anxious', 3: 'optimistic'
  };

  const userChange = changeMap[answers.changeNature];
  const userAgency = agencyMap[answers.agency];
  const userTime = timeMap[answers.timeframe];
  const userRelation = relationMap[answers.relationship];
  const userEmotion = emotionMap[answers.emotionalTone];

  // 各卦のスコアを計算
  for (const hex of Object.values(HEXAGRAM_DATA)) {
    let score = 0;

    // 重み付けスコアリング
    if (hex.changeType === userChange) score += 25;
    if (hex.agencyType === userAgency) score += 20;
    if (hex.timeHorizon === userTime) score += 15;
    if (hex.relationScope === userRelation) score += 20;
    if (hex.emotionalQuality === userEmotion) score += 20;

    // 部分一致ボーナス（隣接カテゴリ）
    // 例：expansion と stability は隣接として扱う
    if (hex.changeType === 'expansion' && userChange === 'stability') score += 10;
    if (hex.changeType === 'stability' && userChange === 'expansion') score += 10;
    if (hex.changeType === 'contraction' && userChange === 'transformation') score += 10;
    if (hex.changeType === 'transformation' && userChange === 'contraction') score += 10;

    scores.set(hex.number, score);
  }

  return scores;
}

function getTopCandidates(scores: Map<number, number>, topK: number = 5): HexagramCandidate[] {
  // スコアでソート
  const sorted = [...scores.entries()].sort((a, b) => b[1] - a[1]);
  const topScores = sorted.slice(0, topK);

  // 合計スコア（確信度計算用）
  const totalScore = topScores.reduce((sum, [, score]) => sum + score, 0);

  return topScores.map(([hexNum, score]) => {
    const hex = HEXAGRAM_DATA[hexNum];
    return {
      hexagramNumber: hex.number,
      name: hex.name,
      confidence: totalScore > 0 ? score / totalScore : 0.2,
      description: hex.situation,
      keywords: hex.keywords
    };
  });
}

// ============================================================
// Phase 2: 適応型質問
// ============================================================

function generateAdditionalQuestions(candidates: HexagramCandidate[]): Phase2Question[] {
  if (candidates.length < 2) return [];

  const top1 = HEXAGRAM_DATA[candidates[0].hexagramNumber];
  const top2 = HEXAGRAM_DATA[candidates[1].hexagramNumber];

  const questions: Phase2Question[] = [];

  // changeType が異なる場合
  if (top1.changeType !== top2.changeType) {
    questions.push({
      id: 'change_clarify',
      question: '変化の方向について、より詳しく教えてください。',
      targetHexagrams: [top1.number, top2.number],
      options: [
        { value: 0, label: '新しいものを作り出す、広げていく' },
        { value: 1, label: '既存のものを守る、維持する' },
        { value: 2, label: '減らす、手放す、縮小する' },
        { value: 3, label: '根本から変える、リセットする' }
      ]
    });
  }

  // agencyType が異なる場合
  if (top1.agencyType !== top2.agencyType) {
    questions.push({
      id: 'agency_clarify',
      question: 'あなたの立場について、より詳しく教えてください。',
      targetHexagrams: [top1.number, top2.number],
      options: [
        { value: 0, label: '自分がリーダーとして引っ張る' },
        { value: 1, label: '誰かのサポート役として動く' },
        { value: 2, label: '状況を観察しながら時機を待つ' }
      ]
    });
  }

  return questions.slice(0, 3); // 最大3問
}

// ============================================================
// 爻レベル定義
// ============================================================

const YAO_DESCRIPTIONS: Record<number, { stage: string; description: string }> = {
  1: { stage: '初期・準備段階', description: 'まだ始まったばかり。準備を整え、基盤を固める時期。焦らず慎重に。' },
  2: { stage: '展開・成長段階', description: '動き出した段階。勢いが出てきているが、まだ油断できない。着実に進める。' },
  3: { stage: '試練・困難段階', description: '困難や障害に直面する段階。ここを乗り越えるかどうかが分かれ目。' },
  4: { stage: '転換・決断段階', description: '重要な転換点。大きな決断が求められる。上への道が開ける可能性。' },
  5: { stage: '成熟・成果段階', description: '最も良い位置。成果が現れ、影響力が発揮できる。しかし慢心に注意。' },
  6: { stage: '終末・移行段階', description: '一つのサイクルの終わり。次への移行期。過剰にならないよう注意。' }
};

// ============================================================
// 公開API
// ============================================================

/**
 * Phase 1: 初期絞り込み
 */
export function processPhase1(answers: Phase1Answers): Phase1Result {
  const scores = calculateHexagramScores(answers);
  const candidates = getTopCandidates(scores, 5);
  const topConfidence = candidates[0]?.confidence || 0;

  const needsAdditional = topConfidence < 0.35; // 35%未満なら追加質問
  const additionalQuestions = needsAdditional ? generateAdditionalQuestions(candidates) : undefined;

  return {
    candidates,
    needsAdditionalQuestions: needsAdditional,
    additionalQuestions,
    topConfidence
  };
}

/**
 * Phase 2: 追加質問処理（スコア調整）
 */
export function processPhase2(
  phase1Answers: Phase1Answers,
  phase2Answers: Record<string, number>
): Phase1Result {
  // Phase 1のスコアを再計算
  const scores = calculateHexagramScores(phase1Answers);

  // Phase 2の回答でスコアを調整
  for (const [questionId, answer] of Object.entries(phase2Answers)) {
    if (questionId === 'change_clarify') {
      // changeType に基づく調整
      for (const [hexNum, score] of scores) {
        const hex = HEXAGRAM_DATA[hexNum];
        const changeMatch: Record<number, HexagramData['changeType']> = {
          0: 'expansion', 1: 'stability', 2: 'contraction', 3: 'transformation'
        };
        if (hex.changeType === changeMatch[answer]) {
          scores.set(hexNum, score + 15);
        }
      }
    }
    if (questionId === 'agency_clarify') {
      for (const [hexNum, score] of scores) {
        const hex = HEXAGRAM_DATA[hexNum];
        const agencyMatch: Record<number, HexagramData['agencyType']> = {
          0: 'active', 1: 'receptive', 2: 'waiting'
        };
        if (hex.agencyType === agencyMatch[answer]) {
          scores.set(hexNum, score + 15);
        }
      }
    }
  }

  const candidates = getTopCandidates(scores, 5);
  const topConfidence = candidates[0]?.confidence || 0;

  return {
    candidates,
    needsAdditionalQuestions: false,
    topConfidence
  };
}

/**
 * Phase 3: ユーザー選択後の爻オプション取得
 */
export function getYaoOptions(hexagramNumber: number): SelectionResult {
  const hex = HEXAGRAM_DATA[hexagramNumber];
  if (!hex) {
    throw new Error(`Hexagram ${hexagramNumber} not found`);
  }

  const yaoOptions: YaoOption[] = Object.entries(YAO_DESCRIPTIONS).map(([yao, desc]) => ({
    yao: parseInt(yao),
    description: desc.description,
    stage: desc.stage
  }));

  return {
    hexagramNumber,
    hexagramName: hex.name,
    yaoOptions
  };
}

/**
 * Phase 4: プレビュー結果生成
 */
export function generatePreview(hexagramNumber: number, yao: number, caseCount: number = 0): PreviewResult {
  const hex = HEXAGRAM_DATA[hexagramNumber];
  if (!hex) {
    throw new Error(`Hexagram ${hexagramNumber} not found`);
  }

  const yaoDesc = YAO_DESCRIPTIONS[yao];
  if (!yaoDesc) {
    throw new Error(`Yao ${yao} not found`);
  }

  return {
    hexagramNumber,
    hexagramName: hex.name,
    yao,
    summary: `【${hex.name}・${yaoDesc.stage}】\n${hex.situation}\n\n${yaoDesc.description}`,
    caseCount,
    distribution: {
      'キャリア転換': Math.floor(caseCount * 0.4),
      '事業立ち上げ': Math.floor(caseCount * 0.3),
      '組織変革': Math.floor(caseCount * 0.2),
      'その他': Math.floor(caseCount * 0.1)
    },
    paidContentPreview: [
      '具体的な類似事例3件の詳細分析',
      'あなたの状況に基づいた90日行動計画',
      '失敗パターンとその回避策',
      '次のステップへの具体的なアドバイス'
    ]
  };
}
