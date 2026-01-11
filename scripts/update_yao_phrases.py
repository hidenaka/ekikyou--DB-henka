import json
import os

filepath = "data/yao_phrases_384.json"

new_data = {
  "46-1": { "classic": "允升", "modern": "信頼され共に上昇する" },
  "46-2": { "classic": "孚乃利用", "modern": "真心で神を祀り吉" },
  "46-3": { "classic": "升虚邑", "modern": "抵抗なく順調に進む" },
  "46-4": { "classic": "王用亨于岐山", "modern": "成功を感謝し奉仕する" },
  "46-5": { "classic": "貞吉升階", "modern": "着実に階段を上る" },
  "46-6": { "classic": "冥升", "modern": "盲目的に進み消耗する" },
  
  "47-1": { "classic": "臀困于株木", "modern": "座り込んで動けない" },
  "47-2": { "classic": "困于酒食", "modern": "享楽に溺れ身を滅ぼす" },
  "47-3": { "classic": "困于石", "modern": "拠り所なく孤立無援" },
  "47-4": { "classic": "来徐徐", "modern": "助けが遅れるが来る" },
  "47-5": { "classic": "劓刖", "modern": "厳罰を受けるも志は折れず" },
  "47-6": { "classic": "困于葛藟", "modern": "迷いを断ち切れば動ける" },
  
  "48-1": { "classic": "井泥不食", "modern": "才能あるも顧みられず" },
  "48-2": { "classic": "井谷射鮒", "modern": "無駄なことに力を使う" },
  "48-3": { "classic": "井渫不食", "modern": "清廉だが用いられず嘆く" },
  "48-4": { "classic": "井甃", "modern": "自らを修養し時に備える" },
  "48-5": { "classic": "井洌", "modern": "清らかな心で人に尽くす" },
  "48-6": { "classic": "井收勿幕", "modern": "惜しみなく徳を施す" },
  
  "49-1": { "classic": "鞏用黄牛之革", "modern": "機熟さず固く守る" },
  "49-2": { "classic": "巳日乃革之", "modern": "準備万端で変革を行う" },
  "49-3": { "classic": "革言三就", "modern": "慎重に検討し決行する" },
  "49-4": { "classic": "悔亡有孚", "modern": "信念が認められ改革成功" },
  "49-5": { "classic": "大人虎変", "modern": "鮮やかに変貌を遂げる" },
  "49-6": { "classic": "君子豹変", "modern": "過ちを改め善に移る" },
  
  "50-1": { "classic": "鼎顛趾", "modern": "古い悪習を一掃する" },
  "50-2": { "classic": "鼎有実", "modern": "才能あれど嫉妬される" },
  "50-3": { "classic": "鼎耳革", "modern": "有能だが機会に恵まれず" },
  "50-4": { "classic": "鼎折足", "modern": "任に耐えず覆す" },
  "50-5": { "classic": "鼎黄耳金鉉", "modern": "中庸の徳で賢者を招く" },
  "50-6": { "classic": "鼎玉鉉", "modern": "柔和な徳で大成する" },
  
  "51-1": { "classic": "震来虩虩", "modern": "危機に万全の備えをする" },
  "51-2": { "classic": "震来厲", "modern": "財を捨てて身を守る" },
  "51-3": { "classic": "蘇蘇", "modern": "動揺してもすぐに立ち直る" },
  "51-4": { "classic": "震遂泥", "modern": "恐怖で足がすくむ" },
  "51-5": { "classic": "震往来厲", "modern": "危機の中、大事を成す" },
  "51-6": { "classic": "震索索", "modern": "災いが隣に迫る" },
  
  "52-1": { "classic": "艮其趾", "modern": "軽挙妄動を慎む" },
  "52-2": { "classic": "艮其腓", "modern": "急に止まれず苦しむ" },
  "52-3": { "classic": "艮其限", "modern": "腰を据えて動かない" },
  "52-4": { "classic": "艮其身", "modern": "独り静かに止まる" },
  "52-5": { "classic": "艮其輔", "modern": "言葉を慎み悔いなし" },
  "52-6": { "classic": "敦艮", "modern": "重厚な徳で吉" },
  
  "53-1": { "classic": "鴻漸于干", "modern": "若手として慎重に進む" },
  "53-2": { "classic": "鴻漸于磐", "modern": "安定した地位を得る" },
  "53-3": { "classic": "鴻漸于陸", "modern": "強引に進み失敗する" },
  "53-4": { "classic": "鴻漸于木", "modern": "安住の地を見つける" },
  "53-5": { "classic": "鴻漸于陵", "modern": "高みを目指し最後に遂げる" },
  "53-6": { "classic": "鴻漸于逵", "modern": "規範となり後世に残る" },
  
  "54-1": { "classic": "帰妹以娣", "modern": "控えめな立場で尽くす" },
  "54-2": { "classic": "眇能視", "modern": "誠実さで信頼を守る" },
  "54-3": { "classic": "帰妹以須", "modern": "分不相応な望みは捨てよ" },
  "54-4": { "classic": "帰妹愆期", "modern": "良縁を待ち時期を遅らす" },
  "54-5": { "classic": "帝乙帰妹", "modern": "謙虚な振る舞いが吉" },
  "54-6": { "classic": "女承筐无实", "modern": "形式だけで中身がない" },
  
  "55-1": { "classic": "遇其配主", "modern": "良きパートナーと共に進む" },
  "55-2": { "classic": "豊其蔀", "modern": "疑われても誠意を貫く" },
  "55-3": { "classic": "豊其沛", "modern": "障害に阻まれ動けない" },
  "55-4": { "classic": "豊其蔀", "modern": "賢者に出会い道が開く" },
  "55-5": { "classic": "来章", "modern": "才能ある人材を登用する" },
  "55-6": { "classic": "豊其屋", "modern": "孤独に陥り凶" },
  
  "56-1": { "classic": "旅瑣瑣", "modern": "些細なことで争うな" },
  "56-2": { "classic": "旅卽次", "modern": "旅先で良き世話役を得る" },
  "56-3": { "classic": "旅焚其次", "modern": "傲慢さで居場所を失う" },
  "56-4": { "classic": "旅于処", "modern": "安住の地なく心休まらず" },
  "56-5": { "classic": "射雉一矢亡", "modern": "名誉ある地位を得る" },
  "56-6": { "classic": "鳥焚其巣", "modern": "高慢さが破滅を招く" },
  
  "57-1": { "classic": "進退", "modern": "迷って決断できない" },
  "57-2": { "classic": "巽在牀下", "modern": "謙虚に神意を問う" },
  "57-3": { "classic": "頻巽", "modern": "媚びへつらい恥をかく" },
  "57-4": { "classic": "悔亡", "modern": "三つの品を得て吉" },
  "57-5": { "classic": "貞吉悔亡", "modern": "始め悪くとも終わり良し" },
  "57-6": { "classic": "巽在牀下", "modern": "考えすぎて機を逃す" },
  
  "58-1": { "classic": "和兌", "modern": "心和やかに人と接する" },
  "58-2": { "classic": "孚兌", "modern": "誠実な喜びを分かち合う" },
  "58-3": { "classic": "来兌", "modern": "目先の快楽に誘われる" },
  "58-4": { "classic": "商兌未寧", "modern": "迷いを捨て正道を行く" },
  "58-5": { "classic": "孚于剥", "modern": "悪の影響を受け危険" },
  "58-6": { "classic": "引兌", "modern": "楽しみに流されず自制せよ" },
  
  "59-1": { "classic": "用拯馬壮", "modern": "強力な援助で難を逃れる" },
  "59-2": { "classic": "渙奔其机", "modern": "身近なものに頼り安泰" },
  "59-3": { "classic": "渙其躬", "modern": "自己犠牲で他を救う" },
  "59-4": { "classic": "渙其群", "modern": "私心を捨て公益に尽くす" },
  "59-5": { "classic": "渙汗其大号", "modern": "大号令で危機を救う" },
  "59-6": { "classic": "渙其血", "modern": "害悪を去り不安解消" },
  
  "60-1": { "classic": "不出戸庭", "modern": "時期を待ち動かない" },
  "60-2": { "classic": "不出門庭", "modern": "好機を逃し凶" },
  "60-3": { "classic": "不節若", "modern": "節度を欠き嘆く" },
  "60-4": { "classic": "安節", "modern": "分相応に満足し吉" },
  "60-5": { "classic": "甘節", "modern": "進んで節制し美徳となる" },
  "60-6": { "classic": "苦節", "modern": "厳しすぎる規律は続かない" },
  
  "61-1": { "classic": "虞吉", "modern": "慎重に相手を選べば吉" },
  "61-2": { "classic": "鳴鶴在陰", "modern": "誠意が感応し呼び合う" },
  "61-3": { "classic": "得敵", "modern": "感情の起伏が激しすぎる" },
  "61-4": { "classic": "月幾望", "modern": "主君を敬い身を引く" },
  "61-5": { "classic": "有孚攣如", "modern": "固い信頼で結ばれる" },
  "61-6": { "classic": "翰音登于天", "modern": "口先だけで信用されない" },
  
  "62-1": { "classic": "飛鳥以凶", "modern": "分をわきまえ静かにせよ" },
  "62-2": { "classic": "弗過其祖", "modern": "目上の代理で功を立てる" },
  "62-3": { "classic": "弗過防之", "modern": "油断せず守りを固めよ" },
  "62-4": { "classic": "无咎弗過", "modern": "柔軟に対応し難を逃れる" },
  "62-5": { "classic": "密雲不雨", "modern": "実力不足で補佐が必要" },
  "62-6": { "classic": "弗遇過之", "modern": "身の程知らずで災い遭う" },
  
  "63-1": { "classic": "曳其輪", "modern": "功を焦らず慎重に" },
  "63-2": { "classic": "婦喪其茀", "modern": "無くした物は戻る" },
  "63-3": { "classic": "高宗伐鬼方", "modern": "3年かけて大業を成す" },
  "63-4": { "classic": "繻有衣袽", "modern": "備えあれば憂いなし" },
  "63-5": { "classic": "東隣殺牛", "modern": "真心は形式に勝る" },
  "63-6": { "classic": "濡其首", "modern": "油断して最後に失敗する" },
  
  "64-1": { "classic": "濡其尾", "modern": "力量不足で挫折する" },
  "64-2": { "classic": "曳其輪", "modern": "中庸を守り吉" },
  "64-3": { "classic": "未済征凶", "modern": "力不足で進めば凶" },
  "64-4": { "classic": "貞吉悔亡", "modern": "長年の戦いに勝利する" },
  "64-5": { "classic": "君子之光", "modern": "輝かしい成果を上げる" },
  "64-6": { "classic": "有孚于飲酒", "modern": "祝宴での深酒を慎む" }
}

current_data = {}
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        current_data = json.load(f)

current_data.update(new_data)

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(current_data, f, ensure_ascii=False, indent=2)

print(f"Updated {filepath} with {len(new_data)} new items.")
