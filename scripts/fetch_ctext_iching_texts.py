#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import urllib.request
from html import unescape

BASE_URL = "https://ctext.org/book-of-changes"
USER_AGENT = "Mozilla/5.0"

LINE_STATEMENT_RE = re.compile(
    r"^(初九|初六|九二|六二|九三|六三|九四|六四|九五|六五|上九|上六|用九|用六)："
)
LINE_NUMBER_MAP = {
    "初九": "1",
    "初六": "1",
    "九二": "2",
    "六二": "2",
    "九三": "3",
    "六三": "3",
    "九四": "4",
    "六四": "4",
    "九五": "5",
    "六五": "5",
    "上九": "6",
    "上六": "6",
    "用九": "use_nine",
    "用六": "use_six",
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as res:
        return res.read().decode("utf-8", errors="ignore")


def parse_index():
    html = fetch(BASE_URL)
    pairs = re.findall(r'href="book-of-changes/([^"]+)"[^>]*>([^<]+)</a>', html)
    start_idx = next(i for i, (slug, _) in enumerate(pairs) if slug == "qian")
    return pairs[start_idx : start_idx + 64]


def parse_pairs(slug):
    html = fetch(f"{BASE_URL}/{slug}")
    pattern = re.compile(
        r'<tr[^>]*>.*?<td class="ctext">(.*?)</td>.*?<td class="etext">(.*?)</td>.*?</tr>',
        re.S,
    )
    pairs = []
    for c, e in pattern.findall(html):
        ctext = unescape(re.sub(r"<[^>]+>", " ", c)).strip()
        etext = unescape(re.sub(r"<[^>]+>", " ", e)).strip()
        if ctext or etext:
            pairs.append((ctext, etext))
    return pairs


def is_line_statement(text):
    return bool(LINE_STATEMENT_RE.match(text))


def parse_hexagram(pairs):
    data = {
        "judgment": None,
        "tuan": None,
        "xiang": None,
        "lines": {},
        "appendix": [],
    }
    if pairs:
        data["judgment"] = {"classic": pairs[0][0], "modern": pairs[0][1]}
    if len(pairs) > 1:
        data["tuan"] = {"classic": pairs[1][0], "modern": pairs[1][1]}
    if len(pairs) > 2:
        data["xiang"] = {"classic": pairs[2][0], "modern": pairs[2][1]}

    i = 3
    while i < len(pairs):
        ctext, etext = pairs[i]
        if is_line_statement(ctext):
            label = ctext.split("：", 1)[0]
            key = LINE_NUMBER_MAP.get(label, label)
            entry = {"label": label, "classic": ctext, "modern": etext}
            if i + 1 < len(pairs) and not is_line_statement(pairs[i + 1][0]):
                entry["xiang"] = {"classic": pairs[i + 1][0], "modern": pairs[i + 1][1]}
                i += 2
            else:
                i += 1
            data["lines"][key] = entry
        else:
            data["appendix"].append({"classic": ctext, "modern": etext})
            i += 1
    return data


def load_hexagram_master(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="CTPから易経の古典本文＋Legge英訳を取得してJSON化する"
    )
    parser.add_argument(
        "--output",
        default="data/reference/iching_texts_ctext_legge.json",
        help="出力JSONパス",
    )
    parser.add_argument(
        "--hex-master",
        default="data/hexagrams/hexagram_master.json",
        help="ローカルの卦マスターJSONパス",
    )
    args = parser.parse_args()

    hex_master = load_hexagram_master(args.hex_master)
    hex_pairs = parse_index()

    output = {
        "metadata": {
            "source": "Chinese Text Project (ctext.org)",
            "base_url": BASE_URL,
            "translation": "James Legge (public domain translation)",
            "retrieved_at": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "license_note": (
                "CTP site is copyrighted; when quoting or citing, link back to the "
                "corresponding CTP page. Legge translation is public domain."
            ),
        },
        "hexagrams": {},
    }

    for idx, (slug, ctext_name) in enumerate(hex_pairs, start=1):
        pairs = parse_pairs(slug)
        parsed = parse_hexagram(pairs)
        local = hex_master.get(str(idx), {})
        output["hexagrams"][str(idx)] = {
            "number": idx,
            "slug": slug,
            "ctext_name": ctext_name,
            "local_name": local.get("name"),
            "local_short": local.get("chinese"),
            "source_url": f"{BASE_URL}/{slug}",
            "judgment": parsed["judgment"],
            "tuan": parsed["tuan"],
            "xiang": parsed["xiang"],
            "lines": parsed["lines"],
            "appendix": parsed["appendix"],
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
