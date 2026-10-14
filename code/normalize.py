"""
Task 2 - rule-based text frontend for IndicF5 (no model of any kind).

normalize(text) rewrites everything that is not already a speakable word into Hindi words in Devanagari:
  * Indian-style cardinal numbers (हज़ार, लाख, करोड़), Devanagari digits included
  * ₹ amounts (with लाख/करोड़ units and paise), percentages, decimals
  * dates dd/mm/yyyy and "<day> <month> <year>", years 1100-1999 read as "उन्नीस सौ ..."
  * abbreviations written in capitals (RBI, ATM, GST ...) spelled out letter by letter
English words written in Latin script are left unchanged (that is a Task 3 problem).

All rules were written from the validation items (and standard Hindi number names); nothing was
derived from the evaluation text.

    python code/normalize.py --kit . --out predictions/t2_normalized.csv
"""
import argparse
import csv
import re
from pathlib import Path

ONES = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
        "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस "
        "छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस "
        "उनचास पचास इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ "
        "पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर "
        "उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे "
        "तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(ONES) == 100

MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]

# English letter names as said in Indian English (RBI -> आर बी आई, ATM -> ए टी एम, GST -> जी एस टी)
LETTERS = {"A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ़", "G": "जी", "H": "एच", "I": "आई",
           "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर",
           "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड"}

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
UNITS = "लाख|करोड़|करोड़|हज़ार|हजार"


def cardinal(n: int) -> str:
    """Indian-system number words: 1,47,382 -> एक लाख सैंतालीस हज़ार तीन सौ बयासी."""
    if n < 100:
        return ONES[n]
    parts = []
    for value, word in ((10 ** 7, "करोड़"), (10 ** 5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            q, n = divmod(n, value)
            parts.append(f"{cardinal(q)} {word}")
    if n:
        parts.append(ONES[n])
    return " ".join(parts)


def year_words(n: int) -> str:
    """Years 1100-1999 are read in pairs (उन्नीस सौ निन्यानवे); others as cardinals (दो हज़ार छब्बीस)."""
    if 1100 <= n <= 1999:
        hi, lo = divmod(n, 100)
        return f"{ONES[hi]} सौ" + (f" {ONES[lo]}" if lo else "")
    return cardinal(n)


def digits_words(s: str) -> str:
    return " ".join(ONES[int(c)] for c in s)


def number_words(s: str) -> str:
    """'1,25,000' / '12951' / '2.3' -> words. Digits after a decimal point are read one by one."""
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{cardinal(int(whole or 0))} दशमलव {digits_words(frac)}"
    return cardinal(int(s))


NUM = r"\d[\d,]*(?:\.\d+)?"


def _money(m):
    amount, unit = m.group(1), m.group(2)
    amount = amount.rstrip(",.")
    if unit:                                   # ₹38 लाख -> अड़तीस लाख रुपये
        return f"{number_words(amount)} {unit} रुपये"
    clean = amount.replace(",", "")
    if "." in clean:                           # ₹2,340.50 -> ... रुपये पचास पैसे
        rupees, paise = clean.split(".", 1)
        paise = (paise + "0")[:2]
        out = f"{cardinal(int(rupees or 0))} रुपये"
        return out + (f" {cardinal(int(paise))} पैसे" if int(paise) else "")
    return f"{cardinal(int(clean))} रुपये"


def _date(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    year = int(y) if len(y) == 4 else 2000 + int(y)
    return f"{cardinal(d)} {MONTHS[mo - 1]} {year_words(year)}"


def _plain_number(m):
    s = m.group(0)
    if "," not in s and "." not in s and len(s) == 4 and 1100 <= int(s) <= 1999:
        return year_words(int(s))
    return number_words(s)


def _abbrev(m):
    return " ".join(LETTERS[c] for c in m.group(0))


def normalize(text: str) -> str:
    t = text.translate(DEV_DIGITS)
    # dates: 12/03/2026, 5-6-2026
    t = re.sub(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)", _date, t)
    # rupee amounts, optionally followed by a unit word
    t = re.sub(rf"(?:₹|रु\.?|Rs\.?)\s*({NUM})(?:\s*({UNITS}))?", _money, t)
    # percentages
    t = re.sub(rf"({NUM})\s*%", lambda m: f"{number_words(m.group(1))} प्रतिशत", t)
    # a hyphen glued to a number (चंद्रयान-3) is read as a pause, not a minus
    t = re.sub(r"(?<=[ऀ-ॿ])-(?=\d)", " ", t)
    # remaining numbers
    t = re.sub(r"\d[\d,]*\d(?:\.\d+)?|\d(?:\.\d+)?", _plain_number, t)
    # abbreviations: two or more capitals (SBI, KYC); spelled letter by letter
    t = re.sub(r"\b[A-Z]{2,}\b", _abbrev, t)
    t = t.replace("₹", " रुपये ")
    t = re.sub(r"\s+([,।?!.])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=".", help="folder that holds data/val_items.csv and data/eval_items.csv")
    ap.add_argument("--out", default="predictions/t2_normalized.csv")
    a = ap.parse_args()
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        with open(Path(a.kit) / "data" / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([r["id"], split, r["text"], normalize(r["text"])])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "text", "normalized_text"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
