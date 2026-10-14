"""
Task 3 text: original text -> Task 2 normalize() -> rule-based English->Devanagari (translit.py).

    python code/make_t3_text.py --kit . --out out/t3_text.csv
Writes id,split,text,t3_text for all 64 items (only Latin-script words differ from the Task 2 text).
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize          # noqa: E402
from translit import transliterate_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=".")
    ap.add_argument("--out", default="out/t3_text.csv")
    a = ap.parse_args()
    rows = []
    for split, name in (("val", "val_items.csv"), ("eval", "eval_items.csv")):
        with open(Path(a.kit) / "data" / name, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append([r["id"], split, r["text"], transliterate_text(normalize(r["text"]))])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "text", "t3_text"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
