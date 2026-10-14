"""
Load IndicF5 once so that the kit's length fix gets applied to the downloaded model code.

The kit's Engine raises "restart the session" the first time it patches the model's remote code.
Every task in this repo runs as its own Python process, so a second process is the restart:
    python code/warmup.py      # may print "patched, run again"
    python code/warmup.py      # loads cleanly
It also prints whether IndicF5's text vocabulary contains Latin letters and digits (a diagnostic
for Task 1: characters missing from the vocabulary cannot be spoken).
"""
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))


def vocab_report():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import yaml
        cfg = yaml.safe_load(open(KIT / "baseline" / "config.yaml", encoding="utf-8"))
        files = [f for f in HfApi().list_repo_files(cfg["repo"], revision=cfg["revision"]) if "vocab" in f.lower()]
        for f in files:
            p = hf_hub_download(cfg["repo"], f, revision=cfg["revision"])
            vocab = [l.rstrip("\n") for l in open(p, encoding="utf-8")]
            latin = sorted({c for c in vocab if len(c) == 1 and c.isascii() and c.isalpha()})
            digits = sorted({c for c in vocab if len(c) == 1 and c.isdigit()})
            print(f"[vocab] {f}: {len(vocab)} symbols; latin letters={''.join(latin) or 'NONE'}; "
                  f"digits={''.join(digits) or 'NONE'}; has '₹'={'₹' in vocab}; has '%'={'%' in vocab}")
    except Exception as e:  # diagnostic only
        print("[vocab] could not inspect vocabulary:", e)


def main():
    from baseline.tts import get_engine
    try:
        get_engine()
    except RuntimeError as e:
        print("patched, run again:", e)
        sys.exit(0)
    print("IndicF5 loaded OK")
    vocab_report()


if __name__ == "__main__":
    main()
