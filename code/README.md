# 24B2509 – TTS bootcamp (Hindi–Hinglish voice cloning with IndicF5)

**Status: midterm.** Task 1 and Task 2 are done. Task 3 (idea) is planned in `idea.md` and not run yet.

**AI assistant disclosure:** code in this folder was written with the help of Claude (Anthropic), an AI coding
assistant. `idea.md` and `report.md` are written by me in my own words.

All runs: Kaggle notebook, 1× NVIDIA Tesla T4, Python 3.13, seed 0. Commands are run from the kit folder (the
folder that contains `metrics.py`), with this `code/` folder copied into it.

## Environment (note on the kit's requirements.txt)

`baseline/requirements.txt` pins IndicF5 to `fb652a96…`, which is the Hugging Face *model* revision and not a
commit of the GitHub code repo, so `pip install -r baseline/requirements.txt` fails (`git fetch … exit code 128`).
I install the same packages explicitly, plus the IndicF5 code at its latest GitHub commit. The model weights are
still the pinned Hugging Face revision from `baseline/config.yaml`.

```bash
pip install "numpy>=2.1,<2.3" "transformers==4.49.0" accelerate soundfile librosa pyyaml pandas "speechbrain>=1.0" \
  cached_path ema_pytorch hydra-core jieba pypinyin pydub torchdiffeq vocos "x_transformers>=1.31.14" \
  transformers_stream_generator safetensors tomli click wandb datasets audioop-lts torchcodec
pip install --no-deps "git+https://github.com/ai4bharat/IndicF5.git@13f7c4d627cc10111aea8fe9c0039462cacacdc7"
export HF_TOKEN=...                      # gated model; terms accepted on huggingface.co/ai4bharat/IndicF5
export PYTHONPATH=code/compat            # soundfile fallback for torchaudio.load (code/compat/sitecustomize.py)
python code/warmup.py; python code/warmup.py   # applies the kit's length fix (first call may ask for a restart)
```

* `audioop-lts`: pydub needs it on Python 3.13.
* `torchcodec` and `code/compat/sitecustomize.py`: torchaudio ≥ 2.9 only reads audio through torchcodec. The shim
  falls back to soundfile if that fails. Audio is unchanged.

`code/kaggle_run1.ipynb` runs everything below on Kaggle (secret `HF_TOKEN`, GPU on, Internet on).

## Task 1 – measure the baseline

```bash
python metrics.py data/baseline_val --json out/t1_baseline.json --csv out/t1_baseline.csv
```

Validation CER 0.2421, SIM 0.8303. This matches `data/baseline_published.json`. Logs: `code/logs/t1_baseline.*`.
`predictions/t1_val_errors.csv` tags the 14 items with CER > 0.20:

* the 6 Hinglish items → `english_word_error`. The English words come out garbled, e.g. val_hg_01
  "office internet slow" is heard as "अजसे … सर्ण … कोई".
* the 8 hard items → `number_error`. Digits, ₹ amounts and dates come out as non-words, e.g. val_hd_01. The
  model's vocabulary has no "₹" symbol.

Tagging used the recognizer transcripts plus listening.

## Task 2 – rule-based text frontend (`code/normalize.py`, no model)

```bash
python code/normalize.py --kit . --out predictions/t2_normalized.csv
python baseline/run_baseline.py --items data/eval_items.csv --out out/t2_norm \
    --text-csv predictions/t2_normalized.csv --text-column normalized_text --seed 0     # -> predictions/t2_norm/
python baseline/run_baseline.py --items data/val_items.csv --out out/val_control \
    --text-csv predictions/t2_normalized.csv --text-column normalized_text --seed 0     # -> predictions/val_control/
python metrics.py predictions/val_control --json out/t2_val_control.json --csv out/t2_val_control.csv
```

The normalizer rewrites the following into Hindi words in Devanagari. English words in Latin script stay
unchanged.

* Indian-system numbers (हज़ार / लाख / करोड़)
* ₹ amounts, with लाख/करोड़ units and paise
* percentages and decimals
* dates (dd/mm/yyyy); years 1100–1999 are read as "उन्नीस सौ …"
* capitalised abbreviations, spelled out letter by letter (RBI → आर बी आई)

All rules were written from the validation items and standard Hindi number names. On the 24 validation
sentences its output matches the `spoken` reference exactly (text-level CER 0).

| validation CER | hard | hindi | hinglish | long | all | SIM |
|---|---|---|---|---|---|---|
| baseline (provided audio) | 0.4583 | 0.0342 | 0.3020 | 0.0318 | 0.2421 | 0.8303 |
| Task 2 (`val_control/`) | **0.0989** | 0.0375 | 0.3020 | 0.0318 | **0.1231** | 0.8371 |

The Hindi items' text is not changed by normalization. Their small difference (0.0342 → 0.0375, one item
val_hi_05) is GPU run-to-run variation: a fresh run of the unchanged baseline on the same machine also gives
0.0375 (`code/logs/repro.json`).

## Task 3 – planned (see `idea.md`)

Not run yet. `results.json` has the target set: `english_word_error` on `hinglish`, building on the Task 2 text,
with `predictions/val_control/` as the control.
