# 24B2509 – TTS bootcamp (Hindi–Hinglish voice cloning with IndicF5)

Tasks 1–4 are done. Parts (1)–(3) of `idea.md` were submitted on the midterm form before Task 3 was run.

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

`code/kaggle_run1.ipynb` runs Tasks 1–2 and `code/kaggle_run2.ipynb` runs Tasks 3–4 on Kaggle (secret `HF_TOKEN`, GPU on,
Internet on).

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

## Task 3 – idea: rule-based English letter-to-sound conversion (`code/translit.py`, no model)

`transliterate_text()` rewrites every Latin-script word in Devanagari the way an Indian speaker reads it. It uses
general spelling rules, not a word list:

* letter groups: sh/ch/th/ph/ck, tion/sion → शन, igh → ाइ, ee/ea → ी, oo → ू, ou → ाउ, ew → यू
* "magic e" vowels; a silent final e; final y → ी
* Indian-English vowel colours: o in a closed syllable → ॉ; a → ै / ा by context; al+k/l → ॉ
* soft c/g; anusvara before stop consonants; doubled consonants are said once

It runs after the Task 2 normalizer, and all other settings are the baseline's (seed 0, one candidate). The rules
were written from the 17 English words in the validation items plus general Indian-English pronunciation, never
from the evaluation text. At the text level, 14/17 validation words match the `spoken` spelling exactly (3.4 %
character difference).

```bash
python code/make_t3_text.py --kit . --out out/t3_text.csv          # id,split,text,t3_text (all 64 items)
python baseline/run_baseline.py --items data/val_items.csv  --out out/val_idea --text-csv out/t3_text.csv --text-column t3_text --seed 0
python baseline/run_baseline.py --items data/eval_items.csv --out out/t3_idea  --text-csv out/t3_text.csv --text-column t3_text --seed 0
python metrics.py out/val_idea --json out/val_idea.json --csv out/val_idea.csv
# out/val_idea -> predictions/val_idea/, out/t3_idea -> predictions/t3_idea/
```

| validation | CER all | hard | hindi | hinglish | long | SIM |
|---|---|---|---|---|---|---|
| baseline | 0.2421 | 0.4583 | 0.0342 | 0.3020 | 0.0318 | 0.8303 |
| control (Task 2 text, `val_control/`) | 0.1231 | 0.0989 | 0.0375 | 0.3020 | 0.0318 | 0.8371 |
| **idea (`val_idea/`)** | **0.0582** | 0.0989 | 0.0375 | **0.0421** | 0.0318 | **0.8411** |

The idea changes only Latin-script words, so only Hinglish moves: 0.302 → 0.042 against the control. Logs are in
`code/logs/val_idea.*` and `code/logs/t3_text.csv`.

## Task 4 – faster: the Task 3 system with `nfe_step=16`

```bash
# same Kaggle session, same GPU (GPU 0), model loaded and one warm-up excluded by run_baseline.py
python baseline/run_baseline.py --items data/val_items.csv --out out/baseline_val --seed 0                    # baseline timing
python baseline/run_baseline.py --items data/val_items.csv --out out/val_fast --text-csv out/t3_text.csv \
    --text-column t3_text --seed 0 --set nfe_step=16                                                           # fast timing
python metrics.py out/val_fast --json out/val_fast.json --csv out/val_fast.csv
python baseline/run_baseline.py --items data/eval_items.csv --out out/t4_fast --text-csv out/t3_text.csv \
    --text-column t3_text --seed 0 --set nfe_step=16                                                           # -> predictions/t4_fast/
```

| | seconds for the 24 validation items | validation CER |
|---|---|---|
| baseline (nfe 32) | 563.6 | 0.2421 |
| fast (Task 3 + nfe 16) | 297.4 | 0.0618 (Task 3: 0.0582; ratio 1.06 ≤ 1.2) |

Speed-up = 563.6 / 297.4 = **1.90×**. Both timing files are in `code/logs/` (`baseline_val_timing.json`,
`val_fast_timing.json`). During both timed runs, the second T4 was busy with the Task 3 synthesis, so the two
timings share the same conditions.
