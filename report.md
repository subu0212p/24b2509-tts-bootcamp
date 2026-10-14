Q1
The largest improvements were val_hg_01 (0.452 → 0.032) and val_hg_06 (0.318 → 0.023), where converting English words to Devanagari significantly improved pronunciation. Only val_hi_05 worsened (0.000 → 0.020), despite identical inputs, suggesting GPU run-to-run variation confirmed by rerunning the baseline. The smallest improvement was val_hg_02 (0.216 → 0.108), where "न्यू" sounded like "निव" due to difficulty with the य consonant cluster.

Q2
I learned that the main limitation was the input script rather than the model's ability to produce English sounds. Although Latin characters exist in its vocabulary, the model had limited training exposure to them. I would test individual transliteration rules on difficult consonant combinations, especially words like "न्यू", before running the full evaluation. This could help identify pronunciation issues earlier.

Q3
My next improvement would handle uncertain transliterations by generating up to four possible Devanagari spellings for each ambiguous English word. For example, alternatives could include ऑ versus आ or न्यू versus नयू. A speech recognizer would then evaluate the generated pronunciations and select the clearest candidate. Since evaluation may contain unseen English words, this approach should improve generalization beyond the current fixed rules.
