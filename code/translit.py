"""
Task 3 idea - rule-based Indian-English spelling -> Devanagari converter for Latin-script words.

No model and no word list: every English word is converted by general letter-to-sound rules
(grapheme patterns with left/right context), written from the validation items and general
Indian-English pronunciation, never from the evaluation text.

Pipeline: original text -> Task 2 normalize() -> transliterate_text() -> IndicF5.

Rules, in short
  * digraphs/trigraphs: sh->श, ch->च, th->थ, ph->फ, ck->क, gh (silent after i), kn->न, wr->र,
    tion/sion->शन, qu->क्व, x->क्स, ng->ंग
  * vowels: ee/ea->ी, oo->ू, ou/ow(+n)->ाउ, ow(final)->ो, ai/ay->े, ew->यू, igh->ाइ,
    magic-e (a_e->े, i_e->ाइ, o_e->ो, u_e->यू), silent final e, final y->ी, ie(final)->ी
  * Indian-English vowel colours: o in a closed syllable -> ॉ (office, shopping),
    a in a closed syllable -> ै (cancel, match), a before ss/st/sk/r -> ा (class, fast, party),
    al+k/l -> ॉ (walk, call), word-initial a + doubled consonant -> ए (address)
  * c/g softening before e/i/y, -ge/-dge -> ज
  * n/m before a consonant -> anusvara (internet -> इंटरनेट, weekend -> वीकेंड)
  * doubled consonants are said once; consonant clusters get a virama; a word-final consonant
    keeps Hindi's silent final schwa
"""
import re

V = "V"   # vowel unit: (V, independent, matra)
C = "C"   # consonant unit: (C, letter)
N = "N"   # nasal mark (anusvara)

CONS = {"b": "ब", "c": "क", "d": "ड", "f": "फ", "g": "ग", "h": "ह", "j": "ज", "k": "क", "l": "ल",
        "m": "म", "n": "न", "p": "प", "q": "क", "r": "र", "s": "स", "t": "ट", "v": "व", "w": "व",
        "x": "क्स", "y": "य", "z": "ज़"}

VOW = {  # name: (independent form, matra)
    "a": ("अ", ""), "aa": ("आ", "ा"), "i": ("इ", "ि"), "ii": ("ई", "ी"), "u": ("उ", "ु"),
    "uu": ("ऊ", "ू"), "e": ("ए", "े"), "ai": ("ऐ", "ै"), "o": ("ओ", "ो"), "au": ("औ", "ौ"),
    "O": ("ऑ", "ॉ"), "aai": ("आइ", "ाइ"), "aau": ("आउ", "ाउ"), "yuu": ("यू", "्यू"),
}
VOWELS = set("aeiou")


def _is_cons(ch):
    return ch.isalpha() and ch not in VOWELS


def _closed(w, j):
    """True if position j (just after a vowel) starts a closed syllable: two consonants follow, or one
    consonant then end of word (ignoring a silent final e)."""
    rest = w[j:]
    if rest.endswith("e") and len(rest) >= 2 and _is_cons(rest[-2]) and len(rest) == 2:
        return False  # magic-e handled elsewhere
    if len(rest) == 0:
        return False
    if len(rest) == 1:
        return _is_cons(rest[0])
    return _is_cons(rest[0]) and (_is_cons(rest[1]) or rest[1] == "")


def word_units(w):
    """Convert one lowercase Latin word into a list of units."""
    u, i, n = [], 0, len(w)

    def magic_e(k):  # vowel at k, then one consonant, then a final 'e'
        if not (k + 2 == n - 1 and _is_cons(w[k + 1]) and w[n - 1] == "e" and w[k + 1] not in "wxy"):
            return False
        # unstressed endings in longer words keep a short i: office, notice, active, favourite
        if w[k] == "i" and any(c in VOWELS for c in w[:k]) and w[k + 1] in "cvt":
            return False
        return True

    while i < n:
        s = w[i:]
        prev = w[i - 1] if i else ""
        # ---- multi-letter patterns
        if s.startswith(("tion", "sion")):
            u += [(C, "श"), (V, "a")]; u += [(C, "न")]; i += 4; continue
        if s.startswith("igh"):
            u.append((V, "aai")); i += 3; continue
        if s.startswith("dge"):
            u.append((C, "ज")); i += 3; continue
        if s.startswith("ck"):
            u.append((C, "क")); i += 2; continue
        if s.startswith("sh"):
            u.append((C, "श")); i += 2; continue
        if s.startswith("ch"):
            u.append((C, "च")); i += 2; continue
        if s.startswith("th"):
            u.append((C, "थ")); i += 2; continue
        if s.startswith("ph"):
            u.append((C, "फ")); i += 2; continue
        if s.startswith("wh"):
            u.append((C, "व")); i += 2; continue
        if s.startswith("qu"):
            u += [(C, "क"), (C, "व")]; i += 2; continue
        if i == 0 and s.startswith("kn"):
            u.append((C, "न")); i += 2; continue
        if i == 0 and s.startswith("wr"):
            u.append((C, "र")); i += 2; continue
        if s.startswith("ng"):
            u += [(N, ""), (C, "ग")]; i += 2; continue
        if s.startswith(("ee", "ea")):
            u.append((V, "ii")); i += 2; continue
        if s.startswith("oo"):
            u.append((V, "uu")); i += 2; continue
        if s.startswith("ou"):
            u.append((V, "aau")); i += 2; continue
        if s.startswith("ow"):
            u.append((V, "o" if i + 2 >= n else "aau")); i += 2; continue
        if s.startswith(("ai", "ay")):
            u.append((V, "e")); i += 2; continue
        if s.startswith("ew"):
            u.append((V, "yuu")); i += 2; continue
        if s.startswith("oa"):
            u.append((V, "o")); i += 2; continue
        if s.startswith("ie") and i + 2 == n:
            u.append((V, "ii")); i += 2; continue
        if s.startswith("al") and i + 2 < n and w[i + 2] in "kl":
            u += [(V, "O")]; i += 1 + (w[i + 2] == "k"); continue   # walk -> वॉक, call -> कॉल
        # ---- single letters
        ch = w[i]
        if ch == "e":
            if i == n - 1 and n > 2:
                i += 1; continue                       # silent final e
            if s.startswith("er") and i + 2 == n:
                u += [(V, "a"), (C, "र")]; i += 2; continue            # proper -> प्रॉपर
            if s.startswith("er") and i + 2 < n and _is_cons(w[i + 2]):
                u += [(V, "a"), (C, "र"), (V, "a")]; i += 2; continue  # internet -> इंटरनेट
            if s.startswith("el") and i + 2 == n:
                u.append((V, "a")); i += 1; continue   # hotel -> होटल
            u.append((V, "e")); i += 1; continue
        if ch == "a":
            if i == n - 1:
                u.append((V, "aa"))
            elif magic_e(i):
                u.append((V, "e"))
            elif s.startswith(("ar", "ss", "st", "sk", "sp")) or w[i + 1:i + 3] in ("ss", "st", "sk", "sp"):
                u.append((V, "aa"))
            elif i == 0 and i + 2 < n and w[i + 1] == w[i + 2]:
                u.append((V, "e"))
            elif _is_cons(w[i + 1]):
                u.append((V, "ai"))
            else:
                u.append((V, "a"))
            i += 1; continue
        if ch == "i":
            u.append((V, "aai" if magic_e(i) else "i")); i += 1; continue
        if ch == "o":
            if i == n - 1 or magic_e(i):
                u.append((V, "o"))
            elif i + 1 < n and w[i + 1] == "v" and i + 2 < n and w[i + 2] in "ie":
                u.append((V, "uu"))                    # movie, move
            elif i + 1 < n and _is_cons(w[i + 1]):
                u.append((V, "O"))                     # office, shopping, online
            else:
                u.append((V, "o"))
            i += 1; continue
        if ch == "u":
            if magic_e(i) or (i + 2 == n and w[i + 1] == "e"):
                u.append((V, "yuu"))
            elif s.startswith("ur"):
                u.append((V, "a"))
            elif i + 1 < n and _is_cons(w[i + 1]) and i + 2 < n and _is_cons(w[i + 2]):
                u.append((V, "a"))                     # function, number
            else:
                u.append((V, "u"))
            i += 1; continue
        if ch == "y":
            if i == 0:
                u.append((C, "य"))
            elif i == n - 1:
                u.append((V, "ii"))
            else:
                u.append((V, "aai"))
            i += 1; continue
        if ch == "c":
            nxt = w[i + 1] if i + 1 < n else ""
            u.append((C, "स" if nxt in ("e", "i", "y") else "क")); i += 1; continue
        if ch == "g":
            nxt = w[i + 1] if i + 1 < n else ""
            soft = nxt == "e" and i + 2 == n                # message -> मैसेज
            u.append((C, "ज" if soft else "ग")); i += 1; continue
        if ch in "nm" and i + 1 < n and w[i + 1] in "kgcjtdpbsfqxz" and u and u[-1][0] == V:
            u.append((N, "")); i += 1; continue
        if ch == "h" and prev and _is_cons(prev):
            i += 1; continue
        if ch in CONS:
            if u and u[-1][0] == C and w[i - 1:i] == ch:   # doubled consonant said once
                i += 1; continue
            u.append((C, CONS[ch])); i += 1; continue
        i += 1
    return u


def render(units):
    out, prev = [], None
    for k, (kind, val) in enumerate(units):
        if kind == C:
            if prev == C:
                out.append("्")
            out.append(val)
        elif kind == V:
            ind, mat = VOW[val]
            out.append(mat if prev == C else ind)
        else:
            out.append("ं")
        prev = kind if kind != N else prev
    return "".join(out)


def transliterate_word(word):
    lower = word.lower()
    if not lower.isalpha():
        return word
    return render(word_units(lower))


def transliterate_text(text):
    """Replace every Latin-script word with its Devanagari reading; everything else is unchanged."""
    return re.sub(r"[A-Za-z]+", lambda m: transliterate_word(m.group(0)), text)


if __name__ == "__main__":
    import sys
    for w in sys.argv[1:]:
        print(w, "->", transliterate_word(w))
