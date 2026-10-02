"""Parallel prompts from FLORES+ devtest (openlanguagedata/flores_plus, gated: accept terms on HF).

One prompt per source article: its first N_SENTENCES sentences, identical content
in every language. Base models continue the passage directly; instruct models get
the passage after a short "continue this text" instruction in the same language.
"""

import random
from collections import defaultdict

from datasets import load_dataset

FLORES = "openlanguagedata/flores_plus"
SPLIT = "devtest"
N_SENTENCES = 3

LANGS = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "zh": "cmn_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "sw": "swh_Latn",
    "yo": "yor_Latn",
}

# Languages written without spaces between sentences.
NO_SPACE_JOIN = {"zh", "ja"}

# Drafted by Claude: native-speaker check needed (esp. sw, yo).
INSTRUCTIONS = {
    "en": "Continue the following text.",
    "es": "Continúa el siguiente texto.",
    "ru": "Продолжи следующий текст.",
    "zh": "请续写以下文本。",
    "ja": "次の文章の続きを書いてください。",
    "ar": "أكمل النص التالي.",
    "hi": "निम्नलिखित पाठ को आगे जारी रखें।",
    "tr": "Aşağıdaki metni devam ettirin.",
    "sw": "Endeleza maandishi yafuatayo.",
    "yo": "Tẹ̀síwájú ọ̀rọ̀ tó wà nísàlẹ̀ yìí.",
}


def articles(lang: str) -> dict[str, list[tuple[int, str]]]:
    """{article url: [(sentence id, text), ...] in id order} for one language."""
    rows = load_dataset(FLORES, LANGS[lang], split=SPLIT)
    by_url = defaultdict(list)
    for r in rows:
        by_url[r["url"]].append((int(r["id"]), r["text"]))
    return {u: sorted(s) for u, s in by_url.items()}


def select_urls(en: dict, n_prompts: int = 200, seed: int = 0) -> list[str]:
    """The articles used in every experiment, chosen once from English."""
    urls = sorted(en)
    random.Random(seed).shuffle(urls)
    return sorted(urls[:n_prompts], key=lambda u: en[u][0][0])


def join_sentences(sentences: list[str], lang: str) -> str:
    return ("" if lang in NO_SPACE_JOIN else " ").join(sentences)


def build_prompts(n_prompts: int = 200, seed: int = 0) -> list[dict]:
    """Returns one record per (article, language).

    Articles are chosen once (from English) and reused for every language, so
    prompt i is the same passage in all 10 languages. Records carry the
    sentence ids so alignment can be re-checked downstream.
    """
    en = articles("en")
    urls = select_urls(en, n_prompts, seed)

    records = []
    for lang in LANGS:
        arts = en if lang == "en" else articles(lang)
        for i, url in enumerate(urls):
            sents = arts[url][:N_SENTENCES]
            assert [s[0] for s in sents] == [s[0] for s in en[url][:N_SENTENCES]], (lang, url)
            passage = join_sentences([t for _, t in sents], lang)
            records.append({
                "prompt_idx": i,
                "lang": lang,
                "url": url,
                "sentence_ids": [s[0] for s in sents],
                "passage": passage,
                "instruction": INSTRUCTIONS[lang],
            })
    return records


def format_prompt(record: dict, tokenizer, instruct: bool) -> str:
    """Prompt text for one model. Base: the raw passage. Instruct: chat template
    with instruction + passage as the user turn, default system prompt."""
    if not instruct:
        return record["passage"]
    msgs = [{"role": "user", "content": f"{record['instruction']}\n\n{record['passage']}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    recs = build_prompts()
    print(f"{len(recs)} records, {len(recs) // len(LANGS)} prompts per language")
    for r in recs:
        if r["prompt_idx"] == 0:
            print(f"[{r['lang']}] {r['passage'][:120]}")
