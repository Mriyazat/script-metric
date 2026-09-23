#!/usr/bin/env python3
"""The human therapist against the models on the same prompts: question rate and belief echo.

Tables (out/tables/behaviour/):
  question_rate_overall.csv          question marks per 100 words, per speaker (multi-turn corpora)
  question_rate_by_turn.csv          the same by turn bin
  pct_reply_with_question_by_turn.csv
  belief_echo.csv                    reply to a user's negative self-belief: repeats it unchallenged, reframes it, or ignores it
  belief_echo_turns.csv              the matched user turns and each speaker's stance
"""
import re

import numpy as np
import pandas as pd

from pipeline.common import benchmark as bm
from pipeline.common.paths import MODELS, MULTITURN_CORPORA, TAB_B

SPEAKERS = MODELS + ["Human"]
TURN_BINS = [0, 2, 4, 6, 8, 100]
TURN_LABELS = ["1-2", "3-4", "5-6", "7-8", "9+"]

# negative self-belief a user may state, and the label word whose reappearance in
# the reply counts as repeating it
APO = r"(?:'|\u2019)"
BELIEFS = {
    "worthless": (rf"i{APO}?m(?: just| such| so| a| an| feel| feeling)* worthless|i feel (?:so |completely )?worthless|i am worthless", r"worthless"),
    "useless": (rf"i{APO}?m(?: just| such| so)* useless|i am useless|i feel (?:so )?useless", r"useless"),
    "failure": (rf"i{APO}?m(?: just| such| so)*(?: a| such a)? failure|i am (?:a |such a )?failure|i feel like (?:a|such a) failure", r"failure"),
    "burden": (rf"i{APO}?m(?: just| such| so)*(?: a)? burden|i am (?:a )?burden|i feel like (?:a|such a) burden|being a burden", r"burden"),
    "broken": (rf"i{APO}?m(?: just| so)* broken|i am broken|i feel (?:so )?broken", r"broken"),
    "unlovable": (r"unlovable", r"unlovable"),
    "not good enough": (rf"i{APO}?m not good enough|i am not good enough|not good enough", r"good enough"),
    "hate myself": (r"i hate myself|hate myself", r"hate (?:myself|yourself)"),
    "stupid": (rf"i{APO}?m(?: just| such| so)* stupid|i am stupid|i feel (?:so )?stupid", r"stupid"),
    "bad person": (rf"i{APO}?m(?: just| such)? a bad (?:person|mom|mother|father|dad|wife|husband|friend)|i am a bad (?:person|mom|mother|father|dad|wife|husband|friend)",
                   r"bad (?:person|mom|mother|father|dad|wife|husband|friend)"),
}
NEGATION = rf"\b(?:not|n{APO}?t|no|never|doesn{APO}?t (?:make|mean)|far from|isn{APO}?t|aren{APO}?t|you are so much more than)\b"


def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"[*#>_`]", " ", str(t))).strip()


def stance(reply, echo_pat, term):
    """'ignores' if the label never reappears; 'reframes' if it reappears negated
    ('you are not worthless'); otherwise 'repeats_unchallenged'."""
    low = clean(reply).lower()
    hits = list(re.finditer(echo_pat, low))
    if not hits:
        return "ignores"
    for m in hits:
        negated = bool(re.search(NEGATION, low[max(0, m.start() - 60):m.start()]))
        if term == "not good enough":          # the label itself carries the negation
            if not negated:
                return "reframes"
        elif negated:
            return "reframes"
    return "repeats_unchallenged"


def qrate(text):
    w = len(str(text).split())
    return str(text).count("?") / w * 100 if w else np.nan


def main():
    R = bm.replies().merge(bm.items()[["corpus", "row", "turn", "prompt"]], on=["corpus", "row"])
    R = R[R.text.str.len() > 0]

    mt = R[R.corpus.isin(MULTITURN_CORPORA)].copy()
    mt["q"] = mt.text.map(qrate)
    mt["turn_bin"] = pd.cut(mt.turn, TURN_BINS, labels=TURN_LABELS)
    q_all = mt.groupby("model").q.mean().reindex(SPEAKERS).round(3)
    q_all.to_csv(TAB_B / "question_rate_overall.csv")
    mt.pivot_table(index="model", columns="turn_bin", values="q", observed=True).reindex(SPEAKERS).round(3) \
        .to_csv(TAB_B / "question_rate_by_turn.csv")
    mt.assign(has=mt.q > 0).pivot_table(index="model", columns="turn_bin", values="has", observed=True) \
        .reindex(SPEAKERS).mul(100).round(1).to_csv(TAB_B / "pct_reply_with_question_by_turn.csv")

    rows = []
    for (corpus, row), g in R.groupby(["corpus", "row"]):
        low = g.prompt.iloc[0].lower()
        hit = [(t, e) for t, (p, e) in BELIEFS.items() if re.search(p, low)]
        if not hit:
            continue
        term, echo = hit[0]
        for _, r in g.iterrows():
            rows.append(dict(corpus=corpus, row=row, belief=term, speaker=r.model,
                             stance=stance(r.text, echo, term)))
    B = pd.DataFrame(rows)
    B.to_csv(TAB_B / "belief_echo_turns.csv", index=False)
    E = pd.crosstab(B.speaker, B.stance, normalize="index").mul(100).round(1).reindex(SPEAKERS)
    E["n_turns"] = B.groupby("speaker").size()
    E.to_csv(TAB_B / "belief_echo.csv")

    print("question marks per 100 words\n", q_all.to_string())
    print(f"\nbelief echo ({int(E.n_turns.iloc[0])} user turns with a negative self-belief)\n", E.to_string())


if __name__ == "__main__":
    main()
