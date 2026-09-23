#!/usr/bin/env python3
"""Typeset the paper's numeric tables from the pipeline's CSV outputs.

    python -m pipeline.paper_tables

Writes one LaTeX fragment per table to out/tables/latex/<name>.tex (table bodies
only; the surrounding table environment and caption live in the paper source),
plus out/tables/latex/numbers.json with every headline number the prose cites.
Every value in the paper's tables is copied from these fragments, so a diff of
this directory against the paper is the audit trail."""
import json

import numpy as np
import pandas as pd

from pipeline.common.paths import DERIVED, MODELS, OUT, TAB

LATEX = TAB / "latex"
LATEX.mkdir(exist_ok=True)


def f3(x):
    return f"{x:.3f}" if pd.notna(x) else "---"


def s3(x):
    """signed, three decimals, LaTeX minus"""
    if pd.isna(x):
        return "---"
    return (r"$-$" if x < 0 else "") + f"{abs(x):.3f}"


def z1(x):
    return f"{x:.1f}" if pd.notna(x) else "---"


def thou(n):
    return f"{int(n):,}".replace(",", "{,}")


def write(name, body):
    import re
    body = re.sub(r"\(-(\d)", r"($-$\1", body)          # negative z inside parentheses
    body = re.sub(r"& -(\d)", r"& $-$\1", body)          # negative z in its own column
    (LATEX / f"{name}.tex").write_text(body)
    print(f"wrote {LATEX / name}.tex")


numbers = {}

# ---------------------------------------------------------------- tab:script-main
val = pd.read_csv(DERIVED / "validation_results.csv").set_index("system")
byc = pd.read_csv(DERIVED / "script_by_corpus.csv")
ceil = pd.read_csv(TAB / "matched_ceiling.csv").set_index("corpus")
rows = []
for m in MODELS:
    r = val.loc[m]
    cb = byc[(byc.model == m) & (byc.corpus == "carebench")].iloc[0]
    ho = byc[(byc.model == m) & (byc.corpus == "hope")].iloc[0]
    frac = ceil.loc[f"counselling ({m}, 20 codes)", "fraction"]
    bold = m == "Claude"
    def b(s):
        return rf"\textbf{{{s}}}" if bold else s
    rows.append(f"{m} & {b(f3(r.SCRIPT))} & {f3(r.C_excess)} & {b(f3(r.M_excess))} & {z1(r.z)} & "
                f"{frac * 100:.0f}\\% & {b(f3(cb.SCRIPT))} & {b(f3(ho.SCRIPT))} & {thou(r.n_events)} \\\\")
write("script_main_models", "\n".join(rows))
numbers["testbed1"] = {m: dict(SCRIPT=float(val.loc[m, "SCRIPT"]), C=float(val.loc[m, "C_excess"]),
                               M=float(val.loc[m, "M_excess"]), z=float(val.loc[m, "z"]),
                               events=int(val.loc[m, "n_events"]),
                               fraction=float(ceil.loc[f"counselling ({m}, 20 codes)", "fraction"]))
                       for m in MODELS}

# ruler block
ver = json.load(open(OUT / "verify_report.json"))
ctrl = ver["V3"]
wmt = pd.read_csv(TAB / "anchor_wmt24.csv")
d2t = pd.read_csv(TAB / "anchor_d2t.csv")
rag = pd.read_csv(TAB / "anchor_ragtruth.csv")
prop = pd.read_csv(TAB / "anchor_propaganda.csv")
prop_h = prop[prop.layer == "human experts"].iloc[0]
prop_c = prop[prop.layer == "random-label control"].iloc[0]
annomi = ceil.loc["AnnoMI human counsellors"]
ther = pd.read_csv(TAB / "therapist_baseline.csv").set_index("speaker")
llm = pd.read_csv(TAB / "llm_annotator_comparison.csv").set_index("model")
wmt_in = wmt[wmt.system != "unbabel-tower70b"]
ruler = [
    (r"Random-label control (Claude's events, labels re-dealt)", s3(ctrl["SCRIPT"]), s3(ctrl["C_excess"]), s3(ctrl["M_excess"]),
     f"{ctrl['z']:.1f}", thou(ctrl["n_events"])),
    (r"MT error spans, WMT24 (8 systems)$^{\ast}$", f"{s3(wmt_in.SCRIPT.min())}--{f3(wmt_in.SCRIPT.max())}", "", "",
     rf"$|z| \le {wmt_in.z.abs().max():.1f}$", thou(wmt.n_events.sum())),
    (r"Data-to-text error spans (4 generators)", f"{f3(d2t.SCRIPT.min())}--{f3(d2t.SCRIPT.max())}", "", "",
     rf"$\le {d2t.z.max():.1f}$", thou(d2t.n_events.sum())),
    (r"Hallucination spans, RAGTruth (6 generators)$^{\ddagger}$", f"{f3(rag.SCRIPT.min())}--{f3(rag.SCRIPT.max())}", "", "",
     rf"$\le {rag.z.max():.1f}$", thou(rag.n_events.sum())),
    (r"Human-written news, 18 propaganda techniques (expert spans)$^{\|}$", f3(prop_h.SCRIPT), f3(prop_h.C_excess), f3(prop_h.M_excess),
     f"{prop_h.z:.1f}", thou(prop_h.n_events)),
    (r"Human counsellors, AnnoMI (4 MI labels)$^{\S}$", f3(annomi.SCRIPT), f3(annomi.C), f3(annomi.M), f"{annomi.z:.1f}", thou(annomi.n_events)),
    (r"Human therapist, surface layer (3 labels, same prompts)$^{\dagger}$", f3(ther.loc["Therapist", "SCRIPT"]), f3(ther.loc["Therapist", "C"]),
     f3(ther.loc["Therapist", "M"]), f"{ther.loc['Therapist', 'z']:.1f}", thou(ther.loc["Therapist", "n_events"])),
    (r"Human therapist, blind LLM layer (20 codes, same prompts)$^{\P}$", f3(llm.loc["Human", "SCRIPT_llm"]), f3(llm.loc["Human", "C_llm"]),
     f3(llm.loc["Human", "M_llm"]), f"{llm.loc['Human', 'z_llm']:.1f}", thou(llm.loc["Human", "n_events_llm"])),
]
write("script_main_ruler", "\n".join(f"{a} & {b} & {c} & {d} & {e} & {f} \\\\" for a, b, c, d, e, f in ruler))
numbers["ruler"] = dict(
    control=dict(SCRIPT=ctrl["SCRIPT"], z=ctrl["z"]),
    wmt=dict(min=float(wmt.SCRIPT.min()), max=float(wmt.SCRIPT.max()), abs_z_inliers=float(wmt_in.z.abs().max()),
             outlier=dict(system="unbabel-tower70b", SCRIPT=float(wmt[wmt.system == "unbabel-tower70b"].SCRIPT.iloc[0]),
                          z=float(wmt[wmt.system == "unbabel-tower70b"].z.iloc[0]),
                          events=int(wmt[wmt.system == "unbabel-tower70b"].n_events.iloc[0]))),
    d2t=dict(min=float(d2t.SCRIPT.min()), max=float(d2t.SCRIPT.max()), zmax=float(d2t.z.max()), zmin=float(d2t.z.min())),
    ragtruth={r.system: dict(SCRIPT=float(r.SCRIPT), z=float(r.z), events=int(r.n_events)) for r in rag.itertuples()},
    propaganda=dict(SCRIPT=float(prop_h.SCRIPT), z=float(prop_h.z), C=float(prop_h.C_excess), M=float(prop_h.M_excess),
                    R_raw=float(prop_h.R_raw), R_null=float(prop_h.R_null), control=float(prop_c.SCRIPT), control_z=float(prop_c.z),
                    fraction=float(ceil.loc["propaganda techniques (human news)", "fraction"])),
    annomi=dict(SCRIPT=float(annomi.SCRIPT), z=float(annomi.z), C=float(annomi.C), M=float(annomi.M),
                ceiling=float(annomi.ceiling), fraction=float(annomi.fraction), events=int(annomi.n_events)),
    therapist_surface=dict(SCRIPT=float(ther.loc["Therapist", "SCRIPT"]), z=float(ther.loc["Therapist", "z"]),
                           models_min=float(ther.loc[MODELS, "SCRIPT"].min()), models_max=float(ther.loc[MODELS, "SCRIPT"].max())),
    therapist_llm=dict(SCRIPT=float(llm.loc["Human", "SCRIPT_llm"]), z=float(llm.loc["Human", "z_llm"]), C=float(llm.loc["Human", "C_llm"]),
                       M=float(llm.loc["Human", "M_llm"]),
                       models_min=float(llm.loc[MODELS, "SCRIPT_llm"].min()), models_max=float(llm.loc[MODELS, "SCRIPT_llm"].max()),
                       models_z_min=float(llm.loc[MODELS, "z_llm"].min()), models_z_max=float(llm.loc[MODELS, "z_llm"].max()),
                       models_C_min=float(llm.loc[MODELS, "C_llm"].min()), models_C_max=float(llm.loc[MODELS, "C_llm"].max())),
    fractions={k: float(v) for k, v in ceil.fraction.items()},
)

# ---------------------------------------------------------------- tab:script-by-corpus (appendix)
rows = []
for m in MODELS:
    cells = []
    for c in ["counselchat", "pair", "carebench", "hope", "ALL"]:
        r = byc[(byc.model == m) & (byc.corpus == c)].iloc[0]
        cells.append(f"{s3(r.SCRIPT)} ({z1(r.z)})")
    cs = [f3(byc[(byc.model == m) & (byc.corpus == c)].C_excess.iloc[0]) for c in ["counselchat", "pair", "carebench", "hope"]]
    rows.append(f"{m} & " + " & ".join(cells) + " & " + " & ".join(cs) + r" \\")
ev = byc[byc.model == "Claude"].set_index("corpus")
write("script_by_corpus", "\n".join(rows))
numbers["by_corpus_events"] = {c: int(byc[byc.corpus == c].n_events.sum()) for c in ["counselchat", "pair", "carebench", "hope"]}

# ---------------------------------------------------------------- tab:mint-systems
M = pd.read_csv(TAB / "mint_systems.csv")
SL = pd.read_csv(TAB / "mint_surface_layer.csv").set_index("system")
order = ["vanilla prompting", "tactic-list prompt", "tactic-history prompt", "verbalized sampling (vanilla prompt)",
         "verbalized sampling (tactic prompt)", "verbalized sampling (history prompt)", "quality-only RL (PsychoCounsel)",
         "quality RL + token diversity (R1-Zero-Div)", "MINT (quality + KL novelty)", "MINT (quality + entropy)",
         "MINT (quality + KL + entropy)"]
rows = []
for fam in order:
    for size in ["1.7B", "4B"]:
        r = M[(M.family == fam) & (M["size"] == size)].iloc[0]
        u = SL.loc[r.system]
        rows.append(f"{fam} & {size} & {f3(r.C)} & {s3(r.M)} & {s3(r.SCRIPT)} & {r.z:.0f} & {s3(r.SCRIPT_matched)} & "
                    f"{f3(u.C)} & {s3(u.M)} & {s3(u.SCRIPT)} & {u.z:.0f} & "
                    f"{r.stickiness:.3f} & {r.empathy:.2f} & {r.events_per_turn:.1f} / {u.events_per_turn:.1f} \\\\")
g = M[M["size"] == "human"].iloc[0]
gu = SL.loc["gold"]
rows.append(r"\midrule")
rows.append(f"Human gold replies & --- & {f3(g.C)} & {s3(g.M)} & {s3(g.SCRIPT)} & {g.z:.1f} & {s3(g.SCRIPT_matched)} & "
            f"{f3(gu.C)} & {s3(gu.M)} & {s3(gu.SCRIPT)} & {gu.z:.1f} & "
            f"{g.stickiness:.3f} & {g.empathy:.2f} & {g.events_per_turn:.1f} / {gu.events_per_turn:.1f} \\\\")
write("mint_systems", "\n".join(rows))
from scipy.stats import spearmanr
L = M[M["size"] != "human"]
numbers["testbed2"] = dict(
    rho_C_empathy=float(spearmanr(L.C, L.empathy)[0]), rho_SCRIPT_empathy=float(spearmanr(L.SCRIPT, L.empathy)[0]),
    rho_matched_empathy=float(spearmanr(L.SCRIPT_matched, L.empathy)[0]),
    rho_C_events=float(spearmanr(L.C, L.events_per_turn)[0]), rho_SCRIPT_events=float(spearmanr(L.SCRIPT, L.events_per_turn)[0]),
    rho_M_events=float(spearmanr(L.M, L.events_per_turn)[0]),
    n_M_negative=int((L.M < 0).sum()), gold=dict(C=float(g.C), M=float(g.M), SCRIPT=float(g.SCRIPT), z=float(g.z)),
    C_min=float(L.C.min()), C_max=float(L.C.max()), SCRIPT_min=float(L.SCRIPT.min()), SCRIPT_max=float(L.SCRIPT.max()),
    systems={r.system: dict(C=float(r.C), M=float(r.M), SCRIPT=float(r.SCRIPT), stickiness=float(r.stickiness),
                            events_per_turn=float(r.events_per_turn)) for r in M.itertuples()})
SLm = SL[SL["size"] != "human"]
gu = SL.loc["gold"]
numbers["testbed2_surface"] = dict(
    SCRIPT_min=float(SLm.SCRIPT.min()), SCRIPT_max=float(SLm.SCRIPT.max()), z_min=float(SLm.z.min()), z_max=float(SLm.z.max()),
    C_min=float(SLm.C.min()), C_max=float(SLm.C.max()), M_min=float(SLm.M.min()), M_max=float(SLm.M.max()),
    n_M_negative=int((SLm.M < 0).sum()), epr_min=float(SLm.events_per_turn.min()), epr_max=float(SLm.events_per_turn.max()),
    gold=dict(SCRIPT=float(gu.SCRIPT), z=float(gu.z), C=float(gu.C), M=float(gu.M), events=int(gu.n_events), epr=float(gu.events_per_turn)),
    rho_SCRIPT_taggerC=float(spearmanr(SLm.SCRIPT, SLm.tagger_C)[0]), rho_C_taggerC=float(spearmanr(SLm.C, SLm.tagger_C)[0]),
    rho_SCRIPT_empathy=float(spearmanr(SLm.SCRIPT, SLm.empathy)[0]), rho_matched_empathy=float(spearmanr(SLm.SCRIPT_matched, SLm.empathy)[0]),
    rho_C_empathy=float(spearmanr(SLm.C, SLm.empathy)[0]),
    rho_SCRIPT_events=float(spearmanr(SLm.SCRIPT, SLm.events_per_turn)[0]), rho_M_events=float(spearmanr(SLm.M, SLm.events_per_turn)[0]),
    rho_SCRIPT_stickiness=float(spearmanr(SLm.SCRIPT, SLm.stickiness)[0]),
    systems={k: dict(SCRIPT=float(v.SCRIPT), z=float(v.z), C=float(v.C), M=float(v.M), epr=float(v.events_per_turn),
                     share_other=float(v.share_other)) for k, v in SL.iterrows()})

# ---------------------------------------------------------------- tab:script-profile
pr = pd.read_csv(DERIVED / "profile_reading.csv").set_index("model")
js = pd.read_csv(DERIVED / "js_fingerprint_distance.csv", index_col=0)
sig = {"Qwen": "twin of Gemini", "Gemini": "twin of Qwen", "GPT": "half of everything is advice",
       "Claude": "questions parked latest", "Llama": "far from everyone"}
# per-model rank-1 at k=5, held-out corpus (case-study confusion diagonal); read from the case-study log if present
conf_diag = {}
log = OUT / "logs" / "30_fig_case_study.txt"
if log.exists():
    lines = log.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("pooled confusion:"):
            for m in MODELS:
                for ln2 in lines[i + 2:i + 8]:          # skip the header line
                    parts = ln2.split()
                    if len(parts) == 6 and parts[0] == m:
                        conf_diag[m] = float(parts[1 + MODELS.index(m)])
rows = []
for m in ["Qwen", "Gemini", "GPT", "Claude", "Llama"]:
    r = pr.loc[m]
    others = js.loc[m].drop(m)
    rows.append(f"{m} & {r.empathy_pos:.2f} & {r.empathy_share:.3f} & {r.advice_pos:.2f} & {r.advice_share:.3f} & "
                f"{r.questions_pos:.2f} & {r.questions_share:.3f} & {sig[m]} & {others.min():.3f} / {others.max():.3f} & "
                f"{conf_diag.get(m, float('nan')):.2f} \\\\")
write("script_profile", "\n".join(rows))
numbers["profile"] = dict(js_qwen_gemini=float(js.loc["Qwen", "Gemini"]), js_llama_min=float(js.loc["Llama"].drop("Llama").min()),
                          js_next_closest=float(sorted(js.values[np.triu_indices(5, 1)])[1]), id_k5=conf_diag,
                          id_k5_mean=float(np.mean(list(conf_diag.values()))) if conf_diag else None)

# ---------------------------------------------------------------- identification curves
idc = pd.read_csv(TAB / "identification_curves.csv")
numbers["identification"] = {s: {int(r.k): float(r.accuracy) for r in g.itertuples()} for s, g in idc.groupby("setting")}
loco = pd.read_csv(TAB / "identification_loco.csv")
numbers["loco"] = dict(loglik=int(loco.loglik_correct.sum()), js=int(loco.js_correct.sum()), n=len(loco))

# ---------------------------------------------------------------- tab:profile-by-corpus
tw = pd.read_csv(TAB / "twin_structure_by_corpus.csv").set_index("corpus")
rows = []
for c in ["counselchat", "pair", "carebench", "hope"]:
    r = tw.loc[c]
    ll = loco[loco.held_out == c]
    rows.append(f"{c} & {r.qwen_gemini:.3f} & {r.next_closest_pair.replace('-', '--')} \\; {r.next_closest_js:.3f} & "
                f"{r.ratio_next_over_twin:.1f} & {r.llama_mean_dist:.3f} & & {int(ll.loglik_correct.sum())} & {int(ll.js_correct.sum())} \\\\")
r = tw.loc["POOLED"]
rows.append(r"\midrule")
rows.append(f"POOLED & {r.qwen_gemini:.3f} & {r.next_closest_pair.replace('-', '--')} \\; {r.next_closest_js:.3f} & "
            f"{r.ratio_next_over_twin:.1f} & {r.llama_mean_dist:.3f} & & \\multicolumn{{2}}{{c}}{{{int(loco.loglik_correct.sum())}/{len(loco)} \\; {int(loco.js_correct.sum())}/{len(loco)}}} \\\\")
write("profile_by_corpus", "\n".join(rows))

# ---------------------------------------------------------------- tab:script-robust
ci = pd.read_csv(TAB / "bootstrap_ci.csv").set_index("model")
ev_ = pd.read_csv(TAB / "estimator_variants.csv").set_index("model")
lt = pd.read_csv(TAB / "length_terciles.csv")
rows = []
for m in MODELS:
    pt = val.loc[m, "SCRIPT"]
    lo = pt - (ci.loc[m, "SCRIPT_mean"] - ci.loc[m, "SCRIPT_lo"])
    hi = pt + (ci.loc[m, "SCRIPT_hi"] - ci.loc[m, "SCRIPT_mean"])
    t = {r.tercile: r for r in lt[lt.model == m].itertuples()}
    rows.append(f"{m} & {f3(pt)} & [{f3(lo)}, {f3(hi)}] & {f3(ev_.loc[m, 'C_knn_excess'])} ({ev_.loc[m, 'z']:.1f}) & "
                f"{s3(ev_.loc[m, 'M2gain_excess'])} ({ev_.loc[m, 'M2_z']:.1f}) & "
                f"{s3(t['short'].SCRIPT)} ({t['short'].z:.1f}) & {s3(t['mid'].SCRIPT)} ({t['mid'].z:.1f}) & {s3(t['long'].SCRIPT)} ({t['long'].z:.1f}) \\\\")
write("script_robust", "\n".join(rows))
pw = pd.read_csv(TAB / "pairwise_tests.csv")
numbers["bootstrap"] = {m: dict(half_lo=float(ci.loc[m, "SCRIPT_mean"] - ci.loc[m, "SCRIPT_lo"]),
                                half_hi=float(ci.loc[m, "SCRIPT_hi"] - ci.loc[m, "SCRIPT_mean"])) for m in MODELS}
numbers["pairwise"] = {r.pair: dict(delta=float(r.delta), lo=float(r.lo), hi=float(r.hi), p_holm=float(r.p_holm)) for r in pw.itertuples()}
numbers["length_terciles"] = {f"{r.model}/{r.tercile}": dict(SCRIPT=float(r.SCRIPT), z=float(r.z), events=int(r.n_events)) for r in lt.itertuples()}
loao = pd.read_csv(TAB / "loao_tier_gap.csv")
meta = pd.read_csv(TAB / "tier_gap_meta.csv")
stats = json.load(open(TAB / "tier_gap_meta_stats.json"))
numbers["annotator_gap"] = dict(loao={r.held_out: dict(gap=float(r.tier_gap), lo=float(r.lo), hi=float(r.hi), p=float(r.p_two_sided),
                                                        claude_minus_next=float(r.claude_minus_next_best)) for r in loao.itertuples()},
                                meta={r.annotator: dict(gap=float(r.gap), se=float(r.se), z=float(r.z)) for r in meta.itertuples()},
                                stats=stats)

# ---------------------------------------------------------------- tab:script-ablations helpers
alt = pd.read_csv(TAB / "alternative_nulls.csv").set_index("model")
la = pd.read_csv(TAB / "length_ablation.csv").set_index("model")
gr = pd.read_csv(TAB / "granularity.csv")
sk = pd.read_csv(TAB / "skipgram.csv").set_index("model")
qm = pd.read_csv(TAB / "quality_matched.csv")
numbers["ablations"] = dict(
    circshift=dict(min=float(alt.excess_circshift.min()), max=float(alt.excess_circshift.max()), zmin=float(alt.z_circ.min()), zmax=float(alt.z_circ.max())),
    withinbin=dict(min=float(alt.excess_withinbin.min()), max=float(alt.excess_withinbin.max()), zmin=float(alt.z_bin.min()), zmax=float(alt.z_bin.max())),
    C_mid_shift_max=float((la.C_mid - la.C_start).abs().max()), dur_gain_max=float(la.dur_gain.max()), dur_gain_zmax=float(la.dur_gain_z.max()),
    granularity={s: dict(min=float(g[g.model.isin(MODELS)].SCRIPT.astype(float).min()), max=float(g[g.model.isin(MODELS)].SCRIPT.astype(float).max()),
                         zmin=float(g[g.model.isin(MODELS)].z.astype(float).min()),
                         id=g[~g.model.isin(MODELS)].iloc[0].SCRIPT + " / " + g[~g.model.isin(MODELS)].iloc[0].z) for s, g in gr.groupby("scheme")},
    skipgram={m: dict(order1=float(sk.loc[m, "order1"]), skip1=float(sk.loc[m, "skip1"]), skip2=float(sk.loc[m, "skip2"]),
                      z=[float(sk.loc[m, "order1_z"]), float(sk.loc[m, "skip1_z"]), float(sk.loc[m, "skip2_z"])]) for m in MODELS},
    quality={f"{r.axis}/{r.model}": dict(raw=float(r.delta_published), matched=float(r.delta_matched), z=float(r.z_matched), verdict=r.verdict,
                                          n_good=int(r.n_events_good), n_bad=int(r.n_events_bad)) for r in qm.itertuples()},
)
rows = []
for axis, label in [("empathy accuracy", r"Accurate vs.\ inaccurate empathy, $\Delta$ raw \,/\, at matched $n$"),
                    ("harmful flag", r"Unflagged vs.\ harmful-flagged, $\Delta$ raw \,/\, at matched $n$")]:
    cells = []
    for m in MODELS:
        r = qm[(qm.axis == axis) & (qm.model == m)].iloc[0]
        cells.append(f"{m} ${s3(r.delta_published).replace('$', '')} / {s3(r.delta_matched).replace('$', '')}$")
    rows.append(f"{label} & " + "; ".join(cells) + r" \\")
write("quality_rows", "\n".join(rows))

# ---------------------------------------------------------------- tab:tie-robustness
tie = pd.read_csv(TAB / "tie_robustness.csv")
rows = []
for layer, lab in [("clinician", "clinician"), ("llm", "LLM")]:
    for r in tie[tie.layer == layer].itertuples():
        name = "Human therapist" if r.system == "Human" else r.system
        rows.append(f"{lab} & {name} & {thou(r.events)} & {r.tied_share:.2f} & {f3(r.C)} & {f3(r.SCRIPT_slot)} & {f3(r.M_slot)} & {r.z_slot:.1f} & & "
                    f"{f3(r.SCRIPT_chain)} & {f3(r.M_chain)} & {r.z_chain:.1f} \\\\")
    if layer == "clinician":
        rows.append(r"\midrule")
write("tie_robustness", "\n".join(rows))
numbers["tie"] = {f"{r.layer}/{r.system}": dict(tied=float(r.tied_share), M_slot=float(r.M_slot), M_chain=float(r.M_chain),
                                                SCRIPT_slot=float(r.SCRIPT_slot), SCRIPT_chain=float(r.SCRIPT_chain),
                                                z_slot=float(r.z_slot), z_chain=float(r.z_chain)) for r in tie.itertuples()}

# ---------------------------------------------------------------- tab:empathy-tactics
et = pd.read_csv(TAB / "anchor_empathy_tactics.csv")
names = {"gpt4-turbo": "GPT-4-turbo", "llama3-70b": "Llama-3-70B", "human": "Human writers (psychology background)",
         "GPT-3.5-turbo": "GPT-3.5-turbo", "GPT-4": "GPT-4 (three variants pooled)", "Llama-2-70B": "Llama-2-70B",
         "Human supporters (Lend-an-Ear)": "Human supporters (Lend an Ear)"}
rows = []
for layer in ["Study-1 human coding", "Real conversations (tagger)"]:
    for r in et[et.layer == layer].itertuples():
        rows.append(f"{names[r.system]} & {f3(r.C)} & {s3(r.M)} & {s3(r.SCRIPT)} & {r.z:.1f} & {s3(r.control)} & "
                    f"{thou(r.n_events)} / {thou(r.n_responses)} & {r.events_per_response:.1f} \\\\")
    if layer == "Study-1 human coding":
        rows.append(r"\midrule")
write("empathy_tactics", "\n".join(rows))
numbers["empathy_tactics"] = {f"{r.layer}/{r.system}": dict(C=float(r.C), M=float(r.M), SCRIPT=float(r.SCRIPT), z=float(r.z),
                                                            events=int(r.n_events), epr=float(r.events_per_response)) for r in et.itertuples()}

# ---------------------------------------------------------------- tab:propaganda
rows = []
for r in prop.itertuples():
    lab = {"human experts": "Human experts (Da San Martino et al.)", "random-label control": "Random-label control (same events)"}.get(r.layer)
    if lab is None:
        setup, ann = r.layer.replace("LLM annotator ", "").split("/")
        pretty = {"deepseek-r1": "DeepSeek-R1", "llama3-3": "Llama 3.3", "claude-3-7-sonnet": "Claude 3.7 Sonnet",
                  "gemini-2-0-flash-thinking": "Gemini 2.0 Flash Thinking", "gpt4o": "GPT-4o", "o3-mini": "o3-mini"}[ann]
        lab = f"{ {'zeroshot': 'zeroshot', '5shot': '5-shot', 'cot': 'CoT'}[setup] } / {pretty}"
    rows.append((r.layer, lab, f"{lab} & {s3(r.SCRIPT)} & {r.z:.1f} & {s3(r.C_excess)} & {s3(r.M_excess)} & {f3(r.R_raw)} & {f3(r.R_null)} & {thou(r.n_events)} \\\\", r.n_events))
top = [x[2] for x in rows if x[0] in ("human experts", "random-label control")]
bottom = [x[2] for x in sorted([x for x in rows if x[0] not in ("human experts", "random-label control")], key=lambda t: -t[3])]
write("propaganda", "\n".join(top + [r"\midrule"] + bottom))
numbers["propaganda_llm_layers"] = {r.layer: dict(SCRIPT=float(r.SCRIPT), z=float(r.z), events=int(r.n_events)) for r in prop.itertuples()}

# ---------------------------------------------------------------- tab:script-by-turn
sbt = pd.read_csv(TAB / "script_by_turn.csv")
rows = []
for sp in MODELS + ["Human"]:
    cells = []
    for layer in ["LLM annotator", "clinicians"]:
        for b in ["T1-3", "T4-7", "T8-10"]:
            r = sbt[(sbt.layer == layer) & (sbt.speaker == sp) & (sbt.bucket == b)]
            cells.append("---" if r.empty else f"{f3(r.C.iloc[0])} / {s3(r.SCRIPT.iloc[0])} ({r.z.iloc[0]:.0f})")
        if layer == "LLM annotator":
            cells.append("")
    rows.append(f"{'Therapist' if sp == 'Human' else sp} & " + " & ".join(cells) + r" \\")
write("script_by_turn", "\n".join(rows))
numbers["script_by_turn"] = dict(C_models_min=float(sbt[sbt.speaker != "Human"].C.min()), C_models_max=float(sbt[sbt.speaker != "Human"].C.max()),
                                 C_human_min=float(sbt[sbt.speaker == "Human"].C.min()), C_human_max=float(sbt[sbt.speaker == "Human"].C.max()))

# ---------------------------------------------------------------- tab:llm-agreement
ag = pd.read_csv(TAB / "llm_agreement.csv")
summ = pd.read_csv(TAB / "llm_agreement_summary.csv").iloc[0]
rows = [f"{r.code} & {thou(r.clinician_spans)} & {r.match_rate * 100:.0f}\\% \\\\" for r in ag.itertuples()]
write("llm_agreement", "\n".join(rows))
numbers["llm_agreement"] = dict(text_overlap=float(summ.text_overlap), code_overlap=float(summ.code_overlap),
                                spans_per_reply_clinician=float(summ.spans_per_reply_clinician), spans_per_reply_llm=float(summ.spans_per_reply_llm),
                                per_code={r.code: float(r.match_rate) for r in ag.itertuples()})
lc = llm.loc[MODELS]
numbers["llm_layer"] = dict(rho_SCRIPT=float(spearmanr(lc.SCRIPT_llm, lc.SCRIPT_clin)[0]), rho_C=float(spearmanr(lc.C_llm, lc.C_clin)[0]),
                            control=float(llm.loc["random-label control (LLM events)", "SCRIPT_llm"]),
                            models={m: dict(SCRIPT=float(lc.loc[m, "SCRIPT_llm"]), z=float(lc.loc[m, "z_llm"]), C=float(lc.loc[m, "C_llm"]),
                                            M=float(lc.loc[m, "M_llm"]), events=int(lc.loc[m, "n_events_llm"])) for m in MODELS})

# ---------------------------------------------------------------- juries, betting, validation, listening
jury = pd.read_csv(TAB / "jury_scripts.csv")
numbers["jury"] = {f"{r.reviewer}/{r.model}": dict(SCRIPT=float(r.SCRIPT), z=float(r.z), events=int(r.n_events)) for r in jury.itertuples()}
bv = pd.read_csv(TAB / "betting_validity.csv")
numbers["betting"] = dict(validity={r.method: dict(rejections=int(r.rejections), streams=int(r.streams)) for r in bv.itertuples()},
                          transfer={r.corpus: dict(rejected=bool(r.rejected), tau=(None if pd.isna(r.tau) else int(r.tau))) for r in pd.read_csv(TAB / "betting_transfer.csv").itertuples()},
                          real_power=pd.read_csv(TAB / "reviewer_qs" / "q7_real_power.csv").to_dict("records"),
                          synthetic_power=pd.read_csv(TAB / "reviewer_qs" / "q7_synthetic_power.csv").to_dict("records"))
numbers["synthetic"] = ver["V4"]
nv = pd.read_csv(TAB / "null_validation.csv")
numbers["null_validation"] = dict(raw_comp_min=float(nv[nv.sweep == "composition"].raw_mean.min()), raw_comp_max=float(nv[nv.sweep == "composition"].raw_mean.max()),
                                  raw_size_min=float(nv[nv.sweep == "sample size"].raw_mean.min()), raw_size_max=float(nv[nv.sweep == "sample size"].raw_mean.max()),
                                  script_abs_max=float(nv.script_mean.abs().max()))
lb = pd.read_csv(TAB / "listening_budget.csv").set_index("speaker")
numbers["listening_budget"] = {s: dict(seat=float(lb.loc[s, "seat_share"]), person=float(lb.loc[s, "person_share"]), turn=float(lb.loc[s, "turn_share"]),
                                       ratio=float(lb.loc[s, "seat_over_person"]), epr=float(lb.loc[s, "events_per_reply"])) for s in lb.index}
q8 = json.load(open(TAB / "reviewer_qs" / "q8_summary.json"))
numbers["testbed2_profiles"] = q8
q2 = pd.read_csv(TAB / "reviewer_qs" / "q2_lexical_labeller.csv")
lm = q2[q2.system != "gold"]
numbers["lexical_labeller"] = dict(models_min=float(lm.SCRIPT_lex.min()), models_max=float(lm.SCRIPT_lex.max()), z_min=float(lm.z_lex.min()),
                                   z_max=float(lm.z_lex.max()), gold=float(q2[q2.system == "gold"].SCRIPT_lex.iloc[0]),
                                   gold_z=float(q2[q2.system == "gold"].z_lex.iloc[0]),
                                   rho_with_tagger=float(spearmanr(lm.SCRIPT_lex, lm.SCRIPT_tagger)[0]))
mom = json.load(open(DERIVED / "figdata" / "mom.json"))
i = mom["codes"].index("DIR")
numbers["momentum_fig"] = dict(n_vin=mom["nvin"], p_dir_after_vin=mom["obs"][i], p_dir_predicted=mom["base"][i], M_excess=mom["M_excess"])

json.dump(numbers, open(LATEX / "numbers.json", "w"), indent=1, default=float)
print(f"wrote {LATEX / 'numbers.json'}")
