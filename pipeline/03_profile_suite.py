import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

from paths import CORPORA, EVENTS, FIG, MODELS, REF, TAB

import script_metric as sm

ltc_seq = LinearSegmentedColormap.from_list("ltc_seq",
    ["#FDF6E3", "#E9D8A6", "#94D2BD", "#0A9396", "#005F73", "#001219"])
ltc_warm = LinearSegmentedColormap.from_list("ltc_warm",
    ["#FDF6E3", "#E9D8A6", "#EE9B00", "#CA6702", "#AE2012", "#9B2226"])
ltc_div = LinearSegmentedColormap.from_list("ltc_div",
    ["#0571b0", "#92c5de", "#f7f7f7", "#f4a582", "#ca0020"])

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}
CMARK = {"counselchat": "o", "pair": "s", "carebench": "^", "hope": "D"}
# Coarse behaviour groups used for display only (the metric itself always
# runs on the full 20-code alphabet). Note: the coding manual defines TEN and
# DIR as poles of one bipolar construct; here DIR is displayed with the
# advice-like codes and TEN with "other" — a display convention, not a claim.
GROUPS = {"empathy (acc)": ["VAC", "NAC", "ASAC", "SAC"],
          "empathy (inacc)": ["VIN", "NIN", "ASIN", "SIN"],
          "advice": ["DIR", "FIX", "RECT"],
          "questions": ["QOP", "QCL"],
          "other": ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}
CODE2G = {c: g for g, cs in GROUPS.items() for c in cs}
NB = 10

E = pd.read_csv(EVENTS)
E["response_id"] = E.corpus + "|" + E.row.astype(str) + "|" + E.model
E["group"] = E.label.map(CODE2G)
E = E.sort_values(["response_id", "position"]).reset_index(drop=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11,
                     "figure.facecolor": "white"})


def build_profile(df):
    _, prof = sm.compute(df[["response_id", "label", "position"]]
                         .sort_values(["response_id", "position"]).reset_index(drop=True),
                         n_bins=NB, n_shuffles=2, seed=0)
    return prof


# ---------------- profiles per (model, corpus) and pooled
prof_mc, prof_m = {}, {}
for m in MODELS:
    prof_m[m] = build_profile(E[E.model == m])
    for c in CORPORA:
        prof_mc[(m, c)] = build_profile(E[(E.model == m) & (E.corpus == c)])

# ---------------- Table: JS matrices per corpus + pooled
rows = []
for c in CORPORA + ["POOLED"]:
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            pa = prof_m[a] if c == "POOLED" else prof_mc[(a, c)]
            pb = prof_m[b] if c == "POOLED" else prof_mc[(b, c)]
            rows.append(dict(corpus=c, pair=f"{a}-{b}",
                             js=round(sm.profile_distance(pa, pb), 4)))
js_long = pd.DataFrame(rows)
js_long.to_csv(TAB / "js_by_corpus.csv", index=False)

# the pooled 5x5 matrix, as its own artefact (the identity figure reads it)
JS = pd.DataFrame(0.0, index=MODELS, columns=MODELS)
for _, r in js_long[js_long.corpus == "POOLED"].iterrows():
    a, b = r["pair"].split("-")
    JS.loc[a, b] = JS.loc[b, a] = r["js"]
JS.to_csv(TAB / "js_fingerprint_distance.csv")
JS.to_csv(REF / "js_fingerprint_distance.csv")

# ---------------- Table: the profile read in words (paper's fingerprint table)
# "questions" here includes hedging (QOP + QCL + TEN), which is why this table
# and the display groups above disagree about TEN on purpose.
PAPER_GROUPS = {**{c: "empathy" for c in ["VAC", "NAC", "ASAC", "SAC",
                                          "VIN", "NIN", "ASIN", "SIN"]},
                **{c: "advice" for c in ["DIR", "FIX", "RECT"]},
                **{c: "questions" for c in ["QOP", "QCL", "TEN"]}}
pr_rows = []
for m in MODELS:
    d = E[E.model == m].copy()
    d["pg"] = d.label.map(PAPER_GROUPS)
    rec = {"model": m}
    for g in ["empathy", "advice", "questions"]:
        s = d[d.pg == g]
        rec[f"{g}_pos"] = round(float(s.position.mean()), 3) if len(s) else np.nan
        rec[f"{g}_share"] = round(len(s) / len(d), 3)
    others = {o: sm.profile_distance(prof_m[m], prof_m[o])
              for o in MODELS if o != m}
    nearest = min(others, key=others.get)
    rec["js_min"] = round(others[nearest], 4)
    rec["js_max"] = round(max(others.values()), 4)
    rec["nearest_model"] = nearest
    pr_rows.append(rec)
PR = pd.DataFrame(pr_rows)
PR.to_csv(TAB / "profile_reading.csv", index=False)
PR.to_csv(REF / "profile_reading.csv", index=False)
print("\n== profile geometry (paper's fingerprint table, left half) ==")
print(PR.to_string(index=False))

# closest-pair summary per corpus
summ = []
for c in CORPORA + ["POOLED"]:
    d = js_long[js_long.corpus == c].set_index("pair").js
    llama_mean = d[[p for p in d.index if "Llama" in p]].mean()
    others = d[[p for p in d.index if "Llama" not in p]]
    summ.append(dict(corpus=c,
                     qwen_gemini=d["Qwen-Gemini"],
                     next_closest_pair=others.drop("Qwen-Gemini").idxmin(),
                     next_closest_js=others.drop("Qwen-Gemini").min(),
                     llama_mean_dist=round(llama_mean, 4),
                     twin_rank=int(d.rank()["Qwen-Gemini"]),
                     ratio_next_over_twin=round(
                         others.drop("Qwen-Gemini").min() / d["Qwen-Gemini"], 1)))
pd.DataFrame(summ).to_csv(TAB / "twin_structure_by_corpus.csv", index=False)
print(pd.DataFrame(summ))

# ---------------- Table: leave-one-corpus-out identification
rows = []
for held in CORPORA:
    enroll = {m: build_profile(E[(E.model == m) & (E.corpus != held)]) for m in MODELS}
    for true_m in MODELS:
        probe = E[(E.model == true_m) & (E.corpus == held)][["response_id", "label", "position"]]
        probe = probe.sort_values(["response_id", "position"]).reset_index(drop=True)
        ll = {m: sm.profile_loglik(probe, enroll[m]) for m in MODELS}
        rk = sorted(ll.items(), key=lambda kv: -kv[1])
        pp = build_profile(E[(E.model == true_m) & (E.corpus == held)])
        js = {m: sm.profile_distance(pp, enroll[m]) for m in MODELS}
        jr = sorted(js.items(), key=lambda kv: kv[1])
        rows.append(dict(held_out=held, true_model=true_m,
                         loglik_pick=rk[0][0], loglik_correct=rk[0][0] == true_m,
                         loglik_margin=round(rk[0][1] - rk[1][1], 4),
                         js_pick=jr[0][0], js_correct=jr[0][0] == true_m,
                         js_margin=round(jr[1][1] - jr[0][1], 4)))
ID = pd.DataFrame(rows)
ID.to_csv(TAB / "identification_loco.csv", index=False)
print(ID.groupby("held_out")[["loglik_correct", "js_correct"]].sum())

# ---------------- FIG: profile atlas — P(bin | code) heatmaps
TOPC = ["SIN", "VIN", "NIN", "VAC", "SAC", "DIR", "FIX", "RECT", "TSH", "QOP", "QCL", "TEN"]
fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.6), sharey=True)
for k, (m, ax) in enumerate(zip(MODELS, axes)):
    d = E[E.model == m]
    Mx = np.zeros((len(TOPC), NB))
    for i, code in enumerate(TOPC):
        x = d[d.label == code].position
        h, _ = np.histogram(x, bins=np.linspace(0, 1, NB + 1))
        Mx[i] = h / h.sum() if h.sum() > 0 else 0
    im = ax.imshow(Mx, cmap=ltc_seq, aspect="auto", vmin=0, vmax=0.35)
    ax.set_title(m, color=MCOLORS[m], fontweight="bold", fontsize=12)
    ax.set_xticks([0, 4.5, 9])
    ax.set_xticklabels(["start", "mid", "end"], fontsize=8)
    if k == 0:
        ax.set_yticks(range(len(TOPC)))
        ax.set_yticklabels(TOPC, fontsize=8.5)
        ax.set_ylabel("behaviour code")
    for y in [4.5, 8.5]:
        ax.axhline(y, color="w", lw=1.4)
fig.suptitle("The script written down — where each behaviour code sits in the reply\n"
             "P(position decile | code); rows: empathy block / advice block / structure block",
             fontsize=13, y=1.04)
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
cb.set_label("share of code's spans", fontsize=9)
fig.savefig(FIG / "fig_profile_atlas.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_profile_atlas.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- FIG: transition lift matrices (group level)
GL = ["empathy (acc)", "empathy (inacc)", "advice", "questions", "other"]
GS = ["emp+", "emp−", "adv", "q", "oth"]
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.7))
for k, (m, ax) in enumerate(zip(MODELS, axes)):
    d = E[E.model == m]
    same = d.response_id.values[1:] == d.response_id.values[:-1]
    prev = d.group.values[:-1][same]
    cur = d.group.values[1:][same]
    T = np.zeros((5, 5))
    for p, cu in zip(prev, cur):
        T[GL.index(p), GL.index(cu)] += 1
    pcur = T.sum(0) / T.sum()
    lift = (T / T.sum(1, keepdims=True)) / pcur[None, :]
    im = ax.imshow(lift, cmap=ltc_div, vmin=0, vmax=2.6)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{lift[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="w" if abs(lift[i, j] - 1.3) > 0.9 else "k")
    ax.set_title(m, color=MCOLORS[m], fontweight="bold", fontsize=12)
    ax.set_xticks(range(5))
    ax.set_xticklabels(GS, fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels(GS if k == 0 else [], fontsize=8)
    if k == 0:
        ax.set_ylabel("previous move")
fig.suptitle("What follows what — transition lift  P(current | previous) / P(current)\n"
             ">1 (red): the move chains after the previous one; <1 (blue): avoided",
             fontsize=13, y=1.10)
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
cb.set_label("lift", fontsize=9)
fig.savefig(FIG / "fig_transition_lift.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_transition_lift.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- FIG: MDS map of all (model x corpus) profiles
keys = [(m, c) for m in MODELS for c in CORPORA]
n = len(keys)
D2 = np.zeros((n, n))
for i, ka in enumerate(keys):
    for j, kb in enumerate(keys):
        if j <= i:
            continue
        d_ = sm.profile_distance(prof_mc[ka], prof_mc[kb])
        D2[i, j] = D2[j, i] = d_ ** 2
J = np.eye(n) - np.ones((n, n)) / n
Bm = -0.5 * J @ D2 @ J
w, V = np.linalg.eigh(Bm)
idx = np.argsort(w)[::-1]
X = V[:, idx[:2]] * np.sqrt(np.maximum(w[idx[:2]], 0))
var = w[idx[:2]] / w[w > 0].sum() * 100
fig, ax = plt.subplots(figsize=(8.6, 6.8))
for i, (m, c) in enumerate(keys):
    ax.scatter(X[i, 0], X[i, 1], s=150, color=MCOLORS[m], marker=CMARK[c],
               edgecolor="k", linewidth=0.6, zorder=3, alpha=0.9)
lab_off = {"Claude": (0, 26), "Llama": (30, 10), "Qwen": (34, 30),
           "Gemini": (-66, -30), "GPT": (58, 4)}
for m in MODELS:
    pts = np.array([X[i] for i, (mm, c) in enumerate(keys) if mm == m])
    cx, cy = pts.mean(0)
    ax.annotate(m, (cx, cy), fontsize=13, fontweight="bold", color=MCOLORS[m],
                ha="center", va="center",
                xytext=lab_off[m], textcoords="offset points")
    if len(pts) > 2:
        hull_order = np.argsort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        ax.add_patch(Polygon(pts[hull_order], closed=True, alpha=0.10,
                             facecolor=MCOLORS[m], edgecolor=MCOLORS[m], lw=1))
handles = [plt.Line2D([], [], marker=CMARK[c], color="k", ls="", ms=8, label=c)
           for c in CORPORA]
ax.legend(handles=handles, title="corpus", fontsize=9, frameon=False, loc="lower right")
ax.set_xlabel(f"MDS dim 1  ({var[0]:.0f}% of JS variance)")
ax.set_ylabel(f"MDS dim 2  ({var[1]:.0f}% of JS variance)")
ax.set_title("Behavioural map — every (model × corpus) profile, embedded by JS distance\n"
             "profiles cluster by MODEL, not by dataset: the fingerprint is the model's",
             fontsize=12)
fig.savefig(FIG / "fig_profile_map.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_profile_map.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- FIG: model fingerprint cards (radar)
feats = {}
for m in MODELS:
    d = E[E.model == m]
    shares = d.group.value_counts(normalize=True)
    q = d[d.group == "questions"]
    adv = d[d.group == "advice"]
    others_js = [sm.profile_distance(prof_m[m], prof_m[o]) for o in MODELS if o != m]
    feats[m] = {
        "advice\nshare": shares.get("advice", 0),
        "question\nshare": shares.get("questions", 0),
        "empathy\nshare": shares.get("empathy (acc)", 0) + shares.get("empathy (inacc)", 0),
        "empathy\naccuracy": (shares.get("empathy (acc)", 0) /
                              max(shares.get("empathy (acc)", 0)
                                  + shares.get("empathy (inacc)", 0), 1e-9)),
        "questions\nparked late": q.position.mean() if len(q) else 0,
        "advice\nstarts early": 1 - adv.position.mean() if len(adv) else 0,
        "distinct-\niveness": float(np.mean(others_js)),
    }
F = pd.DataFrame(feats).T
Fn = (F - F.min()) / (F.max() - F.min())    # scaled across models for display
angles = np.linspace(0, 2 * np.pi, len(F.columns), endpoint=False).tolist()
fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), subplot_kw=dict(polar=True))
for m, ax in zip(MODELS, axes):
    vals = Fn.loc[m].tolist()
    ax.plot(angles + [angles[0]], vals + [vals[0]], color=MCOLORS[m], lw=2)
    ax.fill(angles + [angles[0]], vals + [vals[0]], color=MCOLORS[m], alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(F.columns, fontsize=6.6)
    ax.set_yticks([0.5])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.set_title(m, color=MCOLORS[m], fontweight="bold", fontsize=12, pad=14)
fig.suptitle("Fingerprint cards — seven profile coordinates, scaled across models\n"
             "(raw values in the profile-geometry table; 'distinctiveness' = mean JS distance to the other four)",
             fontsize=12.5, y=1.12)
fig.savefig(FIG / "fig_model_cards.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_model_cards.pdf", bbox_inches="tight")
plt.close(fig)
F.round(3).to_csv(TAB / "fingerprint_card_values.csv")

# ---------------- FIG: identity panels (JS heatmap / geometry / identification)
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), gridspec_kw=dict(wspace=0.5))
# D: JS heatmap — the pooled matrix computed at the top of this file
js = pd.read_csv(REF / "js_fingerprint_distance.csv", index_col=0)
ordm = ["Qwen", "Gemini", "GPT", "Claude", "Llama"]
js = js.loc[ordm, ordm]
ax = axes[0]
im = ax.imshow(js.values, cmap=ltc_warm, vmin=0, vmax=0.18)
ax.set_xticks(range(5))
ax.set_xticklabels(ordm, fontsize=8.5, rotation=45)
ax.set_yticks(range(5))
ax.set_yticklabels(ordm, fontsize=8.5)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{js.values[i, j]:.3f}", ha="center", va="center",
                fontsize=7.2, color="k" if js.values[i, j] < 0.12 else "w")
ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, edgecolor="#00798c", lw=2.2))
ax.set_title("D — JS distance between profiles\ntwins boxed; Llama the outlier", fontsize=10.5)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=7)
# E: two profile coordinates, from the profile geometry computed above
pr = pd.read_csv(REF / "profile_reading.csv")
ax = axes[1]
offsets = {"Qwen": (0, -16), "Gemini": (14, 8), "GPT": (-16, -16),
           "Claude": (-4, 12), "Llama": (4, 12)}
for _, r in pr.iterrows():
    ax.scatter(r.questions_pos, r.questions_share, s=190, color=MCOLORS[r.model],
               edgecolor="k", linewidth=0.7, zorder=3)
    ax.annotate(r.model, (r.questions_pos, r.questions_share),
                textcoords="offset points", xytext=offsets[r.model],
                ha="center", fontsize=9, fontweight="bold", color=MCOLORS[r.model])
ax.set_xlabel("mean position of question spans", fontsize=9)
ax.set_ylabel("share of behaviour that is questions", fontsize=9)
ax.set_title("E — Two profile coordinates separate\ninterviewers from answer-givers", fontsize=10.5)
ax.set_xlim(0.44, 0.72)
ax.set_ylim(-0.02, 0.30)
ax.grid(alpha=0.25, lw=0.5)
# F: identification vs probe size. Values are precomputed by the probe-size
# identification experiment reported in the paper (rank-1 accuracy of
# nearest-profile matching under four generalisation settings).
ax = axes[2]
IDC = pd.read_csv(REF / "identification_curves.csv")
k = sorted(IDC.k.unique())
curves = {s: g.sort_values("k").accuracy.tolist()
          for s, g in IDC.groupby("setting")}
styles = {"within corpus": ("#d1495b", "-"), "held-out corpus": ("#E5A11F", "-"),
          "held-out annotator": ("#00798c", "-"), "both new": ("#66a182", "-"),
          "permutation null": ("#888888", "--")}
for lab, ys in curves.items():
    col, ls = styles[lab]
    ax.plot(k, ys, ls, marker="o", ms=4, lw=1.8, color=col, label=lab)
    ax.annotate(lab, (20, ys[-1]), xytext=(5, 0), textcoords="offset points",
                fontsize=7.4, color=col, va="center")
ax.axhline(0.2, color="k", lw=0.7, ls=":")
ax.annotate("chance", (1, 0.2), xytext=(0, -11), textcoords="offset points", fontsize=7.4)
ax.set_xticks(k)
ax.set_xlabel("probe size k (annotated replies)", fontsize=9)
ax.set_ylabel("rank-1 identification", fontsize=9)
ax.set_xlim(0, 30.5)
ax.set_ylim(0.1, 0.95)
ax.set_title("F — Identification from k replies;\nlabel-permutation null sits at chance",
             fontsize=10.5)
ax.grid(alpha=0.25, lw=0.5)
fig.savefig(FIG / "fig_profile_identity.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_profile_identity.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------- FIG: position-density timelines per behaviour group
bins = np.linspace(0, 1, 21)
gcolors = {"empathy": "#00798c", "advice": "#d1495b", "questions": "#66a182"}
G3 = {"empathy": ["VAC", "NAC", "ASAC", "SAC", "VIN", "NIN", "ASIN", "SIN"],
      "advice": ["DIR", "FIX", "RECT"], "questions": ["QOP", "QCL"]}
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.2), sharey=True)
for k, (m, ax) in enumerate(zip(MODELS, axes)):
    d = E[E.model == m]
    for g, codes in G3.items():
        x = d[d.label.isin(codes)].position
        h, edges = np.histogram(x, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        ax.plot(centers, hs, color=gcolors[g], lw=2, label=g if k == 0 else None)
        ax.fill_between(centers, hs, alpha=0.13, color=gcolors[g])
    ax.set_title(m, fontsize=11.5, color=MCOLORS[m], fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.3)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["start", "mid", "end"], fontsize=8)
    if k == 0:
        ax.set_ylabel("span density")
        ax.legend(fontsize=8, frameon=False)
fig.suptitle("One script, five accents — position density of the three headline groups",
             fontsize=13, y=1.05)
fig.savefig(FIG / "fig_profile_timelines.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_profile_timelines.pdf", bbox_inches="tight")
plt.close(fig)

print("done")
