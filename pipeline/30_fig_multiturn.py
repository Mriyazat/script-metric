import importlib.util
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import EVENTS, FIG, TAB

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}
MODELS = ["Qwen", "Llama", "GPT", "Claude", "Gemini"]
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

GROUPS = {"emp_acc": ["VAC", "NAC", "ASAC", "SAC"],
          "emp_in": ["VIN", "NIN", "ASIN", "SIN"],
          "advice": ["DIR", "FIX", "RECT"],
          "quest": ["QOP", "QCL"],
          "other": ["TSH", "AUR", "LMT", "SEN", "MEN", "INC", "TEN"]}
C2G = {c: g for g, cs in GROUPS.items() for c in cs}
MT = ["carebench", "hope"]

# Conversation/Turn come straight from the downloaded corpora, through the
# same loader stage 08 uses — no side-car spans file needed.
_spec = importlib.util.spec_from_file_location(
    "mt_ext", pathlib.Path(__file__).with_name("08_multiturn_extension.py"))
_mt = importlib.util.module_from_spec(_spec)
_mt.__name__ = "mt_ext"
_spec.loader.exec_module(_mt)
E = _mt.load_events_with_turns()
E["group"] = E.label.map(C2G)
E["phase"] = pd.cut(E.turn, [0, 3, 7, 10],
                    labels=["turns 1–3", "4–7", "8–10"])

mt = pd.read_csv(TAB / "multiturn_extension.csv")
P = mt[(mt.granularity == "5-group") & (mt.corpus == "pooled")].set_index("model")

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5))

# ---- (A) questions share by phase, (B) advice share by phase --------------
shares = (E.groupby(["model", "phase"], observed=True).group
           .value_counts(normalize=True).rename("share").reset_index())
for ax, grp, title in [(axes[0], "quest", "questions collapse"),
                       (axes[1], "advice", "advice rises")]:
    for m in MODELS:
        d = shares[(shares.model == m) & (shares.group == grp)]
        ax.plot(range(3), d.share, "-o", ms=5, lw=1.8, color=MCOLORS[m],
                label=m)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["turns 1–3", "4–7", "8–10"], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, None)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("share of highlighted behaviour", fontsize=10)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

# ---- (C) boundary momentum: raw vs null (they coincide) -------------------
ax = axes[2]
for yi, m in enumerate(MODELS[::-1]):
    raw, nul = P.loc[m, "Mx_raw"], P.loc[m, "Mx_null"]
    ax.plot([nul, raw], [yi, yi], color="#cccccc", lw=2, zorder=1)
    ax.scatter([nul], [yi], s=55, color="#aaaaaa", zorder=3,
               edgecolor="k", linewidth=0.4,
               label="shuffle null" if yi == 0 else None)
    ax.scatter([raw], [yi], s=55, color=MCOLORS[m], zorder=3,
               edgecolor="k", linewidth=0.4)
    ax.text(-0.012, yi, m, ha="right", va="center", fontsize=9,
            color=MCOLORS[m], fontweight="bold")
ax.set_yticks([])
ax.set_xlim(-0.06, 0.16)
ax.set_xlabel("cross-turn transition MI / H(F)", fontsize=9)
ax.set_title("the boundary carries nothing\n(raw $=$ null $\\Rightarrow$ "
             "$M^{\\rightarrow}\\!\\approx 0$, all $|z| < 2$)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)

# ---- (D) conversation choreography C-> ------------------------------------
ax = axes[3]
vals = [P.loc[m, "Cx"] for m in MODELS]
zs = [P.loc[m, "z_Cx"] for m in MODELS]
ax.bar(MODELS, vals, color=[MCOLORS[m] for m in MODELS], width=0.62)
for i, (v, z) in enumerate(zip(vals, zs)):
    ax.text(i, v + 0.0002, f"z={z:.0f}", ha="center", fontsize=8,
            color="#555555")
ax.set_ylabel("$C^{\\rightarrow}$ (excess over null)", fontsize=10)
ax.set_title("turn-index choreography is real\nbut $\\approx$20$\\times$ "
             "weaker than reply-scale $C$", fontsize=10)
ax.tick_params(axis="x", labelsize=8.5, rotation=20)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

for ax, letter in zip(axes, "ABCD"):
    ax.text(-0.08, 1.16, letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold")

fig.suptitle("The script restarts every turn", fontsize=14, y=1.06)
fig.tight_layout()
fig.savefig(FIG / "fig_multiturn.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_multiturn.pdf", bbox_inches="tight")
print("saved", FIG / "fig_multiturn.png")
