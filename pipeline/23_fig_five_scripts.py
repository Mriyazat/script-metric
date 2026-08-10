import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIG, REF

MCOLORS = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
           "Claude": "#d1495b", "Gemini": "#00798c"}
gcolors = {"empathy": "#00798c", "advice": "#d1495b", "questions": "#66a182"}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

pr = pd.read_csv(REF / "profile_reading.csv").set_index("model")
val = pd.read_csv(REF / "validation_results.csv").set_index("system")
order = ["Qwen", "Gemini", "GPT", "Claude", "Llama"]
sigs = {"Qwen": "twin of Gemini  (JS = .016)",
        "Gemini": "twin of Qwen  (JS = .016)",
        "GPT": "half of everything is advice",
        "Claude": "questions parked latest (.64)",
        "Llama": "far from everyone  (JS ≥ .157)"}

fig, ax = plt.subplots(figsize=(11.5, 5.2))
SCALE = 9000
for yi, m in enumerate(order[::-1]):
    r = pr.loc[m]
    ax.hlines(yi, 0, 1, color="#d8d8d8", lw=2, zorder=1)
    for g, pos_c, sh_c in [("empathy", "empathy_pos", "empathy_share"),
                           ("advice", "advice_pos", "advice_share"),
                           ("questions", "questions_pos", "questions_share")]:
        ax.scatter(r[pos_c], yi, s=r[sh_c] * SCALE, color=gcolors[g],
                   alpha=0.85, edgecolor="white", linewidth=1.6, zorder=3)
    v = val.loc[m]
    ax.text(-0.04, yi, m, ha="right", va="center", fontsize=14,
            fontweight="bold", color=MCOLORS[m])
    ax.text(-0.04, yi - 0.24, f"SCRIPT {v.SCRIPT:.3f}  (z = {v.z:.0f})",
            ha="right", va="center", fontsize=8, color="#777777")
    ax.text(1.03, yi, sigs[m], ha="left", va="center", fontsize=10,
            color="#555555", style="italic")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.65, len(order) - 0.2)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("mean position in reply (0 = start, 1 = end)      ·      "
              "marker area = share of annotated behaviour", fontsize=11)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
handles = [plt.scatter([], [], s=180, color=gcolors[g], edgecolor="white",
                       linewidth=1.2, label=g) for g in gcolors]
ax.legend(handles=handles, ncol=3, frameon=False, fontsize=11,
          loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.set_title("Five SCRIPT profiles — same score, five different scripts",
             fontsize=15, pad=34)
fig.savefig(FIG / "fig_five_scripts.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_five_scripts.pdf", bbox_inches="tight")
print("saved")
