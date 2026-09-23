"""Shared visual language of the figures: behaviour-group and model colours, sequential
colour maps, panel tags, and the coloured-text reply renderer."""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GCOL = {"empathy": "#00798c", "advice": "#d1495b",
        "questions": "#66a182", "other": "#9aa0a6"}
GLAB = {"empathy": "Empathy", "advice": "Advice",
        "questions": "Questions", "other": "Other"}
MCOL = {"Qwen": "#5B5F8D", "Llama": "#E5A11F", "GPT": "#66a182",
        "Claude": "#d1495b", "Gemini": "#00798c"}

INK = "#26323a"          # near-black used for text and emphasis
MUTE = "#7c8790"         # secondary text
FAINT = "#e8ebee"        # tracks, separators

ltc_seq = LinearSegmentedColormap.from_list("ltc_seq",
    ["#FDF6E3", "#E9D8A6", "#94D2BD", "#0A9396", "#005F73", "#001219"])
ltc_warm = LinearSegmentedColormap.from_list("ltc_warm",
    ["#FDF6E3", "#E9D8A6", "#EE9B00", "#CA6702", "#AE2012", "#9B2226"])


def setup():
    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.edgecolor": "#c8cdd2",
        "axes.linewidth": 0.8,
        "xtick.color": "#5c666e",
        "ytick.color": "#5c666e",
        "text.color": INK,
        "axes.labelcolor": "#5c666e",
    })


def tag(ax, letter, x=-0.01, y=1.02):
    """Panel letter at the top-left of the axes. The paper's figures do not
    draw panel letters: panels are described positionally in the captions."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left", color=INK)


def despine(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _mono_char_width(ax, fontsize):
    """Measured advance of one DejaVu Sans Mono character, in axes fraction,
    so consecutive coloured runs join seamlessly."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    t = ax.text(0, 0, "M" * 20, transform=ax.transAxes, fontsize=fontsize,
                family="DejaVu Sans Mono")
    w = t.get_window_extent(r).width / 20
    t.remove()
    return w / ax.get_window_extent(r).width


def draw_reply(ax, txt, L, spans, prio, wrap=98, x0=0.006, y0=0.895,
               dy=0.098, fontsize=9.7, mute="#b3bac0"):
    """Render an annotated reply as coloured text: characters inside a span
    take the span's group colour (bold), everything un-annotated stays faint
    grey. Returns ((char_start, char_end, y) per wrapped line, char_width)."""
    cw = _mono_char_width(ax, fontsize)
    char_g = [None] * L
    for s in spans:
        g = s["group"]
        for i in range(s["start"], min(s["end"], L)):
            if char_g[i] is None or prio[g] > prio[char_g[i]]:
                char_g[i] = g
    # wrap on spaces; txt is single-spaced so each line is txt[a:b]
    lines, a = [], 0
    while a < L:
        b = min(a + wrap, L)
        if b < L:
            sp = txt.rfind(" ", a, b + 1)
            b = sp if sp > a else b
        lines.append((a, b))
        a = b + 1
    out = []
    for li, (a, b) in enumerate(lines):
        y = y0 - li * dy
        c = a
        while c < b:
            g = char_g[c]
            c2 = c
            while c2 < b and char_g[c2] == g:
                c2 += 1
            ax.text(x0 + (c - a) * cw, y, txt[c:c2], transform=ax.transAxes,
                    fontsize=fontsize, family="DejaVu Sans Mono", va="center",
                    fontweight="normal" if g is None else "bold",
                    color=(mute if g is None
                           else "#8d96a3" if g == "other" else GCOL[g]),
                    zorder=2)
            c = c2
        out.append((a, b, y))
    return out, cw


def flow(fig, y, text, x=0.5):
    """A small downward arrow with a verb between pipeline stages."""
    fig.text(x - 0.012, y, "\u2193", fontsize=15, color=MUTE,
             ha="right", va="center", fontweight="bold")
    fig.text(x, y, text, fontsize=9.3, color=MUTE, ha="left",
             va="center", style="italic")
