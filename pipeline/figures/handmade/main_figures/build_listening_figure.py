"""
Figure 2: where each behaviour sits, turn by turn, and what moves the behaviour mix.

    a  per speaker and stage of the conversation (turns 1-3, 4-7, 8-10): where each of the twenty
       codes occurs in the response (five position bins), shaded per code in the code's colour
    b  shift of the behaviour mix with the affective intensity of the user's message (hollow)
       and with an explicit safety cue (filled), excess Jensen-Shannon distance over the re-dealing null

Blind LLM layer, multi-turn corpora; reads the pipeline's out/ (run pipeline.external.listening first).

python build_listening_figure.py   # writes out/figures/script_listening_figure.png
"""
import numpy as np
import pandas as pd

from common import FAMILY, FAMILY_COLOR, HAIR, INK, LABELS, card, color, page, render, svg
from pipeline.common.paths import DERIVED, MODELS, TAB          # importable once common is loaded
from pipeline.external.listening import prep, turn_meta

MCOL = {'Qwen': '#5B5F8D', 'Llama': '#E5A11F', 'GPT': '#66a182', 'Claude': '#d1495b', 'Gemini': '#00798c'}
SPEAKERS = ['Human'] + MODELS
NAME = {'Human': 'Therapist', **{m: m for m in MODELS}}
STAGES = ['T1-3', 'T4-7', 'T8-10']
HEAT = {**FAMILY_COLOR, 'other': '#5E6873'}

# shared vertical grid of the rows in b
HEAD, ROW, GAP, SEP = 26, 36, 9, 14


def row_y(r):
    return HEAD + r * (ROW + GAP) + (SEP if r else 0)


BOTTOM = row_y(len(SPEAKERS) - 1) + ROW


def load():
    L = prep(pd.read_csv(DERIVED / 'llm_span_events.csv'), turn_meta())
    L['xb5'] = np.minimum((L.position * 5).astype(int), 4)
    staged = {}
    for sp in SPEAKERS:
        blocks = []
        for tb in STAGES:
            E = L[(L.model == sp) & (L.tb == tb)]
            t = pd.crosstab(E.label, E.xb5).reindex(index=LABELS, columns=range(5)).fillna(0).to_numpy()
            blocks.append(t / max(len(E), 1))       # share of the stage's spans
        staged[sp] = np.hstack(blocks)              # 20 codes x (3 stages x 5 bins)
    us = pd.read_csv(TAB / 'listening_user_state.csv')
    us = us[us.alphabet == '20-code'].pivot_table(index='cue', columns='speaker', values='excess')
    cues = {sp: (float(us.loc['affective intensity', sp]), float(us.loc['safety cue', sp])) for sp in SPEAKERS}
    return staged, cues


def tint(hex_color, v, mx):
    if v == 0:
        return '#F3F0EA'
    a = .07 + .93 * (v / mx) ** .8
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a:.3f})'


def panel_turns(staged, W=780):
    lab_w, gap, sgap, cell_h, head = 44, 16, 4, 11, 22
    bw = (W - lab_w - gap * (len(SPEAKERS) - 1)) / len(SPEAKERS)
    sw = (bw - 2 * sgap) / 3
    cell_w = sw / 5
    o = [f'<text x="{lab_w - 5}" y="{head + i * cell_h + 8.5}" class="rowl" fill="{color(l)}">{l}</text>'
         for i, l in enumerate(LABELS)]
    yb = head + 20 * cell_h
    for k, sp in enumerate(SPEAKERS):
        x = lab_w + k * (bw + gap)
        o.append(f'<text x="{x + bw / 2:.1f}" y="{head - 8}" class="spk" style="text-anchor:middle" '
                 f'fill="{INK if sp == "Human" else MCOL[sp]}">{NAME[sp]}</text>')
        for i, row in enumerate(staged[sp]):
            mx = max(row) or 1
            col = HEAT[FAMILY[LABELS[i]]]
            for s in range(3):
                for j in range(5):
                    o.append(f'<rect x="{x + s * (sw + sgap) + j * cell_w:.1f}" y="{head + i * cell_h}" '
                             f'width="{cell_w - .8:.1f}" height="{cell_h - 1}" fill="{tint(col, row[s * 5 + j], mx)}"/>')
        for s, lab in enumerate(['1–3', '4–7', '8–10']):
            o.append(f'<text x="{x + s * (sw + sgap) + sw / 2:.1f}" y="{yb + 13}" class="tick">{lab}</text>')
        if k == 0:
            o.append(f'<line x1="{x + bw + gap / 2:.1f}" y1="{head - 20}" x2="{x + bw + gap / 2:.1f}" y2="{yb}" stroke="{HAIR}"/>')
    o.append(f'<text x="{lab_w + (W - lab_w) / 2}" y="{yb + 31}" class="rk">turns of the conversation; within each, position in the response from start to end</text>')
    return svg(W, yb + 37, ''.join(o))


def panel_cues(cues, W=380):
    x0, x1, vmax = 84, W - 14, 0.19
    X = lambda v: x0 + (x1 - x0) * v / vmax
    o = [f'<circle cx="{x0 + 5}" cy="{HEAD - 14}" r="4.5" fill="#fff" stroke="{INK}" stroke-width="1.6"/>'
         f'<text x="{x0 + 15}" y="{HEAD - 10}" class="rk" style="text-anchor:start">felt: affective intensity</text>'
         f'<circle cx="{X(0.112):.1f}" cy="{HEAD - 14}" r="4.5" fill="{INK}"/>'
         f'<text x="{X(0.112) + 10:.1f}" y="{HEAD - 10}" class="rk" style="text-anchor:start">stated: safety cue</text>']
    for v in (0, 0.05, 0.10, 0.15):
        o.append(f'<line x1="{X(v):.1f}" y1="{HEAD}" x2="{X(v):.1f}" y2="{BOTTOM}" stroke="{HAIR if v else "#CFCABF"}"/>'
                 f'<text x="{X(v):.1f}" y="{BOTTOM + 15}" class="tick">{v:.2f}</text>')
    for r, sp in enumerate(SPEAKERS):
        y = row_y(r) + ROW / 2
        felt, stated = cues[sp]
        col = INK if sp == 'Human' else MCOL[sp]
        o.append(f'<text x="{x0 - 12}" y="{y + 4.5}" class="spk" fill="{col}">{NAME[sp]}</text>'
                 f'<line x1="{X(felt):.1f}" y1="{y}" x2="{X(stated):.1f}" y2="{y}" stroke="{col}" stroke-opacity=".4" '
                 f'stroke-width="3.2" stroke-linecap="round"/>'
                 f'<circle cx="{X(felt):.1f}" cy="{y}" r="5.5" fill="#fff" stroke="{col}" stroke-width="2"/>'
                 f'<circle cx="{X(stated):.1f}" cy="{y}" r="5.5" fill="{col}"/>')
    ys = row_y(0) + ROW + (SEP + GAP) / 2
    o.append(f'<line x1="0" y1="{ys}" x2="{W}" y2="{ys}" stroke="{HAIR}"/>')
    o.append(f'<text x="{(x0 + x1) / 2}" y="{BOTTOM + 32}" class="rk">shift of the behaviour mix with the cue</text>')
    return svg(W, BOTTOM + 38, ''.join(o))


CSS = """
.spk{font:700 12.5px Heros;text-anchor:end}
.tick{font-size:10.5px}
.panels{grid-template-columns:812px 1fr}
"""


def build():
    staged, cues = load()
    body = f"""<div class="row panels">
{card('a', 'Where each behaviour sits, turn by turn', panel_turns(staged), body_style='flex:1;display:flex;align-items:flex-start;justify-content:center')}
{card('b', 'Response to the user’s cues', panel_cues(cues), body_style='flex:1;display:flex;align-items:center;justify-content:center')}
</div>"""
    return page('Figure 2', CSS, body)


if __name__ == '__main__':
    render(build(), 'script_listening_figure')
