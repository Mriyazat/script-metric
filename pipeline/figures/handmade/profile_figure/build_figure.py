"""
Figure for Sec. 5.4: tied scores, five profiles, the script profile names the model.

Layout (two rows, read left to right, same visual language as the method figure):
    a  Four of five scores tie          ->  b  Five script profiles, one order
    c  Profile distance (JS)            ->  d  Naming the model from k responses  ->  e  Testbed 2: 22 close relatives

Every number is read from script-metric/out (the paper's pipeline output).
Each panel is one function (panel_a ... panel_e) that returns HTML/SVG.
"""
import base64
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent
OUT = PAPER / 'script-metric' / 'out'
METHOD = PAPER / 'figures' / 'script_method_figure'

NUM = json.loads((OUT / 'tables/latex/numbers.json').read_text())
PROFILES = json.loads((METHOD / 'data/profiles.json').read_text())      # P (20x10), T (20x20) per model
R = json.loads((METHOD / 'data/response.json').read_text())             # label order + family
LABELS, FAMILY = R['labels'], dict(zip(R['labels'], R['family']))


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# STYLE (shared with the method figure)

FAMILY_COLOR = {'empathy': '#3A78B5', 'advice': '#D1495B', 'questions': '#3E9A5E', 'other': '#8A949D'}
MODEL_COLOR = {'Qwen': '#6B5B95', 'Llama': '#E0A030', 'GPT': '#4E9C7C', 'Claude': '#D1495B', 'Gemini': '#1F8A9A'}
MODELS = ['Qwen', 'Gemini', 'GPT', 'Claude', 'Llama']     # twins first, Llama last
GOLD = '#E9A400'
INK = '#1E2A36'
MUTE = '#7E8B97'
HAIR = '#E6E2D9'
HEAT_RGB = (31, 59, 87)
PAGE_BG = '#F5F3EE'
M_COLOR = '#1C9A87'

TITLES = {
    'a': 'Four of five scores tie',
    'b': 'Five script profiles, one order',
    'c': 'Profile distance',
    'd': 'Naming the model from <i>k</i> responses',
    'e': 'Testbed 2: 22 close relatives',
}

SIGNATURE = {
    'Qwen': 'twin of Gemini',
    'Gemini': 'twin of Qwen',
    'GPT': 'half of everything is advice',
    'Claude': 'questions placed latest',
    'Llama': 'far from everyone',
}


def rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


def heat_fill(value, row_max):
    if value == 0:
        return '#F3F0EA'
    a = .07 + .93 * (value / row_max) ** .8
    return f'rgba({HEAT_RGB[0]},{HEAT_RGB[1]},{HEAT_RGB[2]},{a:.3f})'


def svg(w, h, body):
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'


def color(code):
    return FAMILY_COLOR[FAMILY[code]]


# a  four of five scores tie
def panel_a():
    W, H = 300, 236
    x0, x1 = 28, W - 16
    lo, hi = 0.06, 0.14
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    order = sorted(MODELS, key=lambda m: NUM['testbed1'][m]['SCRIPT'])
    row_h, top = 30, 26
    o = []
    # axis
    axis_y = top + len(order) * row_h + 6
    o.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="#BDB7AC" stroke-width="1.2"/>')
    for v in (0.06, 0.08, 0.10, 0.12, 0.14):
        o.append(f'<line x1="{X(v):.1f}" y1="{axis_y}" x2="{X(v):.1f}" y2="{axis_y + 4}" stroke="#BDB7AC"/>'
                 f'<text x="{X(v):.1f}" y="{axis_y + 16}" class="tick">{v:.2f}</text>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{axis_y + 31}" class="axl smallcaps">Script</text>')
    # bracket over the four tied models
    tied = [m for m in order if m != 'Claude']
    ys = [top + order.index(m) * row_h + row_h / 2 for m in tied]
    bx = x1 - 6
    o.append(f'<path d="M{bx - 5} {min(ys) - 8} H{bx} V{max(ys) + 8} H{bx - 5}" fill="none" stroke="{MUTE}" stroke-width="1.2"/>')
    for i, m in enumerate(order):
        y = top + i * row_h + row_h / 2
        v = NUM['testbed1'][m]['SCRIPT']
        b = NUM['bootstrap'][m]
        c = MODEL_COLOR[m]
        o.append(f'<line x1="{X(v - b["half_lo"]):.1f}" y1="{y}" x2="{X(v + b["half_hi"]):.1f}" y2="{y}" stroke="{c}" stroke-width="2.4" stroke-linecap="round"/>'
                 f'<circle cx="{X(v):.1f}" cy="{y}" r="5" fill="{c}"/>'
                 f'<text x="{x0 - 6}" y="{y + 4}" class="mlab" fill="{c}">{m}</text>'
                 f'<text x="{X(v):.1f}" y="{y - 9}" class="val">{v:.3f}</text>')
    note = (f'<div class="note">Pooled clinician layer, bootstrap 95% intervals. '
            f'Four scores overlap (unadjusted <i>p</i> &ge; 0.28); only Claude separates (Holm <i>p</i> = 0.025). '
            f'The score says <b>how</b> scripted, not <b>whose</b> script.</div>')
    return svg(W, H, ''.join(o)) + note


# b  five script profiles
def heat_block(M, cell, x=0, y=0):
    o = []
    for i, row in enumerate(M):
        mx = max(row) or 1
        for j, v in enumerate(row):
            o.append(f'<rect x="{x + j * cell:.2f}" y="{y + i * cell:.2f}" width="{cell - .5:.2f}" height="{cell - .5:.2f}" fill="{heat_fill(v, mx)}"/>')
    return ''.join(o)


def profile_reading():
    rows = read_csv(OUT / 'derived/profile_reading.csv')
    return {r['model']: r for r in rows}


def panel_b():
    PR = profile_reading()
    JS = js_matrix()
    cellP, cellT = 7.0, 3.5
    pw, ph = 10 * cellP, 20 * cellP           # 66 x 132
    tw, th = 20 * cellT, 20 * cellT           # 70 x 70
    card_w, gap = 150, 12
    label_w = 40
    top = 30
    W = label_w + 5 * card_w + 4 * gap
    H = top + ph + 14 + th + 92
    o = []
    # row labels once, coloured by family
    for i, l in enumerate(LABELS):
        o.append(f'<text x="{label_w - 6}" y="{top + i * cellP + 5.6:.1f}" class="rowl" fill="{color(l)}">{l}</text>')
    for k, m in enumerate(MODELS):
        x = label_w + k * (card_w + gap)
        c = MODEL_COLOR[m]
        o.append(f'<text x="{x + pw / 2:.1f}" y="{top - 12}" class="mname b" fill="{c}">{m}</text>')
        # P (position table) and T (transition table)
        o.append(heat_block(PROFILES[m]['P'], cellP, x, top))
        tx = x + pw + 8
        o.append(heat_block(PROFILES[m]['T'], cellT, tx, top + (ph - th) / 2))
        if k == 0:
            o.append(f'<text x="{x + pw / 2:.1f}" y="{top + ph + 11}" class="colh">P̂(ℓ | X)&nbsp; start → end</text>'
                     f'<text x="{tx + tw / 2:.1f}" y="{top + ph + 11}" class="colh">T̂(ℓ<tspan class="sub">prev</tspan>, ℓ)</text>')
        # geometry: mean position of the three headline groups, marker area = share
        gy = top + ph + 34
        gx0, gx1 = x + 4, x + card_w - 6
        o.append(f'<line x1="{gx0}" y1="{gy}" x2="{gx1}" y2="{gy}" stroke="#CFCABF"/>'
                 f'<text x="{gx0}" y="{gy + 14}" class="tick" text-anchor="start">start</text>'
                 f'<text x="{gx1}" y="{gy + 14}" class="tick" text-anchor="end">end</text>')
        for fam, key in (('advice', 'advice'), ('empathy', 'empathy'), ('questions', 'questions')):
            pos, share = float(PR[m][f'{key}_pos']), float(PR[m][f'{key}_share'])
            r = 3 + 22 * math.sqrt(share)
            o.append(f'<circle cx="{gx0 + (gx1 - gx0) * pos:.1f}" cy="{gy}" r="{r:.1f}" fill="{rgba(FAMILY_COLOR[fam], .78)}" stroke="#fff" stroke-width="1"/>')
        # signature and nearest neighbour
        near = min((JS[m][n], n) for n in MODELS if n != m)
        o.append(f'<text x="{x + card_w / 2:.1f}" y="{gy + 36}" class="sig">{SIGNATURE[m]}</text>'
                 f'<text x="{x + card_w / 2:.1f}" y="{gy + 50}" class="tick">nearest: {near[1]} (JS {near[0]:.3f})</text>')
    key = (f'<div class="key"><span class="sw" style="background:{FAMILY_COLOR["empathy"]}"></span>empathy'
           f'<span class="sw" style="background:{FAMILY_COLOR["advice"]}"></span>advice'
           f'<span class="sw" style="background:{FAMILY_COLOR["questions"]}"></span>questions'
           f'<span class="sw" style="background:{FAMILY_COLOR["other"]}"></span>other'
           f'<span style="margin-left:14px">tables pooled over each model’s ~800 responses; dots: mean position of a group, area = share of highlighted behaviour</span></div>')
    return svg(W, H, ''.join(o)) + key


# c  profile distance
def js_matrix():
    rows = read_csv(OUT / 'derived/js_fingerprint_distance.csv')
    key = list(rows[0].keys())[0]
    return {r[key]: {m: float(r[m]) for m in MODELS} for r in rows}


def panel_c():
    JS = js_matrix()
    cell, x0, y0 = 40, 52, 58
    W, H = x0 + 5 * cell + 6, y0 + 5 * cell + 6
    vmax = max(JS[a][b] for a in MODELS for b in MODELS)
    o = []
    for i, a in enumerate(MODELS):
        o.append(f'<text x="{x0 - 6}" y="{y0 + i * cell + cell / 2 + 4}" class="mlab" fill="{MODEL_COLOR[a]}">{a}</text>'
                 f'<text transform="translate({x0 + i * cell + cell / 2 + 4},{y0 - 6}) rotate(-90)" class="mlabr" fill="{MODEL_COLOR[a]}">{a}</text>')
        for j, b in enumerate(MODELS):
            v = JS[a][b]
            if i == j:
                o.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell - 2}" height="{cell - 2}" fill="#F3F0EA"/>')
                continue
            closeness = 1 - v / vmax
            o.append(f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell - 2}" height="{cell - 2}" '
                     f'fill="rgba({HEAT_RGB[0]},{HEAT_RGB[1]},{HEAT_RGB[2]},{.06 + .9 * closeness ** 1.6:.3f})"/>'
                     f'<text x="{x0 + j * cell + cell / 2 - 1}" y="{y0 + i * cell + cell / 2 + 4}" class="cellv" '
                     f'fill="{"#fff" if closeness > .55 else INK}">{v:.3f}</text>')
    # twins
    i, j = MODELS.index('Qwen'), MODELS.index('Gemini')
    for a, b in ((i, j), (j, i)):
        o.append(f'<rect x="{x0 + b * cell - 1.5}" y="{y0 + a * cell - 1.5}" width="{cell + 1}" height="{cell + 1}" rx="3" fill="none" stroke="{GOLD}" stroke-width="2.4"/>')
    note = (f'<div class="note">Jensen–Shannon distance between profiles (dark = close). '
            f'<span style="color:{GOLD};font-weight:700">Qwen–Gemini</span> sit at 0.019, half the next-closest pair (0.038), '
            f'and are the closest pair in every one of the four corpora; Llama is &ge; 0.170 from everyone.</div>')
    return svg(W, H, ''.join(o)) + note


# d  the identification game
def build_probe(model='Claude', held_out='carebench', k=5, seed=7):
    """One real round: k held-out responses of one model, scored against profiles enrolled on the other corpora."""
    rows = read_csv(OUT / 'derived/span_events.csv')
    by_resp = defaultdict(list)
    for r in rows:
        by_resp[(r['corpus'], r['row'], r['model'])].append((float(r['position']), r['label']))
    LI = {l: i for i, l in enumerate(LABELS)}

    def slots(events):
        ev = sorted(events)
        out, cur = [], None
        for x, l in ev:
            if cur is None or x > cur[0]:
                cur = (x, [l]); out.append(cur)
            else:
                cur[1].append(l)
        return out

    # enrol: counts over the three other corpora
    P = {m: [[0.0] * 10 for _ in LABELS] for m in MODELS}
    T = {m: [[0.0] * 20 for _ in LABELS] for m in MODELS}
    for (corpus, row, m), ev in by_resp.items():
        if corpus == held_out or m not in P:
            continue
        sl = slots(ev)
        for x, labs in sl:
            b = min(int(x * 10), 9)
            for l in labs:
                P[m][LI[l]][b] += 1
        for (xa, la), (xb, lb) in zip(sl, sl[1:]):
            T[m][LI[la[0]]][LI[lb[0]]] += 1

    def logP(m, l, b):
        row = P[m][LI[l]]
        return math.log((row[b] + .5) / (sum(row) + 5))

    def logT(m, lp, l):
        row = T[m][LI[lp]]
        return math.log((row[LI[l]] + .5) / (sum(row) + 10))

    rng = random.Random(seed)
    cands = [key for key in by_resp if key[0] == held_out and key[2] == model and len(by_resp[key]) >= 6]
    for _ in range(200):
        probe = rng.sample(cands, k)
        scores = {}
        for m in MODELS:
            tot, n = 0.0, 0
            for key in probe:
                sl = slots(by_resp[key])
                for idx, (x, labs) in enumerate(sl):
                    b = min(int(x * 10), 9)
                    for j, l in enumerate(labs):
                        tot += logP(m, l, b); n += 1
                        if j == 0 and idx > 0:
                            tot += logT(m, sl[idx - 1][1][0], l)
            scores[m] = tot / n
        if max(scores, key=scores.get) == model:
            break
    strips = [[(x, l) for x, l in sorted(by_resp[key])] for key in probe]
    return strips, scores


def panel_d():
    strips, scores = build_probe()
    curves = read_csv(OUT / 'tables/identification_curves.csv')
    W, H = 500, 262
    o = []
    # left: the probe
    px0, px1 = 14, 232
    o.append(f'<text x="{px0}" y="12" class="cap2">probe: 5 responses, model hidden</text>')
    for i, ev in enumerate(strips):
        y = 26 + i * 22
        o.append(f'<rect x="{px0}" y="{y}" width="{px1 - px0}" height="14" rx="3" fill="#EFECE5"/>')
        for x, l in ev:
            o.append(f'<rect x="{px0 + (px1 - px0) * x - 2.5:.1f}" y="{y + 2}" width="5" height="10" rx="1" fill="{color(l)}" opacity=".9"/>')
    o.append(f'<text x="{px0}" y="{26 + 5 * 22 + 8}" class="tick" text-anchor="start">start</text>'
             f'<text x="{px1}" y="{26 + 5 * 22 + 8}" class="tick" text-anchor="end">end</text>')
    # arrow down to the bars
    ay = 26 + 5 * 22 + 18
    o.append(f'<path d="M{(px0 + px1) / 2:.1f} {ay} v12" stroke="{INK}" stroke-width="2" fill="none"/>'
             f'<path d="M{(px0 + px1) / 2 - 4:.1f} {ay + 9} l4 6 l4 -6 Z" fill="{INK}"/>')
    # likelihood bars per enrolled profile
    best = max(scores.values())
    by = ay + 30
    bw, bgap = 40, 9
    bx0 = (px0 + px1) / 2 - (5 * bw + 4 * bgap) / 2
    o.append(f'<text x="{px0}" y="{by - 4}" class="cap2">likelihood under each enrolled profile</text>')
    for k, m in enumerate(sorted(MODELS, key=lambda m: MODEL_COLOR[m])):
        x = bx0 + k * (bw + bgap)
        rel = math.exp(scores[m] - best)
        h = 8 + 44 * rel
        win = scores[m] == best
        o.append(f'<rect x="{x}" y="{by + 62 - h:.1f}" width="{bw}" height="{h:.1f}" rx="3" fill="{MODEL_COLOR[m] if win else "#D9D4CA"}"/>'
                 f'<text x="{x + bw / 2}" y="{by + 76}" class="mname{" b" if win else ""}" fill="{MODEL_COLOR[m] if win else "#7E8B97"}">{m}</text>')
        if win:
            o.append(f'<text x="{x + bw / 2}" y="{by + 62 - h - 6:.1f}" class="mname b">m̂</text>')
    # right: accuracy vs k
    cx0, cx1, cy0, cy1 = 288, W - 16, 78, 212
    ks = [1, 5, 10, 20]
    KX = lambda k: cx0 + (cx1 - cx0) * ks.index(k) / (len(ks) - 1)
    KY = lambda a: cy1 - (cy1 - cy0) * a
    o.append(f'<text x="{cx0}" y="12" class="cap2">rank-1 accuracy</text>')
    for a in (0.2, 0.4, 0.6, 0.8):
        o.append(f'<line x1="{cx0}" y1="{KY(a):.1f}" x2="{cx1}" y2="{KY(a):.1f}" stroke="#ECE8E0"/>'
                 f'<text x="{cx0 - 6}" y="{KY(a) + 4:.1f}" class="tick" text-anchor="end">{a:.1f}</text>')
    o.append(f'<line x1="{cx0}" y1="{cy1}" x2="{cx1}" y2="{cy1}" stroke="#BDB7AC"/>')
    for k in ks:
        o.append(f'<text x="{KX(k):.1f}" y="{cy1 + 16}" class="tick">{k}</text>')
    o.append(f'<text x="{(cx0 + cx1) / 2:.1f}" y="{cy1 + 32}" class="axl"><tspan font-style="italic">k</tspan> annotated responses in the probe</text>')
    o.append(f'<line x1="{cx0}" y1="{KY(.2):.1f}" x2="{cx1}" y2="{KY(.2):.1f}" stroke="{MUTE}" stroke-dasharray="3 3"/>'
             f'<text x="{cx0 + 4}" y="{KY(.2) - 5:.1f}" class="tick" style="text-anchor:start">chance 0.20</text>')
    style = {
        'held-out corpus': (INK, 2.6, ''),
        'held-out annotator': (MUTE, 1.8, ''),
        'both new': (MUTE, 1.8, '5 3'),
        'permutation null': ('#B9B3A8', 1.6, '2 3'),
    }
    labels = {'held-out corpus': 'held-out corpus', 'held-out annotator': 'held-out annotator',
              'both new': 'corpus + annotator held out', 'permutation null': 'label permutation'}
    for setting, (col, wdt, dash) in style.items():
        pts = [(int(r['k']), float(r['accuracy'])) for r in curves if r['setting'] == setting]
        pts.sort()
        path = ' '.join(f'{KX(k):.1f},{KY(a):.1f}' for k, a in pts)
        o.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="{wdt}"{" stroke-dasharray=\"" + dash + "\"" if dash else ""}/>')
        for k, a in pts:
            o.append(f'<circle cx="{KX(k):.1f}" cy="{KY(a):.1f}" r="{3 if setting == "held-out corpus" else 2.2}" fill="{col}"/>')
        k, a = pts[-1]
        if setting in ('held-out corpus', 'permutation null'):
            o.append(f'<text x="{KX(k) + 6:.1f}" y="{KY(a) + 4:.1f}" class="cl" fill="{col}">{a:.2f}</text>')
        elif setting == 'both new':
            o.append(f'<text x="{KX(k) + 6:.1f}" y="{KY(a) + 4:.1f}" class="cl" fill="{col}">0.44–0.46</text>')
    # legend
    ly = 24
    for i, (setting, (col, wdt, dash)) in enumerate(style.items()):
        y = ly + i * 14
        o.append(f'<line x1="{cx0 + 6}" y1="{y}" x2="{cx0 + 26}" y2="{y}" stroke="{col}" stroke-width="{wdt}"{" stroke-dasharray=\"" + dash + "\"" if dash else ""}/>'
                 f'<text x="{cx0 + 31}" y="{y + 3.5}" class="lg" fill="{col if col == INK else MUTE}">{labels[setting]}</text>')
    note = (f'<div class="note">Left: one real round. Five held-out Claude responses, scored by mean per-event log-likelihood under each profile '
            f'enrolled on the other three corpora; the highest bar names the model. Right: over 4,000 rounds per point. '
            f'On a held-out corpus the profile names the model from 20 responses in 74% of rounds; every miss is a Qwen–Gemini confusion.</div>')
    return svg(W, H, ''.join(o)) + note


# e  Testbed 2: 22 close relatives
RECIPE = [('baseline1_vanilla', 'vanilla'), ('baseline2_tactic_prompt', 'tactic-list'), ('baseline3_tactic_history', 'tactic-history'),
          ('baseline4_vs_vanilla', 'VS vanilla'), ('baseline5_vs_tactic', 'VS tactic'), ('baseline6_vs_tactic_history', 'VS history'),
          ('baseline7_psychocounsel', 'quality RL'), ('baseline8_r1zerodiv', 'RL + token div.'),
          ('ours1_q_dkl', 'MINT KL'), ('ours2_q_h', 'MINT entropy'), ('ours3_q_dkl_h', 'MINT KL+ent.')]


def panel_e():
    rows = read_csv(OUT / 'tables/reviewer_qs/q8_js_matrix.csv')
    key = list(rows[0].keys())[0]
    JS = {r[key]: r for r in rows}
    order = [f'{rec}_Qwen3-{sz}' for rec, _ in RECIPE for sz in ('1.7B', '4B')]
    S = json.loads((OUT / 'tables/reviewer_qs/q8_summary.json').read_text())
    cell, x0, y0 = 9.6, 90, 24
    n = len(order)
    W, H = x0 + n * cell + 8, y0 + n * cell + 8
    vmax = max(float(JS[a][b]) for a in order for b in order)
    o = []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                o.append(f'<rect x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" width="{cell - .8}" height="{cell - .8}" fill="#F3F0EA"/>')
                continue
            v = float(JS[a][b]); closeness = 1 - v / vmax
            o.append(f'<rect x="{x0 + j * cell:.1f}" y="{y0 + i * cell:.1f}" width="{cell - .8}" height="{cell - .8}" '
                     f'fill="rgba({HEAT_RGB[0]},{HEAT_RGB[1]},{HEAT_RGB[2]},{.05 + .93 * closeness ** 2.2:.3f})"/>')
    # recipe labels and 2x2 same-recipe blocks
    for r, (rec, name) in enumerate(RECIPE):
        y = y0 + 2 * r * cell
        o.append(f'<text x="{x0 - 6}" y="{y + cell + 3.5:.1f}" class="rlab2">{name}</text>'
                 f'<rect x="{x0 + 2 * r * cell - .8:.1f}" y="{y - .8:.1f}" width="{2 * cell + .8:.1f}" height="{2 * cell + .8:.1f}" rx="2" fill="none" stroke="{GOLD}" stroke-width="1.8"/>')
    o.append(f'<text x="{x0 + n * cell / 2:.1f}" y="{y0 - 10}" class="colh">22 systems, ordered by recipe; each pair = 1.7B and 4B</text>')
    # family brackets: prompting vs RL
    bx = x0 + n * cell + 4
    o.append(f'<path d="M{bx} {y0} h4 V{y0 + 12 * cell} h-4" fill="none" stroke="{MUTE}" stroke-width="1.2"/>'
             f'<text transform="translate({bx + 14},{y0 + 6 * cell}) rotate(90)" class="tick">prompting</text>'
             f'<path d="M{bx} {y0 + 12 * cell + 2} h4 V{y0 + 22 * cell} h-4" fill="none" stroke="{MUTE}" stroke-width="1.2"/>'
             f'<text transform="translate({bx + 14},{y0 + 17 * cell}) rotate(90)" class="tick">RL</text>')
    W += 22
    note = (f'<div class="note">Same recipe at the other size (<span style="color:{GOLD};font-weight:700">gold blocks</span>): mean JS '
            f'<b>{S["js_same_recipe_other_size"]:.3f}</b>. Different recipe at the same size: <b>{S["js_same_size_other_recipe"]:.3f}</b>. '
            f'A 40-turn probe picks its own system in <b>{S["probe_self"]}/22</b> (chance 1/22), its own recipe in {S["probe_recipe"]}/22, '
            f'its own size in {S["probe_size"]}/22: the profile tracks the training objective, not the parameter count.</div>')
    return svg(W, H, ''.join(o)) + note


# page
def font_face(file, family, weight, style='normal'):
    data = base64.b64encode((METHOD / 'fonts' / f'{file}.woff').read_bytes()).decode()
    return (f"@font-face{{font-family:'{family}';src:url(data:font/woff;base64,{data}) format('woff');"
            f"font-weight:{weight};font-style:{style}}}")


FONTS = ''.join([
    font_face('texgyreheros-regular', 'Heros', 400),
    font_face('texgyreheros-bold', 'Heros', 700),
    font_face('texgyreheroscn-bold', 'HerosCn', 700),
    font_face('texgyretermes-regular', 'Termes', 400),
    font_face('texgyretermes-italic', 'Termes', 400, 'italic'),
])

CSS = f"""
{FONTS}
:root{{--ink:{INK};--mute:{MUTE};--hair:{HAIR};--gold:{GOLD}}}
*{{box-sizing:border-box}}
body{{margin:0;background:{PAGE_BG};font-family:Heros,Helvetica,Arial,sans-serif;color:var(--ink)}}
#fig{{position:relative;width:1280px;padding:20px;background:{PAGE_BG};display:flex;flex-direction:column;gap:34px}}
.row{{display:grid;gap:34px}}
.r1{{grid-template-columns:336px 1fr}}
.r2{{grid-template-columns:290px 1fr 380px}}
.card{{background:#fff;border:1px solid var(--hair);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;min-width:0}}
svg{{display:block;overflow:visible;max-width:100%;height:auto}}
#arrows{{position:absolute;left:0;top:0;pointer-events:none}}
.ph{{display:flex;align-items:center;gap:9px;margin-bottom:12px}}
.lt{{width:21px;height:21px;border-radius:50%;background:var(--ink);color:#fff;font:700 12px/21px Heros;text-align:center}}
.tt{{font:700 14.5px/1 Heros}}
.io{{margin-left:auto;font:700 9.5px/1 Heros;letter-spacing:.14em;text-transform:uppercase;color:#fff;background:{M_COLOR};padding:4px 7px;border-radius:20px}}
.note{{font:11.2px/1.45 Heros;color:#56636F;margin-top:auto;padding-top:10px}}
.note b{{color:var(--ink)}}
.key{{display:flex;align-items:center;gap:6px;font:10.5px/1.3 Heros;color:var(--mute);margin-top:8px;flex-wrap:wrap}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-left:8px}}
.axl{{font:italic 12px Termes;fill:var(--ink);text-anchor:middle}}
.smallcaps{{font-style:normal;font-variant:small-caps;font-size:13px;letter-spacing:.04em}}
.tick{{font:9.5px Heros;fill:var(--mute);text-anchor:middle}}
.colh{{font:9.5px Heros;fill:var(--mute);text-anchor:middle}}
.rowl{{font:700 6.6px HerosCn;text-anchor:end}}
.rlab2{{font:700 8.6px HerosCn;fill:var(--ink);text-anchor:end}}
.mlab{{font:700 11px Heros;text-anchor:end}}
.mlabr{{font:700 11px Heros;text-anchor:start}}
.mname{{font:11px Heros;text-anchor:middle}}
.mname.b{{font-weight:700}}
.val{{font:700 9.5px HerosCn;fill:var(--ink);text-anchor:middle}}
.sig{{font:italic 11px Termes;fill:var(--ink);text-anchor:middle}}
.cellv{{font:700 9.5px HerosCn;text-anchor:middle}}
.cap2{{font:700 9px Heros;letter-spacing:.12em;text-transform:uppercase;fill:var(--mute)}}
.cl{{font:700 10px HerosCn}}
.lg{{font:9.6px Heros}}
.sub{{font-size:.7em;baseline-shift:sub}}
"""

ARROWS_JS = f"""
(function(){{
  const fig=document.getElementById('fig'), sv=document.getElementById('arrows'), F=fig.getBoundingClientRect();
  sv.setAttribute('width',F.width); sv.setAttribute('height',F.height);
  const c=[...fig.querySelectorAll('.card')].map(e=>{{const r=e.getBoundingClientRect();
    return {{l:r.left-F.left,r:r.right-F.left,t:r.top-F.top,b:r.bottom-F.top}};}});
  const INK='{INK}';
  let o='<defs><marker id="ah" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto">'
       +'<path d="M0 0L10 5L0 10z" fill="'+INK+'"/></marker></defs>';
  function badge(a,b){{
    const x=(a.r+b.l)/2, y=(Math.max(a.t,b.t)+Math.min(a.b,b.b))/2;
    o+='<circle cx="'+x+'" cy="'+y+'" r="13" fill="'+INK+'"/>'
      +'<path d="M'+(x-5.5)+' '+y+'h10M'+(x+.5)+' '+(y-5)+'l5 5-5 5" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
  }}
  badge(c[0],c[1]); badge(c[2],c[3]); badge(c[3],c[4]);
  const x1=(c[1].l+c[1].r)/2, x2=(c[2].l+c[2].r)/2, ym=(c[1].b+c[2].t)/2;
  o+='<path d="M'+x1+' '+c[1].b+'V'+ym+'H'+x2+'V'+(c[2].t-2)+'" fill="none" stroke="'+INK+'" stroke-width="2" marker-end="url(#ah)"/>';
  sv.innerHTML=o;
  document.title = Math.ceil(F.height);
}})();
"""


def card(letter, body, tag=''):
    tag_html = f'<span class="io">{tag}</span>' if tag else ''
    head = f'<div class="ph"><span class="lt">{letter}</span><span class="tt">{TITLES[letter]}</span>{tag_html}</div>'
    return f'<div class="card">{head}{body}</div>'


def build():
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SCRIPT profile figure</title><style>{CSS}</style></head>
<body><div id="fig">
<div class="row r1">
{card('a', panel_a(), 'score')}
{card('b', panel_b(), 'profile')}
</div>
<div class="row r2">
{card('c', panel_c())}
{card('d', panel_d())}
{card('e', panel_e())}
</div>
<svg id="arrows"></svg>
</div>
<script>{ARROWS_JS}</script>
</body></html>"""
    (HERE / 'script_profile_figure.html').write_text(page)


if __name__ == '__main__':
    build()
    print('wrote script_profile_figure.html')
