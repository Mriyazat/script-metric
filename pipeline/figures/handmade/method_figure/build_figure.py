"""
Layout (two rows, read left to right):
    a  Span-annotated response   ->  b  Behavioral events (l, X)
    c  Position & transition profiles  ->  d  Matched null -> Script  ->  e  Script profile -> model

Everything is in the STYLE section below.
Each panel is one function (panel_a ... panel_e) that returns HTML/SVG.
"""
import base64
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'data/response.json').read_text())   # the worked response + pooled tables
PROFILES = json.loads((HERE / 'data/profiles.json').read_text())  # P and T per enrolled model


# STYLE

FAMILY_COLOR = {'empathy': '#3A78B5', 'advice': '#D1495B', 'questions': '#3E9A5E', 'other': '#8A949D'}
FAMILY_PRIORITY = {'questions': 4, 'empathy': 3, 'advice': 2, 'other': 1}  # colour of a multi-label span
C_COLOR = '#7A5CC2'      # choreography
M_COLOR = '#1C9A87'      # momentum (also transition arcs and the "input/output" tags)
GOLD = '#E9A400'         # this response / the traced slot
INK = '#1E2A36'
MUTE = '#7E8B97'
HAIR = '#E6E2D9'
HEAT_RGB = (31, 59, 87)  # heat-map cells
PAGE_BG = '#F5F3EE'

FOCUS_SLOT = 4           # slot traced in gold through a and b (DIR + SIN at p = 379)

TITLES = {
    'a': 'Span-annotated response',
    'b': 'Behavioral events (ℓ, X)',
    'c': 'Position &amp; transition profiles',
    'd': 'Matched null → Script',
    'e': 'Script profile → model',
}


# helpers
FAMILY = dict(zip(R['labels'], R['family']))
LABELS = R['labels']
# Display truncation: the response's final sentence (its last QCL span) is cut so the
# card is shorter; the event, its slot and its transitions are dropped consistently.
TRUNCATE_AT = 649
if TRUNCATE_AT:
    R['text'] = R['text'][:TRUNCATE_AT].rstrip() + ' [\u2026]'
    R['events'] = [e for e in R['events'] if e['p'] < TRUNCATE_AT]
    R['slots'] = [s for s in R['slots'] if s['p'] < TRUNCATE_AT]
    _keep = {e['code'] for e in R['events']}
    R['transitions'] = R['transitions'][:len(R['slots']) - 1] if 'transitions' in R else []
SLOTS, EVENTS = R['slots'], R['events']
esc = html.escape


def color(code):
    return FAMILY_COLOR[FAMILY[code]]


def slot_family(codes):
    """Family used to colour a span that carries several codes."""
    return max((FAMILY[c] for c in codes), key=FAMILY_PRIORITY.get)


def rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'


def heat_fill(value, row_max):
    """Row-normalised cell shade; empty cells get a flat paper colour."""
    if value == 0:
        return '#F3F0EA'
    a = .07 + .93 * (value / row_max) ** .8
    return f'rgba({HEAT_RGB[0]},{HEAT_RGB[1]},{HEAT_RGB[2]},{a:.3f})'


def spread(xs, x0, x1, min_gap):
    """Map positions in [0,1] to pixels; also return pushed-apart copies so stacks don't overlap."""
    true = [x0 + x * (x1 - x0) for x in xs]
    pushed = []
    for p in true:
        pushed.append(max(p, pushed[-1] + min_gap) if pushed else p)
    return true, pushed


def svg(w, h, body, fill=False):
    if fill:
        return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" '
                f'style="width:100%;height:100%;max-height:100%">{body}</svg>')
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'



LI = {l: i for i, l in enumerate(LABELS)}
POS_HITS = sorted({(LI[e['code']], e['X']) for e in EVENTS})
TRANS_HITS = sorted({(LI[a], LI[b]) for a, b in R['transitions']})


# span-annotated response
def panel_a():
    end_of = {e['p']: e['end'] for e in EVENTS}
    text, out, cur = R['text'], [], 0
    for k, s in enumerate(SLOTS):
        p, end = s['p'], end_of[s['p']]
        if p > cur:
            out.append(esc(text[cur:p]))
        fam_color = FAMILY_COLOR[slot_family(s['codes'])]
        focus = ' focus' if k == FOCUS_SLOT else ''
        chips = ''.join(f'<b style="background:{color(c)}">{c}</b>' for c in s['codes'])
        first, _, rest = esc(text[p:end]).partition(' ')
        # chips sit on the span's first word so they wrap with it
        out.append(f'<span class="hl{focus}" style="--c:{fam_color};--hb:{rgba(fam_color, .15)}">'
                   f'<span class="nw"><span class="anc"><span class="tags{focus}">{chips}</span></span>{first}</span>'
                   f'{" " + rest if rest else ""}</span>')
        cur = end
    out.append(esc(text[cur:]))
    return (f'<div class="user"><span class="cap">user</span>{esc(R["user"])}</div>'
            f'<div class="cap">model response · clinician span labels</div>'
            f'<div class="reply">{"".join(out)}</div>')


# behavioral events
def panel_b():
    W, H = 468, 288
    x0, x1 = 26, W - 18
    strip_y, stack_base, axis_y = 8, 196, 222
    chip_w, chip_h, chip_gap = 40, 18, 3
    X = lambda v: x0 + v * (x1 - x0)
    true, pushed = spread([s['x'] for s in SLOTS], x0, x1, 43)
    end_of = {e['p']: e['end'] for e in EVENTS}
    o = []

    # the response as a strip: each highlight at its character extent
    o.append(f'<rect x="{x0}" y="{strip_y}" width="{x1 - x0}" height="16" rx="3" fill="#EFECE5"/>')
    for s in SLOTS:
        a, b = X(s['p'] / R['length']), X(end_of[s['p']] / R['length'])
        o.append(f'<rect x="{a + .8:.1f}" y="{strip_y}" width="{max(b - a - 1.6, 2):.1f}" height="16" rx="2" '
                 f'fill="{FAMILY_COLOR[slot_family(s["codes"])]}" opacity=".85"/>')
    o.append(f'<text x="{x0 - 9}" y="{strip_y + 12}" class="axl" text-anchor="end">r</text>')

    # position axis with 10 bins
    for b in range(10):
        bx = x0 + (x1 - x0) * b / 10
        o.append(f'<rect x="{bx:.1f}" y="{axis_y}" width="{(x1 - x0) / 10:.1f}" height="15" '
                 f'fill="{"#EFECE5" if b % 2 else "#F7F5F0"}"/>'
                 f'<text x="{bx + (x1 - x0) / 20:.1f}" y="{axis_y + 11}" class="bin">{b}</text>')
    o.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="#CFCABF"/>'
             f'<text x="{x1 + 5}" y="{axis_y + 12}" class="axl">X</text>')

    # transitions between successive slots
    for a, b in zip(true, true[1:]):
        depth = 12 + min(30, (b - a) * .34)
        y = axis_y + 19
        o.append(f'<path d="M{a:.1f} {y} C{a:.1f} {y + depth:.1f} {b:.1f} {y + depth:.1f} {b:.1f} {y + 2}" '
                 f'fill="none" stroke="{M_COLOR}" stroke-width="1.6"/>'
                 f'<path d="M{b - 3.6:.1f} {y + 7.5} L{b:.1f} {y + .5} L{b + 3.6:.1f} {y + 7.5} Z" fill="{M_COLOR}"/>')

    # one stack of chips per slot, linked to its span start (above) and its bin (below)
    for k, s in enumerate(SLOTS):
        c, t, n = pushed[k], true[k], len(s['codes'])
        top = stack_base - n * chip_h - (n - 1) * chip_gap
        line, width = (GOLD, 1.6) if k == FOCUS_SLOT else ('#B9B3A8', 1)
        o.append(f'<path d="M{t:.1f} {strip_y + 16} V{strip_y + 26} L{c:.1f} {strip_y + 46} V{top - 3}" '
                 f'fill="none" stroke="{line}" stroke-width="{width}"/>'
                 f'<circle cx="{t:.1f}" cy="{strip_y + 16}" r="2.4" fill="{INK}"/>'
                 f'<path d="M{c:.1f} {stack_base + 2} V{stack_base + 8} L{t:.1f} {axis_y - 8} V{axis_y}" '
                 f'fill="none" stroke="{line}" stroke-width="{width}"/>'
                 f'<circle cx="{t:.1f}" cy="{axis_y}" r="2.6" fill="{INK}"/>')
        for j, code in enumerate(s['codes']):
            y = stack_base - (j + 1) * chip_h - j * chip_gap
            o.append(f'<rect x="{c - chip_w / 2:.1f}" y="{y}" width="{chip_w}" height="{chip_h}" rx="3" fill="{color(code)}"/>'
                     f'<text x="{c:.1f}" y="{y + 12.8}" class="chip">{code}</text>')
        if k == FOCUS_SLOT:
            o.append(f'<rect x="{c - chip_w / 2 - 4:.1f}" y="{top - 4}" width="{chip_w + 8}" '
                     f'height="{stack_base - top + 8}" rx="6" fill="none" stroke="{GOLD}" stroke-width="2"/>')
    return svg(W, H, "".join(o), fill=True)


# position & transition profiles
def heatmap(M, x, y, cell, hits, col_labels):
    o = []
    for i, row in enumerate(M):
        mx = max(row) or 1
        for j, v in enumerate(row):
            o.append(f'<rect x="{x + j * cell}" y="{y + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{heat_fill(v, mx)}"/>')
    for i, j in hits:
        o.append(f'<rect x="{x + j * cell - 1.2}" y="{y + i * cell - 1.2}" width="{cell + 1.4}" height="{cell + 1.4}" '
                 f'rx="1.5" fill="none" stroke="{GOLD}" stroke-width="2.2"/>')
    for j, lab in enumerate(col_labels):
        cx = x + j * cell + (cell - 1) / 2
        if lab in FAMILY:   # code names: rotated
            o.append(f'<text transform="translate({cx + 3.2:.1f},{y - 4}) rotate(-90)" class="colr" fill="{color(lab)}">{lab}</text>')
        else:               # bin numbers
            o.append(f'<text x="{cx}" y="{y - 5}" class="colh">{lab}</text>')
    return ''.join(o)


def hat_title(x, y, letter, rest):
    """Italic table title with a hat over the first letter, e.g. P-hat(l, X)."""
    return (f'<text x="{x}" y="{y}" class="ttl">{letter}{rest}</text>'
            f'<text x="{x + 3.5}" y="{y - 4}" class="ttl hat">ˆ</text>')


def panel_c():
    cell, label_x, top = 12, 38, 58
    px = label_x + 4                  # left edge of P
    tx = px + 10 * cell + 22          # left edge of T
    W, H = tx + 20 * cell + 4, top + 20 * cell + 6
    o = [f'<text x="{label_x}" y="{top + i * cell + 9}" class="rowl" fill="{color(l)}">{l}</text>'
         for i, l in enumerate(LABELS)]
    o.append(heatmap(R['position_counts'], px, top, cell, POS_HITS, [str(b) for b in range(10)]))
    o.append(heatmap(R['transition_counts'], tx, top, cell, TRANS_HITS, LABELS))
    o.append(hat_title(px + 38, 18, 'P', '(ℓ, X)'))
    o.append(hat_title(tx + 92, 18, 'T', '(ℓ<tspan class="sub up">prev</tspan>, ℓ)'))
    tables = svg(W, H, ''.join(o))

    # share of H(L) explained by position (C) and by the previous label given position (M)
    b, w = R['bits'], 428
    k = w / b['H_L']
    c, m = b['I_LX'] * k, b['I_LLprev_X'] * k
    bar = svg(w, 22,
              f'<rect x="0" y="4" width="{c:.1f}" height="14" fill="{C_COLOR}"/>'
              f'<rect x="{c:.1f}" y="4" width="{m:.1f}" height="14" fill="{M_COLOR}"/>'
              f'<rect x="{c + m:.1f}" y="4" width="{w - c - m:.1f}" height="14" fill="#ECE8E0"/>'
              f'<text x="{c / 2:.1f}" y="14.6" class="inb">C</text>'
              f'<text x="{c + m / 2:.1f}" y="14.6" class="inb">M</text>'
              f'<text x="{w - 4}" y="15" class="barl">H(L)</text>')
    key = f'<div class="key"><i></i>this response · pooled over the system’s {R["n_responses"]} responses</div>'
    return f'{tables}{key}<div style="margin:12px 0 10px">{bar}</div>'


# d  matched null -> Script
def shuffle_rows(W):
    """Observed response and its label shuffles over the same positions."""
    x0, x1, sq, gap, row_h = 34, W - 10, 9, 1.4, 50
    _, pushed = spread([s['x'] for s in SLOTS], x0, x1, 17)
    rows = [('obs', [s['codes'] for s in SLOTS])]
    for i, sh in enumerate(R['shuffles']):
        rows.append((f'π{i + 1}', [[e['code'] for e in sh if e['x'] == s['x']] for s in SLOTS]))
    o = []
    for r, (name, cols) in enumerate(rows):
        yb = r * row_h + row_h - 6
        o.append(f'<line x1="{x0 - 4}" y1="{yb + 2}" x2="{x1 + 4}" y2="{yb + 2}" stroke="#D8D3C9"/>'
                 f'<text x="0" y="{yb - 4}" class="rlab{" b" if r == 0 else ""}">{name}</text>')
        for k, codes in enumerate(cols):
            for j, code in enumerate(codes):
                o.append(f'<rect x="{pushed[k] - sq / 2:.1f}" y="{yb - (j + 1) * sq - j * gap:.1f}" '
                         f'width="{sq}" height="{sq}" rx="1.6" fill="{color(code)}"/>')
        if r == 0:
            o.append(f'<line x1="{x0 - 4}" y1="{yb + 8}" x2="{x1 + 4}" y2="{yb + 8}" stroke="{HAIR}" stroke-dasharray="3 3"/>')
    return svg(W, row_h * len(rows) + 4, ''.join(o))


def calibration(W):
    """R_obs = chance + C_ex + M_ex; Script is the part above the shuffle mean."""
    n = R['null']
    x0, x1, vmax = 10, W - 14, .5
    X = lambda v: x0 + (x1 - x0) * v / vmax
    axis_y, bar_y, bar_h = 150, 64, 18
    xz, xn, xo, xc = X(0), X(n['R_null_mean']), X(n['R_obs']), X(n['ceiling'])
    xm = X(n['R_null_mean'] + n['C_excess'])
    o = ['<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="5" height="5" fill="#ECE8E0"/><line x1="0" y1="0" x2="0" y2="5" stroke="#CFC9BE" stroke-width="2"/></pattern></defs>']
    # decomposition bar
    o.append(f'<rect x="{xz:.1f}" y="{bar_y}" width="{xn - xz:.1f}" height="{bar_h}" fill="url(#hatch)"/>'
             f'<rect x="{xn:.1f}" y="{bar_y}" width="{xm - xn:.1f}" height="{bar_h}" fill="{C_COLOR}"/>'
             f'<rect x="{xm:.1f}" y="{bar_y}" width="{xo - xm:.1f}" height="{bar_h}" fill="{M_COLOR}"/>'
             f'<text x="{(xn + xm) / 2:.1f}" y="{bar_y + 13}" class="inb">C<tspan class="sub">ex</tspan></text>'
             f'<text x="{(xm + xo) / 2:.1f}" y="{bar_y + 13}" class="inb">M<tspan class="sub">ex</tspan></text>'
             f'<text x="{(xz + xn) / 2:.1f}" y="{bar_y - 8}" class="rk">chance</text>')
    o.append(f'<path d="M{xn:.1f} {bar_y - 6} V{bar_y - 14} H{xo:.1f} V{bar_y - 6}" fill="none" stroke="{INK}" stroke-width="1.2"/>'
             f'<text x="{(xn + xo) / 2:.1f}" y="{bar_y - 20}" class="rk b smallcaps">Script</text>')
    for x in (xn, xo):
        o.append(f'<line x1="{x:.1f}" y1="{bar_y + bar_h}" x2="{x:.1f}" y2="{axis_y}" stroke="#BDB7AC" stroke-dasharray="2 3"/>')
    # axis, shuffle distribution of R, observed R, ceiling
    o.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="#BDB7AC" stroke-width="1.2"/>'
             f'<line x1="{xz:.1f}" y1="{axis_y - 4}" x2="{xz:.1f}" y2="{axis_y + 4}" stroke="#BDB7AC"/>'
             f'<text x="{xz:.1f}" y="{axis_y + 20}" class="tick">0</text>')
    mu, sd = n['R_null_mean'], n['R_null_sd']
    pts = ' '.join(f'{X(mu + i * sd / 10):.2f},{axis_y - 46 * math.exp(-.5 * (i / 10) ** 2):.2f}' for i in range(-40, 41))
    o.append(f'<polyline points="{pts}" fill="rgba(126,139,151,.3)" stroke="{MUTE}" stroke-width="1"/>')
    o.append(f'<line x1="{xc:.1f}" y1="{bar_y - 14}" x2="{xc:.1f}" y2="{axis_y}" stroke="{INK}" stroke-dasharray="3 2"/>'
             f'<path d="M{xo:.1f} {axis_y - 7} l6 7 l-6 7 l-6 -7 Z" fill="{INK}"/>')
    o.append(f'<text x="{xn:.1f}" y="{axis_y + 22}" class="rk"><tspan text-decoration="overline">R</tspan><tspan class="sup">(s)</tspan></text>'
             f'<text x="{xo:.1f}" y="{axis_y + 22}" class="rk b">R<tspan class="sub">obs</tspan></text>'
             f'<text x="{xc:.1f}" y="{axis_y + 22}" class="rk">ceiling</text>')
    return svg(W, 178, ''.join(o))


def panel_d():
    W = 352
    return f'{shuffle_rows(W)}<div style="margin-top:auto">{calibration(W)}</div>'


# script profile -> model
def thumbnail(M, cell, hits):
    o = []
    for i, row in enumerate(M):
        mx = max(row) or 1
        for j, v in enumerate(row):
            o.append(f'<rect x="{j * cell:.2f}" y="{i * cell:.2f}" width="{cell - .4:.2f}" height="{cell - .4:.2f}" fill="{heat_fill(v, mx)}"/>')
    for i, j in hits:
        o.append(f'<circle cx="{j * cell + cell / 2 - .2:.2f}" cy="{i * cell + cell / 2 - .2:.2f}" r="{max(cell * .3, 1.1):.2f}" fill="{GOLD}"/>')
    return ''.join(o), len(M[0]) * cell, len(M) * cell


def panel_e():
    ll = R['model_loglik']
    models = sorted(ll, key=ll.get, reverse=True)       # best match first
    best = math.exp(ll[models[0]])
    card_w, gap, H = 60, 4, 302
    W = len(models) * card_w + (len(models) - 1) * gap
    o = []
    for k, m in enumerate(models):
        x, win = k * (card_w + gap), k == 0
        o.append(f'<rect x="{x}" y="0" width="{card_w}" height="{H}" rx="7" fill="{"#FFF8E6" if win else "#FAF8F4"}" '
                 f'stroke="{INK if win else "#ECE8E0"}" stroke-width="{1.6 if win else 1}"/>'
                 f'<text x="{x + card_w / 2}" y="17" class="mname{" b" if win else ""}">{m}</text>')
        p, pw, ph = thumbnail(PROFILES[m]['P'], 5, POS_HITS)
        t, tw, th = thumbnail(PROFILES[m]['T'], 2.7, TRANS_HITS)
        o.append(f'<g transform="translate({x + (card_w - pw) / 2:.1f},27)">{p}</g>'
                 f'<g transform="translate({x + (card_w - tw) / 2:.1f},{36 + ph})">{t}</g>')
        # relative likelihood of this response under model m (bars start at zero)
        bar_h = math.exp(ll[m]) / best * 64
        yb = 46 + ph + th
        o.append(f'<rect x="{x + card_w / 2 - 9}" y="{yb + 64 - bar_h:.1f}" width="18" height="{bar_h:.1f}" rx="2" '
                 f'fill="{INK if win else "#D9D4CA"}"/>')
        if win:
            o.append(f'<text x="{x + card_w / 2}" y="{yb + 80}" class="mname b">m̂</text>')
    note = f'<div class="cap">profile per enrolled model · <span style="color:{GOLD}">●</span> this response</div>'
    return note + svg(W, H, ''.join(o))


# page
def font_face(file, family, weight, style='normal'):
    data = base64.b64encode((HERE / 'fonts' / f'{file}.woff').read_bytes()).decode()
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
.r1{{grid-template-columns:700px 1fr}}
.r1 .card{{padding-bottom:8px}}
.r2{{grid-template-columns:462px 1fr 358px}}
.card{{background:#fff;border:1px solid var(--hair);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;min-width:0}}
svg{{display:block;overflow:visible}}
#arrows{{position:absolute;left:0;top:0;pointer-events:none}}

/* panel header */
.ph{{display:flex;align-items:center;gap:9px;margin-bottom:10px}}
.lt{{width:21px;height:21px;border-radius:50%;background:var(--ink);color:#fff;font:700 12px/21px Heros;text-align:center}}
.tt{{font:700 14.5px/1 Heros}}
.io{{margin-left:auto;font:700 9.5px/1 Heros;letter-spacing:.14em;text-transform:uppercase;color:#fff;background:{M_COLOR};padding:4px 7px;border-radius:20px}}
.cap{{font:700 9px/1 Heros;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:0 0 12px}}

/* a */
.user{{font:italic 12.6px/1.42 Termes;color:#56636F;background:#FAF8F4;border-left:3px solid #CFC9BE;padding:8px 11px;border-radius:0 6px 6px 0;margin-bottom:12px}}
.user .cap{{display:inline;margin-right:8px;font-style:normal}}
.reply{{font:14.6px/2.3 Termes;color:#26323D;margin-bottom:-10px}}
.hl{{background:var(--hb);box-shadow:inset 0 -2px 0 var(--c);padding:1px 0;-webkit-box-decoration-break:clone;box-decoration-break:clone}}
.hl.focus{{background:rgba(247,190,40,.28)}}
.nw{{white-space:nowrap}}
.anc{{position:relative;display:inline-block;width:0;height:0}}
.tags{{position:absolute;left:0;bottom:11px;display:flex;gap:1.5px}}
.tags b{{font:700 8.8px/1 HerosCn;color:#fff;padding:2.6px 3.4px 2.2px;border-radius:2.5px}}
.tags.focus{{outline:2px solid var(--gold);outline-offset:1.5px;border-radius:3px}}

/* c */
.key{{display:flex;align-items:center;gap:7px;font:10.5px/1 Heros;color:var(--mute);margin-top:6px}}
.key i{{width:11px;height:11px;border:2.2px solid var(--gold);border-radius:2px}}

/* svg text */
.axl{{font:italic 13px Termes;fill:var(--ink)}}
.ttl{{font:italic 15px Termes;fill:var(--ink)}}
.hat{{font-style:normal;font-size:13px}}
.bin,.colh,.tick{{font:9.5px Heros;fill:var(--mute);text-anchor:middle}}
.chip{{font:700 9.6px HerosCn;fill:#fff;text-anchor:middle}}
.rowl{{font:700 9.2px HerosCn;text-anchor:end}}
.colr{{font:700 8.2px HerosCn}}
.inb{{font:700 10px HerosCn;fill:#fff;text-anchor:middle}}
.barl{{font:10px Heros;fill:var(--mute);text-anchor:end}}
.rlab{{font:italic 12.5px Termes;fill:var(--mute)}}
.rk{{font:italic 12px Termes;fill:var(--mute);text-anchor:middle}}
.b{{fill:var(--ink);font-weight:700}}
.rk.b,.rlab.b{{font-weight:400}}
.smallcaps{{font-style:normal;font-variant:small-caps;font-size:13.5px;letter-spacing:.04em}}
.mname{{font:12px Heros;fill:#56636F;text-anchor:middle}}
.mname.b{{font-weight:700;fill:var(--ink)}}
.sub{{font-size:.7em;baseline-shift:sub}}
.up{{font-style:normal}}
.sup{{font-size:.7em;baseline-shift:super}}
"""

# Pipeline arrows are drawn after layout, from the cards' real positions:
# a->b, c->d, d->e as round badges; b->c as an elbow line between the rows.
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
}})();
"""


def card(letter, body, tag='', body_style=''):
    tag_html = f'<span class="io">{tag}</span>' if tag else ''
    head = f'<div class="ph"><span class="lt">{letter}</span><span class="tt">{TITLES[letter]}</span>{tag_html}</div>'
    return f'<div class="card">{head}<div style="{body_style}">{body}</div></div>' if body_style else \
           f'<div class="card">{head}{body}</div>'


def build():
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SCRIPT method figure</title><style>{CSS}</style></head>
<body><div id="fig">
<div class="row r1">
{card('a', panel_a(), 'input')}
{card('b', panel_b(), body_style='flex:1;display:flex;align-items:center;min-height:0')}
</div>
<div class="row r2">
{card('c', panel_c())}
{card('d', panel_d(), body_style='display:flex;flex-direction:column;flex:1')}
{card('e', panel_e(), 'output')}
</div>
<svg id="arrows"></svg>
</div>
<script>{ARROWS_JS}</script>
</body></html>"""
    (HERE / 'script_method_figure.html').write_text(page)


if __name__ == '__main__':
    build()
