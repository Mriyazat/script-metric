"""
Figure 1: the SCRIPT method, read from input to output.

    a  Span-annotated response (input)  ->  b  Behavioural events (l, X)
    c  Position & transition tables  ->  d  Chance level  ->  e  Script score & profile (output)

Gold outline = the worked response; solid gold = the span traced through a, b and c.
Each panel is one function that returns SVG; the page layout is at the bottom.

python build_method_figure.py   # writes out/figures/script_method_figure.png
"""
import math

from common import (C_COLOR, FAMILY_COLOR, FOCUS_SLOT, GOLD, GOLD_TEXT, HEAT_RGB, INK, LABELS, M_COLOR,
                    MUTE, POS_HITS, PROFILES, R, SLOTS, TRANS_HITS, card, color, heatmap, page,
                    panel_response, render, spread, svg)


def fill(v, mx, power):
    if v == 0:
        return '#F3F0EA'
    a = .06 + .94 * (v / mx) ** power
    return f'rgba({HEAT_RGB[0]},{HEAT_RGB[1]},{HEAT_RGB[2]},{a:.3f})'


def cells(M, x, y, c, gap=1.5, power=.8):
    """Row-normalised shading of a small table."""
    o = []
    for i, row in enumerate(M):
        mx = max(row) or 1
        for j, v in enumerate(row):
            o.append(f'<rect x="{x + j * c:.1f}" y="{y + i * c:.1f}" width="{c - gap:.1f}" height="{c - gap:.1f}" '
                     f'rx="{min(3, c / 6):.1f}" fill="{fill(v, mx, power)}"/>')
    return ''.join(o)


# b  the response in a as events: chips on the position ruler, transitions underneath, key below
def panel_events(W=468):
    x0, x1 = 12, W - 12
    chip_w, chip_h, chip_gap = 40, 18, 3
    maxn = max(len(s['codes']) for s in SLOTS)
    stack_base = 28 + maxn * (chip_h + chip_gap)
    axis_y = stack_base + 26
    X = lambda v: x0 + v * (x1 - x0)
    _, pushed = spread([s['x'] for s in SLOTS], x0 + 8, x1, 43)
    o = [f'<rect x="{x0 - 8}" y="18" width="{x1 - x0 + 16}" height="{axis_y + 52 - 18}" rx="8" fill="none" '
         f'stroke="{GOLD}" stroke-width="1.8"/>',
         f'<rect x="{x1 - 150}" y="11" width="150" height="14" fill="#fff"/>',
         f'<text x="{x1 - 4}" y="22" class="nm" style="text-anchor:end;fill:{GOLD_TEXT}">response (a), span by span</text>']
    for b in range(10):
        bx = x0 + (x1 - x0) * b / 10
        o.append(f'<rect x="{bx:.1f}" y="{axis_y}" width="{(x1 - x0) / 10:.1f}" height="15" '
                 f'fill="{"#EFECE5" if b % 2 else "#F7F5F0"}"/>'
                 f'<text x="{bx + (x1 - x0) / 20:.1f}" y="{axis_y + 11}" class="bin">{b}</text>')
    o.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x1}" y2="{axis_y}" stroke="#CFCABF"/>')
    tx = [X(s['x']) for s in SLOTS]
    for a, b in zip(tx, tx[1:]):
        depth, y = 12 + min(24, (b - a) * .3), axis_y + 19
        o.append(f'<path d="M{a:.1f} {y} C{a:.1f} {y + depth:.1f} {b:.1f} {y + depth:.1f} {b:.1f} {y + 2}" '
                 f'fill="none" stroke="{M_COLOR}" stroke-width="1.6"/>'
                 f'<path d="M{b - 3.6:.1f} {y + 7.5} L{b:.1f} {y + .5} L{b + 3.6:.1f} {y + 7.5} Z" fill="{M_COLOR}"/>')
    # key: chip = label, ruler = position, arc = what follows what
    yk = axis_y + 74
    k1, k2, k3 = x0 + 10, x0 + 150, x0 + 300
    o.append(''.join(f'<rect x="{k1 + i * 7}" y="{yk - 9}" width="6" height="12" rx="1.5" fill="{FAMILY_COLOR[f]}"/>'
                     for i, f in enumerate(['empathy', 'advice', 'questions', 'other']))
             + f'<text x="{k1 + 34}" y="{yk + 1}" class="nm" style="text-anchor:start">label <tspan class="axl">ℓ</tspan> of a span</text>'
             f'<rect x="{k2}" y="{yk - 8}" width="11" height="10" fill="#F7F5F0" stroke="#CFCABF" stroke-width=".8"/>'
             f'<rect x="{k2 + 11}" y="{yk - 8}" width="11" height="10" fill="#EFECE5" stroke="#CFCABF" stroke-width=".8"/>'
             f'<text x="{k2 + 28}" y="{yk + 1}" class="nm" style="text-anchor:start">position <tspan class="axl">X</tspan>, start → end</text>'
             f'<path d="M{k3} {yk - 7} C{k3} {yk + 3} {k3 + 22} {yk + 3} {k3 + 22} {yk - 6}" fill="none" stroke="{M_COLOR}" stroke-width="1.6"/>'
             f'<path d="M{k3 + 19} {yk - 2} L{k3 + 22} {yk - 8} L{k3 + 25} {yk - 2} Z" fill="{M_COLOR}"/>'
             f'<text x="{k3 + 30}" y="{yk + 1}" class="nm" style="text-anchor:start">what follows what → <tspan class="axl">M</tspan></text>')
    for k, s in enumerate(SLOTS):
        c, t, n = pushed[k], tx[k], len(s['codes'])
        top = stack_base - n * chip_h - (n - 1) * chip_gap
        focus = k == FOCUS_SLOT
        o.append(f'<path d="M{c:.1f} {stack_base + 2} V{stack_base + 8} L{t:.1f} {axis_y - 8} V{axis_y}" fill="none" '
                 f'stroke="{GOLD if focus else "#B9B3A8"}" stroke-width="{1.6 if focus else 1}"/>'
                 f'<circle cx="{t:.1f}" cy="{axis_y}" r="2.6" fill="{INK}"/>')
        if focus:
            o.append(f'<rect x="{c - chip_w / 2 - 4:.1f}" y="{top - 4}" width="{chip_w + 8}" height="{stack_base - top + 8}" '
                     f'rx="6" fill="{GOLD}" fill-opacity=".85"/>')
        for j, code in enumerate(s['codes']):
            y = stack_base - (j + 1) * chip_h - j * chip_gap
            o.append(f'<rect x="{c - chip_w / 2:.1f}" y="{y}" width="{chip_w}" height="{chip_h}" rx="3" fill="{color(code)}"/>'
                     f'<text x="{c:.1f}" y="{y + 12.8}" class="chip">{code}</text>')
        if focus:
            o.append(f'<rect x="{c - chip_w / 2 - 4:.1f}" y="{top - 4}" width="{chip_w + 8}" height="{stack_base - top + 8}" '
                     f'rx="6" fill="none" stroke="{GOLD}" stroke-width="2"/>')
    return svg(W, axis_y + 82, ''.join(o))


# c  the system's pooled position and transition tables over the twenty codes
def panel_tables(W=410):
    cell, lab_x, top = 9.5, 36, 58
    px = lab_x + 4
    tx = px + 10 * cell + 26
    o = [f'<text x="{lab_x}" y="{top + i * cell + 8.5}" class="rowl" fill="{color(l)}">{l}</text>'
         for i, l in enumerate(LABELS)]
    o.append(heatmap(R['position_counts'], px, top, cell, POS_HITS, [str(b) for b in range(10)]))
    o.append(heatmap(R['transition_counts'], tx, top, cell, TRANS_HITS, LABELS))
    # the traced span of a and b: its codes at its bin, and the transition into it, filled
    fs = SLOTS[FOCUS_SLOT]
    li = {l: i for i, l in enumerate(LABELS)}
    prev, first = R['transitions'][FOCUS_SLOT - 1]
    for i, j, x in [(li[c], fs['X'], px) for c in fs['codes']] + [(li[prev], li[first], tx)]:
        o.append(f'<rect x="{x + j * cell - 1.2:.1f}" y="{top + i * cell - 1.2:.1f}" width="{cell + 1.4:.1f}" height="{cell + 1.4:.1f}" '
                 f'rx="1.5" fill="{GOLD}" stroke="{GOLD}" stroke-width="2.2"/>')
    o.append(f'<text x="{px + 5 * cell}" y="14" class="ttl2">position</text>'
             f'<text x="{tx + 10 * cell}" y="14" class="ttl2">previous → next</text>')
    yb = top + 20 * cell
    o.append(f'<text x="{px}" y="{yb + 12}" class="rk" style="text-anchor:start;font-size:11px">start</text>'
             f'<text x="{px + 10 * cell - 1}" y="{yb + 12}" class="rk" style="text-anchor:end;font-size:11px">end</text>')
    o.append(f'<rect x="{tx}" y="{yb + 4}" width="9" height="9" rx="1.5" fill="none" stroke="{GOLD}" stroke-width="2"/>'
             f'<text x="{tx + 14}" y="{yb + 12}" class="nm" style="text-anchor:start;font-size:11px;fill:{GOLD_TEXT}">this response</text>'
             f'<rect x="{tx + 96}" y="{yb + 4}" width="9" height="9" rx="1.5" fill="{GOLD}"/>'
             f'<text x="{tx + 110}" y="{yb + 12}" class="nm" style="text-anchor:start;font-size:11px">the traced span</text>')
    yp = yb + 22
    for x, w, col, txt in [(px, 10 * cell, C_COLOR, 'C · position'), (tx, 20 * cell, M_COLOR, 'M · sequence')]:
        o.append(f'<rect x="{x}" y="{yp}" width="{w}" height="20" rx="10" fill="{col}"/>'
                 f'<text x="{x + w / 2}" y="{yp + 14}" class="pill">{txt}</text>')
    o.append(f'<text x="{(px + tx + 20 * cell) / 2}" y="{yp + 40}" class="rk" style="font-size:11.5px">'
             f'pooled over the system’s {R["n_responses"]} responses</text>')
    return svg(W, yp + 46, ''.join(o))


# d  the response and one re-dealing of its labels over the same positions
def panel_chance(W=296):
    x0, x1, sq, gap = 92, W - 8, 11, 1.6
    _, pushed = spread([s['x'] for s in SLOTS], x0, x1, 22)
    obs = [s['codes'] for s in SLOTS]
    sh = [[e['code'] for e in R['shuffles'][0] if e['x'] == s['x']] for s in SLOTS]
    maxn = max(len(c) for c in obs)
    rh = maxn * (sq + gap) + 6
    o = []
    for r, (lab, cols) in enumerate([('this response', obs), ('shuffled', sh)]):
        yb = 6 + r * (rh + 34) + rh
        o.append(f'<text x="0" y="{yb - 4}" class="rlab{" b" if r == 0 else ""}"{f" style=\"fill:{GOLD_TEXT}\"" if r == 0 else ""}>{lab}</text>'
                 f'<line x1="{x0 - 6}" y1="{yb + 2}" x2="{x1 + 4}" y2="{yb + 2}" stroke="#D8D3C9"/>')
        for k, codes in enumerate(cols):
            for j, code in enumerate(codes):
                o.append(f'<rect x="{pushed[k] - sq / 2:.1f}" y="{yb - (j + 1) * sq - j * gap:.1f}" width="{sq}" height="{sq}" '
                         f'rx="2" fill="{color(code)}"/>')
    ya = 6 + rh + 8
    xm = (x0 + x1) / 2
    o.append(f'<path d="M{xm - 40} {ya} C{xm - 40} {ya + 20} {xm + 40} {ya + 4} {xm + 40} {ya + 22}" fill="none" '
             f'stroke="{MUTE}" stroke-width="1.3"/><path d="M{xm + 36} {ya + 17} L{xm + 40} {ya + 24} L{xm + 44} {ya + 17}" fill="{MUTE}"/>'
             f'<text x="{xm - 48}" y="{ya + 15}" class="nm" style="text-anchor:end">labels re-dealt</text>')
    y = 6 + 2 * rh + 34 + 24
    o.append(f'<text x="{W / 2}" y="{y}" class="nm" style="text-anchor:middle">positions kept; repeated to give the chance level</text>')
    return svg(W, y + 6, ''.join(o))


# e  how scripted: observed against shuffled, the gap is Script; whose script: fit to each profile
def panel_output(W=348):
    N = R['null']
    x0, xr = 66, W - 6
    X = lambda v: x0 + (xr - x0) * v / (N['R_obs'] * 1.02)
    xn, xm, xo = X(N['R_null_mean']), X(N['R_null_mean'] + N['C_excess']), X(N['R_obs'])
    o = ['<defs><pattern id="h6" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="5" height="5" fill="#ECE8E0"/><line x1="0" y1="0" x2="0" y2="5" stroke="#CFC9BE" stroke-width="2"/></pattern></defs>',
         '<text x="0" y="10" class="cap2">how scripted</text>']
    yo, ys, h = 36, 60, 16
    o.append(f'<text x="0" y="{yo + 12}" class="rlab b">observed</text>'
             f'<rect x="{x0}" y="{yo}" width="{xn - x0:.1f}" height="{h}" fill="#D9DDE1"/>'
             f'<rect x="{xn:.1f}" y="{yo}" width="{xm - xn:.1f}" height="{h}" fill="{C_COLOR}"/>'
             f'<rect x="{xm:.1f}" y="{yo}" width="{xo - xm:.1f}" height="{h}" fill="{M_COLOR}"/>'
             f'<text x="{(xn + xm) / 2:.1f}" y="{yo + 12}" class="pill">C</text>'
             f'<text x="{(xm + xo) / 2:.1f}" y="{yo + 12}" class="pill">M</text>'
             f'<text x="0" y="{ys + 12}" class="rlab">shuffled</text>'
             f'<rect x="{x0}" y="{ys}" width="{xn - x0:.1f}" height="{h}" fill="url(#h6)"/>'
             f'<line x1="{xn:.1f}" y1="{yo - 4}" x2="{xn:.1f}" y2="{ys + h + 4}" stroke="{INK}" stroke-dasharray="2.5 2"/>'
             f'<path d="M{xn:.1f} {yo - 5} V{yo - 11} H{xo:.1f} V{yo - 5}" fill="none" stroke="{INK}" stroke-width="1.3"/>'
             f'<text x="{(xn + xo) / 2:.1f}" y="{yo - 15}" class="rk b smallcaps">Script {N["R_obs"] - N["R_null_mean"]:.2f}</text>')
    ll = R['model_loglik']
    models = sorted(ll, key=ll.get, reverse=True)
    best = math.exp(ll[models[0]])
    y0, rh, c = ys + h + 26, 34, 1.5
    o.append(f'<text x="0" y="{y0}" class="cap2">whose script</text>')
    bx0 = x0 + 10 * c + 20 * c + 22
    bmax = xr - bx0 - 4
    for k, m in enumerate(models):
        y, win = y0 + 22 + k * rh, k == 0
        w = math.exp(ll[m]) / best * bmax
        if win:
            o.append(f'<rect x="-4" y="{y - 3}" width="{W + 4}" height="{rh - 2}" rx="5" fill="#FFF8E6"/>')
        o.append(f'<text x="0" y="{y + 18}" class="mname{" b" if win else ""}" style="text-anchor:start;font-size:12px">'
                 f'{"✓ " if win else ""}{m}</text>'
                 + cells(PROFILES[m]['P'], x0, y, c, .25) + cells(PROFILES[m]['T'], x0 + 10 * c + 8, y, c, .25)
                 + f'<rect x="{bx0}" y="{y + 11}" width="{w:.1f}" height="8" rx="2" fill="{INK if win else "#D9D4CA"}"/>')
    yl = y0 + 22 + len(models) * rh
    o.append(f'<text x="{x0}" y="{y0 + 13}" class="nm" style="text-anchor:start;font-size:11px">profile tables</text>'
             f'<text x="{bx0}" y="{y0 + 13}" class="nm" style="text-anchor:start;font-size:11px;fill:{GOLD_TEXT}">fit of this response</text>')
    return svg(W, yl, ''.join(o))


CSS = """
.tt{font-size:15px}
.rlab{font:italic 13px Termes;fill:var(--mute)}
.rk{font-size:12.5px}
.mname{font-size:11.5px}
.ttl2{font:italic 13.5px Termes;fill:var(--ink);text-anchor:middle}
.pill{font:700 11.5px HerosCn;fill:#fff;text-anchor:middle}
.cap2{font:700 9px Heros;letter-spacing:.12em;text-transform:uppercase;fill:var(--mute)}
.nm{font:italic 12px Termes;fill:#56636F}
.top{grid-template-columns:700px 1fr}
.bottom{grid-template-columns:1fr 330px 380px}
.top .card:first-child .reply{border:2px solid #E9A400;border-radius:8px;padding:6px 8px 4px;margin:0 -10px 0 -10px;font-size:14.2px}
.top .card:first-child div.cap{color:#B27A00}
"""


def build():
    centre = 'flex:1;display:flex;align-items:center;justify-content:center'
    body = f"""<div class="row top">
{card('a', 'Span-annotated response', panel_response(), 'input', sub='One model response (gold frame), highlighted by clinicians; the filled gold span is traced through (b) and (c)')}
{card('b', 'Behavioral events (ℓ, X)', panel_events(), sub='The response in (a) as events: each highlighted span’s label and position', body_style='flex:1;display:flex;align-items:center;min-height:0')}
</div>
<div class="row bottom">
{card('c', 'Position &amp; transition tables', panel_tables(), sub='Where behaviours sit, and what follows what', body_style=centre)}
{card('d', 'Chance level', panel_chance(), sub='Compare with shuffled responses', body_style=centre)}
{card('e', 'Script score &amp; profile', panel_output(), 'output', sub='How scripted, and whose script?', body_style=centre)}
</div>"""
    arrows = [dict(t='badge', a=0, b=1), dict(t='elbow', a=1, b=2, ax=.5, bx=.5),
              dict(t='badge', a=2, b=3), dict(t='badge', a=3, b=4)]
    return page('SCRIPT method figure', CSS, body, arrows)


if __name__ == '__main__':
    render(build(), 'script_method_figure')
