"""
Shared data, style, page scaffolding and rendering for the two hand-built figures of the main text
(build_method_figure.py: Figure 1, build_listening_figure.py: Figure 2).
"""
import asyncio
import base64
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from pipeline.common.paths import FIG  # noqa: E402  (figures land beside the other paper figures)
R = json.loads((HERE / 'data/response.json').read_text())   # the worked response + pooled tables
PROFILES = json.loads((HERE / 'data/profiles.json').read_text())  # P and T per enrolled model


# STYLE

FAMILY_COLOR = {'empathy': '#3A78B5', 'advice': '#D1495B', 'questions': '#3E9A5E', 'other': '#8A949D'}
FAMILY_PRIORITY = {'questions': 4, 'empathy': 3, 'advice': 2, 'other': 1}  # colour of a multi-label span
C_COLOR = '#7A5CC2'      # choreography
M_COLOR = '#1C9A87'      # momentum (also the transition arcs)
GOLD = '#E9A400'         # this response / the traced slot
GOLD_TEXT = '#B27A00'    # gold dark enough to read as text
TAG_RED = '#C8323F'      # the input / output tags
INK = '#1E2A36'
MUTE = '#7E8B97'
HAIR = '#E6E2D9'
HEAT_RGB = (31, 59, 87)  # heat-map cells
PAGE_BG = '#F5F3EE'

FOCUS_SLOT = 4           # slot traced in gold through a, b and c (DIR + SIN at p = 379)


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


# the worked response, highlighted span by span
def panel_response():
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


# the twenty-code legend under both figures; columns of two, accurate over inaccurate
LEGEND = [('VAC', 'accurate validation'), ('VIN', 'inaccurate validation'),
          ('NAC', 'accurate normalising'), ('NIN', 'inaccurate normalising'),
          ('ASAC', 'accurate autonomy support'), ('ASIN', 'inaccurate autonomy support'),
          ('SAC', 'accurate support'), ('SIN', 'inaccurate support'),
          ('DIR', 'directive'), ('FIX', 'fix-it'),
          ('RECT', 'recommendation'), ('TEN', 'tentative'),
          ('QOP', 'open question'), ('QCL', 'closed question'),
          ('TSH', 'topic shift'), ('LMT', 'language matching'),
          ('MEN', 'minimal encourager'), ('AUR', 'assumes user accuracy'),
          ('SEN', 'sensitivity'), ('INC', 'incoherent')]


def legend():
    items = ''.join(f'<span class="k"><b style="background:{color(code)}">{code}</b><em>{gloss}</em></span>'
                    for code, gloss in LEGEND)
    return f'<div class="legend">{items}</div>'


def card(letter, title, body, tag='', sub='', body_style=''):
    tag_html = f'<span class="io">{tag}</span>' if tag else ''
    head = (f'<div class="ph"><span class="lt">{letter}</span><span class="tt">{title}</span>{tag_html}</div>'
            + (f'<div class="sub2">{sub}</div>' if sub else ''))
    inner = f'<div style="{body_style}">{body}</div>' if body_style else body
    return f'<div class="card">{head}{inner}</div>'


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
#fig{{position:relative;width:1280px;padding:20px;background:{PAGE_BG};display:flex;flex-direction:column;gap:24px}}
#fig .row{{display:grid;gap:24px}}
.card{{background:#fff;border:1px solid var(--hair);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column;min-width:0}}
svg{{display:block;overflow:visible}}
#arrows{{position:absolute;left:0;top:0;pointer-events:none}}

/* panel header */
.ph{{display:flex;align-items:center;gap:9px;margin-bottom:10px}}
.lt{{width:21px;height:21px;border-radius:50%;background:var(--ink);color:#fff;font:700 12px/21px Heros;text-align:center}}
.tt{{font:700 14.5px/1 Heros}}
.io{{margin-left:auto;font:700 9.5px/1 Heros;letter-spacing:.14em;text-transform:uppercase;color:#fff;background:{TAG_RED};padding:4px 7px;border-radius:20px}}
.cap{{font:700 9px/1 Heros;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin:0 0 12px}}
.sub2{{font:11px/1.35 Heros;color:var(--mute);margin:-4px 0 10px 30px}}

/* the worked response */
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

/* code legend */
.legend{{margin-top:-10px;padding:0 4px;display:grid;grid-auto-flow:column;grid-template-rows:auto auto;
  justify-content:space-between;row-gap:4px}}
.legend .k{{display:flex;align-items:center;gap:5px;white-space:nowrap}}
.legend b{{font:700 8.8px/1 HerosCn;color:#fff;padding:2.6px 3.6px 2.2px;border-radius:2.5px;min-width:24px;text-align:center}}
.legend em{{font:italic 11.5px/1 Termes;color:#3A4652}}

/* svg text */
.axl{{font:italic 13px Termes;fill:var(--ink)}}
.bin,.colh,.tick{{font:9.5px Heros;fill:var(--mute);text-anchor:middle}}
.chip{{font:700 9.6px HerosCn;fill:#fff;text-anchor:middle}}
.rowl{{font:700 9.2px HerosCn;text-anchor:end}}
.colr{{font:700 8.2px HerosCn}}
.rlab{{font:italic 12.5px Termes;fill:var(--mute)}}
.rk{{font:italic 12px Termes;fill:var(--mute);text-anchor:middle}}
.b{{fill:var(--ink);font-weight:700}}
.rk.b,.rlab.b{{font-weight:400}}
.smallcaps{{font-style:normal;font-variant:small-caps;font-size:13.5px;letter-spacing:.04em}}
.mname{{font:12px Heros;fill:#56636F;text-anchor:middle}}
.mname.b{{font-weight:700;fill:var(--ink)}}
"""

# Pipeline arrows are drawn after layout, from the cards' real positions.
# specs: badge between horizontally adjacent cards, elbow between rows (from bottom/top edges)
ARROWS_JS = """
(function(){
  const S=%s, INK='%s', GOLD='%s';
  const fig=document.getElementById('fig'), sv=document.getElementById('arrows'), F=fig.getBoundingClientRect();
  sv.setAttribute('width',F.width); sv.setAttribute('height',F.height);
  const c=[...fig.querySelectorAll('.card')].map(e=>{const r=e.getBoundingClientRect();
    return {l:r.left-F.left,r:r.right-F.left,t:r.top-F.top,b:r.bottom-F.top,cx:(r.left+r.right)/2-F.left};});
  let o='<defs>'+['ink','gold'].map((k,i)=>'<marker id="ah'+k+'" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto"><path d="M0 0L10 5L0 10z" fill="'+(i?GOLD:INK)+'"/></marker>').join('')+'</defs>';
  for(const s of S){
    const col=s.trace?GOLD:INK, id=s.trace?'ahgold':'ahink';
    if(s.t==='badge'){
      const a=c[s.a], b=c[s.b], x=(a.r+b.l)/2, y=(Math.max(a.t,b.t)+Math.min(a.b,b.b))/2, d=s.left?-1:1;
      o+='<circle cx="'+x+'" cy="'+y+'" r="13" fill="'+col+'"/>'
        +'<path d="M'+(x-5.5*d)+' '+y+'h'+(10*d)+'M'+(x+.5*d)+' '+(y-5)+'l'+(5*d)+' 5l'+(-5*d)+' 5" fill="none" stroke="'+(s.trace?INK:'#fff')+'" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>';
    } else {
      const a=c[s.a], b=c[s.b], up=s.up;
      const x1=s.ax!=null?a.l+(a.r-a.l)*s.ax:a.cx, x2=s.bx!=null?b.l+(b.r-b.l)*s.bx:b.cx;
      const y1=up?a.t:a.b, y2=up?b.b+2:b.t-2, ym=(y1+y2)/2;
      o+='<path d="M'+x1+' '+y1+'V'+ym+'H'+x2+'V'+y2+'" fill="none" stroke="'+col+'" stroke-width="2"'+(s.trace?' stroke-dasharray="5 4"':'')+' marker-end="url(#'+id+')"/>';
      if(s.label){const lx=(x1+x2)/2;
        o+='<rect x="'+(lx-s.label.length*3.4-8)+'" y="'+(ym-9)+'" width="'+(s.label.length*6.8+16)+'" height="18" rx="9" fill="#F5F3EE"/>'
          +'<text x="'+lx+'" y="'+(ym+4)+'" text-anchor="middle" style="font:italic 12.5px Termes;fill:'+INK+'">'+s.label+'</text>';}
    }
  }
  sv.innerHTML=o;
})();
"""


def page(title, css, body, arrows=None):
    """A figure page: cards in `body`, the code legend underneath, pipeline arrows if given."""
    arrow_svg = '\n<svg id="arrows"></svg>' if arrows else ''
    script = f'\n<script>{ARROWS_JS % (json.dumps(arrows), INK, GOLD)}</script>' if arrows else ''
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}{css}</style></head>
<body><div id="fig">
{body}
{legend()}{arrow_svg}
</div>{script}
</body></html>"""


def render(page_html, name):
    """Render a figure page to out/figures/<name>.png at 2x with a headless browser (needs Playwright)."""
    from playwright.async_api import async_playwright

    async def run():
        async with async_playwright() as p:
            # uses the installed Google Chrome; falls back to Playwright's Chromium
            try:
                browser = await p.chromium.launch(channel='chrome')
            except Exception:
                browser = await p.chromium.launch()
            tab = await browser.new_page(viewport={'width': 1280, 'height': 800}, device_scale_factor=2)
            await tab.set_content(page_html)
            await tab.wait_for_timeout(300)
            await (await tab.query_selector('#fig')).screenshot(path=str(FIG / f'{name}.png'))
            await browser.close()

    FIG.mkdir(parents=True, exist_ok=True)
    asyncio.run(run())
    print(f'wrote {FIG / name}.png')
