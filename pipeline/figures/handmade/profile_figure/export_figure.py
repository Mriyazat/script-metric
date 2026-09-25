"""Render the profile figure page to out/figures/script_profile_figure.png (3x) with headless Chrome.

Called by build_figure.py. No Python dependencies beyond the standard library and Pillow (for the crop).
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from pipeline.common.paths import FIG  # noqa: E402

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
WIDTH, SCALE = 1280, 3


def chrome(*args):
    return subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                           '--no-first-run', '--no-default-browser-check', *args],
                          capture_output=True, text=True, check=False)


def figure_height(html):
    """The page script writes the figure height into document.title."""
    r = chrome('--dump-dom', f'--window-size={WIDTH},2000', html.as_uri())
    m = re.search(r'<title>(\d+)</title>', r.stdout)
    return int(m.group(1)) if m else 1200


def render(page_html):
    with tempfile.NamedTemporaryFile('w', suffix='.html', dir=HERE, delete=False) as f:
        f.write(page_html)
        html = Path(f.name)
    try:
        h = figure_height(html)
        FIG.mkdir(parents=True, exist_ok=True)
        png = FIG / 'script_profile_figure.png'
        chrome(f'--screenshot={png}', f'--window-size={WIDTH},{h}', f'--force-device-scale-factor={SCALE}', html.as_uri())
    finally:
        html.unlink(missing_ok=True)
    im = Image.open(png)
    im = im.crop((0, 0, WIDTH * SCALE, h * SCALE))
    im.save(png)
    print(f'wrote {png} ({im.width}x{im.height}); figure height {h}px')
