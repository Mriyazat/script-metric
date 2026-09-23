"""Render script_profile_figure.html to PNG (3x) and a vector PDF with headless Chrome.

No Python dependencies beyond the standard library and Pillow (for the crop).
"""
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
HTML = HERE / 'script_profile_figure.html'
WIDTH, SCALE = 1280, 3


def chrome(*args):
    return subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars',
                           '--no-first-run', '--no-default-browser-check', *args],
                          capture_output=True, text=True, check=False)


def figure_height():
    """The page script writes the figure height into document.title."""
    r = chrome('--dump-dom', f'--window-size={WIDTH},2000', HTML.as_uri())
    m = re.search(r'<title>(\d+)</title>', r.stdout)
    return int(m.group(1)) if m else 1200


def main():
    h = figure_height()
    png = HERE / 'script_profile_figure.png'
    chrome(f'--screenshot={png}', f'--window-size={WIDTH},{h}', f'--force-device-scale-factor={SCALE}', HTML.as_uri())
    im = Image.open(png)
    im = im.crop((0, 0, WIDTH * SCALE, h * SCALE))
    im.save(png)

    # vector PDF: same page with an exact @page size
    src = HTML.read_text()
    pdf_html = src.replace('</style>', f'@page{{size:{WIDTH}px {h}px;margin:0}}</style>')
    with tempfile.NamedTemporaryFile('w', suffix='.html', dir=HERE, delete=False) as f:
        f.write(pdf_html); tmp = Path(f.name)
    try:
        chrome('--no-pdf-header-footer', f'--print-to-pdf={HERE / "script_profile_figure.pdf"}',
               f'--window-size={WIDTH},{h}', tmp.as_uri())
    finally:
        tmp.unlink(missing_ok=True)
    print(f'wrote script_profile_figure.png ({im.width}x{im.height}) and .pdf; figure height {h}px')


if __name__ == '__main__':
    main()
