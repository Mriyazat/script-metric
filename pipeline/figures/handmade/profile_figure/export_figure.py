"""Render the profile figure page to out/figures/script_profile_figure.png (3x) with a headless browser.

Called by build_figure.py. Needs Playwright (uses the installed Google Chrome, else Playwright's Chromium).
"""
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from pipeline.common.paths import FIG  # noqa: E402

WIDTH, SCALE = 1280, 3


def render(page_html):
    from playwright.async_api import async_playwright

    async def run():
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(channel='chrome')
            except Exception:
                browser = await p.chromium.launch()
            tab = await browser.new_page(viewport={'width': WIDTH, 'height': 2000}, device_scale_factor=SCALE)
            await tab.set_content(page_html)
            await tab.wait_for_timeout(400)
            h = int(await tab.evaluate('document.title'))    # the page script writes the figure height here
            await tab.set_viewport_size({'width': WIDTH, 'height': h})
            await tab.wait_for_timeout(300)
            await tab.screenshot(path=str(png), clip={'x': 0, 'y': 0, 'width': WIDTH, 'height': h})
            await browser.close()
            return h

    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / 'script_profile_figure.png'
    h = asyncio.run(run())
    print(f'wrote {png} ({WIDTH * SCALE}x{h * SCALE})')
