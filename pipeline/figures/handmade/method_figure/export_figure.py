"""Render script_method_figure.html to PNG (3x) and a vector PDF.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
SCALE = 3  # PNG pixel density (3 -> 3840 px wide)


async def main():
    async with async_playwright() as p:
        # uses the installed Google Chrome; fall back to Playwright's Chromium if present
        try:
            browser = await p.chromium.launch(channel='chrome')
        except Exception:
            browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1280, 'height': 800}, device_scale_factor=SCALE)
        await page.goto((HERE / 'script_method_figure.html').as_uri())
        await page.wait_for_timeout(300)
        fig = await page.query_selector('#fig')
        box = await fig.bounding_box()
        await fig.screenshot(path=str(HERE / 'script_method_figure.png'))
        await page.pdf(path=str(HERE / 'script_method_figure.pdf'), print_background=True,
                       width=f"{box['width']}px", height=f"{box['height'] + 1}px",
                       margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})
        await browser.close()


asyncio.run(main())
