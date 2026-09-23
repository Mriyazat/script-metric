# SCRIPT method figure

`script_method_figure.png` is our figure. It is built from `script_method_figure.html`, and `script_method_figure.pdf` is the vector version of the same page.

## Make the image

```bash
python build_figure.py     # writes script_method_figure.html
python export_figure.py    # writes script_method_figure.png (3x) and .pdf
```

`export_figure.py` needs Playwright.


## Where to edit in `build_figure.py`

- **Colours, panel titles, traced slot:** the `STYLE` block at the top.
- **Sizes and spacing inside a panel:** the numbers at the start of each `panel_*` function.
- **Fonts, card layout, column widths:** the `CSS` string. `.r1` and `.r2` set the column widths of the two rows.
- **Pipeline arrows:** `ARROWS_JS`. The arrows are drawn from the cards' positions, so they follow any layout change.

