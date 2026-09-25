# Hand-built Figures 1 and 2

Two figures of the main text are drawn as HTML/SVG pages and rendered to PNG with a headless browser.
Like every other figure of the paper, they are written to `out/figures/`.

| Paper | File | Built by |
|---|---|---|
| Figure 1, the method | `script_method_figure.png` | `build_method_figure.py` |
| Figure 2, where behaviour sits turn by turn, and what moves it | `script_listening_figure.png` | `build_listening_figure.py` |

## Make the images

```bash
python build_method_figure.py      # out/figures/script_method_figure.png
python build_listening_figure.py   # out/figures/script_listening_figure.png (reads the pipeline's out/)
```

Rendering needs Playwright. `build_listening_figure.py` needs the blind-annotator layer and the
listening tables; `bash reproduce.sh fig2` runs everything it depends on.

## Files

- `common.py`: data, fonts, colours, the card layout, pipeline arrows, the twenty-code legend and the renderer shared by both figures.
- `data/response.json`: the worked response of Figure 1 with the pooled tables of its system; `data/profiles.json`: the position and transition tables of each enrolled model (`make_profiles.py` rebuilds it).
- `fonts/`: TeX Gyre Heros and Termes, embedded in the pages. The profile figure (`../profile_figure`) reads the same data and fonts.

## Where to edit

- **Colours, the traced slot:** the `STYLE` block of `common.py`.
- **Sizes and spacing inside a panel:** the numbers at the start of each `panel_*` function.
- **Card layout and column widths:** the `CSS` string of each builder (`.top` / `.bottom` in Figure 1, `.panels` in Figure 2).
- **Pipeline arrows:** the `arrows` list in `build_method_figure.py`; they are drawn from the cards' positions, so they follow any layout change.
