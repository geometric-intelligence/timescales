"""Shared scaffolding for the self-contained interactive HTML reports.

Used by ``8_interactive_report.py`` (sine wave) and ``9_flipflop_report.py``. Holds the
page template, the theme-matching script, and the small plotly helpers both need, so the
two reports stay visually consistent and the styling lives in one place.
"""

import numpy as np
from plotly.offline import get_plotlyjs

COLOR = {"linear": "#2a78d6", "tanh": "#eb6834"}
C_REF, C_GRID = "#898781", "#d8d8d2"
CYCLIC = "Twilight"          # phase is cyclic; a sequential scale would fake a seam
SEQ = "Viridis"
DIVERGING = "RdBu"
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def base_layout(title, height, margin):
    return dict(
        title=dict(text=title, font=dict(size=15)),
        height=height, margin=margin,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12),
    )


def layout3d(title, axis_titles, height=560):
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=C_GRID,
                showbackground=True, zeroline=False)
    out = dict(
        **base_layout(title, height, dict(l=0, r=0, t=92, b=0)),
        scene=dict(xaxis=dict(title=axis_titles[0], **axis),
                   yaxis=dict(title=axis_titles[1], **axis),
                   zaxis=dict(title=axis_titles[2], **axis)),
    )
    # These panels carry a colouring switcher whose buttons live in the top margin. Pin the
    # title to the top-left of the container so the two stack instead of overlapping — a
    # centred title collides with the button row once there are more than a few buttons.
    out["title"] = dict(text=title, font=dict(size=15), xref="container",
                        yref="container", x=0.012, y=0.975, xanchor="left",
                        yanchor="top")
    return out


def marker(vals, scale, cbar, size, show_scale):
    return dict(size=size, color=vals, colorscale=scale, showscale=show_scale,
                cmin=float(np.min(vals)), cmax=float(np.max(vals)),
                colorbar=dict(title=dict(text=cbar), thickness=12, len=0.72))


def vis_switcher(colorings, n_groups, n_extra=0):
    """Buttons that toggle which colouring is visible.

    Restyling ``marker.colorscale`` per trace is ambiguous — plotly cannot tell a list of
    colourscale *names* from a single colourscale *definition* — so each colouring gets its
    own trace and the buttons switch visibility instead. Traces are laid out group-major:
    [g0c0, g0c1, …, g1c0, …] followed by ``n_extra`` always-visible traces.
    """
    n_c = len(colorings)
    buttons = []
    for ci, (label, *_rest) in enumerate(colorings):
        vis = [(c == ci) for _g in range(n_groups) for c in range(n_c)]
        vis += [True] * n_extra
        buttons.append(dict(label=label, method="update", args=[{"visible": vis}]))
    return [dict(type="buttons", direction="right", showactive=True,
                 x=0, y=1.06, xanchor="left", yanchor="top",
                 pad=dict(t=0, r=6), buttons=buttons,
                 font=dict(size=11), bgcolor="rgba(0,0,0,0.04)")]


def slug(title):
    return "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-")


def build_sections(sections):
    """(title, blurb, [fig, …]) triples -> (sections_html, toc_html)."""
    divs, toc = [], []
    for title, blurb, figs in sections:
        sid = slug(title)
        toc.append(f'<a href="#{sid}">{title}</a>')
        plots = "".join(
            f'<div class="plot">'
            f'{f.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)}'
            f'</div>' for f in figs)
        divs.append(f'<section id="{sid}">\n  <h2>{title}</h2>\n'
                    f'  <p class="blurb">{blurb}</p>\n  {plots}\n</section>')
    return "".join(divs), "".join(toc)


_STYLE = """
  :root { color-scheme: light dark;
    --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:rgba(11,11,11,0.10);
    --card:#ffffff; }
  @media (prefers-color-scheme: dark) { :root {
    --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --line:rgba(255,255,255,0.12);
    --card:#232322; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.55; }
  .wrap { max-width:1180px; margin:0 auto; padding:32px 22px 64px; }
  header h1 { font-size:26px; margin:0 0 6px; }
  header p { color:var(--muted); margin:0 0 4px; max-width:80ch; }
  .meta { display:flex; flex-wrap:wrap; gap:10px; margin:18px 0 4px; }
  .chip { font-size:12px; color:var(--muted); border:1px solid var(--line);
    border-radius:999px; padding:4px 11px; }
  nav { display:flex; flex-wrap:wrap; gap:8px 16px; margin:22px 0 0;
    padding-top:16px; border-top:1px solid var(--line); font-size:13px; }
  nav a { color:var(--muted); text-decoration:none;
    border-bottom:1px solid transparent; }
  nav a:hover { color:var(--fg); border-bottom-color:var(--line); }
  section { margin-top:40px; border-top:1px solid var(--line); padding-top:22px;
    scroll-margin-top:16px; }
  h2 { font-size:18px; margin:0 0 6px; }
  .blurb { color:var(--muted); max-width:88ch; margin:0 0 14px; font-size:14px; }
  .plot { background:var(--card); border:1px solid var(--line); border-radius:10px;
    padding:6px; margin-bottom:14px; overflow-x:auto; }
  footer { margin-top:44px; color:var(--muted); font-size:13px;
    border-top:1px solid var(--line); padding-top:16px; }
  code { font-size:12.5px; }
"""

# Plotly bakes its colours in at export time and cannot see the page theme, so restyle every
# figure to match whichever scheme the viewer is actually using, and again if they switch.
# Axis keys are read off each figure's own layout rather than assumed.
_SCRIPT = r"""
(function () {
  function applyTheme() {
    var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var fg = dark ? '#e9e9e3' : '#0b0b0b';
    var grid = dark ? '#3a3a38' : '#d8d8d2';
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      if (!gd.layout) return;
      var up = { 'font.color': fg };
      Object.keys(gd.layout).forEach(function (k) {
        if (/^[xy]axis\d*$/.test(k)) {
          up[k + '.gridcolor'] = grid;
          up[k + '.color'] = fg;
        }
      });
      if (gd.layout.scene) {
        ['xaxis', 'yaxis', 'zaxis'].forEach(function (a) {
          up['scene.' + a + '.gridcolor'] = grid;
          up['scene.' + a + '.color'] = fg;
          up['scene.' + a + '.backgroundcolor'] = 'rgba(0,0,0,0)';
        });
      }
      if (gd.layout.updatemenus) {
        up['updatemenus[0].bgcolor'] = dark ? 'rgba(255,255,255,0.10)'
                                            : 'rgba(0,0,0,0.05)';
        up['updatemenus[0].font.color'] = fg;
      }
      try { Plotly.relayout(gd, up); } catch (e) { /* figure without those keys */ }
    });
  }
  window.addEventListener('load', applyTheme);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);

  // The browser honours a #hash before the plots have sized themselves, so the figures
  // below it then grow and push the target off screen. Re-apply the hash once layout
  // has settled.
  function reanchor() {
    if (!location.hash) return;
    var el = document.getElementById(location.hash.slice(1));
    if (el) el.scrollIntoView();
  }
  window.addEventListener('load', function () {
    reanchor();
    setTimeout(reanchor, 300);
  });
})();
"""


def page(title, intro, chips, sections, source):
    """Assemble the full standalone HTML document."""
    body, toc = build_sections(sections)
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    return (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{_STYLE}</style>\n"
        f"<script>{get_plotlyjs()}</script>\n"
        '<div class="wrap">\n<header>\n'
        f"  <h1>{title}</h1>\n  <p>{intro}</p>\n"
        f'  <div class="meta">{chip_html}</div>\n'
        f"  <nav>{toc}</nav>\n</header>\n"
        f"{body}\n"
        "<footer>\n"
        f"  Generated by <code>{source}</code>.\n"
        "  Findings and their caveats are written up in <code>docs/geometry_notes.md</code>.\n"
        "</footer>\n</div>\n"
        f"<script>{_SCRIPT}</script>\n"
    )
