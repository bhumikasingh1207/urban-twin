"""Planning insights and downloadable report. Every sentence is computed from the project's own
outputs - nothing is hand-written or hard-coded."""
from __future__ import annotations

import html
from datetime import date

import pandas as pd


def make_insights(area: pd.DataFrame, trans: pd.DataFrame, grid: pd.DataFrame, years: tuple[int, int]) -> list[str]:
    ya, yb = years
    a = area[area.year == ya].set_index("class")
    b = area[area.year == yb].set_index("class")
    out = []
    ba, bb = a.loc["Built-up", "area_km2"], b.loc["Built-up", "area_km2"]
    if ba > 0:
        out.append(f"Built-up area changed from {ba:.1f} km2 ({ya}) to {bb:.1f} km2 ({yb}): "
                   f"{bb - ba:+.1f} km2 ({100 * (bb - ba) / ba:+.1f}%).")
    for cls in ("Cropland", "Vegetation"):
        to_built = trans.loc[cls, "Built-up"]
        if to_built > 0:
            out.append(f"{to_built:.1f} km2 of {cls.lower()} (as classified in {ya}) became built-up by {yb}.")
    gv = (a.loc["Vegetation", "area_km2"], b.loc["Vegetation", "area_km2"])
    out.append(f"Vegetation class area: {gv[0]:.1f} km2 -> {gv[1]:.1f} km2 ({gv[1] - gv[0]:+.1f} km2).")
    if len(grid):
        top = grid.sort_values("gain_km2", ascending=False).head(3)
        cells = ", ".join(f"cell ({int(r.row)},{int(r.col)}): +{r.gain_km2:.2f} km2" for r in top.itertuples())
        out.append(f"Fastest-changing grid cells (largest new built-up area): {cells}.")
    out.append("These figures come from classifier output, so they include classification error; "
               "see the Model evaluation page for accuracy.")
    return out


def build_html_report(title: str, insights: list[str], tables: dict[str, pd.DataFrame], metrics_note: str,
                      synthetic: bool) -> str:
    banner = ('<p style="color:#b00;font-weight:bold">SYNTHETIC TEST DATA - results are not real.</p>'
              if synthetic else "")
    parts = [f"<h1>{html.escape(title)}</h1><p>Generated {date.today().isoformat()} by the Urban Digital Twin prototype.</p>",
             banner, "<h2>Insights</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in insights) + "</ul>",
             f"<h2>Model evaluation</h2><p>{html.escape(metrics_note)}</p>"]
    for name, df in tables.items():
        parts.append(f"<h2>{html.escape(name)}</h2>" + df.to_html(index=False, float_format=lambda v: f"{v:.3f}"))
    parts.append("<p><i>Prototype: a data-driven virtual representation built from satellite data; "
                 "it has no real-time link to the physical city.</i></p>")
    return ("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:auto}"
            "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style></head><body>"
            + "".join(parts) + "</body></html>")
