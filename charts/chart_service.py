from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import EntrySignal, SetupSignal, WatchlistItem


class ChartService:
    def __init__(self, out_dir: str = "artifacts/charts") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def render(self, symbol: str, candles: list[list[float]], watch: WatchlistItem, setup: SetupSignal | None = None, entry: EntrySignal | None = None) -> Path:
        fig = make_subplots(rows=1, cols=1)
        x = [c[0] for c in candles]
        fig.add_trace(
            go.Candlestick(x=x, open=[c[1] for c in candles], high=[c[2] for c in candles], low=[c[3] for c in candles], close=[c[4] for c in candles], name=symbol),
            row=1,
            col=1,
        )
        fig.add_hrect(y0=watch.range_low, y1=watch.range_high, line_width=1, fillcolor="RoyalBlue", opacity=0.12, annotation_text="4H Range")
        fig.add_hline(y=watch.midpoint, line_dash="dot", line_color="orange", annotation_text="Midpoint")
        if setup:
            fig.add_hrect(y0=setup.entry_zone_low, y1=setup.entry_zone_high, line_width=0, fillcolor="gold", opacity=0.15, annotation_text="5M reaction zone")
        if entry:
            tp_price = entry.entry_price + (0.40 / 1.0) if entry.side == "BUY" else entry.entry_price - (0.40 / 1.0)
            fig.add_hline(y=entry.entry_price, line_color="white", annotation_text="Entry")
            fig.add_hline(y=entry.sl_price, line_color="red", annotation_text="SL")
            fig.add_hline(y=tp_price, line_color="green", annotation_text="TP ($0.40)")

        fig.update_layout(template="plotly_dark", title=f"{symbol} | Sweep: {watch.sweep_side} | Bias: {watch.bias}")
        path = self.out_dir / f"{symbol.replace('/', '_')}.png"
        fig.write_image(str(path), width=1400, height=800, scale=2)
        return path
