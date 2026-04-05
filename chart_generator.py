"""
telegram/chart_generator.py
============================
Generates annotated candlestick charts for:
  1. Watchlist Alert   — shows POI zone on M15
  2. Entry Signal      — shows entry zone, SL, TP levels, OB/FVG on M15

Uses matplotlib + mplfinance with a dark professional theme.
"""

from __future__ import annotations

import io
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analysis.fvg_detector import FVG, FVGType
from analysis.liquidity import LiquidityLevel
from analysis.market_structure import MarketStructure, SwingPoint
from analysis.order_blocks import OrderBlock, OBType
from analysis.smc_engine import SetupCandidate
from config import config
from signals.entry_engine import EntrySignal
from signals.watchlist import WatchlistEntry
from utils.logger import log


# ── Theme ─────────────────────────────────────────────────────────────────────

DARK = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "text":      "#e6edf3",
    "grid":      "#21262d",
    "bull":      "#3fb950",   # Green
    "bear":      "#f85149",   # Red
    "wick":      "#8b949e",
    "bull_ob":   "#1a4a2e",
    "bull_ob_e": "#3fb950",
    "bear_ob":   "#4a1a1a",
    "bear_ob_e": "#f85149",
    "bull_fvg":  "#1a3a4a",
    "bull_fvg_e":"#58a6ff",
    "bear_fvg":  "#3a1a4a",
    "bear_fvg_e":"#d2a8ff",
    "poi":       "#f0a500",
    "entry":     "#58a6ff",
    "sl":        "#f85149",
    "tp1":       "#3fb950",
    "tp2":       "#2ea043",
    "tp3":       "#1a7f37",
    "sweep":     "#ffa657",
    "liq":       "#7d8590",
}

LIGHT = {
    "bg":        "#ffffff",
    "panel":     "#f6f8fa",
    "text":      "#24292f",
    "grid":      "#d0d7de",
    "bull":      "#1a7f37",
    "bear":      "#cf222e",
    "wick":      "#57606a",
    "bull_ob":   "#d1f0d1",
    "bull_ob_e": "#1a7f37",
    "bear_ob":   "#f7d4d4",
    "bear_ob_e": "#cf222e",
    "bull_fvg":  "#cae8ff",
    "bull_fvg_e":"#0969da",
    "bear_fvg":  "#e8d5f5",
    "bear_fvg_e":"#8250df",
    "poi":       "#d4a72c",
    "entry":     "#0969da",
    "sl":        "#cf222e",
    "tp1":       "#1a7f37",
    "tp2":       "#116329",
    "tp3":       "#044f1e",
    "sweep":     "#953800",
    "liq":       "#57606a",
}


def _theme() -> dict:
    return DARK if config.CHART_STYLE == "dark" else LIGHT


class ChartGenerator:

    def __init__(self):
        self.w = config.CHART_WIDTH_IN
        self.h = config.CHART_HEIGHT_IN
        self.dpi = config.CHART_DPI

    # ── Public API ────────────────────────────────────────────────────────────

    def watchlist_chart(
        self,
        df: pd.DataFrame,
        candidate: SetupCandidate,
        title: str = "",
    ) -> io.BytesIO:
        """Chart for when a POI is added to watchlist."""
        return self._render(
            df=df,
            title=title or f"📋 WATCHLIST — {candidate.symbol} {candidate.direction}",
            poi_top=candidate.poi_top,
            poi_bottom=candidate.poi_bottom,
            direction=candidate.direction,
            obs=candidate.m15_obs[:3],
            fvgs=candidate.m15_fvgs[:3],
            liq_levels=candidate.liq_levels[:5],
            structure=candidate.h1_structure,
        )

    def entry_chart(
        self,
        df: pd.DataFrame,
        signal: EntrySignal,
    ) -> io.BytesIO:
        """Chart for an entry signal with full levels."""
        return self._render(
            df=df,
            title=f"🎯 SNIPER ENTRY — {signal.symbol} {signal.direction}",
            poi_top=signal.entry_zone_top,
            poi_bottom=signal.entry_zone_bottom,
            direction=signal.direction,
            obs=[],
            fvgs=[],
            liq_levels=[],
            structure=None,
            entries=signal.entries,
            stop_loss=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
        )

    # ── Core Renderer ─────────────────────────────────────────────────────────

    def _render(
        self,
        df: pd.DataFrame,
        title: str,
        poi_top: float,
        poi_bottom: float,
        direction: str,
        obs: List[OrderBlock],
        fvgs: List[FVG],
        liq_levels: List[LiquidityLevel],
        structure: Optional[MarketStructure],
        entries: Optional[List[float]] = None,
        stop_loss: Optional[float] = None,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        tp3: Optional[float] = None,
    ) -> io.BytesIO:

        th = _theme()
        n = min(len(df), config.CHART_CANDLES_SHOWN)
        plot_df = df.iloc[-n:].reset_index(drop=True)

        fig, ax = plt.subplots(
            figsize=(self.w, self.h),
            facecolor=th["bg"],
        )
        ax.set_facecolor(th["panel"])

        # ── Draw candles manually ─────────────────────────────────────────────
        self._draw_candles(ax, plot_df, th)

        x_min, x_max = -0.5, len(plot_df) - 0.5
        y_vals = list(plot_df["high"]) + list(plot_df["low"])

        # ── Order Blocks ──────────────────────────────────────────────────────
        for ob in obs:
            ob_idx = ob.index - (len(df) - n)
            if ob_idx < 0:
                continue
            self._draw_zone(
                ax, 0, len(plot_df),
                ob.bottom, ob.top,
                fill_color=th["bull_ob"] if ob.ob_type == OBType.BULLISH else th["bear_ob"],
                edge_color=th["bull_ob_e"] if ob.ob_type == OBType.BULLISH else th["bear_ob_e"],
                label=f"{'Bull' if ob.ob_type == OBType.BULLISH else 'Bear'} OB",
                alpha=0.35,
            )
            y_vals += [ob.bottom, ob.top]

        # ── FVGs ─────────────────────────────────────────────────────────────
        for fvg in fvgs:
            fvg_idx = fvg.index - (len(df) - n)
            if fvg_idx < 0:
                continue
            self._draw_zone(
                ax, max(0, fvg_idx - 1), len(plot_df),
                fvg.bottom, fvg.top,
                fill_color=th["bull_fvg"] if fvg.fvg_type == FVGType.BULLISH else th["bear_fvg"],
                edge_color=th["bull_fvg_e"] if fvg.fvg_type == FVGType.BULLISH else th["bear_fvg_e"],
                label=f"{'Bull' if fvg.fvg_type == FVGType.BULLISH else 'Bear'} FVG",
                alpha=0.30,
                linestyle="--",
            )
            y_vals += [fvg.bottom, fvg.top]

        # ── Liquidity levels ──────────────────────────────────────────────────
        for liq in liq_levels[:5]:
            ax.axhline(
                y=liq.price, color=th["liq"], linewidth=0.7,
                linestyle=":", alpha=0.7,
                label=liq.label,
            )
            ax.text(
                len(plot_df) - 0.5, liq.price, f" {liq.label}",
                color=th["liq"], fontsize=7, va="bottom", ha="right",
            )
            y_vals.append(liq.price)

        # ── Structure swing points ────────────────────────────────────────────
        if structure:
            for sh in structure.swing_highs:
                sh_idx = sh.index - (len(df) - n)
                if 0 <= sh_idx < len(plot_df):
                    ax.annotate(
                        "SH", xy=(sh_idx, plot_df["high"].iloc[sh_idx]),
                        fontsize=7, color=th["bear"], ha="center", va="bottom",
                        fontweight="bold",
                    )
            for sl in structure.swing_lows:
                sl_idx = sl.index - (len(df) - n)
                if 0 <= sl_idx < len(plot_df):
                    ax.annotate(
                        "SL", xy=(sl_idx, plot_df["low"].iloc[sl_idx]),
                        fontsize=7, color=th["bull"], ha="center", va="top",
                        fontweight="bold",
                    )

        # ── POI Zone ──────────────────────────────────────────────────────────
        self._draw_zone(
            ax, 0, len(plot_df),
            poi_bottom, poi_top,
            fill_color=th["poi"],
            edge_color=th["poi"],
            label="POI Zone",
            alpha=0.20,
            linewidth=1.5,
        )
        ax.axhline(
            y=(poi_top + poi_bottom) / 2,
            color=th["poi"], linewidth=1.0, linestyle="-.", alpha=0.9,
        )
        y_vals += [poi_bottom, poi_top]

        # ── Entry / SL / TP levels ────────────────────────────────────────────
        if entries:
            for i, ep in enumerate(entries):
                ax.axhline(
                    y=ep, color=th["entry"], linewidth=0.8,
                    linestyle="--", alpha=0.8,
                )
                if i == 0:
                    ax.text(
                        1, ep, f" E{i+1}: {ep:.5f}",
                        color=th["entry"], fontsize=7, va="bottom",
                    )
            y_vals += entries

        if stop_loss:
            ax.axhline(
                y=stop_loss, color=th["sl"], linewidth=1.2,
                linestyle="-", alpha=0.9,
            )
            ax.text(
                1, stop_loss, f" SL: {stop_loss:.5f}",
                color=th["sl"], fontsize=8, va="bottom", fontweight="bold",
            )
            y_vals.append(stop_loss)

        tp_colors = [th["tp1"], th["tp2"], th["tp3"]]
        for label, price, color in [
            ("TP1", tp1, tp_colors[0]),
            ("TP2", tp2, tp_colors[1]),
            ("TP3", tp3, tp_colors[2]),
        ]:
            if price:
                ax.axhline(
                    y=price, color=color, linewidth=1.2,
                    linestyle="-", alpha=0.9,
                )
                ax.text(
                    1, price, f" {label}: {price:.5f}",
                    color=color, fontsize=8, va="bottom", fontweight="bold",
                )
                y_vals.append(price)

        # ── Axes formatting ───────────────────────────────────────────────────
        y_min = min(y_vals) * 0.9990
        y_max = max(y_vals) * 1.0010
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_facecolor(th["panel"])
        ax.grid(color=th["grid"], linewidth=0.4, linestyle="-")
        ax.spines[:].set_color(th["grid"])
        ax.tick_params(colors=th["text"], labelsize=7)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        # X-axis time labels (every 10 candles)
        tick_idx = list(range(0, len(plot_df), max(1, len(plot_df) // 8)))
        tick_labels = [
            plot_df["time"].iloc[i].strftime("%m/%d\n%H:%M")
            for i in tick_idx
        ]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, fontsize=6, color=th["text"])

        # Title
        ax.set_title(
            title, color=th["text"], fontsize=11, fontweight="bold",
            pad=8, loc="left",
        )

        # Watermark
        ax.text(
            0.99, 0.02, "GOLD HUNTER PRO",
            transform=ax.transAxes,
            color=th["text"], fontsize=8, alpha=0.25,
            ha="right", va="bottom", fontweight="bold",
        )

        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    # ── Drawing Helpers ───────────────────────────────────────────────────────

    def _draw_candles(self, ax, df: pd.DataFrame, th: dict):
        for i, row in df.iterrows():
            is_bull = row["close"] >= row["open"]
            color = th["bull"] if is_bull else th["bear"]

            # Wick
            ax.plot(
                [i, i], [row["low"], row["high"]],
                color=th["wick"], linewidth=0.7, zorder=1,
            )
            # Body
            body_bottom = min(row["open"], row["close"])
            body_top    = max(row["open"], row["close"])
            body_height = max(body_top - body_bottom, 0.0001)
            rect = mpatches.FancyBboxPatch(
                (i - 0.35, body_bottom),
                0.7, body_height,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor=color,
                linewidth=0.3,
                zorder=2,
            )
            ax.add_patch(rect)

    def _draw_zone(
        self, ax, x_start: int, x_end: int,
        y_bottom: float, y_top: float,
        fill_color: str, edge_color: str,
        label: str = "",
        alpha: float = 0.3,
        linestyle: str = "-",
        linewidth: float = 1.0,
    ):
        rect = mpatches.Rectangle(
            (x_start, y_bottom),
            x_end - x_start,
            y_top - y_bottom,
            facecolor=fill_color,
            edgecolor=edge_color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)
        if label:
            ax.text(
                x_start + 0.5, y_top, f" {label}",
                color=edge_color, fontsize=7, va="bottom",
                fontweight="bold", alpha=min(alpha * 3, 1.0),
            )


chart_generator = ChartGenerator()
