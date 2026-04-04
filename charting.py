"""
charting.py — Professional SMC Chart Generator v3 (Pro Trader Edition)

Watchlist chart (15m):
  • Candlestick + volume  |  TradingView dark theme
  • Swing structure labels: HH / HL / LH / LL dots on every swing point
  • BOS: dashed level line + vertical break marker + bold label
  • CHoCH: orange dash-dot segment + direction arrow + dot
  • Liquidity Sweeps: wick spike markers (△/▽) with swept-level annotation
  • OB boxes: extend to right edge, role tag (E-OB/OB), volume star, price labels
  • FVG boxes: dashed outline, fill-% bar, ATR-ratio tag
  • Fibonacci shading: Discount / Premium / OTE (bullish & bearish aware)
  • POI zone: coloured fill + dashed borders + mid label
  • CMP dotted line  |  Watermark  |  Legend panel

Sniper chart (5m):
  • All of the above + Entry zone (40% retrace band)
  • SL (red)  |  TP1-6 (green gradient)
  • Direction arrow  |  Risk:Reward annotation
  • Clamped ↑/↓ edge labels for out-of-view levels
"""

from __future__ import annotations

import logging
import tempfile
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import mplfinance as mpf
import numpy as np
import pandas as pd

from smc import (
    detect_swing_points, detect_bos, detect_order_blocks,
    detect_fvgs, detect_liquidity_sweeps, detect_choch,
    compute_fib_zones, is_in_discount, is_in_premium, is_in_ote,
    Zone, BOS, FibZones, SwingPoint, LiquiditySweep,
)

logger = logging.getLogger("charting")

# ─────────────────────────────────────────────────────────────────────
# TradingView dark theme
# ─────────────────────────────────────────────────────────────────────

TV_DARK = {
    "base_mpl_style": "dark_background",
    "marketcolors": mpf.make_marketcolors(
        up="#26a69a", down="#ef5350", edge="inherit", wick="inherit",
        volume={"up": "#26a69a80", "down": "#ef535080"},
    ),
    "mavcolors": ["#2962ff", "#ff6d00", "#ab47bc"],
    "facecolor": "#131722",
    "gridcolor": "#1e222d",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.edgecolor": "#2a2e39",
        "axes.labelcolor": "#787b86",
        "xtick.color":     "#787b86",
        "ytick.color":     "#787b86",
        "font.size":       9,
    },
}
TV_STYLE = mpf.make_mpf_style(**TV_DARK)

# ── Colour palette ──────────────────────────────────────────────────
C_BOS_BULL  = "#2962ff"
C_BOS_BEAR  = "#ff6d00"
C_CHOCH     = "#ff9100"
C_SWEEP_BUL = "#80d8ff"   # bullish sweep (price swept sell-side liquidity)
C_SWEEP_BEA = "#ff80ab"   # bearish sweep
C_OB_BULL   = "#ff9800"
C_OB_BEAR   = "#ab47bc"
C_FVG_BULL  = "#26a69a"
C_FVG_BEAR  = "#ef5350"
C_DISCOUNT  = "#26a69a14"
C_PREMIUM   = "#ef535014"
C_OTE_BULL  = "#ffd60022"
C_OTE_BEAR  = "#ab47bc22"
C_CMP       = "#ffffffaa"
C_ENTRY     = "#2962ff"
C_SL        = "#ef5350"
C_SH_DOT    = "#ef5350"   # swing high dot
C_SL_DOT    = "#26a69a"   # swing low dot
TP_COLORS   = ["#66bb6a", "#4caf50", "#26a69a", "#00bcd4", "#00e676", "#76ff03"]


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out.set_index("timestamp", inplace=True)
    out.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }, inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _dynamic_ylim(ohlcv: pd.DataFrame, pad_pct: float = 0.13) -> Tuple[float, float]:
    c_high = float(ohlcv["High"].max())
    c_low  = float(ohlcv["Low"].min())
    span   = c_high - c_low
    pad    = span * pad_pct
    return (c_low - pad, c_high + pad)


def _in_view(price: float, y_min: float, y_max: float) -> bool:
    return y_min <= price <= y_max


def _hline(ax, price: float, color: str, lw: float = 0.9, ls: str = "--",
           alpha: float = 0.75, x0: int = 0, x1: int = 0,
           y_min: float = 0.0, y_max: float = 1e18) -> None:
    if not _in_view(price, y_min, y_max):
        return
    ax.hlines(price, xmin=x0, xmax=x1,
              colors=color, linewidths=lw, linestyles=ls, alpha=alpha)


def _label(ax, price: float, x: int, text: str, color: str,
           fs: int = 7, va: str = "bottom",
           y_min: float = 0.0, y_max: float = 1e18,
           clamp: bool = False, c_range: float = 0.0) -> None:
    if _in_view(price, y_min, y_max):
        ax.text(x, price, text, color=color, fontsize=fs,
                fontweight="bold", va=va, ha="left", clip_on=False)
    elif clamp and c_range > 0:
        arrow = "↑" if price > y_max else "↓"
        edge  = y_max * 0.9995 if price > y_max else y_min * 1.0005
        ax.text(x, edge, f"{arrow} {text.strip()}",
                color=color, fontsize=max(fs - 1, 5), fontweight="bold",
                va="top" if price > y_max else "bottom", ha="left", clip_on=False)


def _add_watermark(ax, symbol: str, timeframe: str) -> None:
    ax.text(0.5, 0.5, f"{symbol}\n{timeframe}",
            transform=ax.transAxes, fontsize=28, color="#ffffff06",
            ha="center", va="center", fontweight="bold", zorder=0)


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: Swing structure labels  HH / HL / LH / LL
# ─────────────────────────────────────────────────────────────────────

def _draw_swing_labels(ax, swings: List[SwingPoint], offset: int, n: int,
                       y_min: float, y_max: float) -> None:
    """
    Place HH/HL/LH/LL labels on every confirmed swing point.
    Uses sequential comparison of alternating SH/SL to classify structure.
    """
    highs = [s for s in swings if s.kind == "SH"]
    lows  = [s for s in swings if s.kind == "SL"]

    # label swing highs: HH if higher than prev SH, else LH
    for i, sh in enumerate(highs):
        vis_x = sh.index - offset
        if vis_x < 0 or vis_x >= n:
            continue
        if not _in_view(sh.price, y_min, y_max):
            continue
        tag = "HH" if (i == 0 or sh.price > highs[i - 1].price) else "LH"
        ax.scatter(vis_x, sh.price, color=C_SH_DOT, s=28, zorder=6,
                   marker="^", edgecolors="#ffffff40", linewidths=0.5)
        ax.text(vis_x, sh.price, f" {tag}", color=C_SH_DOT,
                fontsize=6, fontweight="bold", va="bottom", ha="left")

    # label swing lows: HL if higher than prev SL, else LL
    for i, sl in enumerate(lows):
        vis_x = sl.index - offset
        if vis_x < 0 or vis_x >= n:
            continue
        if not _in_view(sl.price, y_min, y_max):
            continue
        tag = "LL" if (i == 0 or sl.price < lows[i - 1].price) else "HL"
        ax.scatter(vis_x, sl.price, color=C_SL_DOT, s=28, zorder=6,
                   marker="v", edgecolors="#ffffff40", linewidths=0.5)
        ax.text(vis_x, sl.price, f" {tag}", color=C_SL_DOT,
                fontsize=6, fontweight="bold", va="top", ha="left")


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: BOS — dashed level + vertical break bar + label
# ─────────────────────────────────────────────────────────────────────

def _draw_bos(ax, bos: Optional[BOS], n: int,
              y_min: float, y_max: float) -> None:
    if bos is None:
        return
    color = C_BOS_BULL if bos.direction == "bullish" else C_BOS_BEAR
    brk_x = min(bos.break_index, n - 1)
    seg0  = max(0, brk_x - 30)

    if not _in_view(bos.swing_price, y_min, y_max):
        return

    # Horizontal level line (dashed)
    ax.hlines(bos.swing_price, xmin=seg0, xmax=n - 1,
              colors=color, linewidths=1.1, linestyles="--", alpha=0.7, zorder=3)

    # Vertical break marker at the break candle
    ax.vlines(brk_x, ymin=y_min, ymax=bos.swing_price,
              colors=color, linewidths=0.5, linestyles=":", alpha=0.4)

    # Bold BOS label with background box
    lbl = "BOS ▲" if bos.direction == "bullish" else "BOS ▼"
    ax.text(brk_x + 0.5, bos.swing_price, f" {lbl}",
            color=color, fontsize=7, fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#131722cc",
                      edgecolor=color, linewidth=0.6), zorder=5)


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: CHoCH — orange dash-dot + direction arrow + dot
# ─────────────────────────────────────────────────────────────────────

def _draw_choch(ax, df_raw: pd.DataFrame, direction: str,
                n_vis: int, offset: int,
                y_min: float, y_max: float) -> None:
    try:
        lb     = 3 if len(df_raw) > 60 else 2
        swings = detect_swing_points(df_raw, lb)
        if not swings:
            return
        sweeps = detect_liquidity_sweeps(df_raw, swings, direction)
        if not sweeps:
            return
        choch = detect_choch(df_raw, sweeps[0], swings)
        if choch is None:
            return

        vis_idx = choch.choch_index - offset
        if not (0 <= vis_idx < n_vis):
            return

        price = float(df_raw["close"].iat[choch.choch_index])
        if not _in_view(price, y_min, y_max):
            return

        seg0 = max(0, vis_idx - 8)
        seg1 = min(n_vis - 1, vis_idx + 15)

        # Dash-dot segment
        ax.hlines(price, xmin=seg0, xmax=seg1,
                  colors=C_CHOCH, linewidths=1.6, linestyles="-.", alpha=0.95, zorder=4)

        # Dot at CHoCH candle
        ax.scatter(vis_idx, price, color=C_CHOCH, s=55, zorder=6,
                   marker="D", edgecolors="#ffffff70", linewidths=0.7)

        # Direction arrow above/below the dot
        dy = (y_max - y_min) * 0.025
        if direction == "bullish":
            ax.annotate("", xy=(vis_idx, price + dy * 1.8),
                        xytext=(vis_idx, price + dy * 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=C_CHOCH, lw=1.5))
        else:
            ax.annotate("", xy=(vis_idx, price - dy * 1.8),
                        xytext=(vis_idx, price - dy * 0.3),
                        arrowprops=dict(arrowstyle="-|>", color=C_CHOCH, lw=1.5))

        # Label with background box
        ax.text(seg0 + 1, price, " CHoCH",
                color=C_CHOCH, fontsize=7, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#131722cc",
                          edgecolor=C_CHOCH, linewidth=0.6), zorder=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: Liquidity Sweeps — spike markers + swept-level line
# ─────────────────────────────────────────────────────────────────────

def _draw_sweeps(ax, df_raw: pd.DataFrame, swings: List[SwingPoint],
                 direction: str, n_vis: int, offset: int,
                 y_min: float, y_max: float) -> None:
    try:
        sweeps = detect_liquidity_sweeps(df_raw, swings, direction)
        if not sweeps:
            return

        color = C_SWEEP_BUL if direction == "bullish" else C_SWEEP_BEA

        for sw in sweeps[:3]:   # draw at most 3 sweeps
            vis_x = sw.sweep_index - offset
            if not (0 <= vis_x < n_vis):
                continue

            # Swept price level (dotted line from sweep to recovery)
            rec_x = min(sw.recovery_index - offset, n_vis - 1) if sw.recovery_index else vis_x
            rec_x = max(rec_x, vis_x)

            if _in_view(sw.swept_price, y_min, y_max):
                ax.hlines(sw.swept_price, xmin=vis_x, xmax=max(rec_x, vis_x + 1),
                          colors=color, linewidths=0.8, linestyles=":", alpha=0.85, zorder=3)

            # Spike triangle marker at the wick extreme
            if direction == "bullish":
                wick_price = float(df_raw["low"].iat[sw.sweep_index])
                marker, va_lbl = "v", "top"
            else:
                wick_price = float(df_raw["high"].iat[sw.sweep_index])
                marker, va_lbl = "^", "bottom"

            if _in_view(wick_price, y_min, y_max):
                ax.scatter(vis_x, wick_price, color=color, s=80, zorder=7,
                           marker=marker, edgecolors="#ffffff60", linewidths=0.7)
                ax.text(vis_x + 0.5, wick_price, " SWEEP",
                        color=color, fontsize=6, fontweight="bold",
                        va=va_lbl,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="#131722cc",
                                  edgecolor=color, linewidth=0.5), zorder=5)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: OB boxes — extended to right edge, price labels
# ─────────────────────────────────────────────────────────────────────

def _draw_ob_boxes(ax, obs: List[Zone], n: int, offset: int,
                   y_min: float, y_max: float) -> None:
    for ob in obs:
        vis_x = ob.ob_index - offset
        if vis_x >= n:
            continue
        vis_x    = max(vis_x, 0)
        box_low  = max(ob.low,  y_min)
        box_high = min(ob.high, y_max)
        if box_high <= box_low:
            continue

        # Extend box from formation to right edge
        box_width = n - vis_x
        color = C_OB_BULL if ob.direction == "bullish" else C_OB_BEAR
        fill  = f"{color}28"

        rect = mpatches.Rectangle(
            (vis_x, box_low), box_width, box_high - box_low,
            facecolor=fill, edgecolor=color, linewidth=1.0,
            linestyle="-", zorder=2, alpha=0.9,
        )
        ax.add_patch(rect)

        # Left edge solid line (zone origin marker)
        ax.vlines(vis_x, ymin=box_low, ymax=box_high,
                  colors=color, linewidths=1.8, alpha=0.9, zorder=3)

        # Role / volume tag (top-right of box)
        role_tag = "E-OB" if ob.role == "extreme" else "D-OB"
        vol_tag  = " ✦" if ob.has_volume else ""
        mid_y    = (box_low + box_high) / 2

        ax.text(n - 1, box_high, f" {role_tag}{vol_tag}",
                color=color, fontsize=6, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#131722cc",
                          edgecolor=color, linewidth=0.5), zorder=5)

        # Price labels on right border
        if _in_view(ob.high, y_min, y_max):
            ax.text(n - 1, ob.high, f" {ob.high:.4f}",
                    color=color, fontsize=5, va="bottom", ha="right", alpha=0.7)
        if _in_view(ob.low, y_min, y_max):
            ax.text(n - 1, ob.low, f" {ob.low:.4f}",
                    color=color, fontsize=5, va="top", ha="right", alpha=0.7)


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: FVG boxes — dashed + fill-% bar inside
# ─────────────────────────────────────────────────────────────────────

def _draw_fvg_boxes(ax, fvgs: List[Zone], n: int, offset: int,
                    y_min: float, y_max: float) -> None:
    for fvg in fvgs:
        vis_x = fvg.ob_index - offset
        if vis_x >= n:
            continue
        vis_x    = max(vis_x, 0)
        box_low  = max(fvg.low,  y_min)
        box_high = min(fvg.high, y_max)
        if box_high <= box_low:
            continue

        box_width = n - vis_x
        color = C_FVG_BULL if fvg.direction == "bullish" else C_FVG_BEAR
        fill  = f"{color}18"

        rect = mpatches.Rectangle(
            (vis_x, box_low), box_width, box_high - box_low,
            facecolor=fill, edgecolor=color, linewidth=0.8,
            linestyle="--", zorder=2, alpha=0.85,
        )
        ax.add_patch(rect)

        # Fill-% bar on left side of box (visual depth indicator)
        if fvg.partial_fill_pct > 0:
            gap_size  = fvg.high - fvg.low
            fill_h    = gap_size * fvg.partial_fill_pct
            if fvg.direction == "bullish":
                fill_y = fvg.low
            else:
                fill_y = fvg.high - fill_h
            fill_y_vis = max(fill_y, y_min)
            fill_h_vis = min(fill_h, y_max - fill_y_vis)
            if fill_h_vis > 0:
                bar = mpatches.Rectangle(
                    (vis_x, fill_y_vis), 1.0, fill_h_vis,
                    facecolor=color, alpha=0.35, zorder=3,
                )
                ax.add_patch(bar)

        # Tag label
        ratio_tag  = f"{fvg.fvg_atr_ratio:.1f}×ATR" if fvg.fvg_atr_ratio > 0 else ""
        fresh_tag  = " ✦" if fvg.is_fresh else ""
        fill_lbl   = f" {fvg.partial_fill_pct * 100:.0f}%filled" if fvg.partial_fill_pct > 0 else ""
        mid_y      = (box_low + box_high) / 2

        ax.text(vis_x + 1.2, mid_y,
                f"FVG {ratio_tag}{fresh_tag}{fill_lbl}",
                color=color, fontsize=5.5, va="center", alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="#131722bb",
                          edgecolor="none"))


# ─────────────────────────────────────────────────────────────────────
# Pro drawer: Fibonacci bands — Discount / Premium / OTE
# ─────────────────────────────────────────────────────────────────────

def _draw_fib_zones(ax, fib: Optional[FibZones], n: int,
                    direction: str, y_min: float, y_max: float) -> None:
    if fib is None:
        return

    ote_color = C_OTE_BULL if direction == "bullish" else C_OTE_BEAR

    # Bearish OTE zone (mirrored from top)
    rng = fib.swing_high - fib.swing_low
    if direction == "bearish":
        bear_ote_lo = fib.swing_high - (1.0 - 0.618) * rng
        bear_ote_hi = fib.swing_high - (1.0 - 0.786) * rng
    else:
        bear_ote_lo = fib.ote_low
        bear_ote_hi = fib.ote_high

    bands = [
        (fib.swing_low,    fib.discount_upper, C_DISCOUNT, "Discount 0–50%"),
        (fib.premium_lower, fib.swing_high,    C_PREMIUM,  "Premium 50–100%"),
        (bear_ote_lo,       bear_ote_hi,       ote_color,  "OTE"),
    ]
    for lo, hi, color, lbl in bands:
        vis_lo = max(lo, y_min)
        vis_hi = min(hi, y_max)
        if vis_hi <= vis_lo:
            continue
        ax.axhspan(vis_lo, vis_hi, color=color, zorder=0)
        ax.text(0.01, (vis_lo + vis_hi) / 2, lbl,
                color="#ffffff50", fontsize=5.5, va="center", ha="left",
                transform=ax.get_yaxis_transform(), zorder=1)

    # EQ 50% line
    eq = (fib.swing_low + fib.swing_high) / 2
    if _in_view(eq, y_min, y_max):
        ax.axhline(eq, color="#ffffff20", linewidth=0.6, linestyle=":", zorder=1)
        ax.text(n - 2, eq, "  EQ 50%",
                color="#ffffff40", fontsize=5.5, va="bottom")


# ─────────────────────────────────────────────────────────────────────
# CMP line
# ─────────────────────────────────────────────────────────────────────

def _draw_cmp(ax, cmp: float, n: int, y_min: float, y_max: float) -> None:
    if not _in_view(cmp, y_min, y_max):
        return
    ax.axhline(cmp, color=C_CMP, linewidth=0.8, linestyle=":", alpha=0.55, zorder=4)
    ax.text(n - 1, cmp, f"  {cmp:.5f}",
            color=C_CMP, fontsize=6.5, va="bottom", ha="left", fontweight="bold")


# ─────────────────────────────────────────────────────────────────────
# Legend panel
# ─────────────────────────────────────────────────────────────────────

def _draw_legend(ax, direction: str) -> None:
    handles = [
        mlines.Line2D([], [], color=C_BOS_BULL if direction == "bullish" else C_BOS_BEAR,
                      lw=1.2, linestyle="--", label="BOS"),
        mlines.Line2D([], [], color=C_CHOCH, lw=1.2, linestyle="-.", label="CHoCH"),
        mpatches.Patch(facecolor=C_OB_BULL + "40", edgecolor=C_OB_BULL, label="Bullish OB"),
        mpatches.Patch(facecolor=C_OB_BEAR + "40", edgecolor=C_OB_BEAR, label="Bearish OB"),
        mpatches.Patch(facecolor=C_FVG_BULL + "30", edgecolor=C_FVG_BULL,
                       linestyle="--", label="Bullish FVG"),
        mpatches.Patch(facecolor=C_FVG_BEAR + "30", edgecolor=C_FVG_BEAR,
                       linestyle="--", label="Bearish FVG"),
        mlines.Line2D([], [], color=C_SWEEP_BUL, lw=0, marker="v",
                      markersize=6, label="Bull Sweep"),
        mlines.Line2D([], [], color=C_SWEEP_BEA, lw=0, marker="^",
                      markersize=6, label="Bear Sweep"),
        mlines.Line2D([], [], color=C_SH_DOT, lw=0, marker="^",
                      markersize=5, label="SH (HH/LH)"),
        mlines.Line2D([], [], color=C_SL_DOT, lw=0, marker="v",
                      markersize=5, label="SL (HL/LL)"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=5.5,
                    framealpha=0.6, facecolor="#1a1e2e", edgecolor="#2a2e39",
                    labelcolor="white", ncol=2, borderpad=0.5,
                    handlelength=1.5, handletextpad=0.4, columnspacing=0.8)
    leg.set_zorder(10)


# ─────────────────────────────────────────────────────────────────────
# Watchlist Chart — 15m full SMC markup
# ─────────────────────────────────────────────────────────────────────

def generate_watchlist_chart(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    poi_high: float,
    poi_low: float,
    score: int = 0,
    tail_candles: int = 80,
    timeframe: str = "15m",
) -> Optional[str]:
    try:
        raw_df = df.copy()
        ohlcv  = _prepare_df(df).tail(tail_candles)
        if len(ohlcv) < 10:
            return None

        n      = len(ohlcv)
        offset = max(len(raw_df) - tail_candles, 0)
        cmp    = float(raw_df["close"].iat[-1])

        y_min, y_max = _dynamic_ylim(ohlcv)
        c_range      = y_max - y_min

        dir_arrow = "▲" if direction == "bullish" else "▼"
        fig, axes = mpf.plot(
            ohlcv, type="candle", style=TV_STYLE, volume=True,
            title=(f"\n{symbol}  |  {timeframe}  |  Score: {score}/100"
                   f"  |  {direction.upper()} {dir_arrow}"),
            figsize=(16, 8), returnfig=True, tight_layout=True,
            ylim=(y_min, y_max),
        )
        ax = axes[0]

        lb     = 3
        swings = detect_swing_points(raw_df, lb)
        bos    = detect_bos(raw_df, swings, lb)
        obs    = detect_order_blocks(raw_df, direction, swings, lb)
        fvgs   = detect_fvgs(raw_df, direction)
        fib    = compute_fib_zones(swings)

        # Draw order: back → front
        _draw_fib_zones(ax, fib, n, direction, y_min, y_max)
        _draw_fvg_boxes(ax, fvgs, n, offset, y_min, y_max)
        _draw_ob_boxes(ax, obs, n, offset, y_min, y_max)
        _draw_bos(ax, bos, n, y_min, y_max)
        _draw_sweeps(ax, raw_df, swings, direction, n, offset, y_min, y_max)
        _draw_choch(ax, raw_df, direction, n, offset, y_min, y_max)
        _draw_swing_labels(ax, swings, offset, n, y_min, y_max)
        _draw_cmp(ax, cmp, n, y_min, y_max)
        _add_watermark(ax, symbol, timeframe)
        _draw_legend(ax, direction)

        # POI zone (topmost layer)
        poi_fill   = "#26a69a38" if direction == "bullish" else "#ef535038"
        poi_border = "#26a69a"   if direction == "bullish" else "#ef5350"
        vis_lo = max(poi_low,  y_min)
        vis_hi = min(poi_high, y_max)
        if vis_hi > vis_lo:
            ax.axhspan(vis_lo, vis_hi, color=poi_fill, zorder=3)
        _hline(ax, poi_high, poi_border, lw=1.1, ls="--", alpha=0.85,
               x0=0, x1=n - 1, y_min=y_min, y_max=y_max)
        _hline(ax, poi_low, poi_border, lw=1.1, ls="--", alpha=0.85,
               x0=0, x1=n - 1, y_min=y_min, y_max=y_max)
        poi_mid = (poi_high + poi_low) / 2
        if _in_view(poi_mid, y_min, y_max):
            ax.text(n - 1, poi_mid, "  ◀ HTF POI",
                    color=poi_border, fontsize=9, fontweight="bold",
                    va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#131722cc",
                              edgecolor=poi_border, linewidth=0.8), zorder=6)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="wl1_", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Watchlist chart saved: %s", tmp.name)
        return tmp.name

    except Exception as exc:
        logger.error("Watchlist chart failed: %s", exc, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────
# Sniper Chart — 5m full SMC markup + entry/SL/TP levels
# ─────────────────────────────────────────────────────────────────────

def generate_sniper_chart(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    entry_high: float,
    entry_low: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    tp4: float = 0.0,
    tp5: float = 0.0,
    tp6: float = 0.0,
    tail_candles: int = 60,
    timeframe: str = "5m",
) -> Optional[str]:
    try:
        raw_df  = df.copy()
        ohlcv   = _prepare_df(df).tail(tail_candles)
        if len(ohlcv) < 10:
            return None

        n       = len(ohlcv)
        offset  = max(len(raw_df) - tail_candles, 0)
        smc_dir = "bullish" if direction == "LONG" else "bearish"
        cmp     = float(raw_df["close"].iat[-1])
        lx      = n - 1

        y_min, y_max = _dynamic_ylim(ohlcv)
        c_range      = y_max - y_min

        fig, axes = mpf.plot(
            ohlcv, type="candle", style=TV_STYLE, volume=True,
            title=f"\n{symbol}  |  {timeframe}  |  {direction}  ◀ SNIPER ENTRY",
            figsize=(16, 8), returnfig=True, tight_layout=True,
            ylim=(y_min, y_max),
        )
        ax = axes[0]

        lb     = 2
        swings = detect_swing_points(raw_df, lb)
        bos    = detect_bos(raw_df, swings, lb)
        obs    = detect_order_blocks(raw_df, smc_dir, swings, lb)
        fvgs   = detect_fvgs(raw_df, smc_dir)
        fib    = compute_fib_zones(swings)

        # SMC layers (back → front)
        _draw_fib_zones(ax, fib, n, smc_dir, y_min, y_max)
        _draw_fvg_boxes(ax, fvgs, n, offset, y_min, y_max)
        _draw_ob_boxes(ax, obs, n, offset, y_min, y_max)
        _draw_bos(ax, bos, n, y_min, y_max)
        _draw_sweeps(ax, raw_df, swings, smc_dir, n, offset, y_min, y_max)
        _draw_choch(ax, raw_df, smc_dir, n, offset, y_min, y_max)
        _draw_swing_labels(ax, swings, offset, n, y_min, y_max)
        _draw_cmp(ax, cmp, n, y_min, y_max)
        _add_watermark(ax, symbol, timeframe)
        _draw_legend(ax, smc_dir)

        # ── Entry zone (40% retrace) ──────────────────────────────────────────────
        entry_mid = (entry_high + entry_low) / 2
        vis_lo_e  = max(entry_low,  y_min)
        vis_hi_e  = min(entry_high, y_max)
        if vis_hi_e > vis_lo_e:
            ax.axhspan(vis_lo_e, vis_hi_e, color="#2962ff22", zorder=4)

        _hline(ax, entry_mid, C_ENTRY, lw=2.0, ls="-", alpha=0.95,
               x0=0, x1=lx, y_min=y_min, y_max=y_max)
        _hline(ax, entry_high, C_ENTRY, lw=0.7, ls=":", alpha=0.45,
               x0=0, x1=lx, y_min=y_min, y_max=y_max)
        _hline(ax, entry_low, C_ENTRY, lw=0.7, ls=":", alpha=0.45,
               x0=0, x1=lx, y_min=y_min, y_max=y_max)
        _label(ax, entry_mid, lx, f"  ENTRY {entry_mid:.5f}",
               C_ENTRY, 8, "center", y_min, y_max, clamp=True, c_range=c_range)

        # ── SL ──────────────────────────────────────────────────────
        _hline(ax, stop_loss, C_SL, lw=1.5, ls="--", alpha=0.95,
               x0=0, x1=lx, y_min=y_min, y_max=y_max)
        _label(ax, stop_loss, lx, f"  ✕ SL {stop_loss:.5f}",
               C_SL, 7, "bottom", y_min, y_max, clamp=True, c_range=c_range)

        # Risk distance annotation
        risk = abs(entry_mid - stop_loss)
        if risk > 0 and _in_view(stop_loss, y_min, y_max) and _in_view(entry_mid, y_min, y_max):
            sl_y  = min(stop_loss, entry_mid)
            ax.annotate("", xy=(lx - 4, entry_mid), xytext=(lx - 4, stop_loss),
                        arrowprops=dict(arrowstyle="<->", color=C_SL, lw=0.9, alpha=0.6))
            ax.text(lx - 3.5, (entry_mid + stop_loss) / 2, "Risk",
                    color=C_SL, fontsize=5.5, va="center", alpha=0.8)

        # ── TPs ─────────────────────────────────────────────────────
        tp_specs = [
            (tp1, "TP1 1:1"), (tp2, "TP2 1:2"), (tp3, "TP3 1:3"),
            (tp4, "TP4 1:4"), (tp5, "TP5 1:6"), (tp6, "TP6 Run"),
        ]
        for idx, (price, lbl) in enumerate(tp_specs):
            if not price:
                continue
            color = TP_COLORS[idx]
            ls    = "-." if idx % 2 == 0 else "--"
            _hline(ax, price, color, lw=0.9, ls=ls, alpha=0.85,
                   x0=0, x1=lx, y_min=y_min, y_max=y_max)
            _label(ax, price, lx, f"  🎯 {lbl} {price:.5f}",
                   color, 6, "bottom", y_min, y_max, clamp=True, c_range=c_range)

        # ── R:R annotation (TP2 used as reference) ──────────────────
        rr = round(abs(tp2 - entry_mid) / risk, 2) if risk > 0 and tp2 else 0
        ax.text(0.99, 0.97,
                f"R:R (TP2) = 1:{rr}   Risk = {risk:.5f}   Dir = {direction}",
                transform=ax.transAxes, fontsize=7, color="#ffffff99",
                ha="right", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722cc",
                          edgecolor="#2a2e39", linewidth=0.7))

        # ── Direction arrow (right margin) ───────────────────────────
        a_color = "#26a69a" if direction == "LONG" else "#ef5350"
        dy      = c_range * 0.12
        if _in_view(entry_mid, y_min, y_max):
            dest = entry_mid + (dy if direction == "LONG" else -dy)
            if _in_view(dest, y_min, y_max):
                ax.annotate("", xy=(lx + 0.5, dest), xytext=(lx + 0.5, entry_mid),
                            arrowprops=dict(arrowstyle="-|>", color=a_color, lw=2.5),
                            annotation_clip=False)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="sniper_", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Sniper chart saved: %s", tmp.name)
        return tmp.name

    except Exception as exc:
        logger.error("Sniper chart failed: %s", exc, exc_info=True)
        return None
