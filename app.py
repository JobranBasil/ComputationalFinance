"""
To Hide or Not to Hide — Darkpool Simulation
solara run app.py
"""

from __future__ import annotations
import threading, time
import solara
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from model import MarketModel

# ======================================================================
#  PALETTE — Robinhood white
# ======================================================================
WHITE    = "#ffffff"
BG       = "#f5f6f8"
SIDEBAR  = "#ffffff"
CARD     = "#ffffff"
BORDER   = "#e3e5ea"
TEXT     = "#1a1a2e"
LABEL    = "#5c6178"
HINT     = "#9ca3b0"
DIVIDER  = "#eeeff2"
AX_BG    = "#f8f9fb"       # chart plot area — just off-white

GREEN    = "#00c805"
RED      = "#ff5000"
CYAN     = "#0088ff"
PURPLE   = "#7b61ff"
AMBER    = "#ff9f1c"
PINK     = "#e84393"
TEAL     = "#00b894"

FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Helvetica, Arial, sans-serif"

# -- matplotlib ---------------------------------------------------------
TICK_C  = "#5c6178"
TITLE_C = "#2d3142"

plt.rcParams.update({
    "figure.facecolor":   "none",
    "axes.facecolor":     AX_BG,
    "axes.edgecolor":     "#dcdee5",
    "axes.labelcolor":    TICK_C,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.color":        TICK_C,
    "ytick.color":        TICK_C,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "text.color":         TEXT,
    "grid.color":         "#e8e9ef",
    "grid.linestyle":     "-",
    "grid.alpha":         0.7,
    "grid.linewidth":     0.5,
    "legend.facecolor":   AX_BG,
    "legend.edgecolor":   "none",
    "legend.fontsize":    8,
    "legend.labelcolor":  TICK_C,
    "font.family":        "sans-serif",
    "font.size":          8.5,
    "figure.dpi":         110,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

# -- CSS ---------------------------------------------------------------
DARK_CSS = f"""
<style>
html, body, .v-application, .v-main, .v-main__wrap {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: {FONT} !important;
}}
.v-application .text--primary {{ color: {TEXT} !important; }}
.theme--light.v-application {{ color: {TEXT} !important; background: {BG} !important; }}
.theme--light.v-sheet {{ background-color: transparent !important; color: {TEXT} !important; }}
.v-label, .v-input__slot label, .v-slider .v-label,
.v-messages__message, .v-text-field__slot input,
.v-input .v-label, .theme--light .v-label,
.theme--light.v-label, .theme--light .v-input,
.solara-slider-label, .solara-slider-value {{
    color: {TEXT} !important;
    font-family: {FONT} !important;
    font-size: 12px !important;
}}
.v-card, .v-sheet.v-card {{ background-color: {CARD} !important; color: {TEXT} !important; }}
.v-input, .v-input input {{ color: {TEXT} !important; }}
.v-slider__thumb {{ background-color: {GREEN} !important; }}
.v-slider__track-fill {{ background-color: {GREEN} !important; }}
.v-slider__track-background {{ background-color: {BORDER} !important; }}
.v-slider__thumb-label {{ color: {WHITE} !important; background: {GREEN} !important; }}
.v-btn.primary {{
    background-color: {GREEN} !important; color: {WHITE} !important;
    border-radius: 20px !important; text-transform: none !important;
    font-weight: 600 !important; box-shadow: none !important;
}}
.v-btn--contained.error {{
    background-color: {RED} !important; color: {WHITE} !important;
    border-radius: 20px !important; text-transform: none !important;
    box-shadow: none !important;
}}
.v-progress-linear__determinate {{ background-color: {GREEN} !important; }}
.v-progress-linear__background {{ background-color: {BORDER} !important; opacity: 1 !important; }}
.v-messages {{ min-height: 0 !important; }}
.v-input__slot {{ margin-bottom: 0 !important; }}
.v-text-field__details {{ display: none !important; }}
.solara-banner, .solara-powered-by,
a[href*="solara.dev"], .v-toolbar__content a {{
    display: none !important;
}}
</style>
"""


# ======================================================================
#  HELPERS
# ======================================================================

def _s(vals):
    return np.array([v if v is not None else np.nan for v in vals], dtype=float)

def _fill(ax, x, y1, y2=0, **kw):
    y1 = np.asarray(y1, dtype=float)
    if isinstance(y2, (list, tuple, np.ndarray)):
        y2 = np.asarray(y2, dtype=float)
        mask = np.isfinite(y1) & np.isfinite(y2)
    else:
        mask = np.isfinite(y1)
    if mask.any():
        ax.fill_between(x, y1, y2, **kw)

def _rolling_vol(mids, window=20):
    arr = np.array(mids, dtype=float)
    lr = np.diff(np.log(arr, where=arr > 0, out=np.full_like(arr, np.nan)))
    vol = np.full(len(arr), np.nan)
    for i in range(window, len(lr)):
        c = lr[i - window:i]; v = c[np.isfinite(c)]
        vol[i + 1] = np.std(v) if len(v) > 2 else np.nan
    return vol


# ======================================================================
#  STATE
# ======================================================================
steps                  = solara.reactive(500)
sigma_v                = solara.reactive(0.02)
start_price            = solara.reactive(100.0)

noise_count            = solara.reactive(3)
noise_participation    = solara.reactive(0.90)
noise_market_prob      = solara.reactive(0.50)

institutional_count    = solara.reactive(2)
institutional_part     = solara.reactive(0.05)
iceberg_prob           = solara.reactive(0.80)
inst_dark_frac         = solara.reactive(0.20)

informed_count         = solara.reactive(1)
informed_participation = solara.reactive(0.50)
informed_sigma_s       = solara.reactive(0.08)
informed_dark_frac     = solara.reactive(0.20)

mmas_gamma             = solara.reactive(0.10)
mmas_kappa             = solara.reactive(50.0)
mmas_sigma             = solara.reactive(0.05)
mmas_horizon           = solara.reactive(5000.0)

tick_speed             = solara.reactive(10)
sim_seed               = solara.reactive(42)

live_history  = solara.reactive([])
live_snapshot = solara.reactive(None)
live_tick     = solara.reactive(0)
is_running    = solara.reactive(False)
is_done       = solara.reactive(False)
_stop = threading.Event()

def _run():
    _stop.clear()
    is_running.set(True); is_done.set(False)
    live_history.set([]); live_snapshot.set(None); live_tick.set(0)
    m = MarketModel(
        steps=steps.value, sigma_v=sigma_v.value,
        start_price=start_price.value,
        noise_count=noise_count.value,
        noise_participation=noise_participation.value,
        noise_market_prob=noise_market_prob.value,
        institutional_count=institutional_count.value,
        iceberg_prob=iceberg_prob.value,
        institutional_participation=institutional_part.value,
        inst_dark_fraction=inst_dark_frac.value,
        informed_count=informed_count.value,
        mmas_gamma=mmas_gamma.value, mmas_kappa=mmas_kappa.value,
        mmas_sigma=mmas_sigma.value, mmas_horizon=mmas_horizon.value,
        informed_participation=informed_participation.value,
        informed_sigma_s=informed_sigma_s.value,
        dark_fraction=informed_dark_frac.value,
        seed=sim_seed.value,
    )
    total = steps.value; batch = max(1, tick_speed.value)
    for t in range(total):
        if _stop.is_set(): break
        m.step()
        if (t+1) % batch == 0 or t == total - 1:
            live_history.set(list(m.history))
            live_snapshot.set(m.latest_snapshot)
            live_tick.set(t + 1)
            time.sleep(0.01)
    is_running.set(False); is_done.set(True)

def start_sim():
    if not is_running.value:
        threading.Thread(target=_run, daemon=True).start()

def stop_sim():
    _stop.set()


# ======================================================================
#  LABELED SLIDERS
# ======================================================================

@solara.component
def LS_Int(label, value, min, max, step=1):
    with solara.Row(style={"align-items": "center", "gap": "0", "margin": "1px 0"}):
        solara.SliderInt(label, value=value, min=min, max=max, step=step)
        solara.Text(f"{value.value}", style={
            "color": GREEN, "font-size": "13px", "font-weight": "700",
            "min-width": "36px", "text-align": "right",
        })

@solara.component
def LS_Float(label, value, min, max, step=0.01):
    v = value.value
    if step >= 1: txt = f"{v:.0f}"
    elif step >= 0.1: txt = f"{v:.1f}"
    elif step >= 0.01: txt = f"{v:.2f}"
    else: txt = f"{v:.3f}"
    with solara.Row(style={"align-items": "center", "gap": "0", "margin": "1px 0"}):
        solara.SliderFloat(label, value=value, min=min, max=max, step=step)
        solara.Text(txt, style={
            "color": GREEN, "font-size": "13px", "font-weight": "700",
            "min-width": "44px", "text-align": "right",
        })

@solara.component
def LS_Pct(label, value, min=0.0, max=1.0, step=0.05):
    """Slider that shows value as percentage."""
    with solara.Row(style={"align-items": "center", "gap": "0", "margin": "1px 0"}):
        solara.SliderFloat(label, value=value, min=min, max=max, step=step)
        solara.Text(f"{value.value:.0%}", style={
            "color": GREEN, "font-size": "13px", "font-weight": "700",
            "min-width": "44px", "text-align": "right",
        })


# ======================================================================
#  CHART CARD STYLE
# ======================================================================

CARD_STYLE = {
    "background": CARD,
    "border": f"1px solid {BORDER}",
    "border-radius": "14px",
    "padding": "14px 14px 6px 14px",
    "flex": "1",
    "min-width": "0",
    "box-shadow": "0 1px 4px rgba(0,0,0,0.06)",
}


# ======================================================================
#  CHARTS
# ======================================================================
FW = 6.0
FH = 2.4

def _fin(fig, ax):
    ax.grid(True, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=8.5)
    fig.tight_layout(pad=0.8)

@solara.component
def ChartPrice(history):
    fig, ax = plt.subplots(figsize=(FW, FH))
    ts   = [h["t"] for h in history]
    mids = _s([h["mid"] for h in history])
    fund = [h["fundamental"] for h in history]
    ax.plot(ts, mids, color=GREEN, lw=1.4, label="Mid Price")
    ax.plot(ts, fund, color=AMBER, lw=1.1, ls="--", label="Fundamental", alpha=0.75)
    _fill(ax, ts, mids, fund, alpha=0.07, color=PURPLE)
    ax.set_title("Mid Price vs Fundamental", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Price"); ax.legend(loc="upper left", framealpha=0.9)
    ax.ticklabel_format(useOffset=False, style="plain")
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def ChartBook(snapshot):
    fig, ax = plt.subplots(figsize=(FW, FH))
    if snapshot is None:
        ax.text(0.5, 0.5, "Waiting...", ha="center", va="center", color=HINT, fontsize=10, transform=ax.transAxes)
    else:
        bids, asks = snapshot["bids"], snapshot["asks"]
        if bids:
            bp, bq = zip(*bids)
            ax.bar(bp, bq, width=0.004, color=GREEN, alpha=0.75)
        if asks:
            ap, aq = zip(*asks)
            ax.bar(ap, [-q for q in aq], width=0.004, color=RED, alpha=0.75)
        ax.axhline(0, color="#ccc", lw=0.5)
        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) / 2
            ax.axvline(mid, color=LABEL, lw=0.6, ls="--", alpha=0.3)
            ax.set_xlim(mid - 0.12, mid + 0.12)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_title(f"Order Book  t={snapshot['t']}", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Qty")
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def ChartSpread(history):
    fig, ax = plt.subplots(figsize=(FW, FH))
    ts = [h["t"] for h in history]
    sp = _s([h["spread"] for h in history])
    _fill(ax, ts, sp, alpha=0.10, color=PURPLE)
    ax.plot(ts, sp, color=PURPLE, lw=1.0)
    ax.set_title("Bid-Ask Spread", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Spread")
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def ChartVol(history):
    fig, ax = plt.subplots(figsize=(FW, FH))
    ts = [h["t"] for h in history]
    vol = _rolling_vol([h["mid"] for h in history], window=20)
    _fill(ax, ts, vol, alpha=0.10, color=RED)
    ax.plot(ts, vol, color=RED, lw=1.0)
    ax.set_title("Realised Volatility (20-tick)", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Vol")
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def ChartDepth(history):
    fig, ax = plt.subplots(figsize=(FW, FH))
    ts = [h["t"] for h in history]
    bd = _s([h["bid_depth"] for h in history])
    ad = _s([h["ask_depth"] for h in history])
    _fill(ax, ts, bd, alpha=0.10, color=GREEN)
    _fill(ax, ts, ad, alpha=0.10, color=RED)
    ax.plot(ts, bd, color=GREEN, lw=1.0, label="Bids")
    ax.plot(ts, ad, color=RED,   lw=1.0, label="Asks")
    ax.set_title("Lit Book Depth", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Depth"); ax.legend(loc="upper right", framealpha=0.9)
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)

@solara.component
def ChartDark(history):
    fig, ax = plt.subplots(figsize=(FW, FH))
    ts  = [h["t"] for h in history]
    dpv = _s([h["dp_volume"] for h in history])
    dpb = _s([h["dp_bid_depth"] for h in history])
    dpa = _s([h["dp_ask_depth"] for h in history])
    ax.bar(ts, dpv, color=AMBER, alpha=0.3, width=1.0, label="Volume")
    ax.plot(ts, dpb, color=GREEN, lw=0.8, alpha=0.65, label="Bid Depth")
    ax.plot(ts, dpa, color=RED,   lw=0.8, alpha=0.65, label="Ask Depth")
    ax.set_title("Dark Pool Activity", fontsize=10, fontweight="700", color=TITLE_C, loc="left", pad=8)
    ax.set_ylabel("Qty"); ax.legend(loc="upper right", framealpha=0.9)
    _fin(fig, ax); solara.FigureMatplotlib(fig); plt.close(fig)


# ======================================================================
#  PILL
# ======================================================================

@solara.component
def Pill(label, value, color=GREEN):
    with solara.Row(style={
        "background": BG, "border": f"1px solid {BORDER}",
        "border-radius": "20px", "padding": "4px 14px",
        "align-items": "center", "gap": "6px",
    }):
        solara.Text(label, style={
            "color": LABEL, "font-size": "10px",
            "text-transform": "uppercase", "letter-spacing": "0.3px",
        })
        solara.Text(str(value), style={
            "color": color, "font-size": "13px", "font-weight": "700",
        })


# ======================================================================
#  PAGE
# ======================================================================

@solara.component
def Page():
    solara.HTML(tag="div", unsafe_innerHTML=DARK_CSS)

    with solara.Column(style={"background": BG, "min-height": "100vh", "padding": "0"}):

        # header
        with solara.Row(style={
            "background": WHITE, "padding": "12px 28px",
            "border-bottom": f"1px solid {BORDER}",
            "align-items": "baseline", "gap": "10px",
        }):
            solara.Text("To Hide or Not to Hide", style={
                "color": TEXT, "font-size": "16px", "font-weight": "700",
            })
            solara.Text("Darkpool Simulation", style={
                "color": LABEL, "font-size": "13px", "font-weight": "400",
            })

        with solara.Row(style={"gap": "0", "flex": "1"}):

            # ─── SIDEBAR ───
            with solara.Column(style={
                "width": "300px", "min-width": "300px",
                "background": SIDEBAR,
                "border-right": f"1px solid {BORDER}",
                "padding": "12px 14px",
                "overflow-y": "auto",
                "max-height": "calc(100vh - 50px)",
            }):
                def _hdr(t):
                    solara.Text(t, style={
                        "color": LABEL, "font-size": "10px", "font-weight": "600",
                        "letter-spacing": "1px", "text-transform": "uppercase",
                        "margin-top": "14px", "margin-bottom": "0px",
                        "padding-bottom": "4px",
                        "border-bottom": f"1px solid {DIVIDER}",
                    })

                _hdr("Simulation")
                LS_Int("Steps", value=steps, min=100, max=3000, step=100)
                LS_Float("Start Price", value=start_price, min=50, max=200, step=1.0)
                LS_Int("Ticks / refresh", value=tick_speed, min=1, max=50)
                LS_Int("Seed", value=sim_seed, min=1, max=9999)

                _hdr("Fundamental")
                LS_Float("sigma_v", value=sigma_v, min=0.005, max=0.05, step=0.005)

                _hdr("Noise Traders")
                LS_Int("Count", value=noise_count, min=1, max=10)
                LS_Pct("Participation", value=noise_participation, min=0.1, max=1.0, step=0.05)
                LS_Pct("Mkt Order %", value=noise_market_prob, min=0.1, max=0.9, step=0.05)

                _hdr("Market Maker (A-S)")
                LS_Float("gamma (Risk Aversion)", value=mmas_gamma, min=0.01, max=1.0, step=0.01)
                LS_Float("kappa (Liquidity)", value=mmas_kappa, min=5.0, max=200.0, step=5.0)
                LS_Float("sigma (Vol Est)", value=mmas_sigma, min=0.01, max=0.20, step=0.01)
                LS_Float("T (Horizon)", value=mmas_horizon, min=500.0, max=10000.0, step=500.0)

                _hdr("Institutional Traders")
                LS_Int("Count", value=institutional_count, min=0, max=6)
                LS_Pct("Participation", value=institutional_part, min=0.01, max=0.30, step=0.01)
                LS_Pct("Iceberg Prob", value=iceberg_prob, min=0.0, max=1.0, step=0.05)
                LS_Pct("Dark Fraction", value=inst_dark_frac, min=0.0, max=1.0, step=0.05)

                _hdr("Informed Traders")
                LS_Int("Count", value=informed_count, min=0, max=5)
                LS_Pct("Participation", value=informed_participation, min=0.1, max=1.0, step=0.05)
                LS_Float("sigma_s", value=informed_sigma_s, min=0.01, max=0.30, step=0.01)
                LS_Pct("Dark Fraction", value=informed_dark_frac, min=0.0, max=1.0, step=0.05)

                with solara.Row(style={"gap": "8px", "margin-top": "18px"}):
                    solara.Button(
                        "Run" if not is_running.value else "Running...",
                        on_click=start_sim, disabled=is_running.value,
                        color="primary",
                        style={"flex": "1", "font-weight": "600"},
                    )
                    solara.Button(
                        "Stop", on_click=stop_sim,
                        disabled=not is_running.value, color="error",
                        style={"font-weight": "600"},
                    )

            # ─── MAIN ───
            with solara.Column(style={
                "flex": "1", "padding": "14px 18px",
                "overflow-y": "auto",
                "max-height": "calc(100vh - 50px)",
                "background": BG,
            }):
                history  = live_history.value
                snapshot = live_snapshot.value
                cur_t    = live_tick.value
                tot_t    = steps.value

                if not history:
                    with solara.Column(style={
                        "align-items": "center", "justify-content": "center",
                        "height": "70vh", "gap": "6px",
                    }):
                        solara.Text("Configure parameters and press Run", style={
                            "color": LABEL, "font-size": "15px",
                        })
                        solara.Text("How do dark pools affect price efficiency, volatility, and liquidity?", style={
                            "color": HINT, "font-size": "12px",
                        })
                else:
                    pct = int(100 * cur_t / tot_t) if tot_t > 0 else 0
                    tag = "LIVE" if is_running.value else ("DONE" if is_done.value else "")
                    tag_c = GREEN if is_running.value else CYAN

                    with solara.Row(style={
                        "align-items": "center", "gap": "8px",
                        "margin-bottom": "8px", "flex-wrap": "wrap",
                    }):
                        Pill("Tick", f"{cur_t}/{tot_t}", tag_c)
                        Pill("Dark Frac", f"{informed_dark_frac.value:.0%}", AMBER)
                        Pill("Iceberg", f"{iceberg_prob.value:.0%}", PURPLE)
                        mids_v = [h["mid"] for h in history if h["mid"] is not None]
                        spreads_v = [h["spread"] for h in history if h["spread"] is not None]
                        if mids_v:
                            Pill("Avg Mid", f"${np.mean(mids_v):.2f}", GREEN)
                        if spreads_v:
                            Pill("Avg Spread", f"{np.mean(spreads_v):.4f}", PURPLE)
                        misp = [h["mispricing"] for h in history if h["mispricing"] is not None]
                        if misp:
                            Pill("|Mispricing|", f"{np.mean(misp):.4f}", PINK)
                        if tag:
                            solara.Text(tag, style={
                                "color": tag_c, "font-size": "11px", "font-weight": "700",
                            })

                    solara.ProgressLinear(value=pct)

                    with solara.Row(style={"gap": "12px", "margin-top": "12px"}):
                        with solara.Column(style=CARD_STYLE):
                            ChartPrice(history)
                        with solara.Column(style=CARD_STYLE):
                            ChartBook(snapshot)

                    with solara.Row(style={"gap": "12px", "margin-top": "12px"}):
                        with solara.Column(style=CARD_STYLE):
                            ChartSpread(history)
                        with solara.Column(style=CARD_STYLE):
                            ChartVol(history)

                    with solara.Row(style={"gap": "12px", "margin-top": "12px"}):
                        with solara.Column(style=CARD_STYLE):
                            ChartDepth(history)
                        with solara.Column(style=CARD_STYLE):
                            ChartDark(history)
