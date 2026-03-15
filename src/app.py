"""
SolaraViz dashboard for the order-book ABM.
Usage: python -m solara run src.app
"""

import numpy as np
import solara
from matplotlib.figure import Figure

from .model import OrderBookModel
from mesa.visualization import SolaraViz
from mesa.visualization.utils import update_counter


# ---------------------------------------------------------------------------
# Custom matplotlib components
# ---------------------------------------------------------------------------

@solara.component
def MidPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["Mid"], color="#E24A33", linewidth=1)
        ax.set_title("Mid-price diffusion")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mid")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def SpreadPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["Spread"], color="#E24A33", linewidth=1)
        ax.set_title("Spread")
        ax.set_xlabel("Step")
        ax.set_ylabel("Spread")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def OBIPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["OBI"], color="#E24A33", linewidth=1)
        ax.set_title("Orderbook Imbalance")
        ax.set_xlabel("Step")
        ax.set_ylabel("OBI")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def MicropricePlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["Microprice"], color="#E24A33", linewidth=1)
        ax.set_title("Microprice")
        ax.set_xlabel("Step")
        ax.set_ylabel("Microprice")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def InventoryPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        inv = df["MMAS_Inventory"]
        steps = range(len(inv))
        ax.plot(steps, inv, color="steelblue", linewidth=1, label="MMAS Inventory")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(steps, inv, 0,
                        where=inv > 0, alpha=0.2, color="green", label="Long")
        ax.fill_between(steps, inv, 0,
                        where=inv < 0, alpha=0.2, color="red", label="Short")
        ax.set_title("MarketMakerAS Inventory Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Inventory")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def DPVolumePlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["DPRecentVolume"], color="#E24A33", linewidth=1)
        ax.set_title("Dark Pool: Recent Traded Volume (20-tick)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Volume")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def DepthPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        ax.plot(df["TotalBidDepth"], color="#348ABD", linewidth=1, label="Bid Depth")
        ax.plot(df["TotalAskDepth"], color="#E24A33", linewidth=1, label="Ask Depth")
        ax.set_title("Total Visible Depth")
        ax.set_xlabel("Step")
        ax.set_ylabel("Quantity")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def MidVsFundamentalPlot(model):
    update_counter.get()
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if len(df) > 0:
        fund_series = model.fundamental.history[:len(df)]
        steps = range(len(df))
        ax.plot(steps, df["Mid"], color="steelblue", linewidth=1.2, label="Mid Price")
        ax.plot(steps, fund_series, color="crimson", linewidth=1.2, linestyle="--",
                label="Fundamental Value")
        ax.fill_between(steps, df["Mid"], fund_series, alpha=0.15, color="purple",
                        label="Mispricing")
        ax.set_title("Mid Price vs Fundamental Value")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def DemandCurvePlot(model):
    update_counter.get()
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    bids = model.book.top_n_levels("buy", 20)
    asks = model.book.top_n_levels("sell", 20)
    if bids:
        bid_prices = [p for p, _ in bids]
        bid_cum = [sum(q for p, q in bids if p >= price) for price in bid_prices]
        ax.step(bid_cum, bid_prices, where="post", color="#348ABD", label="Bids")
    if asks:
        ask_prices = [p for p, _ in asks]
        ask_cum = [sum(q for p, q in asks if p <= price) for price in ask_prices]
        ax.step(ask_cum, ask_prices, where="post", color="#E24A33", label="Asks")
    ax.set_xlabel("Cumulative Quantity")
    ax.set_ylabel("Price")
    ax.set_title("Demand Curve")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def SnapshotPlot(model):
    update_counter.get()
    snapshots = model.snapshots
    if not snapshots:
        solara.Text("No snapshots yet")
        return
    # Show the latest snapshot
    snap = snapshots[-1]
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    bids, asks = snap["bids"], snap["asks"]
    tick = 0.01
    if len(bids) >= 2:
        tick = abs(bids[0][0] - bids[1][0]) or 0.01
    bar_width = tick * 0.4
    half_width = tick * 22
    if bids:
        bp, bq = zip(*bids)
        ax.bar(bp, bq, width=bar_width, color="#348ABD", label="Bids")
    else:
        bp = []
    if asks:
        ap, aq = zip(*asks)
        ax.bar(ap, [-q for q in aq], width=bar_width, color="#E24A33", label="Asks")
    else:
        ap = []
    ax.axhline(0, linewidth=1, color="black")
    if bids and asks:
        mid = (ap[0] + bp[0]) / 2
        ax.axvline(mid, linewidth=1, color="black", linestyle="--")
        ax.set_xlim(mid - half_width, mid + half_width)
    ax.set_title(f"Order Book Snapshot (t={snap['t']})")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.ticklabel_format(useOffset=False, style="plain")
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


@solara.component
def VWAPPlot(model):
    update_counter.get()
    fig = Figure(figsize=(8, 3))
    ax = fig.subplots()
    # Compute VWAP per step from trade records
    if model.trade_records:
        import pandas as pd
        tdf = pd.DataFrame(model.trade_records)
        vwap_series = tdf.groupby("t").apply(
            lambda g: np.sum(g["Price"] * g["Qty"]) / np.sum(g["Qty"]) if np.sum(g["Qty"]) > 0 else np.nan
        )
        ax.plot(vwap_series.index, vwap_series.values, color="#E24A33", linewidth=1)
    ax.set_title("Volume-Weighted Average Price (VWAP)")
    ax.set_xlabel("Step")
    ax.set_ylabel("VWAP")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    solara.FigureMatplotlib(fig, format="png")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

model = OrderBookModel(steps_to_run=1000, seed=42)

page = SolaraViz(
    model,
    components=[
        MidPlot,
        SpreadPlot,
        OBIPlot,
        InventoryPlot,
        MicropricePlot,
        MidVsFundamentalPlot,
        VWAPPlot,
        DPVolumePlot,
        DepthPlot,
        DemandCurvePlot,
        SnapshotPlot,
    ],
    play_interval=50,
    name="Order Book ABM",
)

page