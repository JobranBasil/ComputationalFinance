# To Hide or Not to Hide — Darkpool Simulation

## Quick Start

```bash
cd abm_project
pip install -r requirements.txt
solara run app.py
```

Open **http://localhost:8765**

## Layout

6 charts in a 3×2 grid — no scrolling:

| Left | Right |
|------|-------|
| Mid Price vs Fundamental (price efficiency) | Live Order Book Depth |
| Bid-Ask Spread (liquidity) | Rolling Realised Volatility |
| Lit Book Depth (bid vs ask) | Dark Pool Activity |

## Key Slider: Dark Fraction

Controls what percentage of the informed trader's volume is routed to the
dark pool vs the lit book. This is the main experimental variable for
studying how dark pools affect price efficiency, volatility, and liquidity.
