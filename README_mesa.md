# Mesa Integration

Old `run.py` got split into two files. The actual logic went into a class called `OrderBookModel` inside a new file `model.py`. The new `run.py` is just a few lines that creates that class and calls run on it.

The old `main()` function, where the book was set up, the dark pool created, the agents list built — that's now the `__init__` of `OrderBookModel`. Same code, same order, same numbers.

The old `run_simulation()` function, the big for loop that stepped through time — that's now the `step()` method on `OrderBookModel`. Each call is one tick. The agent loop inside it is the same, the dark pool tick is the same, the MMAS inventory notification logic is the same, the sanity checks are the same.

All analytics functions like `microprice`, `vwap`, `top_n_obi` and all plotting functions like `plot_series`, `plot_snapshots` — they were copied into `model.py` without any changes. Same names, same logic.

In `agents.py`, all agent logic is identical. `act()` is still `act()`, `act_dark()` is still `act_dark()`, `trader_id` is still `trader_id`. The only change is that `self.rng` became `self.agent_rng`  

`orderbook.py`, `dark_pool.py`, and `fundemental.py` were not touched at all.

So when something needs to change, the same places apply as before. Agent behavior is still in `agents.py`. Which agents run and with what parameters is now in `model.py` `__init__()` instead of `main()`. What happens each tick is now in `model.py` `step()` instead of `run_simulation()`. Adding a plot follows the same pattern, just inside `save_plots()` in `model.py`.

There's also a new file `app.py` that provides a live browser dashboard where the simulation can be watched step by step. It's completely optional and doesn't affect anything else.

## How to run

Headless (saves PNGs + CSVs):
```bash
python -m src.run
```

Interactive browser dashboard:
```bash
python -m solara run src.app
```