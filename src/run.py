"""
Entry point for the order-book ABM simulation.
Usage: python -m src.run
"""

import sys
import logging
from .model import OrderBookModel


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    model = OrderBookModel(
        steps_to_run=1000,
        seed=42,
        tick=0.01,
        max_depth_levels=20,
        best_bid=100.05,
        best_ask=100.10,
        init_levels=10,
    )

    model.run()
    model.save_plots()


if __name__ == "__main__":
    main()