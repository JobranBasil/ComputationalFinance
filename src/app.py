"""
SolaraViz dashboard for the order-book ABM.
Usage: python -m solara run src.app
"""

from .model import OrderBookModel
from mesa.visualization import SolaraViz, make_plot_component

model = OrderBookModel(steps_to_run=1000, seed=42)

page = SolaraViz(
    model,
    components=[
        make_plot_component(measure="Mid"),
        make_plot_component(measure="Spread"),
        make_plot_component(measure="OBI"),
        make_plot_component(measure="MMAS_Inventory"),
        make_plot_component(measure="DPRecentVolume"),
        make_plot_component(measure="Microprice"),
    ],
    model_params={
        "steps_to_run": {
            "type": "SliderInt",
            "value": 1000,
            "label": "Steps",
            "min": 100,
            "max": 5000,
            "step": 100,
        },
        "seed": {
            "type": "SliderInt",
            "value": 42,
            "label": "Seed",
            "min": 1,
            "max": 100,
        },
    },
    name="Order Book ABM",
)

page