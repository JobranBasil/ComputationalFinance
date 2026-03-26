import numpy as np
class FundamentalProcess:
    # random-walk fundamental price
    def __init__(self, start: float = 100.0, sigma_v: float = 0.03, rng: np.random.Generator = None):
        self.value = start
        self.sigma_v = sigma_v
        self.rng = rng or np.random.default_rng(99)
        self.history: list[float] = [start]

    def step(self) -> float:
        # call once per step, not per agent
        self.value += self.sigma_v * self.rng.standard_normal()
        self.history.append(self.value)
        return self.value

    def observe(self, sigma_s: float, rng: np.random.Generator) -> float:
        # noisy private signal
        return self.value + sigma_s * rng.standard_normal()