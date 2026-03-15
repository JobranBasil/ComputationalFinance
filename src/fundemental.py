import numpy as np
class FundamentalProcess:
    """
    Shared random-walk fundamental price.
    V_t = V_{t-1} + sigma_v * ε,  ε ~ N(0,1)
    All InformedTraders reference the same instance.
    """
    def __init__(self, start: float = 100.0, sigma_v: float = 0.03, rng: np.random.Generator = None):
        self.value = start
        self.sigma_v = sigma_v
        self.rng = rng or np.random.default_rng(99)
        self.history: list[float] = [start]

    def step(self) -> float:
        """Advance by one tick. Call once per simulation step, NOT per agent."""
        self.value += self.sigma_v * self.rng.standard_normal()
        self.history.append(self.value)
        return self.value

    def observe(self, sigma_s: float, rng: np.random.Generator) -> float:
        """Each informed trader calls this to get their private noisy signal."""
        return self.value + sigma_s * rng.standard_normal()