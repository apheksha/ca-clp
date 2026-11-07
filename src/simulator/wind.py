"""Wind disturbance model for the flight simulator: steady mean wind + Ornstein-Uhlenbeck gusts + optional power-law shear with height. Seeded and reproducible."""
from __future__ import annotations
import math
import random


class WindModel:
    def __init__(self, mean=(0.0, 0.0), gust_sigma: float = 0.0, gust_tau_s: float = 8.0, shear_alpha: float = 0.0, z_ref_m: float = 10.0, seed: int = 0):
        self.mean = (float(mean[0]), float(mean[1])) if mean else (0.0, 0.0)
        self.sigma, self.tau, self.alpha, self.z_ref = gust_sigma, max(gust_tau_s, 1e-3), shear_alpha, z_ref_m
        self.rng = random.Random(f"wind|{seed}")
        self.gx = self.gy = 0.0

    def step(self, dt: float) -> None:
        if self.sigma <= 0:
            return
        a = math.exp(-dt / self.tau)
        s = self.sigma * math.sqrt(1 - a * a)
        self.gx = a * self.gx + s * self.rng.gauss(0, 1)
        self.gy = a * self.gy + s * self.rng.gauss(0, 1)

    def at(self, z_agl: float = 10.0) -> tuple[float, float]:
        f = (max(z_agl, 1.0) / self.z_ref) ** self.alpha if self.alpha else 1.0
        return (self.mean[0] * f + self.gx, self.mean[1] * f + self.gy)

    @property
    def steady(self) -> tuple[float, float]:
        return self.mean
