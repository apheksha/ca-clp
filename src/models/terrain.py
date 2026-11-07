"""Terrain model: a regular elevation grid (metres, same vertical datum as mission altitudes), bilinear interpolation.

Sources: SRTM .hgt tiles (src/geodata/srtm.py), any DEM resampled to a grid, or analytic test terrain. Heights outside the grid are clamped to the nearest edge cell."""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class TerrainGrid:
    x0: float            # map x of heights[0][0] (west edge), map units
    y0: float            # map y of heights[0][0] (south edge)
    cell: float          # cell size, map units
    heights: list        # heights[row][col], row = +y, col = +x, metres

    def __post_init__(self):
        if self.cell <= 0:
            raise ValueError("terrain cell must be > 0")
        if not self.heights or not self.heights[0]:
            raise ValueError("terrain heights must be a non-empty 2D list")
        w = len(self.heights[0])
        if any(len(r) != w for r in self.heights):
            raise ValueError("terrain rows must have equal length")

    @property
    def rows(self) -> int:
        return len(self.heights)

    @property
    def cols(self) -> int:
        return len(self.heights[0])

    def height_at(self, x: float, y: float) -> float:
        fx = min(max((x - self.x0) / self.cell, 0.0), self.cols - 1.0)
        fy = min(max((y - self.y0) / self.cell, 0.0), self.rows - 1.0)
        c0, r0 = int(math.floor(fx)), int(math.floor(fy))
        c1, r1 = min(c0 + 1, self.cols - 1), min(r0 + 1, self.rows - 1)
        tx, ty = fx - c0, fy - r0
        h = self.heights
        return (h[r0][c0] * (1 - tx) * (1 - ty) + h[r0][c1] * tx * (1 - ty) + h[r1][c0] * (1 - tx) * ty + h[r1][c1] * tx * ty)

    def sample_segment(self, a, b, step: float | None = None):
        """Yield (t, x, y, ground_height) along segment ab with spacing <= step (default: half a cell). Includes both ends."""
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        step = step or self.cell / 2
        n = max(1, int(math.ceil(L / step)))
        for k in range(n + 1):
            t = k / n
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            yield t, x, y, self.height_at(x, y)

    def max_height_on(self, pts, step=None) -> float:
        return max(h for a, b in zip(pts, pts[1:]) for _, _, _, h in self.sample_segment(a, b, step))

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "cell": self.cell, "heights": self.heights}

    @staticmethod
    def from_dict(d: dict) -> "TerrainGrid":
        return TerrainGrid(float(d["x0"]), float(d["y0"]), float(d["cell"]), [[float(v) for v in r] for r in d["heights"]])
