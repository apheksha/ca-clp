"""Terrain from SRTM `.hgt` tiles (big-endian int16, N x N samples, rows north->south; SRTM1 = 3601, SRTM3 = 1201; any square size is accepted, which the tests use).
Void cells (-32768) are filled from the nearest valid neighbour. The tile origin comes from the file name (e.g. N47E008.hgt = south-west corner 47N 8E).
Fetching tiles is not done here; download them from the provider of your choice (NASA/USGS SRTM, Copernicus DEM, ...) and pass the paths."""
from __future__ import annotations
import math
import os
import re
import struct
from src.geodata.projection import LocalFrame
from src.models.terrain import TerrainGrid

VOID = -32768


class HgtTile:
    def __init__(self, lat0: int, lon0: int, n: int, data: list):
        self.lat0, self.lon0, self.n, self.data = lat0, lon0, n, data    # data[row][col]; row 0 = north edge

    def height(self, lat: float, lon: float) -> float:
        fy = (self.lat0 + 1 - lat) * (self.n - 1)
        fx = (lon - self.lon0) * (self.n - 1)
        fx = min(max(fx, 0.0), self.n - 1.0)
        fy = min(max(fy, 0.0), self.n - 1.0)
        c0, r0 = int(fx), int(fy)
        c1, r1 = min(c0 + 1, self.n - 1), min(r0 + 1, self.n - 1)
        tx, ty = fx - c0, fy - r0
        d = self.data
        return d[r0][c0] * (1 - tx) * (1 - ty) + d[r0][c1] * tx * (1 - ty) + d[r1][c0] * (1 - tx) * ty + d[r1][c1] * tx * ty

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat0 <= lat <= self.lat0 + 1 and self.lon0 <= lon <= self.lon0 + 1


def read_hgt(path: str) -> HgtTile:
    m = re.match(r"([NS])(\d{2})([EW])(\d{3})", os.path.basename(path).upper())
    if not m:
        raise ValueError(f"cannot read the tile origin from file name {os.path.basename(path)!r} (expected e.g. N47E008.hgt)")
    lat0 = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon0 = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    with open(path, "rb") as fh:
        raw = fh.read()
    n = int(round(math.sqrt(len(raw) / 2)))
    if n * n * 2 != len(raw) or n < 2:
        raise ValueError(f"{path}: size {len(raw)} bytes is not a square int16 grid")
    vals = list(struct.unpack(f">{n * n}h", raw))
    data = [vals[r * n:(r + 1) * n] for r in range(n)]
    _fill_voids(data)
    return HgtTile(lat0, lon0, n, [[float(v) for v in row] for row in data])


def _fill_voids(data: list) -> None:
    n = len(data)
    voids = [(r, c) for r in range(n) for c in range(len(data[0])) if data[r][c] == VOID]
    if not voids:
        return
    valid = [(r, c) for r in range(n) for c in range(len(data[0])) if data[r][c] != VOID]
    if not valid:
        raise ValueError("tile contains only voids")
    for r, c in voids:
        rr, cc = min(valid, key=lambda p: (p[0] - r) ** 2 + (p[1] - c) ** 2)
        data[r][c] = data[rr][cc]


def terrain_from_tiles(tiles: list[HgtTile], frame: LocalFrame, bbox_m: tuple, cell_m: float = 30.0) -> TerrainGrid:
    """Resample the tiles onto a regular local-metre grid covering bbox_m = (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bbox_m
    cols = max(2, int(math.ceil((xmax - xmin) / cell_m)) + 1)
    rows = max(2, int(math.ceil((ymax - ymin) / cell_m)) + 1)
    heights = []
    for r in range(rows):
        row = []
        for c in range(cols):
            lat, lon = frame.to_latlon(xmin + c * cell_m, ymin + r * cell_m)
            t = next((t for t in tiles if t.contains(lat, lon)), None)
            if t is None:
                raise ValueError(f"no terrain tile covers lat {lat:.5f}, lon {lon:.5f}")
            row.append(t.height(lat, lon))
        heights.append(row)
    return TerrainGrid(xmin, ymin, cell_m, heights)
