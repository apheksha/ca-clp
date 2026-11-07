# Geodata: authoritative airspace, terrain, buildings (v10)

Loaders only; **no data is bundled or downloaded** (no network access in this code base). `data/geodata_samples/` contains SYNTHETIC fixtures (`scripts/make_geodata_fixtures.py`) that exercise the code.

| Input | Loader | Notes |
|---|---|---|
| Airspace restrictions (GeoJSON Polygon/MultiPolygon/Point+radius_m) | `src/geodata/airspace.py` | real boundaries kept (concave allowed, simplified by Douglas-Peucker); vertical limits in m / ft / FL, GND/AGL limits converted with the terrain; holes ignored (conservative) |
| Buildings / obstacles (OSM Overpass JSON `out geometry;` or GeoJSON) | `src/geodata/osm.py` | prisms with height from `height` / `building:levels` / default (default heights are guesses; reported in the description) |
| Terrain (SRTM .hgt tiles) | `src/geodata/srtm.py` | void filling, resampling to a local metre grid (`TerrainGrid`) |
| Scenario assembly + oracle verdict | `src/geodata/scenario_builder.py`, `python run.py geoscenario --config cfg.yaml` | config example: `data/geodata_samples/geo_sample.yaml` |

Verifier/oracle support: terrain-clearance family (min/max AGL, sampled continuously along every flight segment, take-off/landing exemption radius), 3D zones, terrain-following witness plans.

Where to get data (check licences yourself; none of this was verified here): national AIP / UAS geozone services (OpenAIP exports, FAA UAS Facility Map, EAD/EUROCONTROL, national U-space geozones), OpenStreetMap via Overpass, SRTM / Copernicus DEM tiles.
Caveats: coordinates use a local equirectangular projection (fine for tens of km); heights above ground vs. above sea level must be consistent between airspace, terrain and plan (see `docs/VERIFIER_RULES.md`); OSM completeness varies.

## Bengaluru 3D set (v12)
`data/geodata_bengaluru/` holds two real OSM building exports (overpass-turbo GeoJSON around IISc/MSRIT and raw Overpass JSON for the area east of it;
60,330 buildings after de-duplication) and the SRTM tile N13E077.
```
python scripts/build_bengaluru_set.py --buildings data/geodata_bengaluru/bengaluru_buildings.geojson data/geodata_bengaluru/bengaluru_buildings_east.json \
    --hgt data/geodata_bengaluru/N13E077.hgt --out benchmark/bengaluru --n-per-difficulty 10
```
30 missions (10 easy / 10 medium / 10 hard) between named buildings; band 20–120 m above the ground, followed along every leg; buildings that reach the
band floor are 3D obstacles over the convex hull of their footprint; buildings without height data are assumed 10 m (below the band). A mission contains
EVERY obstacle building that a plan within the distance budget could reach (reachability ellipses, iterated with the witness; see the script docstring),
so leaving the other buildings out never changes a verdict; the terrain grid is cropped to the same region. Budgets as in the airport set
(slack 1.40/1.20/1.08, energy 1.30/1.12, k_v = 30). Candidates with more than 60 reachable buildings are skipped (counted in `build_log_*.json`).
The grid A* baseline is 2D (flies at the band floor) and is not used on this set.
