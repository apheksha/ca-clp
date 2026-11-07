# Real geodata for the Bengaluru 3D set

| File | Content | Source and licence |
|---|---|---|
| `bengaluru_buildings.geojson` | 15,260 building footprints (Polygon features) between 13.014–13.045 N and 77.544–77.575 E (IISc, Mathikere, Malleswaram, Yeshwanthpur). 17 carry a `height` tag and 102 a `building:levels` tag; the rest have no height. | OpenStreetMap contributors, exported with overpass-turbo on 2026-11-05 (timestamp inside the file). © OpenStreetMap contributors, Open Database License (ODbL) 1.0 — attribution required, share-alike for derived databases. |
| `bengaluru_buildings_east.json` | 46,802 building ways (raw Overpass JSON) between 13.002–13.048 N and 77.577–77.624 E (the area directly east of the first file). 32 carry a `height` tag and 538 a `building:levels` tag. | OpenStreetMap contributors via the Overpass API, 2026-11-06 (timestamp inside the file); ODbL 1.0. |
| `N13E077.hgt` | SRTM 1 arc-second elevation tile, 3601×3601 samples, south-west corner 13 N 77 E. | NASA/USGS Shuttle Radar Topography Mission; public domain. |

`scripts/build_bengaluru_set.py` merges both building files (60,330 buildings after de-duplication; 96 reach the 20 m band floor) and turns them into
`benchmark/bengaluru/` (30 missions; see `START_HERE.md`). `build_log_*.json` record how many candidate missions were skipped and why.
What is real: footprints, tagged heights, ground elevation, site names and positions. What is modeled: the AGL altitude band, the 10 m height assumed for
buildings without height data, and the distance/energy budgets. OSM heights are incomplete, so the set under-represents tall buildings; it is a test of
3D geometry and terrain handling, not an obstacle survey.
