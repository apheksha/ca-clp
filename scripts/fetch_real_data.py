"""Download the real OurAirports dataset and build real-airspace scenarios.  Needs internet (free, no key).
   python scripts/fetch_real_data.py            (or: python run.py real)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark.real_airspace import fetch, build_real_scenarios

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else fetch()
    sc = build_real_scenarios(p, "benchmark/real", "benchmark/metadata")
    print(f"built {len(sc)} real-airspace scenarios in benchmark/real")
