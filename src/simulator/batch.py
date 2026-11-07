"""Batch execution of the plans stored in a raw run folder: every RELEASED, verified plan is flown in the built-in simulator (wind, gusts, localization error) and the
EXECUTED trajectory is evaluated. Answers: 'of the plans the verifier accepted, how many would the vehicle actually fly without incident?'"""
from __future__ import annotations
import csv
import glob
import json
import os
from dataclasses import replace
from src.models.schemas import Mission, FlightPlan
from src.simulator.sim import simulate, SimConfig
from src.simulator.evaluation import evaluate_execution
from src.utils.io import read_jsonl


def prepare_mission(m: Mission, vehicle: str | dict, unit_m: float | None, wind=None, payload_kg: float = 0.0) -> Mission:
    d = m.to_dict()
    d["vehicle"] = vehicle
    d["payload_kg"] = payload_kg
    if unit_m is not None:
        d["unit_m"] = unit_m
    if wind is not None:
        d["wind"] = list(wind)
    return Mission.from_dict(d)


def simulate_run(run_dir: str, vehicle="generic_quad_large", unit_m: float | None = None, wind=(0.0, 0.0), gust_sigma: float = 1.0, seeds=(0,), conditions: list | None = None,
                 limit: int | None = None, out_dir: str | None = None, winds: dict | None = None, run_seeds: list | None = None, altitude_margin_m: float = 0.0) -> str:
    """Delegates to the consistent study (src/simulator/study.py). `wind` = a single steady wind; `winds` = named scenarios {name: (east, north)} (default: calm + 4 m/s crosswind)."""
    from src.simulator.study import simulate_run_study, DEFAULT_WINDS
    w = winds if winds is not None else (DEFAULT_WINDS if tuple(wind) == (0.0, 0.0) else {"given": tuple(wind)})
    return simulate_run_study(run_dir, vehicle, unit_m, w, gust_sigma, seeds, conditions, limit, out_dir, run_seeds=run_seeds, altitude_margin_m=altitude_margin_m)
