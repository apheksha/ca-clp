"""Vehicle profiles (physical parameters) shared by the physical energy model, the dynamics checker and the flight simulator.

All numbers come from `config/vehicles.yaml` (REPRESENTATIVE class parameters, not calibrated to a particular airframe). A mission references a profile by name
(`Mission.vehicle = "generic_quad_small"`) or embeds the parameter dict directly."""
from __future__ import annotations
import os
from dataclasses import dataclass, asdict, fields
import yaml

RHO = 1.225     # air density, kg/m^3 (ISA sea level)
G0 = 9.80665

_FIELDS = None


@dataclass
class VehicleProfile:
    name: str = "custom"
    kind: str = "multirotor"                 # multirotor | fixed_wing
    mass_kg: float = 1.5
    rotor_disc_area_m2: float = 0.2          # multirotor only
    lift_to_drag: float = 12.0               # fixed_wing only
    efficiency: float = 0.6
    drag_area_m2: float = 0.012              # multirotor parasitic Cd*A
    avionics_w: float = 8.0
    battery_wh: float = 60.0
    usable_fraction: float = 0.8
    v_cruise: float = 12.0
    v_max: float = 18.0
    v_climb_max: float = 3.0
    v_descend_max: float = 2.5
    max_accel: float = 3.0
    max_lateral_accel: float = 4.0
    min_turn_radius_m: float = 8.0
    hover_time_per_waypoint_s: float = 2.0
    payload_max_kg: float = 0.5
    tracking_gain: float = 0.8
    acceptance_radius_m: float = 3.0
    sensor_range_m: float = 60.0
    gps_sigma_m: float = 1.2
    gps_bias_tau_s: float = 60.0
    gps_bias_sigma_m: float = 1.5
    gps_dropout_p: float = 0.0

    def validate(self) -> "VehicleProfile":
        if self.kind not in ("multirotor", "fixed_wing"):
            raise ValueError(f"unknown vehicle kind {self.kind!r}")
        for nm in ("mass_kg", "efficiency", "battery_wh", "v_cruise", "v_max", "v_climb_max", "v_descend_max", "max_accel"):
            if not getattr(self, nm) > 0:
                raise ValueError(f"vehicle.{nm} must be > 0")
        if not 0 < self.efficiency <= 1 or not 0 < self.usable_fraction <= 1:
            raise ValueError("efficiency and usable_fraction must be in (0, 1]")
        if self.kind == "multirotor" and not self.rotor_disc_area_m2 > 0:
            raise ValueError("rotor_disc_area_m2 must be > 0")
        if self.v_cruise > self.v_max:
            raise ValueError("v_cruise must be <= v_max")
        return self

    @property
    def usable_wh(self) -> float:
        return self.battery_wh * self.usable_fraction

    def to_dict(self) -> dict:
        return asdict(self)


def _load_all(path: str | None = None) -> dict:
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "vehicles.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["profiles"]


def get_vehicle(spec, path: str | None = None) -> VehicleProfile:
    """spec: profile name (str), parameter dict (optionally with `base:` profile name), or a VehicleProfile."""
    if isinstance(spec, VehicleProfile):
        return spec
    known = {f.name for f in fields(VehicleProfile)}
    if isinstance(spec, str):
        prof = _load_all(path)
        if spec not in prof:
            raise KeyError(f"unknown vehicle profile {spec!r}; available: {sorted(prof)}")
        return VehicleProfile(name=spec, **prof[spec]).validate()
    if isinstance(spec, dict):
        d = dict(spec)
        base = d.pop("base", None)
        out = {}
        if base:
            out.update(_load_all(path)[base])
        out.update(d)
        bad = set(out) - known
        if bad:
            raise ValueError(f"unknown vehicle parameters: {sorted(bad)}")
        out.setdefault("name", base or "custom")
        return VehicleProfile(**out).validate()
    raise TypeError("vehicle must be a profile name, a dict or a VehicleProfile")
