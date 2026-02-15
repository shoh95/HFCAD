"""HFCBattACDesign_SH_260117 (OOP rewrite)

This module is an object-oriented refactor of the legacy script:
  - configuration is explicit (dataclasses)
  - responsibilities are separated (powertrain, FC system, cooling, mass estimation, solver)
  - the numerical approach and default constants are intentionally kept close to the legacy behaviour
"""

from __future__ import annotations

import argparse
import configparser
import shutil
import threading
import sys

import time
import math
import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, get_args, get_origin

import numpy as np
import pandas as pd

from ambiance import Atmosphere

from Conversions import Conversions
from Weight_Estimation import WeightEst

from models.stack_functions import cell_model, stack_model, mass_flow_stack
from models.compressor_performance import compressor_performance_model
from models.compressor_mass import compressor_mass_model
from models.humidifier import humidifier_model
from models.heat_exchanger import heat_exchanger_model


logger = logging.getLogger(__name__)


# ============================
# Configuration (dataclasses)
# ============================


@dataclass(frozen=True)
class MissionProfile:
    """Discrete mission timeline used for plotting/export.

    The legacy script stores 20 time points and uses hard-coded power arrays of the same length.
    """

    times_min: Tuple[float, ...] = (
        0,
        5,
        15,
        16,
        23,
        25,
        70,
        71,
        100,
        110,
        112,
        116,
        118,
        135,
        135,
        135,
        140,
        150,
        160,
        160,
    )


@dataclass(frozen=True)
class FlightPoint:
    altitude_m: float
    mach: float


@dataclass(frozen=True)
class FuelCellArchitecture:
    n_stacks_series: int = 1
    n_stacks_parallel: int = 4
    volt_req_v: float = 700.0


@dataclass(frozen=True)
class FuelCellOperation:
    """Fuel cell operating assumptions."""

    oversizing: float = 0.30
    # Cruise compressor beta policy:
    # - True: beta = 1.2 / rho_cruise for cruise and cruise_charger
    # - False: beta = 1.05
    # - numeric: beta = value for cruise only (cruise_charger remains 1.05)
    beta: str = "False"


@dataclass(frozen=True)
class HybridSplit:
    """Hybridization parameter psi per flight phase.

    Convention preserved from the legacy script:
      - psi > 0 : battery discharges (assists propulsion)
      - psi < 0 : battery charges (absorbs power)
    """

    psi_takeoff: float = 0.25
    psi_climb: float = 0.25
    psi_cruise_charger: float = -0.053


@dataclass(frozen=True)
class PowertrainEfficiencies:
    """Electrical chain efficiencies."""

    eta_converter: float = 0.97
    eta_pdu: float = 0.98
    eta_inverter: float = 0.97
    eta_motor: float = 0.95
    eta_prop: float = 0.80

    # Used only for the legacy 'pnet' printout.
    eta_em: float = 0.9215


@dataclass(frozen=True)
class PowertrainDensities:
    """Specific power assumptions (W/kg)."""

    rhofc_w_per_kg: float = 3000.0
    rhopmad_w_per_kg: float = 15000.0
    rhoem_w_per_kg: float = 7000.0


@dataclass(frozen=True)
class PowerToWeightRatios:
    """Shaft power-to-weight ratios.

    Input-file units (new standard): **kW/kg**.

    The legacy implementation used W/N, via:
      P_shaft = MTOM * g * (W/N)
    With kW/kg, the physically equivalent form is:
      P_shaft = MTOM * (kW/kg) * 1000

    Backward compatibility:
    - If an input file still provides *_w_per_n values, this code will auto-convert
      them to kW/kg (heuristic based on magnitude).
    """

    p_w_climb_kw_per_kg: float = 0.1139
    p_w_cruise_kw_per_kg: float = 0.0876
    p_w_takeoff_kw_per_kg: float = 0.0739


@dataclass(frozen=True)
class CoolingConfig:
    dT_K: float = 60.0


@dataclass(frozen=True)
class HydrogenConfig:
    """Hydrogen storage and mission-range parameters."""

    rho_h2_kg_m3: float = 70.0
    eta_storage: float = 0.40
    eta_vol: float = 0.50
    coversize: float = 3.0

    # Mission/range parameters (legacy)
    range_total_m: float = 500_000.0
    h_to_m: float = 0.0
    vv_mps: float = 8.0


@dataclass(frozen=True)
class WingConfig:
    """Wing sizing and structural-weight parameters."""

    # Input-file units (new standard): kg/m^2 (mass-based wing loading).
    # Backward compatibility: if an input file provides wing_loading_pa, it will be
    # auto-converted to kg/m^2 (heuristic based on magnitude).
    wing_loading_kg_per_m2: float = 2830.24 / 9.81
    aspect_ratio: float = 10.0
    t_c: float = 0.20

    # Ultimate load factor (legacy uses 3*1.5)
    n_ult: float = 3.0 * 1.5

    # Sweep (deg), taper ratio
    lc4_deg: float = 0.0
    taper: float = 0.42


@dataclass(frozen=True)
class FuselageConfig:
    """Fuselage and cabin layout parameters."""

    dfus_m: float = 1.85

    # Geometric/empirical coefficients (legacy)
    finerat_fus_nose: float = 0.60
    finerat_fus_tail: float = 0.80

    # Fineness ratios used for nose/tail lengths
    fnose: float = 1.0
    ftail: float = 2.0

    # Cabin layout
    lseat_m: float = 0.8
    npass: int = 18
    nseat_abreast: int = 2
    ldoor_m: float = 1.0

    # Legacy tail arm value (stored in ft in the original code)
    lht_ft: float = 27.0

    # Compatibility switch:
    # The legacy script overwrote k_c_nose with a tail-fit polynomial (likely a bug).
    # Keeping this True preserves the legacy behaviour.
    legacy_kc_overwrite_bug: bool = True


@dataclass(frozen=True)
class WeightsConfig:
    """Non-structural mass items and payload assumptions."""

    oemmisc_base_kg: float = 0.0
    payload_kg: float = 2400.0

    # Battery energy density (kWh/kg)
    rhobatt_kwh_per_kg: float = 0.30

    # Battery mass reserve fraction used in legacy: mbatt = mbatt_old / (1 - reserve_fraction)
    battery_reserve_fraction: float = 0.25


@dataclass(frozen=True)
class FlightConfig:
    """Cruise and takeoff conditions."""

    h_cr_m: float = 3000.0
    mach_cr: float = 0.35

    # Takeoff modelling point (legacy)
    h_takeoff_m: float = 0.0
    mach_takeoff: float = 0.16


@dataclass(frozen=True)
class SolverConfig:
    power_tol_w: float = 100.0
    mtom_tol_kg: float = 1.0

    max_outer_iter: int = 50
    max_inner_iter: int = 50

    # Outer-loop MTOM closure algorithm:
    #   - 'newton' (default): safeguarded Newton/secant with trust region + bracketing fallback
    #   - 'fixed_point': legacy behaviour (mtom <- mtom_est)
    mtom_solver: str = "newton"

    # Newton/secant safeguards
    newton_max_rel_step: float = 0.50  # limit |ΔMTOM| ≤ this * MTOM (per outer iteration)
    newton_min_step_kg: float = 0.1  # avoid stalling when far from tolerance
    newton_slope_eps: float = 1e-6  # treat |dr/dMTOM| below this as unreliable

    # Damping / relaxation factors (applied to derivative-based steps)
    newton_relax_init: float = 1.0
    newton_relax_min: float = 0.1
    newton_relax_decrease: float = 0.5
    newton_relax_increase: float = 1.2

    # Fixed-point fallback relaxation (mtom_next = mtom + relax * residual)
    newton_fp_relax: float = 1.0

    # Bracketing: when a sign change in the residual is detected, keep the root bracketed
    # and fall back to bisection/regula-falsi if the Newton proposal is unsafe.
    newton_use_bracketing: bool = True



@dataclass(frozen=True)
class ConstraintSizingConfig:
    """ADRpy-based initial sizing / constraint coupling configuration.

    When enabled, HFCAD can call ADRpy's constraint analysis to derive:
      - wing loading (W/S) [Pa] -> mapped to HFCAD's wing_loading_kg_per_m2
      - power-to-weight ratios (P/W) [kW/kg] for takeoff, climb, cruise

    Two selection modes are supported:
      - selection = 'min_combined_pw': choose W/S that minimises ADRpy's combined P/W requirement
      - selection = 'min_mtom': scan W/S and run HFCAD convergence to minimise MTOM
    """

    enable: bool = False

    # 'min_combined_pw' (fast) or 'min_mtom' (coupled, expensive).
    selection: str = "min_combined_pw"

    # Wing-loading scan bounds in Pa (N/m^2). HFCAD uses kg/m^2 internally.
    ws_min_pa: float = 1500.0
    ws_max_pa: float = 3000.0
    ws_step_pa: float = 25.0

    # Optional refinement passes for selection='min_mtom' (0 disables refinement).
    ws_refine_passes: int = 0
    ws_refine_span_fraction: float = 0.20  # +/- span around current best (fraction of full range)

    # Multiplicative margin applied to all constraint-derived P/W outputs.
    pw_margin_fraction: float = 0.0

    # ADRpy propulsion identifier (affects altitude correction models). For H2 FC/electric, use 'electric'.
    propulsion_type: str = "electric"

    # Map which ADRpy constraint curve is used to populate HFCAD's phase P/W inputs.
    # Valid keys from ADRpy.powerrequired(): 'take-off', 'climb', 'cruise', 'turn', 'servceil', 'combined'
    takeoff_constraint: str = "take-off"
    climb_constraint: str = "climb"
    cruise_constraint: str = "cruise"

    # If True, force all three phase P/W values (takeoff/climb/cruise) to use the same curve (combined).
    use_combined_for_phases: bool = False

    # Scan verbosity control for selection='min_mtom'
    scan_quiet: bool = True

    # Optional artifact exports (CSV) written to the output directory.
    write_trade_csv: bool = True


@dataclass(frozen=True)
class ConstraintBriefConfig:
    """Design brief inputs for ADRpy constraint analysis.

    Defaults are set to match the uploaded notebook:
      'Constraint analysis - RIMP9 design case.ipynb'
    """

    # Take-off requirements
    rwyelevation_m: float = 0.0
    groundrun_m: float = 800.0

    # Turn requirements
    stloadfactor: float = 1.5
    turnalt_m: float = 3000.0
    turnspeed_ktas: float = 160.0

    # Climb requirements
    climbalt_m: float = 0.0
    climbspeed_kias: float = 170.0
    climbrate_fpm: float = 1200.0

    # Cruise requirements
    cruisealt_m: float = 3000.0
    cruisespeed_ktas: float = 220.0
    cruisethrustfact: float = 0.75

    # Service ceiling requirements
    servceil_m: float = 5000.0
    secclimbspd_kias: float = 120.0

    # Required clean stall speed
    vstallclean_kcas: float = 110.0


@dataclass(frozen=True)
class ConstraintPerformanceConfig:
    """Aerodynamic and propulsive assumptions for ADRpy constraint analysis."""

    # Take-off / low-speed aerodynamic coefficients
    cdto: float = 0.0414
    clto: float = 1.3
    clmax_to: float = 1.75

    # Clean (cruise/turn) aerodynamics
    clmax_clean: float = 1.45
    cdmin_clean: float = 0.0254

    # Rolling friction coefficient
    mu_r: float = 0.02

    # Propeller efficiency by phase (ADRpy uses phase-specific values)
    etaprop_takeoff: float = 0.70
    etaprop_climb: float = 0.80
    etaprop_cruise: float = 0.85
    etaprop_turn: float = 0.85
    etaprop_servceil: float = 0.80


@dataclass(frozen=True)
class ConstraintGeometryConfig:
    """Geometry inputs required by ADRpy's drag build-up / induced drag models."""

    # Leading-edge and mid-thickness sweep angles (deg). If unknown, set both ~0-5 deg.
    sweep_le_deg: float = 2.0
    sweep_mt_deg: float = 0.0


@dataclass(frozen=True)
class DesignConfig:
    """Top-level configuration for the design run."""

    mission: MissionProfile = MissionProfile()

    flight: FlightConfig = FlightConfig()

    fuel_cell_arch: FuelCellArchitecture = FuelCellArchitecture()
    fuel_cell_op: FuelCellOperation = FuelCellOperation()

    hybrid: HybridSplit = HybridSplit()
    eff: PowertrainEfficiencies = PowertrainEfficiencies()
    densities: PowertrainDensities = PowertrainDensities()
    p_w: PowerToWeightRatios = PowerToWeightRatios()

    cooling: CoolingConfig = CoolingConfig()
    hydrogen: HydrogenConfig = HydrogenConfig()

    wing: WingConfig = WingConfig()
    fuselage: FuselageConfig = FuselageConfig()
    weights: WeightsConfig = WeightsConfig()

    solver: SolverConfig = SolverConfig()

    # ADRpy-based constraint analysis coupling (optional)
    constraint_sizing: ConstraintSizingConfig = ConstraintSizingConfig()
    constraint_brief: ConstraintBriefConfig = ConstraintBriefConfig()
    constraint_performance: ConstraintPerformanceConfig = ConstraintPerformanceConfig()
    constraint_geometry: ConstraintGeometryConfig = ConstraintGeometryConfig()

    # Default initial guesses (legacy)
    initial_mtom_kg: float = 8000.0
    initial_total_power_guess_w: float = 200_000.0


# ============================
# Input file handling
# ============================


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _resolve_cruise_betas(
    beta_setting: object,
    *,
    cruise_density_kg_per_m3: float,
) -> Tuple[float, float]:
    """Resolve (beta_cruise, beta_cruise_charger) from user setting."""

    beta_default = 1.05

    if cruise_density_kg_per_m3 <= 0.0:
        raise ValueError(
            f"Cruise density must be positive to compute beta from density, got {cruise_density_kg_per_m3}."
        )

    if isinstance(beta_setting, bool):
        if beta_setting:
            beta_dyn = 1.2 / float(cruise_density_kg_per_m3)
            return float(beta_dyn), float(beta_dyn)
        return beta_default, beta_default

    if isinstance(beta_setting, (int, float)):
        beta_num = float(beta_setting)
        if beta_num <= 0.0:
            raise ValueError(f"fuel_cell_op.beta must be > 0 when numeric, got {beta_num}.")
        return beta_num, beta_default

    token = str(beta_setting).strip()
    if token == "":
        return beta_default, beta_default

    low = token.lower()
    if low in {"1", "true", "yes", "y", "on"}:
        beta_dyn = 1.2 / float(cruise_density_kg_per_m3)
        return float(beta_dyn), float(beta_dyn)
    if low in {"0", "false", "no", "n", "off"}:
        return beta_default, beta_default

    try:
        beta_num = float(token)
    except ValueError as e:
        raise ValueError(
            "Invalid fuel_cell_op.beta value. Use True/False or a numeric value (for cruise only)."
        ) from e

    if beta_num <= 0.0:
        raise ValueError(f"fuel_cell_op.beta must be > 0 when numeric, got {beta_num}.")

    # Numeric beta is applied only to cruise; cruise_charger remains fixed at default.
    return beta_num, beta_default


def _split_list(value: str) -> List[str]:
    """Split a comma/space separated list string into tokens."""

    v = value.strip().replace("\n", " ")
    # Remove common bracket wrappers
    if (v.startswith("(") and v.endswith(")")) or (v.startswith("[") and v.endswith("]")):
        v = v[1:-1].strip()

    if "," in v:
        parts = [p.strip() for p in v.split(",")]
    else:
        parts = [p.strip() for p in v.split()]

    return [p for p in parts if p]


def _normalize_p_w_section(section_dict: Dict[str, str]) -> Dict[str, str]:
    """Normalize [p_w] section keys/units.

    New standard:
      - p_w_*_kw_per_kg in kW/kg

    Backward compatibility:
      - accepts p_w_*_w_per_n (W/N) and converts to kW/kg
      - if a *_w_per_n key has a small magnitude (<=1), treat it as already kW/kg
        to avoid double-conversion when users update units but keep legacy key names.
    """

    g = 9.81
    out = dict(section_dict)

    alias = {
        "p_w_climb_w_per_n": "p_w_climb_kw_per_kg",
        "p_w_cruise_w_per_n": "p_w_cruise_kw_per_kg",
        "p_w_takeoff_w_per_n": "p_w_takeoff_kw_per_kg",
    }

    for old_key, new_key in alias.items():
        if new_key in out:
            # New key wins; drop the legacy key if present.
            out.pop(old_key, None)
            continue

        if old_key not in out:
            continue

        v = float(out[old_key])
        # Heuristic: W/N values are typically O(5~20). kW/kg is typically O(0.05~0.2).
        if v > 1.0:
            v = v * g / 1000.0
        out[new_key] = f"{v}"
        out.pop(old_key, None)

    return out


def _normalize_wing_section(section_dict: Dict[str, str]) -> Dict[str, str]:
    """Normalize [wing] section keys/units.

    New standard:
      - wing_loading_kg_per_m2 in kg/m^2

    Backward compatibility:
      - accepts wing_loading_pa (Pa) and converts to kg/m^2
      - if wing_loading_pa is already in the typical kg/m^2 range (<1000), treat it
        as kg/m^2 to avoid double-conversion.
    """

    g = 9.81
    out = dict(section_dict)

    new_key = "wing_loading_kg_per_m2"
    old_key = "wing_loading_pa"

    if new_key in out:
        out.pop(old_key, None)
        return out

    if old_key in out:
        v = float(out[old_key])
        # Heuristic: Pa values for commuter-class wing loading are typically >1000.
        if v > 1000.0:
            v = v / g
        out[new_key] = f"{v}"
        out.pop(old_key, None)

    return out


def _coerce_value(raw: str, typ):
    """Coerce a string value from the input file into the annotated field type."""
    s = raw.strip()

    # NEW: handle postponed annotations represented as strings (e.g., 'float', 'int', 'bool', 'str')
    if isinstance(typ, str):
        t = typ.replace(" ", "")
        if t in ("float", "builtins.float"):
            return float(s)
        if t in ("int", "builtins.int"):
            return int(float(s))  # tolerates "3000.0"
        if t in ("str", "builtins.str"):
            return s
        if t in ("bool", "builtins.bool"):
            v = s.lower()
            if v in ("1", "true", "yes", "y", "on"):
                return True
            if v in ("0", "false", "no", "n", "off"):
                return False
            raise ValueError(f"Invalid boolean literal: {raw!r}")

        # keep your existing tuple/list string handlers here too
        if t in ("Tuple[float,...]", "tuple[float,...]"):
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return tuple(float(p) for p in parts)
        if t in ("Tuple[int,...]", "tuple[int,...]"):
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return tuple(int(float(p)) for p in parts)
        if t in ("List[float]", "list[float]"):
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [float(p) for p in parts]
        if t in ("List[int]", "list[int]"):
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [int(float(p)) for p in parts]
        

    raw = raw.strip()

    # Strip quotes for simple string values
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]

    if typ is str:
        return raw
    if typ is float:
        return float(raw)
    if typ is int:
        return int(raw)
    if typ is bool:
        return _parse_bool(raw)

    origin = get_origin(typ)
    args = get_args(typ)

    if origin in (tuple, Tuple):
        # Tuple[T, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            elem_t = args[0]
            return tuple(_coerce_value(p, elem_t) for p in _split_list(raw))
        # Fixed-length tuple
        parts = _split_list(raw)
        if len(parts) != len(args):
            raise ValueError(f"Expected {len(args)} values for {typ}, got {len(parts)}: {raw!r}")
        return tuple(_coerce_value(p, t) for p, t in zip(parts, args))

    raise TypeError(f"Unsupported config field type {typ!r} for value {raw!r}")


def _update_dataclass_from_section(default_obj, section_name: str, section) -> object:
    """Return a new dataclass instance with fields overridden from a config section."""

    if not section:
        return default_obj

    allowed = {f.name for f in fields(default_obj)}
    unknown = sorted(set(section.keys()) - allowed)
    if unknown:
        raise KeyError(
            f"Unknown keys in section [{section_name}]: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}"
        )

    updates = {}
    for f in fields(default_obj):
        if f.name in section:
            updates[f.name] = _coerce_value(section[f.name], f.type)

    return replace(default_obj, **updates) if updates else default_obj


def load_design_config(input_path: Path) -> DesignConfig:
    """Load DesignConfig from an INI-style text file.

    The file extension can be .txt; the syntax is INI-like:
      [section]
      key = value

    Sections map to the nested dataclasses in DesignConfig:
      mission, flight, fuel_cell_arch, fuel_cell_op, hybrid, eff, densities, p_w,
      cooling, hydrogen, wing, fuselage, weights, solver,
      constraint_sizing, constraint_brief, constraint_performance, constraint_geometry,
      design
    """

    cfg_default = DesignConfig()

    cp = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
    )
    cp.optionxform = str

    read_ok = cp.read(str(input_path))
    if not read_ok:
        raise FileNotFoundError(
            f"Input file not found or unreadable: {input_path}. "
            "Create it (or run with --write-template) and try again."
        )

    def section(name: str):
        return cp[name] if cp.has_section(name) else None

    def section_dict(name: str) -> Optional[Dict[str, str]]:
        sec = section(name)
        return dict(sec) if sec is not None else None

    mission = _update_dataclass_from_section(cfg_default.mission, "mission", section("mission"))
    flight = _update_dataclass_from_section(cfg_default.flight, "flight", section("flight"))

    fuel_cell_arch = _update_dataclass_from_section(cfg_default.fuel_cell_arch, "fuel_cell_arch", section("fuel_cell_arch"))
    fuel_cell_op = _update_dataclass_from_section(cfg_default.fuel_cell_op, "fuel_cell_op", section("fuel_cell_op"))

    hybrid = _update_dataclass_from_section(cfg_default.hybrid, "hybrid", section("hybrid"))
    eff = _update_dataclass_from_section(cfg_default.eff, "eff", section("eff"))
    densities = _update_dataclass_from_section(cfg_default.densities, "densities", section("densities"))

    p_w_sec = section_dict("p_w")
    if p_w_sec is not None:
        p_w_sec = _normalize_p_w_section(p_w_sec)
    p_w = _update_dataclass_from_section(cfg_default.p_w, "p_w", p_w_sec)

    cooling = _update_dataclass_from_section(cfg_default.cooling, "cooling", section("cooling"))
    hydrogen = _update_dataclass_from_section(cfg_default.hydrogen, "hydrogen", section("hydrogen"))

    wing_sec = section_dict("wing")
    if wing_sec is not None:
        wing_sec = _normalize_wing_section(wing_sec)
    wing = _update_dataclass_from_section(cfg_default.wing, "wing", wing_sec)
    fuselage = _update_dataclass_from_section(cfg_default.fuselage, "fuselage", section("fuselage"))
    weights = _update_dataclass_from_section(cfg_default.weights, "weights", section("weights"))

    solver = _update_dataclass_from_section(cfg_default.solver, "solver", section("solver"))

    constraint_sizing = _update_dataclass_from_section(
        cfg_default.constraint_sizing, "constraint_sizing", section("constraint_sizing")
    )
    constraint_brief = _update_dataclass_from_section(
        cfg_default.constraint_brief, "constraint_brief", section("constraint_brief")
    )
    constraint_performance = _update_dataclass_from_section(
        cfg_default.constraint_performance, "constraint_performance", section("constraint_performance")
    )
    constraint_geometry = _update_dataclass_from_section(
        cfg_default.constraint_geometry, "constraint_geometry", section("constraint_geometry")
    )

    # Top-level fields
    design_section = section("design")
    if design_section is not None:
        allowed_design = {"initial_mtom_kg", "initial_total_power_guess_w"}
        unknown_design = sorted(set(design_section.keys()) - allowed_design)
        if unknown_design:
            raise KeyError(
                f"Unknown keys in section [design]: {', '.join(unknown_design)}. "
                f"Allowed keys: {', '.join(sorted(allowed_design))}"
            )

    initial_mtom_kg = cfg_default.initial_mtom_kg
    if design_section is not None and "initial_mtom_kg" in design_section:
        initial_mtom_kg = float(design_section["initial_mtom_kg"])

    initial_total_power_guess_w = cfg_default.initial_total_power_guess_w
    if design_section is not None and "initial_total_power_guess_w" in design_section:
        initial_total_power_guess_w = float(design_section["initial_total_power_guess_w"])

    return DesignConfig(
        mission=mission,
        flight=flight,
        fuel_cell_arch=fuel_cell_arch,
        fuel_cell_op=fuel_cell_op,
        hybrid=hybrid,
        eff=eff,
        densities=densities,
        p_w=p_w,
        cooling=cooling,
        hydrogen=hydrogen,
        wing=wing,
        fuselage=fuselage,
        weights=weights,
        solver=solver,
        constraint_sizing=constraint_sizing,
        constraint_brief=constraint_brief,
        constraint_performance=constraint_performance,
        constraint_geometry=constraint_geometry,
        initial_mtom_kg=float(initial_mtom_kg),
        initial_total_power_guess_w=float(initial_total_power_guess_w),
    )


def write_input_template(path: Path, cfg: Optional[DesignConfig] = None) -> None:
    """Write a complete input file template with the current default values."""

    if cfg is None:
        cfg = DesignConfig()

    def fmt(v):
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, float):
            # Keep a stable representation without scientific notation unless necessary.
            return f"{v:.12g}"
        if isinstance(v, tuple):
            return ", ".join(fmt(x) for x in v)
        return str(v)

    lines: List[str] = []
    lines.append("# HFCAD input file (INI-style).")
    lines.append("# Edit values as needed. Units are indicated in the parameter names.")
    lines.append("# Lines starting with '#' or ';' are comments.")
    lines.append("")

    def section(name: str, obj) -> None:
        lines.append(f"[{name}]")
        for f in fields(obj):
            lines.append(f"{f.name} = {fmt(getattr(obj, f.name))}")
        lines.append("")

    section("mission", cfg.mission)
    section("flight", cfg.flight)
    section("fuel_cell_arch", cfg.fuel_cell_arch)
    section("fuel_cell_op", cfg.fuel_cell_op)
    section("hybrid", cfg.hybrid)
    section("eff", cfg.eff)
    section("densities", cfg.densities)
    section("p_w", cfg.p_w)
    section("cooling", cfg.cooling)
    section("hydrogen", cfg.hydrogen)
    section("wing", cfg.wing)
    section("fuselage", cfg.fuselage)
    section("weights", cfg.weights)
    section("solver", cfg.solver)

    section("constraint_sizing", cfg.constraint_sizing)
    section("constraint_brief", cfg.constraint_brief)
    section("constraint_performance", cfg.constraint_performance)
    section("constraint_geometry", cfg.constraint_geometry)

    lines.append("[design]")
    lines.append(f"initial_mtom_kg = {fmt(cfg.initial_mtom_kg)}")
    lines.append(f"initial_total_power_guess_w = {fmt(cfg.initial_total_power_guess_w)}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

# ============================
# Result objects (dataclasses)
# ============================


@dataclass(frozen=True)
class PowerSplitResult:
    p_fuelcell_w: float
    p_battery_w: float  # >0 discharge, <0 charge

    @property
    def p_bus_required_w(self) -> float:
        """Total bus power that must be produced by the electrical sources.

        Preserves legacy behaviour: charging is modelled as additional required generation.
        """

        return self.p_fuelcell_w + abs(self.p_battery_w)


@dataclass(frozen=True)
class FuelCellSystemSizingResult:
    """Per-nacelle fuel cell system sizing output."""

    m_sys_kg: float
    m_stacks_kg: float
    m_comp_kg: float
    m_humid_kg: float
    m_hx_kg: float

    eta_fcsys: float
    mdot_h2_kgps: float

    power_comp_w: float
    q_all_w: float

    v_cr_mps: float

    dim_stack_m: Tuple[float, float, float]
    dim_hx_m: Tuple[float, float, float]
    res_stack: Tuple[float, float]

    figs: Tuple[object, ...] = ()  # Plotly figures, optional


@dataclass(frozen=True)
class PhasePowerResult:
    """Solved power balance for a phase (all nacelles combined)."""

    name: str
    mtom_kg: float

    p_shaft_w: float

    p_fuelcell_w: float
    p_battery_w: float
    p_bus_required_w: float

    p_comp_w: float
    p_cooling_w: float

    p_total_w: float

    heat_rejected_kw: float
    mdot_h2_kgps: float

    # Optional detailed per-nacelle sizing at the converged point
    nacelle: FuelCellSystemSizingResult


@dataclass(frozen=True)
class MassBreakdown:
    mtom_kg: float

    # Propulsion / powertrain
    m_fc_system_kg: float
    m_pmad_kg: float
    m_e_motor_kg: float
    m_powertrain_total_kg: float

    # Energy storage
    m_fuel_kg: float
    m_tank_kg: float
    m_battery_kg: float

    # Airframe
    w_wing_kg: float
    w_fus_kg: float
    w_ht_kg: float
    w_vt_kg: float
    w_lnd_main_kg: float
    w_lnd_nose_kg: float

    # Misc systems
    w_motor_misc_kg: float
    w_flight_control_kg: float
    w_els_kg: float
    w_iae_kg: float
    w_hydraulics_kg: float
    w_furnishings_kg: float

    oem_misc_kg: float
    oem_kg: float

    payload_kg: float

    # Geometry of interest
    wing_span_m: float
    wing_area_m2: float
    fuselage_length_m: float
    tank_length_m: float
    tank_volume_m3: float

    # Derived/aux
    p_fuelcell_engine_w: float
    p_fuelcell_taxing_w: float

    # For reporting
    nacelle_design_power_kw_per_kg: float
    nacelle_stack_dim_m: Tuple[float, float, float]
    nacelle_hx_dim_m: Tuple[float, float, float]


# ============================
# Domain models / services
# ============================


class HybridPowertrain:
    """Handles power split calculations between fuel cell and battery."""

    def __init__(self, eff: PowertrainEfficiencies):
        self._eff = eff

    def split_shaft_power(self, p_shaft_w: float, psi: float) -> PowerSplitResult:
        """Compute fuel-cell and battery powers at the electrical bus.

        This preserves the legacy algebra exactly.
        """

        eta_chain = (
            self._eff.eta_inverter
            * self._eff.eta_motor
            * self._eff.eta_pdu
            * self._eff.eta_prop
        )

        denom = psi + self._eff.eta_converter * (1.0 - psi)
        if denom == 0:
            raise ZeroDivisionError("Invalid hybrid split: psi + eta_converter*(1-psi) equals zero.")

        p_fuelcell = (1.0 / eta_chain) * ((1.0 - psi) / denom) * p_shaft_w
        p_battery = (1.0 / eta_chain) * (psi / denom) * p_shaft_w

        return PowerSplitResult(p_fuelcell_w=float(p_fuelcell), p_battery_w=float(p_battery))


class CoolingSystem:
    """Simple cooling power model (legacy correlation)."""

    def __init__(self, *, cruise_altitude_m: float, cruise_mach: float, cfg: CoolingConfig):
        self._cfg = cfg

        # Legacy uses cruise ambient temperature to compute f_dT and then reuses it for all phases.
        t_air = float(Atmosphere(cruise_altitude_m).temperature[0])
        dT = float(cfg.dT_K)

        self._f_dT = 0.0038 * ((t_air / dT) ** 2) + 0.0352 * (t_air / dT) + 0.1817

    def power_required_w(self, heat_rejected_kw: float) -> float:
        """Compute cooling system electrical power (W).

        Parameters
        ----------
        heat_rejected_kw:
            Heat rejected (kW). This is consistent with legacy usage.
        """

        return float((0.371 * heat_rejected_kw + 1.33) * self._f_dT * 1000.0)


class FuelCellSystemModel:
    """Fuel cell system sizing model (per nacelle)."""

    def __init__(self, arch: FuelCellArchitecture, comp_stl_path: Optional[Path] = None):
        self._arch = arch
        self._comp_stl_path = comp_stl_path

    def size_nacelle(
        self,
        *,
        power_fc_sys_w: float,
        flight_point: FlightPoint,
        beta: float,
        oversizing: float,
        comp_bool: bool = True,
        make_fig: bool = False,
        verbose: bool = False,
        comp_tol_w: float = 1.0,
        max_comp_iter: int = 50,
    ) -> FuelCellSystemSizingResult:
        """Size a fuel cell system for a single nacelle.

        Parameters are aligned with the legacy `FuelCellSystem.size_system` routine.
        """

        # Atmospheric conditions
        atm = Atmosphere(flight_point.altitude_m)
        c = float(atm.speed_of_sound[0])
        v_cr = float(flight_point.mach * c)
        p = float(atm.pressure[0])
        p_tot = float(p * (1 + 0.4 / 2 * flight_point.mach**2) ** (1.4 / 0.4))
        t = float(atm.temperature[0])
        t_tot = float(t * (1 + 0.4 / 2 * flight_point.mach**2))
        rho = float(atm.density[0])
        mu = float(atm.dynamic_viscosity[0])

        if verbose:
            try:
                logger.info(f"Reynolds_number: {rho * v_cr * 1.8 / mu:,.0f}")
            except Exception:
                pass

        # Other inputs
        cell_temp = 273.15 + 80.0
        mu_f = 0.95

        # Cathode inlet pressure
        pres_cathode_in = float(beta * p_tot if comp_bool else p_tot)

        # Cell model (cached and optionally figure-producing)
        pres_h = float(Atmosphere(0).pressure[0])
        volt_cell, power_dens_cell, eta_cell, fig = cell_model(
            pres_cathode_in,
            pres_h,
            cell_temp,
            oversizing,
            make_fig=make_fig,
        )

        figs: Tuple[object, ...] = (fig,) if fig is not None else ()

        # Compressor
        if comp_bool:
            power_req_new = float(power_fc_sys_w)
            power_comp = 0.0
            geom_comp = None
            rho_humid_in = rho
            m_dot_comp = None

            tol = max(float(comp_tol_w), 1e-6 * abs(power_req_new))

            for _ in range(int(max_comp_iter)):
                power_req = power_req_new
                geom_comp, power_comp, rho_humid_in, m_dot_comp = compressor_performance_model(
                    power_req,
                    volt_cell,
                    float(beta),
                    p_tot,
                    t_tot,
                    mu,
                )
                power_req_new = float(power_fc_sys_w) + float(power_comp)
                if abs(power_req_new - power_req) <= tol:
                    break
            else:
                if verbose:
                    logger.warning(
                        "Compressor iteration hit max_comp_iter=%s (|ΔP|=%.3f W, tol=%.3f W).",
                        max_comp_iter,
                        abs(power_req_new - power_req),
                        tol,
                    )

            m_comp = (
                float(compressor_mass_model(geom_comp, power_comp, stl_path=self._comp_stl_path))
                if geom_comp is not None
                else 0.0
            )
        else:
            m_comp = 0.0
            power_comp = 0.0
            power_req_new = float(power_fc_sys_w)
            m_dot_comp = float(mass_flow_stack(power_req_new, volt_cell))
            rho_humid_in = rho

        # Remaining BOP models
        m_humid = float(humidifier_model(m_dot_comp, rho_humid_in))
        q_all, m_hx, dim_hx = heat_exchanger_model(
            power_req_new,
            volt_cell,
            cell_temp,
            mu_f,
            v_cr,
            float(flight_point.mach),
            p_tot,
            t_tot,
            rho,
            mu,
        )

        # Stack model
        m_stacks, dim_stack, res_stack = stack_model(
            self._arch.n_stacks_series,
            self._arch.volt_req_v,
            volt_cell,
            power_req_new,
            power_dens_cell,
        )

        # Aggregate
        m_sys = float(m_stacks + m_comp + m_humid + m_hx)

        eta_fcsys = float(eta_cell * float(power_fc_sys_w) / (float(power_comp) + float(power_fc_sys_w)) * mu_f)

        mdot_h2 = float(1.05e-8 * (float(power_comp) + float(power_fc_sys_w)) / volt_cell)

        if verbose:
            logger.info(f"Stack prop output power: {power_fc_sys_w/1000:,.0f} kW, Pcomp: {power_comp/1000:,.1f} kW")
            logger.info(f"Cell efficiency: {eta_cell:,.3f}, Output efficiency: {eta_fcsys:,.3f}")
            logger.info(f"mdot_h2: {mdot_h2*1000:,.2f} g/s")

        return FuelCellSystemSizingResult(
            m_sys_kg=float(m_sys),
            m_stacks_kg=float(m_stacks),
            m_comp_kg=float(m_comp),
            m_humid_kg=float(m_humid),
            m_hx_kg=float(m_hx),
            eta_fcsys=float(eta_fcsys),
            mdot_h2_kgps=float(mdot_h2),
            power_comp_w=float(power_comp),
            q_all_w=float(q_all),
            v_cr_mps=float(v_cr),
            dim_stack_m=(float(dim_stack[0]), float(dim_stack[1]), float(dim_stack[2])),
            dim_hx_m=(float(dim_hx[0]), float(dim_hx[1]), float(dim_hx[2])),
            res_stack=(float(res_stack[0]), float(res_stack[1])),
            figs=figs,
        )


class PhasePowerSolver:
    """Solves the coupled power balance for a flight phase."""

    def __init__(
        self,
        *,
        config: DesignConfig,
        powertrain: HybridPowertrain,
        fc_model: FuelCellSystemModel,
        cooling: CoolingSystem,
    ):
        self._cfg = config
        self._powertrain = powertrain
        self._fc = fc_model
        self._cooling = cooling

    def solve(
        self,
        *,
        name: str,
        mtom_kg: float,
        p_w_kw_per_kg: float,
        flight_point: FlightPoint,
        psi: float,
        beta: float,
        initial_total_power_w: float,
        oversizing: float,
    ) -> PhasePowerResult:
        """Fixed-point iteration for total electrical power including auxiliaries."""

        p_total = float(initial_total_power_w)

        for inner_iter in range(1, self._cfg.solver.max_inner_iter + 1):
            # New standard input: kW/kg
            p_shaft = float(mtom_kg * p_w_kw_per_kg * 1000.0)

            split = self._powertrain.split_shaft_power(p_shaft, psi)
            p_bus_required = split.p_bus_required_w

            # Fuel-cell net electrical output used for per-nacelle sizing.
            #
            # IMPORTANT:
            # - size_nacelle(power_fc_sys_w=...) expects the **net** FC electrical output that goes to the propulsive bus
            #   (it accounts for compressor parasitics internally via: power_req_new = power_fc_sys_w + power_comp).
            # - Therefore, do NOT base this on p_total (which already includes compressor/cooling), or auxiliaries
            #   get double-counted and p_total can inflate substantially (often ~2× in climb).
            #
            # Battery convention: p_battery_w > 0 discharge, p_battery_w < 0 charge.
            # - Discharge: FC must provide only its share (subtract battery contribution).
            # - Charge: FC must provide propulsion plus charging demand.
            p_fc_sys_total = split.p_bus_required_w - max(split.p_battery_w, 0.0)

            # Guardrail: ensure sizing power stays positive.
            p_fc_sys_total = max(p_fc_sys_total, 1.0)

            p_fc_sys_per_nacelle = p_fc_sys_total / self._cfg.fuel_cell_arch.n_stacks_parallel

            nacelle = self._fc.size_nacelle(
                power_fc_sys_w=p_fc_sys_per_nacelle,
                flight_point=flight_point,
                beta=beta,
                oversizing=oversizing,
                comp_bool=True,
                make_fig=False,
                verbose=True,
            )

            p_comp_total = nacelle.power_comp_w * self._cfg.fuel_cell_arch.n_stacks_parallel
            heat_rejected_kw = (self._cfg.fuel_cell_arch.n_stacks_parallel * nacelle.q_all_w) / 1000.0
            p_cooling = self._cooling.power_required_w(heat_rejected_kw)

            p_total_new = float(p_bus_required + p_comp_total + p_cooling)

            if abs(p_total_new - p_total) <= self._cfg.solver.power_tol_w:
                return PhasePowerResult(
                    name=name,
                    mtom_kg=float(mtom_kg),
                    p_shaft_w=float(p_shaft),
                    p_fuelcell_w=float(split.p_fuelcell_w),
                    p_battery_w=float(split.p_battery_w),
                    p_bus_required_w=float(p_bus_required),
                    p_comp_w=float(p_comp_total),
                    p_cooling_w=float(p_cooling),
                    p_total_w=float(p_total_new),
                    heat_rejected_kw=float(heat_rejected_kw),
                    mdot_h2_kgps=float(nacelle.mdot_h2_kgps * self._cfg.fuel_cell_arch.n_stacks_parallel),
                    nacelle=nacelle,
                )

            p_total = p_total_new

        raise RuntimeError(f"Phase '{name}' did not converge within {self._cfg.solver.max_inner_iter} iterations.")


class MassEstimator:
    """Computes MTOM from phase results and empirical weight models."""

    def __init__(self, *, cfg: DesignConfig, comp_stl_path: Optional[Path] = None):
        self._cfg = cfg
        self._comp_stl_path = comp_stl_path
        self._conv = Conversions()

        # Pre-compute cruise environment
        atm_cr = Atmosphere(cfg.flight.h_cr_m)
        self._c_cr_mps = float(atm_cr.speed_of_sound[0])
        self._v_cr_mps = float(cfg.flight.mach_cr * self._c_cr_mps)
        self._rho_cr = float(atm_cr.density[0])
        self._beta_cruise, _ = _resolve_cruise_betas(
            cfg.fuel_cell_op.beta,
            cruise_density_kg_per_m3=self._rho_cr,
        )

        # Wing loading (mass-based) in kg/m^2
        self._w_s_kg_per_m2 = float(cfg.wing.wing_loading_kg_per_m2)

    @property
    def v_cruise_mps(self) -> float:
        return self._v_cr_mps

    def estimate(
        self,
        *,
        mtom_guess_kg: float,
        climb: PhasePowerResult,
        cruise: PhasePowerResult,
    ) -> MassBreakdown:
        cfg = self._cfg
        conv = self._conv

        # -----------------
        # Fuel cell sizing at climb (design) power
        # -----------------
        # Fuel-cell net output used for sizing (consistent with PhasePowerSolver).
        power_fc_sys_total = climb.p_bus_required_w - max(climb.p_battery_w, 0.0)
        power_fc_sys_total = max(power_fc_sys_total, 1.0)
        power_fc_sys_per_nacelle = power_fc_sys_total / cfg.fuel_cell_arch.n_stacks_parallel

        nacelle_design = FuelCellSystemModel(
            cfg.fuel_cell_arch,
            comp_stl_path=self._comp_stl_path,
        ).size_nacelle(
            power_fc_sys_w=power_fc_sys_per_nacelle,
            flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
            beta=self._beta_cruise,
            oversizing=cfg.fuel_cell_op.oversizing,
            comp_bool=True,
            make_fig=False,
            verbose=True,
        )

        # Mass per nacelle: use cruise-governing compressor mass for MTOM summation.
        m_comp_cruise_per_nacelle = float(cruise.nacelle.m_comp_kg)
        m_comp_climb_per_nacelle = float(climb.nacelle.m_comp_kg)
        m_comp_max_per_nacelle = max(m_comp_cruise_per_nacelle, m_comp_climb_per_nacelle)
        if m_comp_max_per_nacelle > m_comp_cruise_per_nacelle:
            logger.warning(
                "Compressor mass max is not at cruise (cruise=%.3f kg, climb=%.3f kg). "
                "Using cruise compressor mass in MTOM summation.",
                m_comp_cruise_per_nacelle,
                m_comp_climb_per_nacelle,
            )

        m_fc_system_per_nacelle = float(
            nacelle_design.m_stacks_kg
            + nacelle_design.m_humid_kg
            + m_comp_cruise_per_nacelle
            + nacelle_design.m_hx_kg
        )
        m_fc_system = float(cfg.fuel_cell_arch.n_stacks_parallel * m_fc_system_per_nacelle)

        # -----------------
        # Battery mass (legacy)
        # -----------------
        # mbatt_old = (Pbat_climb * 0.234) / (rhobatt * 1000)
        # mbatt = mbatt_old / (1 - reserve_fraction)
        mbatt_old = (climb.p_battery_w * 0.234) / (cfg.weights.rhobatt_kwh_per_kg * 1000.0)
        m_battery = float(mbatt_old / (1.0 - cfg.weights.battery_reserve_fraction))

        # -----------------
        # PMAD and motor sizing (legacy)
        # -----------------
        # NOTE: legacy PPDU expression includes a thermal term (heat_rejected_kw) without unit conversion.
        # This is retained for backward compatibility.
        ppdu_w = (climb.p_fuelcell_w + climb.heat_rejected_kw + climb.p_cooling_w) * cfg.eff.eta_converter + climb.p_battery_w
        pem_w = (climb.p_fuelcell_w * cfg.eff.eta_converter + climb.p_battery_w) * cfg.eff.eta_pdu * cfg.eff.eta_inverter

        m_e_motor = float(pem_w / cfg.densities.rhoem_w_per_kg)
        m_pmad = float(ppdu_w / cfg.densities.rhopmad_w_per_kg)

        m_powertrain_total = float(m_fc_system + m_pmad + m_e_motor)

        # -----------------
        # Fuel mass and tank sizing
        # -----------------
        # Range split (legacy)
        t_climb_s = (cfg.flight.h_cr_m - cfg.hydrogen.h_to_m) / cfg.hydrogen.vv_mps
        v_climb_mps = float(math.sqrt(cfg.hydrogen.vv_mps**2 + self._v_cr_mps**2))
        r_climb_m = float(v_climb_mps * t_climb_s)
        r_descent_m = r_climb_m
        r_cruise_m = float(cfg.hydrogen.range_total_m - r_climb_m - r_descent_m)

        mdot_cruise = float(cruise.mdot_h2_kgps)
        mdot_climb = float(climb.mdot_h2_kgps)

        # Legacy uses v_cr from the FC sizing model; we use the precomputed cruise TAS.
        m_fuel = float(
            mdot_cruise * (r_cruise_m + r_descent_m) / self._v_cr_mps
            + mdot_climb * (r_climb_m) / v_climb_mps
        )

        tank_volume_m3 = float(m_fuel * cfg.hydrogen.coversize / cfg.hydrogen.rho_h2_kg_m3 / cfg.hydrogen.eta_vol)
        tank_length_m = float(tank_volume_m3 / (math.pi * (cfg.fuselage.dfus_m / 2.0) ** 2))

        m_tank = float(m_fuel * cfg.hydrogen.coversize * (1.0 / cfg.hydrogen.eta_storage - 1.0))

        # -----------------
        # Wing sizing (geometry)
        # -----------------
        # New standard input is wing loading in kg/m^2 (mass/area).
        # Use metric directly to avoid mixing force-based and mass-based definitions.
        wing_area_m2 = float(mtom_guess_kg / self._w_s_kg_per_m2)
        wing_span_m = float(math.sqrt(cfg.wing.aspect_ratio * wing_area_m2))

        # -----------------
        # Fuselage sizing (geometry + wetted area)
        # -----------------
        dfus_ft = float(conv.meter_feet(cfg.fuselage.dfus_m, "ft"))
        hfus_ft = dfus_ft

        # Nose/tail coefficients
        fr_n = cfg.fuselage.finerat_fus_nose
        fr_t = cfg.fuselage.finerat_fus_tail

        k_w_nose = -0.603291 * fr_n**2 + 2.17154 * fr_n - 0.425122
        k_w_tail = -0.603291 * fr_t**2 + 2.17154 * fr_t - 0.425122

        k_c_nose = -1.72626 * fr_n**3 + 4.43622 * fr_n**2 - 3.05539 * fr_n + 1.3414
        k_c_tail = -1.72626 * fr_t**3 + 4.43622 * fr_t**2 - 3.05539 * fr_t + 1.3414

        if cfg.fuselage.legacy_kc_overwrite_bug:
            # Preserve legacy behaviour (tail polynomial overwrote k_c_nose)
            k_c_used = k_c_tail
        else:
            k_c_used = k_c_nose

        circum_fus_ft = float(2.0 * k_c_used * (dfus_ft + hfus_ft))

        lnose_ft = float(dfus_ft * cfg.fuselage.fnose)
        ltail_ft = float(dfus_ft * cfg.fuselage.ftail)

        lcabin_m = float((cfg.fuselage.npass / cfg.fuselage.nseat_abreast) * cfg.fuselage.lseat_m + cfg.fuselage.ldoor_m)
        lcabin_ft = float(conv.meter_feet(lcabin_m, "ft"))

        ltank_ft = float(conv.meter_feet(tank_length_m, "ft"))
        lfus_ft = float(lnose_ft + ltail_ft + lcabin_ft + ltank_ft)

        swet_fus_ft2 = float(
            circum_fus_ft * ((lcabin_ft + ltank_ft) + k_w_nose * lnose_ft + k_w_tail * ltail_ft)
        )
        swet_fus_m2 = float(conv.m2_ft2(swet_fus_ft2, "m2"))

        fuselage_length_m = float(conv.meter_feet(lfus_ft, "meter"))

        # -----------------
        # Empirical weight estimation
        # -----------------
        west = WeightEst(
            W=mtom_guess_kg,
            W_fw=0.0,
            S=wing_area_m2,
            b=wing_span_m,
            rho_a_cruise=self._rho_cr,
            v_cruise=self._v_cr_mps,
            t_r_HT=0.09,
            S_HT=1.82,
            S_VT=2.54,
            t_r_VT=0.09 * 1.361,
            L_HT_act=float(conv.meter_feet(cfg.fuselage.lht_ft, "meter")),
            b_HT=3.0,
            b_VT=2.5,
            FL=fuselage_length_m,
            Wf_mm=float(cfg.fuselage.dfus_m * 1000.0),
            hf_mm=float(cfg.fuselage.dfus_m * 1000.0),
            W_press=0.0,
            l_n_mm=float(conv.meter_feet(lnose_ft, "meter") * 1000.0),
            Croot=1.5,
            tc_r=cfg.wing.t_c,
            n_ult=cfg.wing.n_ult,
            Swet_fus=swet_fus_m2,
        )

        (
            w_wing_imp,
            w_wing,
            w_ht_imp,
            w_ht,
            w_vt_imp,
            w_vt,
            w_fus_imp,
            w_fus,
            w_lnd_main_imp,
            w_lnd_main,
            w_lnd_nose_imp,
            w_lnd_nose,
        ) = west.raymermethod(
            A=cfg.wing.aspect_ratio,
            theta_c4=(math.radians(cfg.wing.lc4_deg)),
            lamda=cfg.wing.taper,
            tc=cfg.wing.t_c,
            theta_c4_HT=(math.radians(2.671)),
            lamda_HT=0.4,
            theta_c4_VT=(math.radians(19.53)),
            lamda_VT=0.4,
            LD=14.5,
            Kmp=1.0,
            W_l=float(conv.kg_pound(8000.0, "pound")),
            N_l=3.0 * 1.5,
            L_m=float(conv.meter_inch(0.9, "inch")),
            N_mw=1,
            N_mss=1,
            V_stall=float(conv.km1h_kn(58.32, "kn")),
            Knp=1.0,
            L_n=float(conv.meter_inch(0.90, "inch")),
            N_nw=1,
            K_uht=1.0,
            F_w=float(conv.meter_feet(1.11, "ft")),
            L_t=float(conv.meter_feet(10.74, "ft")),
            K_y=float(conv.meter_feet(3.22, "ft")),
            A_ht=5,
            S_e=float(conv.m2_ft2(0.73, "ft2")),
            H_t__H_v=1.0,
            K_z=float(conv.meter_feet(10.7, "ft")),
            t_c=0.09,
        )

        (
            w_motor_misc,
            w_flight_control,
            w_els,
            w_iae,
            w_furnishings,
            w_hydraulics,
        ) = west.miscweightest(
            pow_max=380,
            N_f=6,
            N_m=2,
            S_cs=float(conv.m2_ft2(8.36, "ft2")),
            Iyaw=4.49 * 10**6,
            R_kva=50,
            L_a=float(conv.meter_feet(20.0, "ft")),
            N_gen=1,
            N_e=4,
            W_TO=float(conv.kg_pound(mtom_guess_kg, "pound")),
            N_c=2,
            W_c=float(conv.kg_pound(10 * 19.0, "pound")),
            S_f=float(swet_fus_ft2 * 23.0),
            N_pil=2,
        )

        oem_misc = float(
            cfg.weights.oemmisc_base_kg
            + w_ht
            + w_vt
            + w_lnd_main
            + w_lnd_nose
            + w_flight_control
            + w_els
            + w_iae
            + w_hydraulics
            + w_furnishings
        )

        oem = float(oem_misc + m_powertrain_total + m_tank + w_wing + w_fus + m_battery)
        mtom = float(oem + m_fuel + cfg.weights.payload_kg)

        p_total_climb_w = float(climb.p_total_w)
        p_fuelcell_engine_w = float(p_total_climb_w * 0.05)
        p_fuelcell_taxing_w = float(p_total_climb_w * 0.10)

        nacelle_pd_kw_per_kg = float((power_fc_sys_per_nacelle / 1000.0) / m_fc_system_per_nacelle)

        return MassBreakdown(
            mtom_kg=mtom,
            m_fc_system_kg=float(m_fc_system),
            m_pmad_kg=float(m_pmad),
            m_e_motor_kg=float(m_e_motor),
            m_powertrain_total_kg=float(m_powertrain_total),
            m_fuel_kg=float(m_fuel),
            m_tank_kg=float(m_tank),
            m_battery_kg=float(m_battery),
            w_wing_kg=float(w_wing),
            w_fus_kg=float(w_fus),
            w_ht_kg=float(w_ht),
            w_vt_kg=float(w_vt),
            w_lnd_main_kg=float(w_lnd_main),
            w_lnd_nose_kg=float(w_lnd_nose),
            w_motor_misc_kg=float(w_motor_misc),
            w_flight_control_kg=float(w_flight_control),
            w_els_kg=float(w_els),
            w_iae_kg=float(w_iae),
            w_hydraulics_kg=float(w_hydraulics),
            w_furnishings_kg=float(w_furnishings),
            oem_misc_kg=float(oem_misc),
            oem_kg=float(oem),
            payload_kg=float(cfg.weights.payload_kg),
            wing_span_m=float(wing_span_m),
            wing_area_m2=float(wing_area_m2),
            fuselage_length_m=float(fuselage_length_m),
            tank_length_m=float(tank_length_m),
            tank_volume_m3=float(tank_volume_m3),
            p_fuelcell_engine_w=float(p_fuelcell_engine_w),
            p_fuelcell_taxing_w=float(p_fuelcell_taxing_w),
            nacelle_design_power_kw_per_kg=float(nacelle_pd_kw_per_kg),
            nacelle_stack_dim_m=nacelle_design.dim_stack_m,
            nacelle_hx_dim_m=nacelle_design.dim_hx_m,
        )


class HybridFuelCellAircraftDesign:
    """Top-level orchestrator."""

    def __init__(self, cfg: DesignConfig, out_dir: Optional[Path] = None):
        self._cfg = cfg
        self._comp_stl_path = (out_dir / "media" / "comp.stl") if out_dir is not None else None
        self._rho_cr = float(Atmosphere(cfg.flight.h_cr_m).density[0])
        self._beta_cruise, self._beta_cruise_charger = _resolve_cruise_betas(
            cfg.fuel_cell_op.beta,
            cruise_density_kg_per_m3=self._rho_cr,
        )

        self._powertrain = HybridPowertrain(cfg.eff)
        self._cooling = CoolingSystem(
            cruise_altitude_m=cfg.flight.h_cr_m,
            cruise_mach=cfg.flight.mach_cr,
            cfg=cfg.cooling,
        )
        self._fc_model = FuelCellSystemModel(cfg.fuel_cell_arch, comp_stl_path=self._comp_stl_path)
        self._phase_solver = PhasePowerSolver(
            config=cfg,
            powertrain=self._powertrain,
            fc_model=self._fc_model,
            cooling=self._cooling,
        )
        self._mass_estimator = MassEstimator(cfg=cfg, comp_stl_path=self._comp_stl_path)

    def run(
        self,
        *,
        initial_mtom_kg: Optional[float] = None,
        initial_total_power_guess_w: Optional[float] = None,
    ) -> Tuple[Dict[str, PhasePowerResult], MassBreakdown]:
        cfg = self._cfg

        # -----------------
        # Initial conditions
        # -----------------
        mtom = float(cfg.initial_mtom_kg if initial_mtom_kg is None else initial_mtom_kg)
        if not math.isfinite(mtom) or mtom <= 0.0:
            raise ValueError(f"initial_mtom_kg must be a positive finite number, got {mtom!r}")

        p0 = float(
            cfg.initial_total_power_guess_w if initial_total_power_guess_w is None else initial_total_power_guess_w
        )
        if (not math.isfinite(p0)) or p0 <= 0.0:
            p0 = float(cfg.initial_total_power_guess_w)

        ptotal_guess = {
            "cruise": p0,
            "cruise_charger": p0,
            "takeoff": p0,
            "climb": p0,
        }

        phases: Dict[str, PhasePowerResult] = {}

        # -----------------
        # MTOM convergence (outer loop)
        # -----------------
        # Solve the MTOM mass-closure equation:
        #     residual(mtom) = mtom_estimated(mtom) - mtom = 0
        #
        # The legacy approach used a fixed-point iteration (mtom <- mtom_est).
        # That can converge very slowly when d(mtom_est)/d(mtom) ~ 1.
        #
        # Here we use a safeguarded Newton/secant update with:
        #   - trust-region step limiting (max relative MTOM change per iteration)
        #   - adaptive damping (relaxation factor)
        #   - optional bracketing: if a sign-change is observed, keep the root bracketed
        #     and fall back to regula-falsi / bisection when the Newton proposal is unsafe.
        solver_mode = str(cfg.solver.mtom_solver).strip().lower()
        use_newton = solver_mode not in {"fixed_point", "legacy", "picard"}

        prev_mtom: Optional[float] = None
        prev_res: Optional[float] = None

        # Bracketing points for residual sign (if encountered)
        pos_pt: Optional[Tuple[float, float]] = None  # (mtom, residual) with residual > 0
        neg_pt: Optional[Tuple[float, float]] = None  # (mtom, residual) with residual < 0

        relax = float(cfg.solver.newton_relax_init)

        def _clamp_step(m_current: float, m_proposed: float) -> float:
            """Clamp MTOM proposal to keep the outer loop robust."""

            m_next = float(m_proposed)

            if not math.isfinite(m_next):
                return float(m_current)

            # Enforce positivity
            if m_next <= 0.0:
                m_next = max(1.0, 0.1 * m_current)

            # Trust region: limit relative step size
            max_rel = float(cfg.solver.newton_max_rel_step)
            if max_rel > 0.0:
                max_abs = max_rel * max(m_current, 1.0)
                dm = m_next - m_current
                if abs(dm) > max_abs:
                    m_next = m_current + math.copysign(max_abs, dm)

            return float(m_next)

        for outer_iter in range(1, cfg.solver.max_outer_iter + 1):
            logger.info("======================================================")
            logger.info(f"======================= ITER {outer_iter} =======================")

            # NOTE: For MTOM closure we only need the phases that drive the mass model
            #       (climb for sizing, cruise for cruise-governing compressor mass).
            #       takeoff / cruise_charger are computed once after MTOM converges.

            # Cruise (FC only -> psi = 0)
            phases["cruise"] = self._phase_solver.solve(
                name="cruise",
                mtom_kg=mtom,
                p_w_kw_per_kg=cfg.p_w.p_w_cruise_kw_per_kg,
                flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
                psi=0.0,
                beta=self._beta_cruise,
                initial_total_power_w=ptotal_guess["cruise"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["cruise"] = phases["cruise"].p_total_w

            # Climb
            phases["climb"] = self._phase_solver.solve(
                name="climb",
                mtom_kg=mtom,
                p_w_kw_per_kg=cfg.p_w.p_w_climb_kw_per_kg,
                flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
                psi=cfg.hybrid.psi_climb,
                beta=1.05,
                initial_total_power_w=ptotal_guess["climb"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["climb"] = phases["climb"].p_total_w

            mass = self._mass_estimator.estimate(
                mtom_guess_kg=mtom,
                climb=phases["climb"],
                cruise=phases["cruise"],
            )

            residual = float(mass.mtom_kg - mtom)

            logger.info("\n-----------------------")
            logger.info(
                f"ptotal_climb: {phases['climb'].p_total_w/1000:,.0f} kW, "
                f"mtom_est: {mass.mtom_kg:,.0f} kg, mtom: {mtom:,.0f} kg, "
                f"residual: {residual:+,.2f} kg\n"
            )

            if abs(residual) <= cfg.solver.mtom_tol_kg:
                # Compute remaining phases at the converged MTOM (for reporting/exports)
                phases["cruise_charger"] = self._phase_solver.solve(
                    name="cruise_charger",
                    mtom_kg=mtom,
                    p_w_kw_per_kg=cfg.p_w.p_w_cruise_kw_per_kg,
                    flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
                    psi=cfg.hybrid.psi_cruise_charger,
                    beta=self._beta_cruise_charger,
                    initial_total_power_w=ptotal_guess["cruise_charger"],
                    oversizing=cfg.fuel_cell_op.oversizing,
                )
                ptotal_guess["cruise_charger"] = phases["cruise_charger"].p_total_w

                phases["takeoff"] = self._phase_solver.solve(
                    name="takeoff",
                    mtom_kg=mtom,
                    p_w_kw_per_kg=cfg.p_w.p_w_takeoff_kw_per_kg,
                    flight_point=FlightPoint(cfg.flight.h_takeoff_m, cfg.flight.mach_takeoff),
                    psi=cfg.hybrid.psi_takeoff,
                    beta=1.05,
                    initial_total_power_w=ptotal_guess["takeoff"],
                    oversizing=cfg.fuel_cell_op.oversizing,
                )
                ptotal_guess["takeoff"] = phases["takeoff"].p_total_w

                logger.info("\nCONVERGED")
                return phases, mass

            # -----------------
            # MTOM update
            # -----------------
            mtom_next: float
            step_note = "fixed_point"

            if not use_newton:
                # Legacy behaviour: pure fixed-point update
                mtom_next = _clamp_step(mtom, float(mass.mtom_kg))
            else:
                # Adapt damping based on progress from previous iteration
                if prev_res is not None and math.isfinite(prev_res):
                    if abs(residual) < abs(prev_res):
                        relax = min(1.0, relax * float(cfg.solver.newton_relax_increase))
                    else:
                        relax = max(float(cfg.solver.newton_relax_min), relax * float(cfg.solver.newton_relax_decrease))

                # Update bracket points (optional)
                if bool(cfg.solver.newton_use_bracketing) and math.isfinite(residual):
                    # Keep one point on each side of the root (if available). When both signs exist,
                    # prefer updates that tighten the MTOM bracket (smaller interval).
                    if residual > 0.0:
                        if pos_pt is None:
                            pos_pt = (mtom, residual)
                        elif neg_pt is None:
                            # No opposite sign yet; keep the most recent positive point
                            pos_pt = (mtom, residual)
                        else:
                            if abs(mtom - neg_pt[0]) < abs(pos_pt[0] - neg_pt[0]):
                                pos_pt = (mtom, residual)
                    elif residual < 0.0:
                        if neg_pt is None:
                            neg_pt = (mtom, residual)
                        elif pos_pt is None:
                            # No opposite sign yet; keep the most recent negative point
                            neg_pt = (mtom, residual)
                        else:
                            if abs(mtom - pos_pt[0]) < abs(neg_pt[0] - pos_pt[0]):
                                neg_pt = (mtom, residual)

                mtom_prop: Optional[float] = None

                # Newton step with secant slope (1 evaluation/iter after the first)
                if prev_mtom is not None and prev_res is not None:
                    dm = mtom - prev_mtom
                    dr = residual - prev_res
                    if dm != 0.0 and dr != 0.0:
                        slope = dr / dm  # ≈ d(residual)/d(mtom)
                        if math.isfinite(slope) and abs(slope) > float(cfg.solver.newton_slope_eps):
                            mtom_newton = mtom - residual / slope
                            mtom_prop = mtom + relax * (mtom_newton - mtom)
                            step_note = "newton"

                # If bracketed, ensure the proposal stays inside (fall back if needed)
                if bool(cfg.solver.newton_use_bracketing) and (pos_pt is not None) and (neg_pt is not None):
                    x_pos, f_pos = pos_pt
                    x_neg, f_neg = neg_pt
                    lo = min(x_pos, x_neg)
                    hi = max(x_pos, x_neg)

                    def _in_bracket(x: float) -> bool:
                        return math.isfinite(x) and (lo <= x <= hi)

                    if (mtom_prop is None) or (not _in_bracket(float(mtom_prop))):
                        # Regula falsi (secant across bracket endpoints): always inside the bracket
                        if f_pos != f_neg:
                            mtom_rf = x_pos - f_pos * (x_neg - x_pos) / (f_neg - f_pos)
                        else:
                            mtom_rf = 0.5 * (lo + hi)

                        if _in_bracket(float(mtom_rf)):
                            mtom_prop = float(mtom_rf)
                            step_note = "regula_falsi"
                        else:
                            mtom_prop = 0.5 * (lo + hi)
                            step_note = "bisection"

                # Fallback: relaxed fixed-point step (still 1 eval/iter)
                if mtom_prop is None or (not math.isfinite(float(mtom_prop))):
                    mtom_prop = mtom + float(cfg.solver.newton_fp_relax) * residual
                    step_note = "fixed_point_fallback"

                mtom_next = _clamp_step(mtom, float(mtom_prop))

                # Prevent stalling when far from the tolerance
                if abs(mtom_next - mtom) < float(cfg.solver.newton_min_step_kg) and abs(residual) > cfg.solver.mtom_tol_kg:
                    mtom_next = _clamp_step(mtom, mtom + math.copysign(float(cfg.solver.newton_min_step_kg), residual))

                # Keep inside bracket after clamping (if we have one)
                if bool(cfg.solver.newton_use_bracketing) and (pos_pt is not None) and (neg_pt is not None):
                    lo = min(pos_pt[0], neg_pt[0])
                    hi = max(pos_pt[0], neg_pt[0])
                    if mtom_next < lo or mtom_next > hi:
                        mtom_next = _clamp_step(mtom, 0.5 * (lo + hi))
                        step_note = "bisection(clamped)"

                logger.info(
                    f"MTOM update: {step_note}, relax={relax:.3f} -> mtom_next={mtom_next:,.2f} kg\n"
                )

            prev_mtom, prev_res = mtom, residual
            mtom = float(mtom_next)

        raise RuntimeError(f"MTOM did not converge within {cfg.solver.max_outer_iter} outer iterations")



# ============================
# ADRpy coupling (constraint analysis)
# ============================

_G0_MPS2 = 9.80665


@dataclass(frozen=True)
class ADRpyDesignPoint:
    """Selected (W/S, P/W) point to be fed into the hybrid sizing loop."""

    wing_loading_pa: float
    wing_loading_kg_per_m2: float

    p_w_takeoff_kw_per_kg: float
    p_w_climb_kw_per_kg: float
    p_w_cruise_kw_per_kg: float


class ADRpyConstraintAnalyzer:
    """Thin adapter around ADRpy constraintanalysis for HFCAD.

    Key design decision:
      - ADRpy is used as the *performance feasibility* model (P/W requirement vs W/S).
      - HFCAD is used as the *mass closure* model (MTOM vs wing area and installed power).
    """

    def __init__(self, cfg: DesignConfig):
        self._cfg = cfg

        try:
            from ADRpy import unitconversions as adr_co  # type: ignore
            from ADRpy import constraintanalysis as adr_ca  # type: ignore
            from ADRpy import atmospheres as adr_at  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ADRpy is required for constraint-based sizing. "
                "Vendor the ADRpy package into app/ADRpy or install it into your environment."
            ) from e

        self._co = adr_co
        self._ca = adr_ca
        self._atm = adr_at.Atmosphere()

    def _build_concept(self, *, mtom_kg: float):
        cfg = self._cfg
        b = cfg.constraint_brief
        p = cfg.constraint_performance
        g = cfg.constraint_geometry

        designbrief = {
            "rwyelevation_m": float(b.rwyelevation_m),
            "groundrun_m": float(b.groundrun_m),
            "stloadfactor": float(b.stloadfactor),
            "turnalt_m": float(b.turnalt_m),
            "turnspeed_ktas": float(b.turnspeed_ktas),
            "climbalt_m": float(b.climbalt_m),
            "climbspeed_kias": float(b.climbspeed_kias),
            "climbrate_fpm": float(b.climbrate_fpm),
            "cruisealt_m": float(b.cruisealt_m),
            "cruisespeed_ktas": float(b.cruisespeed_ktas),
            "cruisethrustfact": float(b.cruisethrustfact),
            "servceil_m": float(b.servceil_m),
            "secclimbspd_kias": float(b.secclimbspd_kias),
            "vstallclean_kcas": float(b.vstallclean_kcas),
        }

        designdefinition = {
            "aspectratio": float(cfg.wing.aspect_ratio),
            "sweep_le_deg": float(g.sweep_le_deg),
            "sweep_mt_deg": float(g.sweep_mt_deg),
            "weightfractions": {"turn": 1.0, "climb": 1.0, "cruise": 1.0, "servceil": 1.0},
            "weight_n": float(self._co.kg2n(float(mtom_kg))),
        }

        designperformance = {
            "CDTO": float(p.cdto),
            "CLTO": float(p.clto),
            "CLmaxTO": float(p.clmax_to),
            "CLmaxclean": float(p.clmax_clean),
            "mu_R": float(p.mu_r),
            "CDminclean": float(p.cdmin_clean),
            "etaprop": {
                "take-off": float(p.etaprop_takeoff),
                "climb": float(p.etaprop_climb),
                "cruise": float(p.etaprop_cruise),
                "turn": float(p.etaprop_turn),
                "servceil": float(p.etaprop_servceil),
            },
        }

        propulsion = str(cfg.constraint_sizing.propulsion_type)
        return self._ca.AircraftConcept(designbrief, designdefinition, designperformance, self._atm, propulsion)

    def power_to_weight_curves_kw_per_kg(
        self,
        *,
        wingloading_pa: np.ndarray,
        mtom_kg: float,
    ) -> Tuple[Dict[str, np.ndarray], Optional[float]]:
        """Return ADRpy P/W requirement curves in kW/kg for each constraint and the clean-stall W/S limit."""

        concept = self._build_concept(mtom_kg=float(mtom_kg))

        preq_hp = concept.powerrequired(wingloading_pa, tow_kg=float(mtom_kg), feasibleonly=True, map2sl=True)

        curves: Dict[str, np.ndarray] = {}
        for k, v in preq_hp.items():
            # Convert hp -> kW -> kW/kg
            curves[k] = np.asarray(self._co.hp2kw(v), dtype=float) / float(mtom_kg)

        # Clean stall maximum W/S (Pa). Not always defined if CLmaxclean missing.
        try:
            wsmax_cleanstall_pa = float(concept.wsmaxcleanstall_pa())
        except Exception:
            wsmax_cleanstall_pa = None

        return curves, wsmax_cleanstall_pa

    @staticmethod
    def _select_index_min(curve: np.ndarray, valid_mask: np.ndarray) -> int:
        idxs = np.where(valid_mask)[0]
        if idxs.size == 0:
            raise RuntimeError("No feasible design points in the wing-loading grid (check inputs / stall limits).")
        sub = curve[idxs]
        # Use nanargmin on the masked subset (should be finite, but be defensive)
        j = int(np.nanargmin(sub))
        return int(idxs[j])

    def select_design_point_min_combined_pw(
        self,
        *,
        wingloading_pa: np.ndarray,
        mtom_kg: float,
    ) -> Tuple[ADRpyDesignPoint, Dict[str, np.ndarray], Optional[float]]:
        """Choose W/S that minimises ADRpy combined P/W (kW/kg)."""

        curves, wsmax_cleanstall_pa = self.power_to_weight_curves_kw_per_kg(
            wingloading_pa=wingloading_pa, mtom_kg=float(mtom_kg)
        )

        combined = np.asarray(curves.get("combined"), dtype=float)
        valid = np.isfinite(combined)

        if wsmax_cleanstall_pa is not None:
            valid &= wingloading_pa <= float(wsmax_cleanstall_pa)

        i_opt = self._select_index_min(combined, valid_mask=valid)

        dp = self._design_point_from_curves(wingloading_pa=wingloading_pa, curves=curves, idx=i_opt)
        return dp, curves, wsmax_cleanstall_pa

    def _design_point_from_curves(
        self,
        *,
        wingloading_pa: np.ndarray,
        curves: Dict[str, np.ndarray],
        idx: int,
    ) -> ADRpyDesignPoint:
        """Map ADRpy curves -> HFCAD inputs at a specific wingloading index."""

        cs = self._cfg.constraint_sizing

        def _get(curve_key: str) -> float:
            if curve_key not in curves:
                raise KeyError(f"ADRpy curve '{curve_key}' not found. Available: {sorted(curves.keys())}")
            return float(np.asarray(curves[curve_key], dtype=float)[idx])

        if cs.use_combined_for_phases:
            p_takeoff = _get("combined")
            p_climb = _get("combined")
            p_cruise = _get("combined")
        else:
            p_takeoff = _get(str(cs.takeoff_constraint))
            p_climb = _get(str(cs.climb_constraint))
            p_cruise = _get(str(cs.cruise_constraint))

        margin = 1.0 + float(cs.pw_margin_fraction)
        p_takeoff *= margin
        p_climb *= margin
        p_cruise *= margin

        ws_pa = float(wingloading_pa[idx])
        ws_kgm2 = ws_pa / _G0_MPS2

        return ADRpyDesignPoint(
            wing_loading_pa=ws_pa,
            wing_loading_kg_per_m2=ws_kgm2,
            p_w_takeoff_kw_per_kg=p_takeoff,
            p_w_climb_kw_per_kg=p_climb,
            p_w_cruise_kw_per_kg=p_cruise,
        )


class CoupledConstraintSizingRunner:
    """Runs constraint-based sizing and couples it to the hybrid mass-closure loop."""

    def __init__(self, cfg: DesignConfig):
        self._cfg = cfg
        self._adr = ADRpyConstraintAnalyzer(cfg)

    def _wingloading_grid_pa(self) -> np.ndarray:
        cs = self._cfg.constraint_sizing
        if cs.ws_step_pa <= 0:
            raise ValueError("constraint_sizing.ws_step_pa must be > 0")
        if cs.ws_max_pa <= cs.ws_min_pa:
            raise ValueError("constraint_sizing.ws_max_pa must be > ws_min_pa")

        # Include end point
        n = int(math.floor((cs.ws_max_pa - cs.ws_min_pa) / cs.ws_step_pa)) + 1
        grid = cs.ws_min_pa + cs.ws_step_pa * np.arange(n, dtype=float)
        return np.asarray(grid, dtype=float)

    def _apply_design_point_to_cfg(self, base_cfg: DesignConfig, dp: ADRpyDesignPoint) -> DesignConfig:
        new_wing = replace(base_cfg.wing, wing_loading_kg_per_m2=float(dp.wing_loading_kg_per_m2))
        new_pw = replace(
            base_cfg.p_w,
            p_w_takeoff_kw_per_kg=float(dp.p_w_takeoff_kw_per_kg),
            p_w_climb_kw_per_kg=float(dp.p_w_climb_kw_per_kg),
            p_w_cruise_kw_per_kg=float(dp.p_w_cruise_kw_per_kg),
        )
        return replace(base_cfg, wing=new_wing, p_w=new_pw)

    def run(
        self,
        *,
        out_dir: Optional[Path] = None,
    ) -> Tuple[DesignConfig, Dict[str, PhasePowerResult], MassBreakdown, Optional[pd.DataFrame]]:
        cs = self._cfg.constraint_sizing
        if not cs.enable:
            design = HybridFuelCellAircraftDesign(self._cfg, out_dir=out_dir)
            phases, mass = design.run()
            return self._cfg, phases, mass, None

        selection = str(cs.selection).strip().lower()

        if selection == "min_mtom":
            return self._run_min_mtom(out_dir=out_dir)

        # Default: min combined P/W (fast)
        return self._run_min_combined_pw(out_dir=out_dir)

    def _run_min_combined_pw(
        self,
        *,
        out_dir: Optional[Path],
    ) -> Tuple[DesignConfig, Dict[str, PhasePowerResult], MassBreakdown, Optional[pd.DataFrame]]:
        cfg = self._cfg
        ws_grid = self._wingloading_grid_pa()

        dp, curves, wsmax_cleanstall_pa = self._adr.select_design_point_min_combined_pw(
            wingloading_pa=ws_grid, mtom_kg=float(cfg.initial_mtom_kg)
        )

        cfg_used = self._apply_design_point_to_cfg(cfg, dp)

        logger.info(
            "ADRpy constraint sizing (min_combined_pw): W/S=%.1f Pa (%.2f kg/m^2), "
            "P/W_TO=%.5f kW/kg, P/W_climb=%.5f kW/kg, P/W_cruise=%.5f kW/kg",
            dp.wing_loading_pa,
            dp.wing_loading_kg_per_m2,
            dp.p_w_takeoff_kw_per_kg,
            dp.p_w_climb_kw_per_kg,
            dp.p_w_cruise_kw_per_kg,
        )
        if wsmax_cleanstall_pa is not None and dp.wing_loading_pa > wsmax_cleanstall_pa:
            logger.warning(
                "Selected W/S exceeds clean stall W/S max (%.1f Pa). Check vstallclean_kcas / CLmax_clean.",
                wsmax_cleanstall_pa,
            )

        design = HybridFuelCellAircraftDesign(cfg_used, out_dir=out_dir)
        phases, mass = design.run()

        # Optional single-point dataframe
        df = pd.DataFrame(
            [
                {
                    "wing_loading_pa": dp.wing_loading_pa,
                    "wing_loading_kg_per_m2": dp.wing_loading_kg_per_m2,
                    "p_w_takeoff_kw_per_kg": dp.p_w_takeoff_kw_per_kg,
                    "p_w_climb_kw_per_kg": dp.p_w_climb_kw_per_kg,
                    "p_w_cruise_kw_per_kg": dp.p_w_cruise_kw_per_kg,
                    "mtom_kg": mass.mtom_kg,
                }
            ]
        )

        if out_dir is not None and cfg.constraint_sizing.write_trade_csv:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(out_dir / "constraint_sizing_trade.csv"), index=False)

        return cfg_used, phases, mass, df

    def _run_min_mtom(
        self,
        *,
        out_dir: Optional[Path],
    ) -> Tuple[DesignConfig, Dict[str, PhasePowerResult], MassBreakdown, Optional[pd.DataFrame]]:
        cfg = self._cfg
        cs = cfg.constraint_sizing

        ws_grid = self._wingloading_grid_pa()

        # Compute constraint curves once (P/W vs W/S is independent of MTOM for this model).
        curves, wsmax_cleanstall_pa = self._adr.power_to_weight_curves_kw_per_kg(
            wingloading_pa=ws_grid, mtom_kg=float(cfg.initial_mtom_kg)
        )

        combined = np.asarray(curves.get("combined"), dtype=float)
        valid = np.isfinite(combined)
        if wsmax_cleanstall_pa is not None:
            valid &= ws_grid <= float(wsmax_cleanstall_pa)

        idxs = np.where(valid)[0]
        if idxs.size == 0:
            raise RuntimeError("No feasible W/S points for selection='min_mtom' (check constraints / stall limit).")

        # Optional: silence iterative logs during trade scan
        old_level = logger.level
        if cs.scan_quiet:
            logger.setLevel(logging.WARNING)

        rows: List[Dict[str, float]] = []
        best: Optional[Tuple[float, DesignConfig, Dict[str, PhasePowerResult], MassBreakdown, ADRpyDesignPoint]] = None

        # Seed guesses
        mtom_seed = float(cfg.initial_mtom_kg)
        p_seed = float(cfg.initial_total_power_guess_w)

        try:
            for idx in idxs:
                dp = self._adr._design_point_from_curves(wingloading_pa=ws_grid, curves=curves, idx=int(idx))
                cfg_i = self._apply_design_point_to_cfg(cfg, dp)

                # Run mass-closure
                design = HybridFuelCellAircraftDesign(cfg_i, out_dir=out_dir)
                phases_i, mass_i = design.run(initial_mtom_kg=mtom_seed, initial_total_power_guess_w=p_seed)

                # Update seeds for next point (helps convergence across the scan)
                mtom_seed = float(mass_i.mtom_kg)
                p_seed = float(phases_i["climb"].p_total_w)

                rows.append(
                    {
                        "wing_loading_pa": dp.wing_loading_pa,
                        "wing_loading_kg_per_m2": dp.wing_loading_kg_per_m2,
                        "p_w_takeoff_kw_per_kg": dp.p_w_takeoff_kw_per_kg,
                        "p_w_climb_kw_per_kg": dp.p_w_climb_kw_per_kg,
                        "p_w_cruise_kw_per_kg": dp.p_w_cruise_kw_per_kg,
                        "mtom_kg": float(mass_i.mtom_kg),
                        "p_total_climb_kw": float(phases_i["climb"].p_total_w) / 1000.0,
                        "p_total_cruise_kw": float(phases_i["cruise"].p_total_w) / 1000.0,
                        "p_total_takeoff_kw": float(phases_i["takeoff"].p_total_w) / 1000.0,
                    }
                )

                if best is None or mass_i.mtom_kg < best[0]:
                    best = (float(mass_i.mtom_kg), cfg_i, phases_i, mass_i, dp)

        finally:
            if cs.scan_quiet:
                logger.setLevel(old_level)

        if best is None:
            raise RuntimeError("No feasible designs found during selection='min_mtom' scan.")

        best_mtom, best_cfg, best_phases, best_mass, best_dp = best

        logger.info(
            "ADRpy+HFCAD coupled sizing (min_mtom): W/S=%.1f Pa (%.2f kg/m^2), MTOM=%.1f kg, "
            "P/W_TO=%.5f, P/W_climb=%.5f, P/W_cruise=%.5f (kW/kg)",
            best_dp.wing_loading_pa,
            best_dp.wing_loading_kg_per_m2,
            best_mtom,
            best_dp.p_w_takeoff_kw_per_kg,
            best_dp.p_w_climb_kw_per_kg,
            best_dp.p_w_cruise_kw_per_kg,
        )

        df = pd.DataFrame(rows)
        if out_dir is not None and cs.write_trade_csv:
            out_dir.mkdir(parents=True, exist_ok=True)
            df.sort_values("wing_loading_pa", inplace=True)
            df.to_csv(str(out_dir / "constraint_sizing_trade.csv"), index=False)

        return best_cfg, best_phases, best_mass, df


# ============================
# Output utilities
# ============================


class OutputWriter:
    def __init__(self, cfg: DesignConfig):
        self._cfg = cfg
        rho_cr = float(Atmosphere(cfg.flight.h_cr_m).density[0])
        self._beta_cruise, _ = _resolve_cruise_betas(
            cfg.fuel_cell_op.beta,
            cruise_density_kg_per_m3=rho_cr,
        )

    def write_pemfc_figure(
        self,
        *,
        nacelle_power_w: float,
        out_dir: Path,
        timeout_s: float = 30.0,
    ) -> None:
        """Generate PEMFC polarization figure once (expensive)."""

        try:
            fc = FuelCellSystemModel(self._cfg.fuel_cell_arch, comp_stl_path=out_dir / "media" / "comp.stl")
            res = fc.size_nacelle(
                power_fc_sys_w=nacelle_power_w,
                flight_point=FlightPoint(self._cfg.flight.h_cr_m, self._cfg.flight.mach_cr),
                beta=self._beta_cruise,
                oversizing=self._cfg.fuel_cell_op.oversizing,
                comp_bool=True,
                make_fig=True,
                verbose=False,
            )
            if not res.figs:
                return

            fig = res.figs[-1]
            figs_dir = out_dir / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)
            save_path = figs_dir / "pemfc_fig.png"
            write_error: List[Exception] = []
            done = threading.Event()

            def _save_image() -> None:
                try:
                    fig.write_image(str(save_path))
                except Exception as e:  # pragma: no cover - runtime/tooling dependent
                    write_error.append(e)
                finally:
                    done.set()

            t = threading.Thread(target=_save_image, daemon=True)
            t.start()
            if not done.wait(timeout=float(timeout_s)):
                logger.warning(
                    "Skipping PEMFC figure export: timed out after %.1f s while writing %s",
                    timeout_s,
                    save_path,
                )
                return

            if write_error:
                raise write_error[0]
        except Exception as e:
            logger.warning("Could not generate PEMFC figure: %s", e)

    def write_mission_profile_outputs(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
        show_plot: bool = False,
    ) -> None:
        """Reproduce legacy mission power plot and Excel export."""

        import matplotlib.pyplot as plt

        cfg = self._cfg

        # Legacy hybrid power allocation outputs
        pfc_ready = mass.p_fuelcell_engine_w
        pfc_taxing = mass.p_fuelcell_taxing_w

        # Total (incl auxiliaries) FC power during phases
        pfc_climb = phases["climb"].p_total_w - phases["climb"].p_battery_w
        pbat_climb = phases["climb"].p_battery_w

        pfc_takeoff = phases["takeoff"].p_total_w - phases["takeoff"].p_battery_w
        pbat_takeoff = phases["takeoff"].p_battery_w

        pfc_cruise_charger = phases["cruise_charger"].p_total_w
        pbat_charge = phases["cruise_charger"].p_battery_w

        pfc_cruise = phases["cruise"].p_total_w

        logger.info(f"Pfuel_ready: {pfc_ready/1000:,.0f} kW, Pfuel_taxing: {pfc_taxing/1000:,.0f} kW, Pfuel_climb: {pfc_climb/1000:,.0f} kW, Pfuel_cruise: {pfc_cruise/1000:,.0f} kW, Pfuel_cruise_charger: {pfc_cruise_charger/1000:,.0f} kW, Pbat_climb: {pbat_climb/1000:,.0f} kW, Pbat_charge: {pbat_charge/1000:,.0f} kW")

        power_fc = [
            pfc_ready,
            pfc_taxing,
            pfc_takeoff,
            pfc_climb,
            pfc_climb,
            pfc_cruise_charger,
            pfc_cruise_charger,
            pfc_cruise,
            pfc_cruise,
            pfc_taxing,
            pfc_climb,
            pfc_climb,
            pfc_cruise,
            pfc_cruise,
            pfc_cruise,
            pfc_cruise,
            pfc_taxing,
            pfc_taxing,
            pfc_taxing,
            pfc_ready,
        ]
        power_bat = [
            0.0,
            0.0,
            pbat_takeoff,
            pbat_climb,
            pbat_climb,
            pbat_charge,
            pbat_charge,
            0.0,
            0.0,
            0.0,
            pbat_climb,
            pbat_climb,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        x = np.array(cfg.mission.times_min, dtype=float)
        y_bat_kw = np.array(power_bat, dtype=float) / 1000.0
        y_fc_kw = np.array(power_fc, dtype=float) / 1000.0
        y_total_kw = y_bat_kw + y_fc_kw

        # Plot
        plt.plot(x, y_bat_kw, linestyle="solid", color="gray", label="Battery")
        plt.plot(x, y_fc_kw, linestyle="dashed", color="orange", label="Fuel Cell")
        plt.plot(x, y_total_kw, linestyle="solid", color="blue", label="Total")

        plt.xlabel("Time(min)")
        plt.ylabel("Power(kW)")
        plt.axis([0, 180, -200, 1400])
        plt.title("Power Mission Profile")
        plt.legend(loc="upper right")
        plt.grid(True)

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_dir / "Power Mission Profile.png"), dpi=400)
        if show_plot:
            plt.show()
        plt.close()

        # Excel export
        req_pow = np.vstack([y_total_kw, y_fc_kw, y_bat_kw]).T
        df = pd.DataFrame(req_pow, columns=["ReqPow_AC", "ReqPow_FC", "ReqPow_Batt"]).T
        df.to_excel(str(out_dir / "ReqPowDATA.xlsx"), index=True)

    def write_converged_text(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
    ) -> None:
        """Write converged summary text to the output directory."""

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "ConvergedData.txt"
        summary_text = _converged_summary_text(phases=phases, mass=mass, cfg=self._cfg)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")


# ============================
# Entry point
# ============================


def _converged_summary_text(
    phases: Dict[str, PhasePowerResult],
    mass: MassBreakdown,
    cfg: DesignConfig,
) -> str:
    """Converged summary used for console output and text export."""

    pnet = phases["climb"].p_total_w * cfg.eff.eta_pdu * cfg.eff.eta_em
    nac = phases["climb"].nacelle
    comp_mass_report_kg = float(phases["cruise"].nacelle.m_comp_kg)
    p_to_w_converged_w_per_kg = phases["climb"].p_total_w / mass.mtom_kg
    wing_loading_kg_per_m2 = mass.mtom_kg / mass.wing_area_m2

    lines = [
        "=============================================================",
        "========================= CONVERGED =========================",
        "=============================================================",
        f"S_wing: {mass.wing_area_m2:,.2f} m^2",
        f"b_wing: {mass.wing_span_m:,.2f} m",
        f"Lfus: {mass.fuselage_length_m:,.2f} m",
        f"Ltank: {mass.tank_length_m:,.2f} m",
        f"Ptotal_climb: {phases['climb'].p_total_w/1000:,.0f} kW",
        f"Ptotal_cruise: {phases['cruise'].p_total_w/1000:,.0f} kW",
        f"Ptotal_takeoff: {phases['takeoff'].p_total_w/1000:,.0f} kW",
        f"Pelectricnet: {phases['climb'].p_bus_required_w/1000:,.0f} kW",
        f"Pcomp: {phases['climb'].p_comp_w/1000:,.0f} kW",
        f"Pcoolingsystem: {phases['climb'].p_cooling_w/1000:,.0f} kW",
        f"Pnet: {pnet/1000:,.0f} kW",
        f"eta_pt: {cfg.eff.eta_em*cfg.eff.eta_pdu:,.4f}",
        "",
        "Per Nacelle",
        (
            f"Stack(1/{cfg.fuel_cell_arch.n_stacks_parallel}): {nac.m_stacks_kg:,.0f} kg, "
            f"Compressor: {comp_mass_report_kg:,.0f} kg, Humidifier: {nac.m_humid_kg:,.0f} kg, HX: {nac.m_hx_kg:,.0f} kg"
        ),
        f"Power density of Nacelle System: {mass.nacelle_design_power_kw_per_kg:,.3f} kW/kg",
        f"dim_hx: dX={mass.nacelle_hx_dim_m[0]:,.3f} m, dY={mass.nacelle_hx_dim_m[1]:,.3f} m, dZ={mass.nacelle_hx_dim_m[2]:,.3f} m",
        f"dim_stack: dX={mass.nacelle_stack_dim_m[2]:,.3f} m, dY={mass.nacelle_stack_dim_m[0]:,.3f} m, dZ={mass.nacelle_stack_dim_m[1]:,.3f} m",
        "",
        "ALL Nacelles",
        f"FCS(FC+Humidifier+Comp+Hx): {mass.m_fc_system_kg:,.0f} kg",
        f"mPMAD: {mass.m_pmad_kg:,.0f} kg",
        f"Electric Motors: {mass.m_e_motor_kg:,.0f} kg",
        "",
        "-----------------------",
        f"Powertrain(FCS+PMAD+Motors): {mass.m_powertrain_total_kg:,.0f} kg",
        f"mtank: {mass.m_tank_kg:,.0f} kg",
        f"W_wing: {mass.w_wing_kg:,.0f} kg",
        f"W_HT: {mass.w_ht_kg:,.0f} kg",
        f"W_VT: {mass.w_vt_kg:,.0f} kg",
        f"W_fus: {mass.w_fus_kg:,.0f} kg",
        f"W_lndgearmain: {mass.w_lnd_main_kg:,.0f} kg",
        f"W_lndgearnose: {mass.w_lnd_nose_kg:,.0f} kg",
        f"W_motor: {mass.w_motor_misc_kg:,.0f} kg",
        f"W_flight_control: {mass.w_flight_control_kg:,.0f} kg",
        f"W_els: {mass.w_els_kg:,.0f} kg",
        f"W_iae: {mass.w_iae_kg:,.0f} kg",
        f"W_hydraulics: {mass.w_hydraulics_kg:,.0f} kg",
        f"W_fur: {mass.w_furnishings_kg:,.0f} kg",
        f"OEM: {mass.oem_kg:,.0f} kg",
        f"OEMmisc: {mass.oem_misc_kg:,.0f} kg",
        f"mfuel: {mass.m_fuel_kg:,.1f} kg",
        f"mbatt: {mass.m_battery_kg:,.1f} kg",
        f"mdot_H2(cruise): {phases['cruise'].mdot_h2_kgps*1000:,.1f} g/s",
        "-----------------------",
        f"MTOM: {mass.mtom_kg:,.0f} kg",
        f"Power-to-weight (converged): {p_to_w_converged_w_per_kg:,.2f} W/kg",
        f"Wing loading (MTOM/S_wing): {wing_loading_kg_per_m2:,.2f} kg/m^2",
        "",
        f"Vtankex: {mass.tank_volume_m3:,.1f} m^3",
        "========================== END ==============================",
    ]
    if cfg.constraint_sizing.enable:
        hdr = [
            f"Constraint sizing: ENABLED ({cfg.constraint_sizing.selection})",
            f"Input W/S: {cfg.wing.wing_loading_kg_per_m2:,.2f} kg/m^2 ({cfg.wing.wing_loading_kg_per_m2*_G0_MPS2:,.1f} Pa)",
            f"Input P/W [kW/kg]: takeoff {cfg.p_w.p_w_takeoff_kw_per_kg:.5f}, climb {cfg.p_w.p_w_climb_kw_per_kg:.5f}, cruise {cfg.p_w.p_w_cruise_kw_per_kg:.5f}",
            "-------------------------------------------------------------",
        ]
        lines = lines[:3] + hdr + lines[3:]
    return "\n".join(lines)


def _print_summary(phases: Dict[str, PhasePowerResult], mass: MassBreakdown, cfg: DesignConfig) -> None:
    """Console report similar to the legacy script."""

    print()
    print(_converged_summary_text(phases=phases, mass=mass, cfg=cfg))


def _output_subdir_from_input(input_path: Path) -> str:
    stem = input_path.stem
    if stem.startswith("input_"):
        return stem[len("input_") :]
    return stem


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    default_input = Path(__file__).with_name("input_HFCAD.txt")

    parser = argparse.ArgumentParser(
        description="Hybrid Fuel-Cell/Battery Aircraft Design (HFCAD) - OOP",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(default_input),
        help="Path to input file (INI-style). Default: input_HFCAD.txt next to this script.",
    )
    parser.add_argument(
        "--outdir",
        default=str(Path.cwd()),
        help="Output directory for plots, ReqPowDATA.xlsx, and converged text summary.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the mission profile plot window.",
    )
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a full template input file to --input and exit.",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    if args.write_template:
        write_input_template(input_path, DesignConfig())
        logger.info("Wrote template input file: %s", input_path)
        return

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Run with --write-template to generate a template."
        )

    cfg = load_design_config(input_path)

    out_root = Path(args.outdir).expanduser()
    out_dir = out_root / _output_subdir_from_input(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy the input file into the output folder for traceability
    input_copy_path = out_dir / input_path.name
    if input_copy_path.resolve() != input_path.resolve():
        shutil.copy2(input_path, input_copy_path)

    if cfg.constraint_sizing.enable:
        runner = CoupledConstraintSizingRunner(cfg)
        cfg_used, phases, mass, _trade_df = runner.run(out_dir=out_dir)
    else:
        cfg_used = cfg
        design = HybridFuelCellAircraftDesign(cfg_used, out_dir=out_dir)
        phases, mass = design.run()

    _print_summary(phases, mass, cfg_used)

    writer = OutputWriter(cfg_used)
    writer.write_converged_text(phases=phases, mass=mass, out_dir=out_dir)

    # Fuel cell figure (per nacelle at design point)
    nacelle_power_w = phases["climb"].p_total_w / cfg_used.fuel_cell_arch.n_stacks_parallel
    writer.write_pemfc_figure(nacelle_power_w=nacelle_power_w, out_dir=out_dir)

    # Mission profile plot and Excel
    writer.write_mission_profile_outputs(
        phases=phases,
        mass=mass,
        out_dir=out_dir,
        show_plot=bool(args.show_plot),
    )


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("\n\n=============================================================")
    print(f'Execution time: {elapsed_time:.1f} seconds')
    print("=============================================================")
