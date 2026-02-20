"""HFCBattACDesign_SH_260117 (OOP rewrite)

This module is an object-oriented refactor of the legacy script:
  - configuration is explicit (dataclasses)
  - responsibilities are separated (powertrain, FC system, cooling, mass estimation, solver)
  - the numerical approach and default constants are intentionally kept close to the legacy behaviour
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tempfile

import time
import math
import logging
from functools import lru_cache
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from pprint import pformat
from typing import Dict, Iterable, List, Optional, Tuple, get_args, get_origin

import numpy as np
import pandas as pd

from ambiance import Atmosphere

try:
    from pint import UnitRegistry
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This project now uses the 'pint' units library (missing dependency).\n"
        "Install it with:\n"
        "  pip install pint\n"
    ) from exc

try:  # pragma: no cover
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore
        """Fallback decorator when numba is unavailable."""
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


_ureg = UnitRegistry()
Q_ = _ureg.Quantity
_G0 = Q_(9.80665, "meter / second ** 2")
_G0_MPS2 = float(_G0.to("meter / second ** 2").magnitude)
_LOITER_PW_RATIO_TO_CRUISE = 0.8
_SEA_LEVEL_PRESSURE_PA = float(Atmosphere(0.0).pressure[0])
# Fast compressor-mass surrogate coefficients (calibrated against CAD model outputs).
_COMP_MASS_C0 = -1.07005783
_COMP_MASS_C1 = 0.518571993
_COMP_MASS_C2 = 262.906263
_COMP_MASS_C3 = 208.871692
_COMP_MASS_C4 = 1162.02321
_COMP_MASS_C5 = 2272.77970
_COMP_MASS_C6 = -70.9721798
_COMP_MASS_C7 = 55.3164451
_COMP_MASS_C8 = -63.9226029

# Ensure common aviation shorthand is available.
# Pint usually includes 'knot' and 'nautical_mile', but some installations/older versions
# may miss the 'kn'/'kt' aliases.
try:  # pragma: no cover
    _ureg.Unit("knot")
except Exception:  # pragma: no cover
    # Fallback definition from base units (1 kn = 1852 m / h).
    _ureg.define("knot = 1852 * meter / hour")

for _alias in ("kn", "kt"):
    try:  # pragma: no cover
        _ureg.Unit(_alias)
    except Exception:  # pragma: no cover
        try:
            _ureg.define(f"{_alias} = knot")
        except Exception:
            pass


@njit(cache=True, fastmath=True)
def _total_pressure_pa(p_static_pa: float, mach: float) -> float:
    return p_static_pa * (1.0 + 0.2 * mach * mach) ** 3.5


@njit(cache=True, fastmath=True)
def _total_temperature_k(t_static_k: float, mach: float) -> float:
    return t_static_k * (1.0 + 0.2 * mach * mach)


@njit(cache=True, fastmath=True)
def _compressor_mass_surrogate_kg(
    power_comp_w: float,
    d1h: float,
    d2: float,
    d2s: float,
    d3: float,
    la: float,
    b1: float,
    b2: float,
    b3: float,
) -> float:
    p_kw = power_comp_w * 1e-3
    d2_sq = d2 * d2
    m = (
        _COMP_MASS_C0
        + _COMP_MASS_C1 * p_kw
        + _COMP_MASS_C2 * d2 * d2_sq
        + _COMP_MASS_C3 * d3 * d3 * d3
        + _COMP_MASS_C4 * la * d2_sq
        + _COMP_MASS_C5 * (b1 + b2 + b3) * d2_sq
        + _COMP_MASS_C6 * d2
        + _COMP_MASS_C7 * d3
        + _COMP_MASS_C8 * la
        + 0.0 * d1h
        + 0.0 * d2s
    )
    return m if m > 0.0 else 0.0


@lru_cache(maxsize=128)
def _atmosphere_at_altitude(altitude_m_rounded: float) -> Tuple[float, float, float, float, float]:
    atm = Atmosphere(float(altitude_m_rounded))
    return (
        float(atm.speed_of_sound[0]),
        float(atm.pressure[0]),
        float(atm.temperature[0]),
        float(atm.density[0]),
        float(atm.dynamic_viscosity[0]),
    )

class UnitConverter:
    """Unit conversion helper backed by Pint.

    The legacy code depended on a custom `Conversions` module. This wrapper keeps
    the call sites readable while delegating all conversions to Pint.

    All methods return *plain floats* (magnitudes) to stay compatible with the
    existing numerical code paths.
    """

    def __init__(self, ureg: Optional["UnitRegistry"] = None):
        self.ureg = ureg or _ureg

    @staticmethod
    def _norm_unit(unit: str) -> str:
        return unit.strip().lower().replace(" ", "")

    def meter_feet_q(self, value: float, out_unit: str):
        """Convert between meters and feet.

        Args:
            value: numeric magnitude in the *input* unit implied by out_unit.
            out_unit: 'ft' to treat input as meters and return feet;
                      'meter'/'m' to treat input as feet and return meters.
        """
        out = self._norm_unit(out_unit)
        if out in {"ft", "foot", "feet"}:
            return (value * self.ureg.meter).to(self.ureg.foot)
        if out in {"m", "meter", "metre"}:
            return (value * self.ureg.foot).to(self.ureg.meter)
        raise ValueError(f"meter_feet: unsupported output unit {out_unit!r}")

    def meter_feet(self, value: float, out_unit: str) -> float:
        return float(self.meter_feet_q(value, out_unit).magnitude)

    def meter_inch_q(self, value: float, out_unit: str):
        """Convert between meters and inches."""
        out = self._norm_unit(out_unit)
        if out in {"in", "inch", "inches"}:
            return (value * self.ureg.meter).to(self.ureg.inch)
        if out in {"m", "meter", "metre"}:
            return (value * self.ureg.inch).to(self.ureg.meter)
        raise ValueError(f"meter_inch: unsupported output unit {out_unit!r}")

    def meter_inch(self, value: float, out_unit: str) -> float:
        return float(self.meter_inch_q(value, out_unit).magnitude)

    def m2_ft2_q(self, value: float, out_unit: str):
        """Convert between square meters and square feet."""
        out = self._norm_unit(out_unit)
        if out in {"ft2", "ft^2", "ft**2", "sqft"}:
            return (value * (self.ureg.meter ** 2)).to(self.ureg.foot ** 2)
        if out in {"m2", "m^2", "m**2", "sqm"}:
            return (value * (self.ureg.foot ** 2)).to(self.ureg.meter ** 2)
        raise ValueError(f"m2_ft2: unsupported output unit {out_unit!r}")

    def m2_ft2(self, value: float, out_unit: str) -> float:
        return float(self.m2_ft2_q(value, out_unit).magnitude)

    def kg_pound_q(self, value: float, out_unit: str):
        """Convert between kilograms and pounds (avoirdupois mass)."""
        out = self._norm_unit(out_unit)
        if out in {"lb", "lbs", "pound", "pounds"}:
            return (value * self.ureg.kilogram).to(self.ureg.pound)
        if out in {"kg", "kilogram", "kilograms"}:
            return (value * self.ureg.pound).to(self.ureg.kilogram)
        raise ValueError(f"kg_pound: unsupported output unit {out_unit!r}")

    def kg_pound(self, value: float, out_unit: str) -> float:
        return float(self.kg_pound_q(value, out_unit).magnitude)

    def km1h_kn_q(self, value: float, out_unit: str):
        """Convert between km/h and knots."""
        out = self._norm_unit(out_unit)
        kmph = self.ureg.kilometer / self.ureg.hour

        if out in {"kn", "knot", "knots", "kt", "kts"}:
            return (value * kmph).to(self.ureg.knot)
        if out in {"km/h", "kmh", "kmph"}:
            return (value * self.ureg.knot).to(kmph)
        raise ValueError(f"km1h_kn: unsupported output unit {out_unit!r}")

    def km1h_kn(self, value: float, out_unit: str) -> float:
        return float(self.km1h_kn_q(value, out_unit).magnitude)
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
    wing_loading_kg_per_m2: float = float((Q_(2830.24, "pascal") / _G0).to("kilogram / meter ** 2").magnitude)
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
class TailConfig:
    """Inputs for `TailCompu`-style horizontal/vertical tail sizing."""

    enable: bool = True

    # Tail arm coefficients relative to fuselage length
    ll_ht: float = 0.49380955
    ll_vt: float = 0.48267689

    # Non-dimensional tail volume coefficients
    c_ht: float = 0.9
    c_vt: float = 0.09

    # Sizing geometry assumptions
    # Legacy fixed main-wing x location in meters from nose (retained for compatibility).
    l_wing_m: float = 8.7
    # If true, use `main_wing_loacation_m` as wing root x-location.
    # If false, place wing root at `0.4 * fuselage_length`.
    main_wing_location: bool = False
    main_wing_loacation_m: float = 8.7
    a_ht: float = 5.0
    lamda_ht: float = 0.4
    a_vt: float = 1.92
    lamda_vt: float = 0.4

    # Wing reference point/airfoil placement
    x_fuse_mm: float = 0.0
    ac: float = 0.25
    wingpos: str = "rand"
    theta_le_deg: float = 5.0
    theta_le_ht_deg: float = 10.0
    theta_le_vt_deg: float = 25.0

    # Empirical cap offsets used by the original TailCompu formulation
    delta_hs_cap_length_mm: float = 600.0
    delta_vs_cap_length_mm: float = 300.0

    # Tail iteration settings
    max_iter: int = 200
    tol_m: float = 1e-5


@dataclass(frozen=True)
class WeightsConfig:
    """Non-structural mass items and payload assumptions."""

    oemmisc_base_kg: float = 0.0
    payload_kg: float = 2400.0

    # Battery energy density (kWh/kg)
    rhobatt_kwh_per_kg: float = 0.30

    # Battery sizing mode:
    #   - "fixed_time": use `battery_use_time_hr`
    #   - "TO_Climb_only": sum battery energy for takeoff (fixed 1 min) + climb duration
    battery_sizing_mode: str = "fixed_time"
    battery_use_time_hr: float = 0.234

    # Battery energy reserve fraction (kWh basis):
    battery_reserve_fraction: float = 0.25


@dataclass(frozen=True)
class FlightConfig:
    """Cruise and takeoff conditions."""

    h_cr_m: float = 3000.0
    mach_cr: float = 0.35
    loiter_speed_ratio_to_cruise: float = 0.83

    # Takeoff modelling point (legacy)
    h_takeoff_m: float = 0.0
    mach_takeoff: float = 0.16


@dataclass(frozen=True)
class SolverConfig:
    power_tol_w: float = 100.0
    mtom_tol_kg: float = 1.0

    max_outer_iter: int = 50
    max_inner_iter: int = 50

    # Compressor mass model in FC sizing loop:
    #   - 'surrogate' (default): fast geometry-based regression (recommended for sweeps)
    #   - 'cadquery': full CAD mass build/export (slow, highest fidelity)
    compressor_mass_mode: str = "surrogate"

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
    # If bracketing is enabled but a sign-changing bracket is not found for many iterations,
    # allow guarded Newton/secant anyway (still requiring slope <= 0 and standard safeguards).
    newton_prebracket_allow_after_iters: int = 8
    # Initial MTOM guess policy: perform a few directed probes around the initial guess
    # to obtain a sign-changing bracket earlier (faster Newton/regula-falsi activation).
    initial_bracket_probe_enable: bool = True
    initial_bracket_probe_iters: int = 2
    initial_bracket_probe_rel_span: float = 0.08
    initial_bracket_probe_min_span_kg: float = 150.0
    # Bracket-stall guard: if bracketed bisection becomes numerically stuck, drop the
    # bracket and take one fixed-point step to escape local cycling/noise.
    bracket_stall_reset_iters: int = 4
    bracket_min_span_kg: float = 0.5

    # Infeasibility guard: stop early if residual stays on one side and does not improve
    # enough despite moving MTOM in the expected direction.
    infeasible_pos_streak_iters: int = 5
    infeasible_min_residual_reduction: float = 0.05
    infeasible_min_mtom_drop_kg: float = 1.0



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
    # Auto-expand W/S scan range if ADRpy feasible window appears clipped by current bounds.
    ws_auto_widen_enable: bool = True
    ws_auto_widen_factor: float = 1.50
    ws_auto_widen_max_passes: int = 2

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
    tail: TailConfig = TailConfig()
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


def _resolve_loiter_speed_ratio(loiter_speed_ratio_to_cruise: float) -> float:
    """Validate and return loiter speed ratio relative to cruise speed."""

    ratio = float(loiter_speed_ratio_to_cruise)
    if ratio <= 0.0:
        raise ValueError(
            f"flight.loiter_speed_ratio_to_cruise must be > 0, got {loiter_speed_ratio_to_cruise}."
        )
    return ratio


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
            v = float((Q_(v, "watt / newton") * _G0).to("kilowatt / kilogram").magnitude)
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
            v = float((Q_(v, "pascal") / _G0).to("kilogram / meter ** 2").magnitude)
        out[new_key] = f"{v}"
        out.pop(old_key, None)

    return out


def _parse_bool_literal(value: str) -> Optional[bool]:
    """Parse common boolean literals and return None when not boolean-like."""
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_float_like(raw: str) -> float:
    """Coerce a token to float, accepting boolean literals as 1.0/0.0."""
    b = _parse_bool_literal(raw)
    if b is not None:
        return 1.0 if b else 0.0
    return float(raw)


def _coerce_int_like(raw: str) -> int:
    """Coerce a token to int, accepting decimal and boolean literals."""
    b = _parse_bool_literal(raw)
    if b is not None:
        return 1 if b else 0
    return int(float(raw))  # tolerates inputs like "3000.0"


def _strip_wrapping_quotes(raw: str) -> str:
    """Remove a single matching pair of wrapping quotes from a token."""
    if len(raw) >= 2 and ((raw[0] == raw[-1] == '"') or (raw[0] == raw[-1] == "'")):
        return raw[1:-1]
    return raw


def _coerce_value(raw: str, typ):
    """Coerce a string value from the input file into the annotated field type."""
    s = raw.strip()

    # NEW: handle postponed annotations represented as strings (e.g., 'float', 'int', 'bool', 'str')
    if isinstance(typ, str):
        t = typ.replace(" ", "")
        if t in ("float", "builtins.float"):
            return _coerce_float_like(s)
        if t in ("int", "builtins.int"):
            return _coerce_int_like(s)
        if t in ("str", "builtins.str"):
            return _strip_wrapping_quotes(s)
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
    raw = _strip_wrapping_quotes(raw)

    if typ is str:
        return raw
    if typ is float:
        return _coerce_float_like(raw)
    if typ is int:
        return _coerce_int_like(raw)
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
            raw_value = section[f.name]
            try:
                updates[f.name] = _coerce_value(raw_value, f.type)
            except Exception as exc:
                raise ValueError(
                    f"Invalid value in [{section_name}] {f.name}={raw_value!r} "
                    f"(expected {f.type!r})."
                ) from exc

    return replace(default_obj, **updates) if updates else default_obj


def _sync_legacy_flight_and_hydrogen_from_constraint_brief(
    *,
    flight: FlightConfig,
    hydrogen: HydrogenConfig,
    brief: ConstraintBriefConfig,
) -> Tuple[FlightConfig, HydrogenConfig]:
    """Map constraint brief inputs into legacy flight/hydrogen fields.

    Source-of-truth mapping:
      - flight.h_cr_m      <- constraint_brief.cruisealt_m
      - flight.mach_cr     <- constraint_brief.cruisespeed_ktas (converted to Mach at cruise altitude)
      - flight.h_takeoff_m <- constraint_brief.rwyelevation_m
      - hydrogen.vv_mps    <- constraint_brief.climbrate_fpm (converted to m/s)
    """

    h_cr_m = float(brief.cruisealt_m)
    h_takeoff_m = float(brief.rwyelevation_m)

    v_cruise = Q_(float(brief.cruisespeed_ktas), "knot").to("meter / second")
    v_cruise_mps = float(v_cruise.to("meter / second").magnitude)
    if v_cruise_mps <= 0.0:
        raise ValueError(
            f"constraint_brief.cruisespeed_ktas must be > 0 to compute flight.mach_cr, got {brief.cruisespeed_ktas}."
        )

    a_cr_mps = float(Atmosphere(h_cr_m).speed_of_sound[0])
    if a_cr_mps <= 0.0:
        raise ValueError(f"Invalid speed of sound at constraint_brief.cruisealt_m={h_cr_m}.")
    mach_cr = v_cruise_mps / a_cr_mps

    climb_rate = Q_(float(brief.climbrate_fpm), "foot / minute").to("meter / second")
    vv_mps = float(climb_rate.to("meter / second").magnitude)
    if vv_mps <= 0.0:
        raise ValueError(
            f"constraint_brief.climbrate_fpm must be > 0 to compute hydrogen.vv_mps, got {brief.climbrate_fpm}."
        )

    return (
        replace(
            flight,
            h_cr_m=float(h_cr_m),
            mach_cr=float(mach_cr),
            h_takeoff_m=float(h_takeoff_m),
        ),
        replace(hydrogen, vv_mps=float(vv_mps)),
    )


def load_design_config(input_path: Path) -> DesignConfig:
    """Load DesignConfig from an INI-style text file.

    The file extension can be .txt; the syntax is INI-like:
      [section]
      key = value

    Sections map to the nested dataclasses in DesignConfig:
      mission, flight, fuel_cell_arch, fuel_cell_op, hybrid, eff, densities, p_w,
      cooling, hydrogen, wing, fuselage, weights, solver,
      constraint_sizing, constraint_brief, constraint_performance, constraint_geometry, tail,
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
    tail = _update_dataclass_from_section(cfg_default.tail, "tail", section("tail"))
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

    # Keep legacy fields synchronized from constraint_brief so one parameter set
    # drives both ADRpy and the core flight-point sizing logic.
    flight, hydrogen = _sync_legacy_flight_and_hydrogen_from_constraint_brief(
        flight=flight,
        hydrogen=hydrogen,
        brief=constraint_brief,
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
        tail=tail,
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
    section("tail", cfg.tail)
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
class TailSizingResult:
    """Computed tail sizing outputs used by legacy Raymer sizing inputs."""

    s_ht_m2: float
    s_vt_m2: float
    b_ht_m: float
    b_vt_m: float
    l_ht_act_m: float
    l_vt_act_m: float
    x_true_ht_m: float
    x_true_vt_m: float
    c_ht_act: float
    c_vt_act: float


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
    s_ht_m2: float
    s_vt_m2: float
    b_ht_m: float
    b_vt_m: float
    l_ht_act_m: float
    l_vt_act_m: float
    x_true_ht_m: float
    x_true_vt_m: float

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
        t_air_k = float(Atmosphere(cruise_altitude_m).temperature[0])
        d_t_k = max(float(cfg.dT_K), 1e-9)
        t_ratio = t_air_k / d_t_k

        self._f_dT = 0.0038 * (t_ratio**2) + 0.0352 * t_ratio + 0.1817

    def power_required_w(self, heat_rejected_kw: float) -> float:
        """Compute cooling system electrical power (W).

        Parameters
        ----------
        heat_rejected_kw:
            Heat rejected (kW). This is consistent with legacy usage.
        """

        cooling_power_kw = (0.371 * float(heat_rejected_kw) + 1.33) * self._f_dT
        return float(cooling_power_kw * 1000.0)


class FuelCellSystemModel:
    """Fuel cell system sizing model (per nacelle)."""

    def __init__(
        self,
        arch: FuelCellArchitecture,
        comp_stl_path: Optional[Path] = None,
        comp_mass_mode: Optional[str] = None,
    ):
        self._arch = arch
        self._comp_stl_path = comp_stl_path
        self._nacelle_cache: Dict[Tuple[float, float, float, float, float, bool], FuelCellSystemSizingResult] = {}
        env_override = os.getenv("HFCAD_COMP_MASS_MODE")
        selected_mode = env_override if env_override is not None else comp_mass_mode
        self._comp_mass_mode = self._normalize_comp_mass_mode(selected_mode)

    @staticmethod
    def _normalize_comp_mass_mode(value: Optional[str]) -> str:
        v = "surrogate" if value is None else str(value).strip().lower()
        if v in {"surrogate", "fast", "approx", "regression"}:
            return "surrogate"
        if v in {"cadquery", "cad", "full"}:
            return "cadquery"
        logger.warning(
            "Unknown solver.compressor_mass_mode=%r. Falling back to 'surrogate'.",
            value,
        )
        return "surrogate"

    @staticmethod
    def _round_key(x: float, quantum: float) -> float:
        if quantum <= 0.0:
            return float(x)
        return float(round(float(x) / quantum) * quantum)

    def _compressor_mass_kg(self, *, geom_comp: Optional[Iterable[float]], power_comp_w: float) -> float:
        if geom_comp is None:
            return 0.0
        geom = tuple(float(x) for x in geom_comp)

        if self._comp_mass_mode in {"cad", "cadquery", "full"}:
            return float(compressor_mass_model(list(geom), float(power_comp_w), stl_path=self._comp_stl_path))

        if len(geom) >= 8:
            return float(
                _compressor_mass_surrogate_kg(
                    float(power_comp_w),
                    geom[0],
                    geom[1],
                    geom[2],
                    geom[3],
                    geom[4],
                    geom[5],
                    geom[6],
                    geom[7],
                )
            )

        # Fallback linear trend in case geometry format changes unexpectedly.
        return max(0.0, 0.52 * float(power_comp_w) * 1e-3 + 0.2)

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

        power_fc_sys_w = float(power_fc_sys_w)
        altitude_m = float(flight_point.altitude_m)
        mach = float(flight_point.mach)
        beta = float(beta)
        oversizing = float(oversizing)

        # Quantized cache key for hot fixed-point iterations.
        if not make_fig:
            power_quantum_w = 0.0 if self._comp_mass_mode in {"cad", "cadquery", "full"} else 50.0
            cache_key = (
                self._round_key(power_fc_sys_w, power_quantum_w),
                self._round_key(altitude_m, 1e-3),
                self._round_key(mach, 1e-6),
                self._round_key(beta, 1e-6),
                self._round_key(oversizing, 1e-6),
                bool(comp_bool),
            )
            cached = self._nacelle_cache.get(cache_key)
            if cached is not None:
                return cached

        # Atmospheric conditions
        c_mps, p_pa, t_k, rho_kg_m3, mu_pa_s = _atmosphere_at_altitude(round(altitude_m, 3))
        v_cr_mps = mach * c_mps
        p_tot_pa = _total_pressure_pa(p_pa, mach)
        t_tot_k = _total_temperature_k(t_k, mach)

        if verbose:
            try:
                reynolds = rho_kg_m3 * v_cr_mps * 1.8 / mu_pa_s
                logger.info(f"Reynolds_number: {reynolds:,.0f}")
            except Exception:
                pass

        # Other inputs
        cell_temp_k = 273.15 + 80.0
        mu_f = 0.95

        # Cathode inlet pressure
        pres_cathode_in_pa = beta * p_tot_pa if comp_bool else p_tot_pa

        # Cell model (cached and optionally figure-producing)
        volt_cell, power_dens_cell, eta_cell, fig = cell_model(
            float(pres_cathode_in_pa),
            float(_SEA_LEVEL_PRESSURE_PA),
            float(cell_temp_k),
            oversizing,
            make_fig=make_fig,
        )

        figs: Tuple[object, ...] = (fig,) if fig is not None else ()

        # Compressor
        if comp_bool:
            power_req_new = float(power_fc_sys_w)
            power_comp_w = 0.0
            geom_comp = None
            rho_humid_in = float(rho_kg_m3)
            m_dot_comp = None

            tol = max(float(comp_tol_w), 1e-6 * abs(power_req_new))

            for _ in range(int(max_comp_iter)):
                power_req = power_req_new
                geom_comp, power_comp_w, rho_humid_in, m_dot_comp = compressor_performance_model(
                    power_req,
                    volt_cell,
                    beta,
                    float(p_tot_pa),
                    float(t_tot_k),
                    float(mu_pa_s),
                )
                power_comp_w = float(power_comp_w)
                power_req_new = power_fc_sys_w + power_comp_w
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

            m_comp = self._compressor_mass_kg(geom_comp=geom_comp, power_comp_w=float(power_comp_w))
        else:
            m_comp = 0.0
            power_comp_w = 0.0
            power_req_new = float(power_fc_sys_w)
            m_dot_comp = float(mass_flow_stack(float(power_req_new), volt_cell))
            rho_humid_in = float(rho_kg_m3)

        # Remaining BOP models
        m_humid = float(humidifier_model(m_dot_comp, rho_humid_in))
        q_all, m_hx, dim_hx = heat_exchanger_model(
            float(power_req_new),
            volt_cell,
            float(cell_temp_k),
            mu_f,
            float(v_cr_mps),
            float(mach),
            float(p_tot_pa),
            float(t_tot_k),
            float(rho_kg_m3),
            float(mu_pa_s),
        )

        # Stack model
        m_stacks, dim_stack, res_stack = stack_model(
            self._arch.n_stacks_series,
            self._arch.volt_req_v,
            volt_cell,
            float(power_req_new),
            power_dens_cell,
        )

        # Aggregate
        m_sys = float(m_stacks + m_comp + m_humid + m_hx)

        p_den = power_comp_w + power_fc_sys_w
        if p_den <= 0.0:
            eta_fcsys = 0.0
        else:
            eta_fcsys = float(eta_cell * power_fc_sys_w / p_den * mu_f)

        mdot_h2 = float(1.05e-8 * p_den / volt_cell)

        if verbose:
            logger.info(f"Stack prop output power: {power_fc_sys_w/1000:,.0f} kW, Pcomp: {power_comp_w/1000:,.1f} kW")
            logger.info(f"Cell efficiency: {eta_cell:,.3f}, Output efficiency: {eta_fcsys:,.3f}")
            logger.info(f"mdot_h2: {mdot_h2*1000:,.2f} g/s")

        result = FuelCellSystemSizingResult(
            m_sys_kg=float(m_sys),
            m_stacks_kg=float(m_stacks),
            m_comp_kg=float(m_comp),
            m_humid_kg=float(m_humid),
            m_hx_kg=float(m_hx),
            eta_fcsys=float(eta_fcsys),
            mdot_h2_kgps=float(mdot_h2),
            power_comp_w=float(power_comp_w),
            q_all_w=float(q_all),
            v_cr_mps=float(v_cr_mps),
            dim_stack_m=(float(dim_stack[0]), float(dim_stack[1]), float(dim_stack[2])),
            dim_hx_m=(float(dim_hx[0]), float(dim_hx[1]), float(dim_hx[2])),
            res_stack=(float(res_stack[0]), float(res_stack[1])),
            figs=figs,
        )

        if not make_fig:
            self._nacelle_cache[cache_key] = result
            # Keep cache bounded without adding another dependency.
            if len(self._nacelle_cache) > 4096:
                self._nacelle_cache.clear()

        return result


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
        """Fixed-point iteration for total electrical power (excluding cooling)."""

        p_total = float(initial_total_power_w)

        for inner_iter in range(1, self._cfg.solver.max_inner_iter + 1):
            # New standard input: kW/kg
            p_shaft = float(mtom_kg) * float(p_w_kw_per_kg) * 1000.0

            split = self._powertrain.split_shaft_power(float(p_shaft), psi)
            p_bus_required = float(split.p_bus_required_w)

            # Fuel-cell net electrical output used for per-nacelle sizing.
            #
            # Coupled fixed-point form:
            # - p_total includes propulsion bus demand + compressor auxiliary.
            # - when battery discharges (p_battery > 0), subtract that contribution.
            # - when battery charges (p_battery < 0), charging demand is already represented
            #   in p_total and must be supplied by FC.
            p_fc_sys_total = float(p_total - max(float(split.p_battery_w), 0.0))

            # Guardrail: ensure sizing power stays positive.
            if p_fc_sys_total <= 0.0:
                p_fc_sys_total = max(float(split.p_fuelcell_w), 1.0)

            p_fc_sys_per_nacelle = p_fc_sys_total / self._cfg.fuel_cell_arch.n_stacks_parallel

            nacelle = self._fc.size_nacelle(
                power_fc_sys_w=p_fc_sys_per_nacelle,
                flight_point=flight_point,
                beta=beta,
                oversizing=oversizing,
                comp_bool=True,
                make_fig=False,
                verbose=logger.isEnabledFor(logging.DEBUG),
            )

            p_comp_total = float(nacelle.power_comp_w * self._cfg.fuel_cell_arch.n_stacks_parallel)
            heat_rejected_w = float(self._cfg.fuel_cell_arch.n_stacks_parallel * nacelle.q_all_w)
            p_cooling = float(self._cooling.power_required_w(heat_rejected_w / 1000.0))

            # Per user requirement, total power excludes cooling load.
            # Cooling is still computed and reported via p_cooling_w.
            p_total_new = float(p_bus_required + p_comp_total)

            if abs(float(p_total_new - p_total)) <= self._cfg.solver.power_tol_w:
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
                    heat_rejected_kw=float(heat_rejected_w / 1000.0),
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
        self._conv = UnitConverter()

        # Pre-compute cruise environment
        atm_cr = Atmosphere(cfg.flight.h_cr_m)
        self._c_cr = Q_(float(atm_cr.speed_of_sound[0]), "meter / second")
        self._v_cr = Q_(float(cfg.flight.mach_cr), "dimensionless") * self._c_cr
        self._v_cr_mps = float(self._v_cr.to("meter / second").magnitude)
        self._rho_cr_q = Q_(float(atm_cr.density[0]), "kilogram / meter ** 3")
        self._rho_cr = float(self._rho_cr_q.to("kilogram / meter ** 3").magnitude)
        self._beta_cruise, _ = _resolve_cruise_betas(
            cfg.fuel_cell_op.beta,
            cruise_density_kg_per_m3=self._rho_cr,
        )

        # Wing loading (mass-based) in kg/m^2
        self._w_s = Q_(float(cfg.wing.wing_loading_kg_per_m2), "kilogram / meter ** 2")
        self._w_s_kg_per_m2 = float(self._w_s.to("kilogram / meter ** 2").magnitude)

    @property
    def v_cruise_mps(self) -> float:
        return self._v_cr_mps

    def _wing_planform_geometry(self, *, S: float, b: float, lamda: float) -> Tuple[float, float, float, float]:
        """Return simple tapered-wing geometric parameters: Croot, Ctip, Cbar, Ybar."""

        if b <= 0.0 or S <= 0.0:
            raise ValueError("Wing area and span must be positive for tail sizing.")
        if lamda <= 0.0:
            raise ValueError("Wing taper ratio must be positive for tail sizing.")

        croot = 2 * S / (b * (1 + lamda))
        ctip = lamda * croot
        cbar = 2.0 / 3.0 * croot * ((1 + lamda + lamda**2) / (1 + lamda))
        ybar = b / 6.0 * ((1 + 2 * lamda) / (1 + lamda))
        return croot, ctip, cbar, ybar

    def _size_tails(self, *, FL: float, S: float, b: float, lamda: float, cfg_tail) -> TailSizingResult:
        """Compute tail geometry using TailCompu-style sizing iteration."""

        from math import sqrt, tan

        min_tail_arm = 0.05
        if not cfg_tail.enable:
            raise RuntimeError("Tail sizing disabled, fallback should skip this method.")

        if FL <= 0.0 or S <= 0.0 or b <= 0.0:
            raise ValueError("Fuselage length, wing area, and span must be positive for tail sizing.")

        _, _, cbar, ybar = self._wing_planform_geometry(S=S, b=b, lamda=lamda)

        if bool(cfg_tail.main_wing_location):
            l_wing = float(cfg_tail.main_wing_loacation_m)
        else:
            l_wing = float(FL * 0.3)

        # Keep a hard validity check for root location placement.
        if l_wing <= 0.0:
            raise ValueError("Computed wing-root x location is invalid for tail sizing.")

        ll_ht = float(cfg_tail.ll_ht)
        ll_vt = float(cfg_tail.ll_vt)
        c_ht = float(cfg_tail.c_ht)
        c_vt = float(cfg_tail.c_vt)
        a_ht = float(cfg_tail.a_ht)
        lamda_ht = float(cfg_tail.lamda_ht)
        a_vt = float(cfg_tail.a_vt)
        lamda_vt = float(cfg_tail.lamda_vt)
        x_fuse_mm = float(cfg_tail.x_fuse_mm)
        ac = float(cfg_tail.ac)
        theta_le = math.radians(float(cfg_tail.theta_le_deg))
        theta_le_ht = math.radians(float(cfg_tail.theta_le_ht_deg))
        theta_le_vt = math.radians(float(cfg_tail.theta_le_vt_deg))

        delta_hs_mm = float(cfg_tail.delta_hs_cap_length_mm)
        delta_vs_mm = float(cfg_tail.delta_vs_cap_length_mm)
        wingpos = str(cfg_tail.wingpos).strip().lower()

        ll_ht = max(min_tail_arm, ll_ht)
        ll_vt = max(min_tail_arm, ll_vt)

        if a_ht < 3.5:
            a_ht = 3.5
        elif a_ht > 4.5:
            a_ht = 4.5

        if lamda <= 0.0 or lamda_ht <= 0.0 or lamda_vt <= 0.0:
            raise ValueError("Tail sizing requires positive wing/tail taper ratios.")

        k = 1
        l_wing_ac = ybar * tan(theta_le) + cbar * ac

        while k <= int(cfg_tail.max_iter):
            if not math.isfinite(ll_ht) or not math.isfinite(ll_vt):
                raise RuntimeError("Tail sizing diverged to non-finite arm values.")

            l_ht = ll_ht * FL
            l_vt = ll_vt * FL
            if l_ht <= 0.0 or l_vt <= 0.0:
                raise RuntimeError("Tail arm became non-positive during sizing.")

            s_ht = c_ht * cbar * S / l_ht
            s_vt = c_vt * b * S / l_vt
            if not (math.isfinite(s_ht) and math.isfinite(s_vt)) or s_ht <= 0.0 or s_vt <= 0.0:
                raise RuntimeError("Tail sizing produced non-positive tail areas.")

            b_ht = sqrt(a_ht * s_ht)
            b_vt = sqrt(a_vt * s_vt)

            ybar_ht = b_ht / 6.0 * ((1 + 2 * lamda_ht) / (1 + lamda_ht))
            cbar_ht = 2.0 / 3.0 * (2 * s_ht / (b_ht * (1 + lamda_ht))) * ((1 + lamda_ht + lamda_ht**2) / (1 + lamda_ht))
            croot_ht = 2 * s_ht / (b_ht * (1 + lamda_ht))

            ybar_vt = b_vt / 6.0 * ((1 + 2 * lamda_vt) / (1 + lamda_vt))
            cbar_vt = 2.0 / 3.0 * (2 * s_vt / (b_vt * (1 + lamda_vt))) * ((1 + lamda_vt + lamda_vt**2) / (1 + lamda_vt))
            croot_vt = 2 * s_vt / (b_vt * (1 + lamda_vt))

            x_wing_mm = l_wing * 1000.0
            x_ht_mm = (FL - croot_ht - delta_hs_mm / 1000.0) * 1000.0
            x_vt_mm = (FL - croot_vt - delta_vs_mm / 1000.0) * 1000.0

            x_true_wing_mm = x_wing_mm - x_fuse_mm
            x_true_ht_mm = x_ht_mm - x_fuse_mm
            x_true_vt_mm = x_vt_mm - x_fuse_mm

            l_ac_wing_mm = l_wing_ac * 1000.0
            l_ac_ht = ybar_ht * tan(theta_le_ht) + 0.25 * cbar_ht
            l_ac_vt = ybar_vt * tan(theta_le_vt) + 0.25 * cbar_vt
            l_ac_ht_mm = l_ac_ht * 1000.0
            l_ac_vt_mm = l_ac_vt * 1000.0

            x_ac_wing_mm = x_true_wing_mm + l_ac_wing_mm
            x_ac_ht_mm = x_true_ht_mm + l_ac_ht_mm
            x_ac_vt_mm = x_true_vt_mm + l_ac_vt_mm

            l_ht_act_m = (x_ac_ht_mm - x_ac_wing_mm) / 1000.0
            l_vt_act_m = (x_ac_vt_mm - x_ac_wing_mm) / 1000.0
            if not (math.isfinite(l_ht_act_m) and math.isfinite(l_vt_act_m)):
                raise RuntimeError("Tail sizing produced invalid actual tail moment arms.")
            # Guard transient geometry inversions by using target arms as a recovery step.
            # This keeps sizing from dropping to legacy defaults when intermediate iterations
            # briefly become infeasible.
            if l_ht_act_m <= 0.0:
                l_ht_act_m = max(min_tail_arm, l_ht)
            if l_vt_act_m <= 0.0:
                l_vt_act_m = max(min_tail_arm, l_vt)

            # Keep parity with the legacy expression used by TailCompu:
            # Cbar_HT = 2/3*S_HT / (b_HT*(1+lamda_HT)) * (1+lamda_HT+lamda_HT^2)/(1+lamda_HT)
            c_ht_act = s_ht / (cbar * S / l_ht_act_m)
            c_vt_act = s_vt / (b * S / l_vt_act_m)
            if not (math.isfinite(c_ht_act) and math.isfinite(c_vt_act)) or c_ht_act <= 0.0 or c_vt_act <= 0.0:
                raise RuntimeError("Tail sizing produced invalid correction coefficients.")

            l_ac_ht_act = l_ht_act_m
            l_ac_vt_act = l_vt_act_m

            delta_ht = abs(l_ac_ht_act - l_ht)
            delta_vt = abs(l_ac_vt_act - l_vt)
            ll_ht_rcmd = c_ht_act * cbar * S / s_ht / FL
            ll_vt_rcmd = c_vt_act * b * S / s_vt / FL

            if delta_ht <= cfg_tail.tol_m and delta_vt <= cfg_tail.tol_m:
                return TailSizingResult(
                    s_ht_m2=float(s_ht),
                    s_vt_m2=float(s_vt),
                    b_ht_m=float(b_ht),
                    b_vt_m=float(b_vt),
                    l_ht_act_m=float(l_ht_act_m),
                    l_vt_act_m=float(l_vt_act_m),
                    x_true_ht_m=float(x_true_ht_mm) / 1000.0,
                    x_true_vt_m=float(x_true_vt_mm) / 1000.0,
                    c_ht_act=float(c_ht_act),
                    c_vt_act=float(c_vt_act),
                )

            ll_ht = ll_ht_rcmd
            ll_vt = ll_vt_rcmd
            if wingpos == "fwd":
                ll_ht = max(0.05, ll_ht)
            elif wingpos == "aft":
                ll_vt = max(0.05, ll_vt)
            else:
                ll_ht = max(min_tail_arm, ll_ht)
                ll_vt = max(min_tail_arm, ll_vt)
            k += 1

            # Explicit convergence guard to avoid pathological loops.
            if not math.isfinite(ll_ht) or not math.isfinite(ll_vt):
                raise RuntimeError("Tail sizing diverged to non-finite values.")

        raise RuntimeError(
            f"Tail sizing did not converge in {cfg_tail.max_iter} iterations. "
            f"Final deltas: HT={delta_ht:.3e}, VT={delta_vt:.3e}"
        )

    def estimate(
        self,
        *,
        mtom_guess_kg: float,
        climb: PhasePowerResult,
        cruise: PhasePowerResult,
        loiter: PhasePowerResult,
        takeoff: PhasePowerResult,
        cruise_charger: Optional[PhasePowerResult] = None,
    ) -> MassBreakdown:
        cfg = self._cfg
        conv = self._conv

        # -----------------
        # Fuel cell sizing at governing max-FC-power phase
        # -----------------
        sizing_phases = {
            "takeoff": takeoff,
            "climb": climb,
            "cruise": cruise,
            "loiter": loiter,
        }
        if cruise_charger is not None:
            sizing_phases["cruise_charger"] = cruise_charger
        nacelle_design_phase, nacelle_design_result, power_fc_sys_total = _max_phase_fc_sizing_power(sizing_phases)
        power_fc_sys_total = max(float(power_fc_sys_total), 1.0)
        power_fc_sys_per_nacelle = power_fc_sys_total / cfg.fuel_cell_arch.n_stacks_parallel

        logger.info(
            "Fuel-cell system mass representative phase: %s (P_fc_sys=%.3f kW).",
            nacelle_design_phase,
            power_fc_sys_total / 1000.0,
        )
        nacelle_design = nacelle_design_result.nacelle

        # Mass per nacelle: use a single governing phase for stack/compressor/humidifier/HX.
        m_fc_system_per_nacelle = float(
            nacelle_design.m_stacks_kg
            + nacelle_design.m_humid_kg
            + nacelle_design.m_comp_kg
            + nacelle_design.m_hx_kg
        )
        m_fc_system = float(cfg.fuel_cell_arch.n_stacks_parallel * m_fc_system_per_nacelle)

        # -----------------
        # Mission kinematics shared by battery sizing and fuel accounting
        # -----------------
        brief = cfg.constraint_brief

        climb_rate = Q_(float(brief.climbrate_fpm), "foot / minute").to("meter / second")
        if float(climb_rate.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.climbrate_fpm must be > 0, got {brief.climbrate_fpm}.")

        v_climb = Q_(float(brief.climbspeed_kias), "knot").to("meter / second")
        v_cruise = Q_(float(brief.cruisespeed_ktas), "knot").to("meter / second")
        loiter_speed_ratio = _resolve_loiter_speed_ratio(cfg.flight.loiter_speed_ratio_to_cruise)
        v_loiter = loiter_speed_ratio * v_cruise
        if float(v_climb.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.climbspeed_kias must be > 0, got {brief.climbspeed_kias}.")
        if float(v_cruise.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.cruisespeed_ktas must be > 0, got {brief.cruisespeed_ktas}.")

        # Use runway elevation as mission climb start altitude; climbalt_m is the constraint evaluation altitude.
        h_delta = Q_(max(float(brief.cruisealt_m) - float(brief.rwyelevation_m), 0.0), "meter")
        t_climb = (h_delta / climb_rate).to("second")
        r_climb = (v_climb * t_climb).to("meter")

        # -----------------
        # Battery mass
        # -----------------
        # mbatt_old = (Ebat_discharge) / rhobatt
        # mbatt = mbatt_old * (1 + reserve_fraction)
        p_battery_takeoff_discharge = Q_(max(float(takeoff.p_battery_w), 0.0), "watt")
        p_battery_climb_discharge = Q_(max(float(climb.p_battery_w), 0.0), "watt")
        rhobatt = Q_(float(cfg.weights.rhobatt_kwh_per_kg), "kilowatt_hour / kilogram")
        if float(cfg.weights.battery_reserve_fraction) < 0.0:
            raise ValueError(
                f"weights.battery_reserve_fraction must be >= 0, got {cfg.weights.battery_reserve_fraction}."
            )
        battery_mode = str(cfg.weights.battery_sizing_mode).strip().lower()
        if battery_mode in ("to_climb_only", "climb_only"):
            # Backward compatible alias: legacy "climb_only" is accepted, but now
            # battery sizing includes both takeoff and climb battery discharge energy.
            t_takeoff = Q_(1.0, "minute").to("hour")
            battery_discharge_energy = (
                p_battery_takeoff_discharge * t_takeoff
                + p_battery_climb_discharge * t_climb.to("hour")
            )
        elif battery_mode == "fixed_time":
            if float(cfg.weights.battery_use_time_hr) < 0.0:
                raise ValueError(f"weights.battery_use_time_hr must be >= 0, got {cfg.weights.battery_use_time_hr}.")
            battery_use_time = Q_(float(cfg.weights.battery_use_time_hr), "hour")
            battery_discharge_energy = p_battery_climb_discharge * battery_use_time
        else:
            raise ValueError(
                "weights.battery_sizing_mode must be one of: "
                "'TO_Climb_only', 'fixed_time'. "
                f"Got {cfg.weights.battery_sizing_mode!r}."
            )

        m_battery = float(
            (
                battery_discharge_energy
                * (1.0 + cfg.weights.battery_reserve_fraction)
                / rhobatt
            ).to("kilogram").magnitude
        )

        # -----------------
        # PMAD and motor sizing (legacy)
        # -----------------
        # NOTE: legacy PPDU expression includes a thermal term (heat_rejected_kw) without unit conversion.
        # This is retained for backward compatibility.
        ppdu_w = (
            (
                Q_(float(climb.p_fuelcell_w), "watt")
                + Q_(float(climb.heat_rejected_kw), "watt")
                + Q_(float(climb.p_cooling_w), "watt")
            )
            * cfg.eff.eta_converter
            + Q_(float(climb.p_battery_w), "watt")
        )
        pem_w = (
            (
                Q_(float(climb.p_fuelcell_w), "watt") * cfg.eff.eta_converter
                + Q_(float(climb.p_battery_w), "watt")
            )
            * cfg.eff.eta_pdu
            * cfg.eff.eta_inverter
        )

        m_e_motor = float((pem_w / Q_(float(cfg.densities.rhoem_w_per_kg), "watt / kilogram")).to("kilogram").magnitude)
        m_pmad = float((ppdu_w / Q_(float(cfg.densities.rhopmad_w_per_kg), "watt / kilogram")).to("kilogram").magnitude)

        m_powertrain_total = float(m_fc_system + m_pmad + m_e_motor)

        # -----------------
        # Fuel mass and tank sizing
        # -----------------
        # Mission timeline for fuel: ready(0), taxi(5 min), takeoff(1 min), climb(variable),
        # cruise(variable), loiter(15 min at loiter speed), landing(1 min, no fuel).
        # This intentionally does not use cfg.mission.times_min.
        t_ready = Q_(0.0, "minute").to("second")
        t_taxi = Q_(5.0, "minute").to("second")
        t_takeoff = Q_(1.0, "minute").to("second")
        t_loiter = Q_(15.0, "minute").to("second")
        t_landing = Q_(1.0, "minute").to("second")

        mdot_climb = Q_(float(climb.mdot_h2_kgps), "kilogram / second")
        mdot_cruise = Q_(float(cruise.mdot_h2_kgps), "kilogram / second")
        mdot_loiter = Q_(float(loiter.mdot_h2_kgps), "kilogram / second")
        mdot_takeoff = Q_(float(takeoff.mdot_h2_kgps), "kilogram / second")
        mdot_taxi = 0.10 * mdot_climb
        mdot_landing = Q_(0.0, "kilogram / second")

        r_loiter = (v_loiter * t_loiter).to("meter")
        range_total = Q_(float(cfg.hydrogen.range_total_m), "meter")
        r_cruise = (range_total - r_climb - r_loiter).to("meter")
        if float(r_cruise.to("meter").magnitude) < 0.0:
            logger.warning(
                "Total range %.1f km is smaller than climb+loiter range %.1f km. "
                "Setting cruise segment to zero for fuel accounting.",
                float(range_total.to("kilometer").magnitude),
                float((r_climb + r_loiter).to("kilometer").magnitude),
            )
            r_cruise = Q_(0.0, "meter")
        t_cruise = (r_cruise / v_cruise).to("second")

        m_fuel = (
            mdot_takeoff * t_takeoff
            + mdot_climb * t_climb
            + mdot_cruise * t_cruise
            + mdot_loiter * t_loiter
            + mdot_taxi * t_taxi
            + mdot_landing * t_landing
            + Q_(0.0, "kilogram / second") * t_ready
        ).to("kilogram")

        tank_volume = (
            m_fuel
            * float(cfg.hydrogen.coversize)
            / Q_(float(cfg.hydrogen.rho_h2_kg_m3), "kilogram / meter ** 3")
            / float(cfg.hydrogen.eta_vol)
        ).to("meter ** 3")
        tank_length = (tank_volume / (math.pi * (Q_(float(cfg.fuselage.dfus_m), "meter") / 2.0) ** 2)).to("meter")

        m_tank = (m_fuel * float(cfg.hydrogen.coversize) * (1.0 / float(cfg.hydrogen.eta_storage) - 1.0)).to("kilogram")
        m_fuel_kg = float(m_fuel.to("kilogram").magnitude)
        tank_volume_m3 = float(tank_volume.to("meter ** 3").magnitude)
        tank_length_m = float(tank_length.to("meter").magnitude)
        m_tank_kg = float(m_tank.to("kilogram").magnitude)

        # -----------------
        # Wing sizing (geometry)
        # -----------------
        # New standard input is wing loading in kg/m^2 (mass/area).
        # Use metric directly to avoid mixing force-based and mass-based definitions.
        wing_area = (Q_(float(mtom_guess_kg), "kilogram") / self._w_s).to("meter ** 2")
        wing_span = (float(cfg.wing.aspect_ratio) * wing_area) ** 0.5
        wing_area_m2 = float(wing_area.to("meter ** 2").magnitude)
        wing_span_m = float(wing_span.to("meter").magnitude)

        # -----------------
        # Fuselage sizing (geometry + wetted area)
        # -----------------
        dfus_ft = conv.meter_feet_q(cfg.fuselage.dfus_m, "ft")
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

        circum_fus_ft = (2.0 * k_c_used * (dfus_ft + hfus_ft)).to("foot")

        lnose_ft = (dfus_ft * float(cfg.fuselage.fnose)).to("foot")
        ltail_ft = (dfus_ft * float(cfg.fuselage.ftail)).to("foot")

        lcabin_m = float((cfg.fuselage.npass / cfg.fuselage.nseat_abreast) * cfg.fuselage.lseat_m + cfg.fuselage.ldoor_m)
        lcabin_ft = conv.meter_feet_q(lcabin_m, "ft")

        ltank_ft = tank_length.to("foot")
        lfus_ft = (lnose_ft + ltail_ft + lcabin_ft + ltank_ft).to("foot")

        swet_fus_ft2 = (
            circum_fus_ft * ((lcabin_ft + ltank_ft) + k_w_nose * lnose_ft + k_w_tail * ltail_ft)
        ).to("foot ** 2")
        swet_fus_m2 = float(swet_fus_ft2.to("meter ** 2").magnitude)

        fuselage_length_m = float(lfus_ft.to("meter").magnitude)
        croot_wing_m, _, _, _ = self._wing_planform_geometry(
            S=wing_area_m2, b=wing_span_m, lamda=cfg.wing.taper
        )

        # -----------------
        # Empirical tail geometry via TailCompu-style sizing
        # -----------------
        try:
            tail_sizing = self._size_tails(
                FL=fuselage_length_m,
                S=wing_area_m2,
                b=wing_span_m,
                lamda=cfg.wing.taper,
                cfg_tail=cfg.tail,
            )
            t_r_HT = 0.09
            t_r_VT = 0.09 * 1.361
            s_ht = tail_sizing.s_ht_m2
            s_vt = tail_sizing.s_vt_m2
            b_ht = tail_sizing.b_ht_m
            b_vt = tail_sizing.b_vt_m
            l_ht_act = tail_sizing.l_ht_act_m
            l_vt_act = tail_sizing.l_vt_act_m
            x_true_ht_m = tail_sizing.x_true_ht_m
            x_true_vt_m = tail_sizing.x_true_vt_m
        except Exception as exc:
            logger.warning(
                "Tail sizing fallback to legacy defaults due to: %s", exc
            )
            t_r_HT = 0.09
            t_r_VT = 0.09 * 1.361
            s_ht = 1.82
            s_vt = 2.54
            b_ht = 3.0
            b_vt = 2.5
            l_ht_act = float(conv.meter_feet(cfg.fuselage.lht_ft, "meter"))
            l_vt_act = float(conv.meter_feet(cfg.fuselage.lht_ft, "meter"))
            x_true_ht_m = l_ht_act
            x_true_vt_m = l_vt_act

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
            t_r_HT=t_r_HT,
            S_HT=s_ht,
            S_VT=s_vt,
            t_r_VT=t_r_VT,
            L_HT_act=l_ht_act,
            b_HT=b_ht,
            b_VT=b_vt,
            FL=fuselage_length_m,
            Wf_mm=float(cfg.fuselage.dfus_m * 1000.0),
            hf_mm=float(cfg.fuselage.dfus_m * 1000.0),
            W_press=0.0,
            l_n_mm=float(lnose_ft.to("meter").magnitude * 1000.0),
            Croot=float(croot_wing_m),
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
            S_f=float(swet_fus_ft2.to("foot ** 2").magnitude * 23.0),
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

        oem = float(oem_misc + m_powertrain_total + m_tank_kg + w_wing + w_fus + m_battery)
        mtom = float(oem + m_fuel_kg + cfg.weights.payload_kg)

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
            m_fuel_kg=float(m_fuel_kg),
            m_tank_kg=float(m_tank_kg),
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
            s_ht_m2=float(s_ht),
            s_vt_m2=float(s_vt),
            b_ht_m=float(b_ht),
            b_vt_m=float(b_vt),
            l_ht_act_m=float(l_ht_act),
            l_vt_act_m=float(l_vt_act),
            x_true_ht_m=float(x_true_ht_m),
            x_true_vt_m=float(x_true_vt_m),
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
        self._fc_model = FuelCellSystemModel(
            cfg.fuel_cell_arch,
            comp_stl_path=self._comp_stl_path,
            comp_mass_mode=cfg.solver.compressor_mass_mode,
        )
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
            "loiter": p0,
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
        pos_residual_streak: List[Tuple[float, float]] = []  # recent (mtom, residual) for residual > 0
        prebracket_iter_count = 0
        bracket_stall_count = 0
        initial_probe_base_mtom = float(mtom)
        initial_probe_dir: Optional[float] = None
        initial_probe_attempts = 0

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

            # NOTE: MTOM closure uses cruise/loiter/climb/takeoff/cruise_charger:
            #   - max FC-power phase among takeoff/climb/cruise/loiter/cruise_charger:
            #     representative FC-system masses
            #   - takeoff+climb: battery sizing (when weights.battery_sizing_mode=TO_Climb_only)
            #   - takeoff: mission-timeline fuel accounting
            loiter_p_w_kw_per_kg = _LOITER_PW_RATIO_TO_CRUISE * float(cfg.p_w.p_w_cruise_kw_per_kg)

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

            # Loiter uses a fixed fraction of cruise P/W and lower speed.
            loiter_speed_ratio = _resolve_loiter_speed_ratio(cfg.flight.loiter_speed_ratio_to_cruise)
            loiter_mach = float(cfg.flight.mach_cr) * loiter_speed_ratio
            phases["loiter"] = self._phase_solver.solve(
                name="loiter",
                mtom_kg=mtom,
                p_w_kw_per_kg=loiter_p_w_kw_per_kg,
                flight_point=FlightPoint(cfg.flight.h_cr_m, loiter_mach),
                psi=0.0,
                beta=self._beta_cruise,
                initial_total_power_w=ptotal_guess["loiter"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["loiter"] = phases["loiter"].p_total_w

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

            # Takeoff (needed for mission fuel accounting in mass closure)
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

            # Cruise with battery charging (included in representative FC mass phase search)
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

            mass = self._mass_estimator.estimate(
                mtom_guess_kg=mtom,
                climb=phases["climb"],
                cruise=phases["cruise"],
                loiter=phases["loiter"],
                takeoff=phases["takeoff"],
                cruise_charger=phases["cruise_charger"],
            )

            residual = float(mass.mtom_kg - mtom)

            logger.info("\n-----------------------")
            logger.info(
                f"ptotal_climb: {phases['climb'].p_total_w/1000:,.0f} kW, "
                f"mtom_est: {mass.mtom_kg:,.0f} kg, mtom: {mtom:,.0f} kg, "
                f"residual: {residual:+,.2f} kg\n"
            )

            if abs(residual) <= cfg.solver.mtom_tol_kg:
                logger.info("\nCONVERGED")
                _log_converged_state(
                    state=f"MTOM iteration {outer_iter}",
                    phases=phases,
                    mass=mass,
                    cfg=cfg,
                )
                return phases, mass

            # Early stop for likely infeasible closure:
            # residual remains positive while MTOM is pushed down, but residual barely drops.
            if math.isfinite(residual) and residual > 0.0:
                pos_residual_streak.append((float(mtom), float(residual)))
            else:
                pos_residual_streak.clear()

            streak_n = max(1, int(cfg.solver.infeasible_pos_streak_iters))
            if len(pos_residual_streak) >= streak_n:
                mtom_start, res_start = pos_residual_streak[-streak_n]
                mtom_drop = float(mtom_start - mtom)
                if res_start != 0.0:
                    residual_reduction = float((res_start - residual) / abs(res_start))
                else:
                    residual_reduction = 0.0

                if (
                    mtom_drop >= float(cfg.solver.infeasible_min_mtom_drop_kg)
                    and residual_reduction < float(cfg.solver.infeasible_min_residual_reduction)
                ):
                    raise RuntimeError(
                        "Infeasible MTOM closure detected: residual stayed positive for "
                        f"{streak_n} iterations while MTOM dropped by {mtom_drop:,.2f} kg "
                        f"but residual reduction was only {100.0 * residual_reduction:.2f}%."
                    )

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
                bracketed = (pos_pt is not None) and (neg_pt is not None)
                prebracket_newton_gate = max(0, int(cfg.solver.newton_prebracket_allow_after_iters))

                if bool(cfg.solver.newton_use_bracketing) and (not bracketed):
                    prebracket_iter_count += 1
                else:
                    prebracket_iter_count = 0

                did_initial_probe = False
                # Initial directed probe around the starting MTOM guess to trigger earlier sign change.
                if (
                    bool(cfg.solver.newton_use_bracketing)
                    and bool(cfg.solver.initial_bracket_probe_enable)
                    and (not bracketed)
                    and (initial_probe_attempts < max(0, int(cfg.solver.initial_bracket_probe_iters)))
                ):
                    if math.isfinite(residual) and residual != 0.0:
                        if initial_probe_dir is None:
                            initial_probe_dir = 1.0 if residual > 0.0 else -1.0
                        span0 = max(
                            float(cfg.solver.initial_bracket_probe_min_span_kg),
                            float(cfg.solver.initial_bracket_probe_rel_span) * max(initial_probe_base_mtom, 1.0),
                        )
                        span = span0 * (2.0 ** initial_probe_attempts)
                        mtom_prop = float(initial_probe_base_mtom + initial_probe_dir * span)
                        step_note = f"initial_bracket_probe_{initial_probe_attempts + 1}"
                        initial_probe_attempts += 1
                        did_initial_probe = True

                # If configured, Newton is disabled before the first sign-changing bracket exists.
                if not did_initial_probe:
                    if (
                        bool(cfg.solver.newton_use_bracketing)
                        and (not bracketed)
                        and (prebracket_iter_count <= prebracket_newton_gate)
                    ):
                        mtom_prop = mtom + float(cfg.solver.newton_fp_relax) * residual
                        step_note = "fixed_point_prebracket"
                    else:
                        # Newton step with secant slope (1 evaluation/iter after the first)
                        if prev_mtom is not None and prev_res is not None:
                            dm = mtom - prev_mtom
                            dr = residual - prev_res
                            if dm != 0.0 and dr != 0.0:
                                slope = dr / dm  # ≈ d(residual)/d(mtom)
                                if math.isfinite(slope) and abs(slope) > float(cfg.solver.newton_slope_eps):
                                    if slope <= 0.0:
                                        mtom_newton = mtom - residual / slope
                                        mtom_prop = mtom + relax * (mtom_newton - mtom)
                                        if bool(cfg.solver.newton_use_bracketing) and (not bracketed):
                                            step_note = "newton_prebracket"
                                        else:
                                            step_note = "newton"
                                    else:
                                        # Positive slope is unsafe for this residual convention.
                                        mtom_prop = None
                                        step_note = "newton_reject_slope_pos"

                # If bracketed, ensure the proposal stays inside (fall back if needed)
                if bool(cfg.solver.newton_use_bracketing) and bracketed:
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

                    # Guard against bracket collapse / oscillatory residual noise:
                    # repeated bisection with tiny/no MTOM movement can deadlock progress.
                    tiny_step = abs(mtom_next - mtom) < float(cfg.solver.newton_min_step_kg)
                    tight_bracket = (hi - lo) < float(cfg.solver.bracket_min_span_kg)
                    bisect_like = step_note in {"bisection", "bisection(clamped)"}
                    if bisect_like and tiny_step and tight_bracket and abs(residual) > cfg.solver.mtom_tol_kg:
                        bracket_stall_count += 1
                    else:
                        bracket_stall_count = 0

                    if bracket_stall_count >= max(1, int(cfg.solver.bracket_stall_reset_iters)):
                        pos_pt = None
                        neg_pt = None
                        prebracket_iter_count = 0
                        bracket_stall_count = 0
                        mtom_next = _clamp_step(mtom, mtom + float(cfg.solver.newton_fp_relax) * residual)
                        step_note = "bracket_reset_fixed_point"
                else:
                    bracket_stall_count = 0

                logger.info(
                    f"MTOM update: {step_note}, relax={relax:.3f} -> mtom_next={mtom_next:,.2f} kg\n"
                )

            prev_mtom, prev_res = mtom, residual
            mtom = float(mtom_next)

        raise RuntimeError(f"MTOM did not converge within {cfg.solver.max_outer_iter} outer iterations")



# ============================
# ADRpy coupling (constraint analysis)
# ============================


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

    def curve_minima_kw_per_kg(
        self,
        *,
        wingloading_pa: np.ndarray,
        curves: Dict[str, np.ndarray],
        wsmax_cleanstall_pa: Optional[float],
    ) -> Dict[str, Dict[str, float]]:
        """Return per-curve minimum P/W points over finite + stall-feasible W/S."""

        minima: Dict[str, Dict[str, float]] = {}
        for key, raw_curve in curves.items():
            curve = np.asarray(raw_curve, dtype=float)
            valid = np.isfinite(curve)
            if wsmax_cleanstall_pa is not None:
                valid &= wingloading_pa <= float(wsmax_cleanstall_pa)
            idxs = np.where(valid)[0]
            if idxs.size == 0:
                continue
            i_min = self._select_index_min(curve, valid_mask=valid)
            ws_pa = float(wingloading_pa[i_min])
            ws_kgm2 = float((Q_(ws_pa, "pascal") / _G0).to("kilogram / meter ** 2").magnitude)
            minima[str(key)] = {
                "idx": float(i_min),
                "wing_loading_pa": ws_pa,
                "wing_loading_kg_per_m2": ws_kgm2,
                "p_w_kw_per_kg": float(curve[i_min]),
            }
        return minima

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
        ws_kgm2 = float((Q_(ws_pa, "pascal") / _G0).to("kilogram / meter ** 2").magnitude)

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
        cs = cfg.constraint_sizing
        ws_step_pa = float(cs.ws_step_pa)
        ws_scan_min_pa = float(cs.ws_min_pa)
        ws_scan_max_pa = float(cs.ws_max_pa)
        if ws_step_pa <= 0.0:
            raise ValueError("constraint_sizing.ws_step_pa must be > 0")
        if ws_scan_max_pa <= ws_scan_min_pa:
            raise ValueError("constraint_sizing.ws_max_pa must be > ws_min_pa")
        auto_widen_enable = bool(cs.ws_auto_widen_enable)
        auto_widen_factor = float(cs.ws_auto_widen_factor)
        auto_widen_max_passes = max(0, int(cs.ws_auto_widen_max_passes))
        if auto_widen_enable and auto_widen_factor <= 1.0:
            logger.warning(
                "ws_auto_widen_enable=True but ws_auto_widen_factor<=1.0; disabling auto-widen.",
            )
            auto_widen_enable = False

        def _wingloading_grid_from_bounds(*, ws_min_pa: float, ws_max_pa: float) -> np.ndarray:
            if ws_max_pa <= ws_min_pa:
                raise ValueError("constraint_sizing.ws_max_pa must be > ws_min_pa")
            n = int(math.floor((ws_max_pa - ws_min_pa) / ws_step_pa)) + 1
            grid = ws_min_pa + ws_step_pa * np.arange(max(1, n), dtype=float)
            if grid[-1] < ws_max_pa - 1e-9:
                grid = np.append(grid, ws_max_pa)
            return np.asarray(grid, dtype=float)

        seed_expand_pass = 0
        ws_grid = np.asarray([], dtype=float)
        dp: ADRpyDesignPoint
        curves: Dict[str, np.ndarray]
        wsmax_cleanstall_pa: Optional[float] = None
        while True:
            ws_grid = _wingloading_grid_from_bounds(ws_min_pa=ws_scan_min_pa, ws_max_pa=ws_scan_max_pa)
            dp, curves, wsmax_cleanstall_pa = self._adr.select_design_point_min_combined_pw(
                wingloading_pa=ws_grid, mtom_kg=float(cfg.initial_mtom_kg)
            )
            combined = np.asarray(curves.get("combined"), dtype=float)
            valid = np.isfinite(combined)
            if wsmax_cleanstall_pa is not None:
                valid &= ws_grid <= float(wsmax_cleanstall_pa)
            idxs = np.where(valid)[0]
            if idxs.size == 0:
                raise RuntimeError("No feasible W/S points for selection='min_combined_pw'.")

            ws_feasible = np.asarray(ws_grid[idxs], dtype=float)
            ws_feasible_min = float(np.min(ws_feasible))
            ws_feasible_max = float(np.max(ws_feasible))

            ws_tol_pa = max(1e-9, 0.51 * ws_step_pa)
            touches_lower = math.isclose(ws_feasible_min, ws_scan_min_pa, abs_tol=ws_tol_pa)
            touches_upper = math.isclose(ws_feasible_max, ws_scan_max_pa, abs_tol=ws_tol_pa)
            seed_on_boundary = (
                math.isclose(dp.wing_loading_pa, ws_feasible_min, abs_tol=ws_tol_pa)
                or math.isclose(dp.wing_loading_pa, ws_feasible_max, abs_tol=ws_tol_pa)
            )
            should_widen = (
                auto_widen_enable
                and seed_expand_pass < auto_widen_max_passes
                and (touches_lower or touches_upper or seed_on_boundary)
            )
            if not should_widen:
                if seed_on_boundary:
                    logger.warning(
                        "ADRpy minimum combined P/W remains on feasible W/S boundary (%.1f Pa). "
                        "Proceeding with current W/S range %.1f..%.1f Pa.",
                        dp.wing_loading_pa,
                        ws_scan_min_pa,
                        ws_scan_max_pa,
                    )
                break

            old_min_pa = ws_scan_min_pa
            old_max_pa = ws_scan_max_pa
            old_span_pa = old_max_pa - old_min_pa
            new_span_pa = old_span_pa * auto_widen_factor
            ws_center_pa = 0.5 * (old_min_pa + old_max_pa)
            ws_scan_min_pa = max(ws_step_pa, ws_center_pa - 0.5 * new_span_pa)
            ws_scan_max_pa = ws_center_pa + 0.5 * new_span_pa
            logger.info(
                "Auto-widening W/S scan window for min_combined_pw: %.1f..%.1f -> %.1f..%.1f Pa "
                "(factor %.3f, lower_hit=%s, upper_hit=%s, min_on_edge=%s).",
                old_min_pa,
                old_max_pa,
                ws_scan_min_pa,
                ws_scan_max_pa,
                auto_widen_factor,
                str(touches_lower),
                str(touches_upper),
                str(seed_on_boundary),
            )
            seed_expand_pass += 1

        curve_minima = self._adr.curve_minima_kw_per_kg(
            wingloading_pa=ws_grid,
            curves=curves,
            wsmax_cleanstall_pa=wsmax_cleanstall_pa,
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

        for label, curve_key in (
            ("combined", "combined"),
            ("takeoff", str(cs.takeoff_constraint)),
            ("climb", str(cs.climb_constraint)),
            ("cruise", str(cs.cruise_constraint)),
        ):
            m = curve_minima.get(curve_key)
            if m is None:
                continue
            logger.info(
                "Curve minimum (%s/%s): W/S=%.1f Pa (%.2f kg/m^2), P/W=%.5f kW/kg",
                label,
                curve_key,
                float(m["wing_loading_pa"]),
                float(m["wing_loading_kg_per_m2"]),
                float(m["p_w_kw_per_kg"]),
            )

        design = HybridFuelCellAircraftDesign(cfg_used, out_dir=out_dir)
        phases, mass = design.run()

        row: Dict[str, object] = {
            "wing_loading_pa": dp.wing_loading_pa,
            "wing_loading_kg_per_m2": dp.wing_loading_kg_per_m2,
            "p_w_takeoff_kw_per_kg": dp.p_w_takeoff_kw_per_kg,
            "p_w_climb_kw_per_kg": dp.p_w_climb_kw_per_kg,
            "p_w_cruise_kw_per_kg": dp.p_w_cruise_kw_per_kg,
            "mtom_kg": mass.mtom_kg,
        }
        for label, curve_key in (
            ("combined", "combined"),
            ("takeoff", str(cs.takeoff_constraint)),
            ("climb", str(cs.climb_constraint)),
            ("cruise", str(cs.cruise_constraint)),
        ):
            m = curve_minima.get(curve_key)
            if m is None:
                continue
            row[f"min_{label}_curve_key"] = curve_key
            row[f"min_{label}_wing_loading_pa"] = float(m["wing_loading_pa"])
            row[f"min_{label}_wing_loading_kg_per_m2"] = float(m["wing_loading_kg_per_m2"])
            row[f"min_{label}_p_w_kw_per_kg"] = float(m["p_w_kw_per_kg"])

        df = pd.DataFrame([row])

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

        ws_step_pa = float(cs.ws_step_pa)
        if ws_step_pa <= 0.0:
            raise ValueError("constraint_sizing.ws_step_pa must be > 0")
        ws_scan_min_pa = float(cs.ws_min_pa)
        ws_scan_max_pa = float(cs.ws_max_pa)
        if ws_scan_max_pa <= ws_scan_min_pa:
            raise ValueError("constraint_sizing.ws_max_pa must be > ws_min_pa")

        refine_passes = max(0, int(cs.ws_refine_passes))
        refine_span_fraction = float(cs.ws_refine_span_fraction)
        auto_widen_enable = bool(cs.ws_auto_widen_enable)
        auto_widen_factor = float(cs.ws_auto_widen_factor)
        auto_widen_max_passes = max(0, int(cs.ws_auto_widen_max_passes))
        if auto_widen_enable and auto_widen_factor <= 1.0:
            logger.warning(
                "ws_auto_widen_enable=True but ws_auto_widen_factor<=1.0; disabling auto-widen.",
            )
            auto_widen_enable = False

        # For min_mtom, always focus the first MTOM scan near ADRpy minimum combined P/W.
        # If refine span fraction is invalid, keep refinement disabled but still use a sane focus span.
        focus_span_fraction = refine_span_fraction if refine_span_fraction > 0.0 else 0.20
        if refine_passes > 0 and refine_span_fraction <= 0.0:
            logger.warning(
                "ws_refine_passes=%d requested but ws_refine_span_fraction<=0; disabling refinement.",
                refine_passes,
            )
            refine_passes = 0

        # Optional: silence iterative logs during trade scan
        old_level = logger.level
        if cs.scan_quiet:
            logger.setLevel(logging.WARNING)

        BestScanResult = Tuple[float, DesignConfig, Dict[str, PhasePowerResult], MassBreakdown, ADRpyDesignPoint]
        rows: List[Dict[str, object]] = []
        best_global: Optional[BestScanResult] = None
        best_near_seed: Optional[BestScanResult] = None
        successful_samples: List[Tuple[float, float]] = []  # (W/S [Pa], MTOM [kg]) for converged points
        skipped_points = 0
        scanned_points = 0
        feasible_points = 0
        seen_ws_keys: set[float] = set()

        # Seed guesses
        mtom_seed = float(cfg.initial_mtom_kg)
        p_seed = float(cfg.initial_total_power_guess_w)

        def _wingloading_grid_from_bounds(*, ws_min_pa: float, ws_max_pa: float) -> np.ndarray:
            if ws_max_pa <= ws_min_pa:
                raise ValueError("constraint_sizing.ws_max_pa must be > ws_min_pa")
            # Include end point when bounds are not an exact multiple of step.
            n = int(math.floor((ws_max_pa - ws_min_pa) / ws_step_pa)) + 1
            grid = ws_min_pa + ws_step_pa * np.arange(max(1, n), dtype=float)
            if grid[-1] < ws_max_pa - 1e-9:
                grid = np.append(grid, ws_max_pa)
            return np.asarray(grid, dtype=float)

        def _build_refine_grid(
            *,
            ws_center_pa: float,
            ws_half_span_pa: float,
            ws_step_pa: float,
        ) -> np.ndarray:
            ws_min_local = max(float(ws_scan_min_pa), float(ws_center_pa) - float(ws_half_span_pa))
            ws_max_local = min(float(ws_scan_max_pa), float(ws_center_pa) + float(ws_half_span_pa))
            if ws_max_local <= ws_min_local:
                return np.asarray([float(ws_center_pa)], dtype=float)
            n = int(math.floor((ws_max_local - ws_min_local) / float(ws_step_pa))) + 1
            grid = ws_min_local + float(ws_step_pa) * np.arange(max(1, n), dtype=float)
            if grid[-1] < ws_max_local - 1e-9:
                grid = np.append(grid, ws_max_local)
            grid = np.append(grid, float(ws_center_pa))
            grid = np.clip(grid, float(ws_scan_min_pa), float(ws_scan_max_pa))
            return np.asarray(np.unique(np.round(grid, 9)), dtype=float)

        def _center_out_order(
            *,
            idxs: np.ndarray,
            ws_grid: np.ndarray,
            ws_center_pa: Optional[float],
            combined: np.ndarray,
        ) -> np.ndarray:
            """Prioritize candidates nearest to center; tie-break by lower combined P/W."""
            idxs_arr = np.asarray(idxs, dtype=int)
            if idxs_arr.size <= 1 or ws_center_pa is None or not math.isfinite(float(ws_center_pa)):
                return idxs_arr
            order = sorted(
                idxs_arr.tolist(),
                key=lambda i: (
                    abs(float(ws_grid[int(i)]) - float(ws_center_pa)),
                    float(combined[int(i)]),
                    float(ws_grid[int(i)]),
                ),
            )
            return np.asarray(order, dtype=int)

        # ADRpy pre-selection: identify feasible W/S region and center the MTOM search
        # around the point that minimises combined P/W.
        seed_expand_pass = 0
        ws_seed_pa = float("nan")
        pw_seed_kw_per_kg = float("nan")
        ws_seed_feasible = np.asarray([], dtype=float)
        wsmax_cleanstall_pa: Optional[float] = None
        while True:
            ws_grid_user = _wingloading_grid_from_bounds(ws_min_pa=ws_scan_min_pa, ws_max_pa=ws_scan_max_pa)
            curves_seed, wsmax_cleanstall_pa = self._adr.power_to_weight_curves_kw_per_kg(
                wingloading_pa=ws_grid_user, mtom_kg=float(cfg.initial_mtom_kg)
            )
            combined_seed = np.asarray(curves_seed.get("combined"), dtype=float)
            valid_seed = np.isfinite(combined_seed)
            if wsmax_cleanstall_pa is not None:
                valid_seed &= ws_grid_user <= float(wsmax_cleanstall_pa)

            idxs_seed = np.where(valid_seed)[0]
            if idxs_seed.size == 0:
                raise RuntimeError("No feasible W/S points for selection='min_mtom' (check constraints / stall limit).")

            ws_feasible = np.asarray(ws_grid_user[idxs_seed], dtype=float)
            ws_seed_feasible = np.asarray(ws_feasible, dtype=float)
            ws_feasible_min = float(np.min(ws_feasible))
            ws_feasible_max = float(np.max(ws_feasible))
            i_pw_seed = self._adr._select_index_min(combined_seed, valid_mask=valid_seed)
            ws_seed_pa = float(ws_grid_user[i_pw_seed])
            pw_seed_kw_per_kg = float(combined_seed[i_pw_seed])

            logger.info(
                "ADRpy feasible W/S window (seed pass %d): %.1f .. %.1f Pa (%.2f .. %.2f kg/m^2), %d points",
                seed_expand_pass,
                ws_feasible_min,
                ws_feasible_max,
                ws_feasible_min / _G0_MPS2,
                ws_feasible_max / _G0_MPS2,
                int(ws_feasible.size),
            )

            ws_tol_pa = max(1e-9, 0.51 * ws_step_pa)
            touches_lower = math.isclose(ws_feasible_min, ws_scan_min_pa, abs_tol=ws_tol_pa)
            touches_upper = math.isclose(ws_feasible_max, ws_scan_max_pa, abs_tol=ws_tol_pa)
            seed_on_boundary = (
                math.isclose(ws_seed_pa, ws_feasible_min, abs_tol=ws_tol_pa)
                or math.isclose(ws_seed_pa, ws_feasible_max, abs_tol=ws_tol_pa)
            )
            should_widen = (
                auto_widen_enable
                and seed_expand_pass < auto_widen_max_passes
                and (touches_lower or touches_upper or seed_on_boundary)
            )
            if not should_widen:
                if seed_on_boundary:
                    logger.warning(
                        "ADRpy minimum combined P/W seed remains on feasible W/S boundary (%.1f Pa). "
                        "Proceeding with current W/S range %.1f..%.1f Pa.",
                        ws_seed_pa,
                        ws_scan_min_pa,
                        ws_scan_max_pa,
                    )
                break

            old_min_pa = ws_scan_min_pa
            old_max_pa = ws_scan_max_pa
            old_span_pa = old_max_pa - old_min_pa
            new_span_pa = old_span_pa * auto_widen_factor
            ws_center_pa = 0.5 * (old_min_pa + old_max_pa)
            ws_scan_min_pa = max(ws_step_pa, ws_center_pa - 0.5 * new_span_pa)
            ws_scan_max_pa = ws_center_pa + 0.5 * new_span_pa

            logger.info(
                "Auto-widening W/S scan window for seed analysis: %.1f..%.1f -> %.1f..%.1f Pa "
                "(factor %.3f, lower_hit=%s, upper_hit=%s, seed_on_edge=%s).",
                old_min_pa,
                old_max_pa,
                ws_scan_min_pa,
                ws_scan_max_pa,
                auto_widen_factor,
                str(touches_lower),
                str(touches_upper),
                str(seed_on_boundary),
            )
            seed_expand_pass += 1

        ws_full_span_pa = float(ws_scan_max_pa - ws_scan_min_pa)

        ws_seed_half_span_pa = max(float(cs.ws_step_pa), ws_full_span_pa * focus_span_fraction)
        ws_seed_neighborhood_pa = float(ws_seed_half_span_pa)
        ws_grid_focus = _build_refine_grid(
            ws_center_pa=ws_seed_pa,
            ws_half_span_pa=ws_seed_half_span_pa,
            ws_step_pa=float(cs.ws_step_pa),
        )
        logger.info(
            "min_mtom seed from ADRpy min combined P/W: W/S=%.1f Pa, combined P/W=%.5f kW/kg; "
            "initial focus window %.1f..%.1f Pa (%d points)",
            ws_seed_pa,
            pw_seed_kw_per_kg,
            float(np.min(ws_grid_focus)),
            float(np.max(ws_grid_focus)),
            int(ws_grid_focus.size),
        )
        logger.info(
            "min_mtom near-seed selection window: |W/S - W/S_seed| <= %.1f Pa",
            ws_seed_neighborhood_pa,
        )

        def _is_near_seed(ws_pa: float) -> bool:
            return abs(float(ws_pa) - float(ws_seed_pa)) <= float(ws_seed_neighborhood_pa) + 1e-9

        def _candidate_rank(item: BestScanResult) -> Tuple[float, float, float]:
            return (
                float(item[0]),
                abs(float(item[4].wing_loading_pa) - float(ws_seed_pa)),
                float(item[4].wing_loading_pa),
            )

        def _build_nonuniform_seed_grid(
            *,
            ws_candidates_pa: np.ndarray,
            ws_center_pa: float,
            max_points: int,
        ) -> np.ndarray:
            """Sub-sample candidate W/S values with higher density near center."""
            ws_sorted = np.asarray(np.unique(np.round(np.asarray(ws_candidates_pa, dtype=float), 9)), dtype=float)
            if ws_sorted.size == 0:
                return np.asarray([], dtype=float)

            max_points_int = max(1, int(max_points))
            if ws_sorted.size <= max_points_int:
                return ws_sorted

            ws_min = float(ws_sorted[0])
            ws_max = float(ws_sorted[-1])
            center = float(np.clip(float(ws_center_pa), ws_min, ws_max))
            left_span = max(center - ws_min, 1e-9)
            right_span = max(ws_max - center, 1e-9)

            picked: List[float] = []
            picked_keys: set[float] = set()

            def _add_nearest(target_pa: float) -> None:
                idx = int(np.argmin(np.abs(ws_sorted - float(target_pa))))
                ws_val = float(ws_sorted[idx])
                key = round(ws_val, 9)
                if key in picked_keys:
                    return
                picked_keys.add(key)
                picked.append(ws_val)

            # Dense near center; progressively coarser outward.
            fractions = [0.0, -0.06, 0.06, -0.14, 0.14, -0.26, 0.26, -0.40, 0.40, -0.58, 0.58, -0.78, 0.78, -1.0, 1.0]
            for frac in fractions:
                span = left_span if frac < 0.0 else right_span
                _add_nearest(center + float(frac) * span)
                if len(picked) >= max_points_int:
                    break

            # Fill remaining slots with broad quantiles for robustness.
            if len(picked) < max_points_int:
                for q in np.linspace(0.0, 1.0, max_points_int):
                    _add_nearest(float(ws_min + float(q) * (ws_max - ws_min)))
                    if len(picked) >= max_points_int:
                        break

            return np.asarray(np.unique(np.round(np.asarray(picked, dtype=float), 9)), dtype=float)

        def _decide_local_seed_window() -> Tuple[float, float, float]:
            """Choose local-search center/span/step from global-stage converged samples."""
            if not successful_samples:
                half_span = max(float(cs.ws_step_pa), float(ws_seed_neighborhood_pa) * 0.40)
                step = max(float(cs.ws_step_pa) * 0.5, 1.0)
                return float(ws_seed_pa), float(half_span), float(step)

            near_pool = [(ws, mtom) for (ws, mtom) in successful_samples if _is_near_seed(ws)]
            pool = near_pool if near_pool else successful_samples
            pool_sorted = sorted(pool, key=lambda x: (float(x[1]), abs(float(x[0]) - float(ws_seed_pa)), float(x[0])))
            top = pool_sorted[: min(5, len(pool_sorted))]

            m0 = float(top[0][1])
            ws_vals = np.asarray([float(ws) for (ws, _) in top], dtype=float)
            dm = np.asarray([max(1.0, float(m) - m0) for (_, m) in top], dtype=float)
            w = 1.0 / dm
            ws_center = float(np.sum(w * ws_vals) / np.sum(w))
            ws_center = float(
                np.clip(
                    ws_center,
                    float(ws_seed_pa) - float(ws_seed_neighborhood_pa),
                    float(ws_seed_pa) + float(ws_seed_neighborhood_pa),
                )
            )

            spread = float(np.max(ws_vals) - np.min(ws_vals)) if ws_vals.size > 1 else float(cs.ws_step_pa)
            if len(pool_sorted) > 1:
                neighbor_gap = abs(float(pool_sorted[1][0]) - float(pool_sorted[0][0]))
            else:
                neighbor_gap = float(cs.ws_step_pa)

            half_span = max(
                float(cs.ws_step_pa),
                min(
                    float(ws_seed_neighborhood_pa),
                    max(2.0 * float(cs.ws_step_pa), 0.75 * spread + neighbor_gap),
                ),
            )
            step = max(float(cs.ws_step_pa) * 0.5, 1.0)
            return float(ws_center), float(half_span), float(step)

        def _scan_ws_grid(
            *,
            ws_grid: np.ndarray,
            pass_id: int,
            stage: str,
            ws_center_pa: Optional[float] = None,
        ) -> None:
            nonlocal best_global
            nonlocal best_near_seed
            nonlocal mtom_seed
            nonlocal p_seed
            nonlocal skipped_points
            nonlocal scanned_points
            nonlocal feasible_points

            # Compute ADRpy curves for this pass grid (feasibility model).
            curves, wsmax_cleanstall_pa = self._adr.power_to_weight_curves_kw_per_kg(
                wingloading_pa=ws_grid, mtom_kg=float(cfg.initial_mtom_kg)
            )

            combined = np.asarray(curves.get("combined"), dtype=float)
            valid = np.isfinite(combined)
            if wsmax_cleanstall_pa is not None:
                valid &= ws_grid <= float(wsmax_cleanstall_pa)

            idxs = np.where(valid)[0]
            if idxs.size == 0:
                logger.warning(
                    "No feasible W/S points in %s pass %d (grid %.1f..%.1f Pa).",
                    stage,
                    pass_id,
                    float(np.min(ws_grid)),
                    float(np.max(ws_grid)),
                )
                return
            feasible_points += int(idxs.size)

            ws_feasible = np.asarray(ws_grid[idxs], dtype=float)
            idxs_eval = _center_out_order(
                idxs=idxs,
                ws_grid=ws_grid,
                ws_center_pa=ws_center_pa,
                combined=combined,
            )
            logger.info(
                "ADRpy feasible W/S window (%s pass %d): %.1f .. %.1f Pa (%.2f .. %.2f kg/m^2), %d points",
                stage,
                pass_id,
                float(np.min(ws_feasible)),
                float(np.max(ws_feasible)),
                float(np.min(ws_feasible)) / _G0_MPS2,
                float(np.max(ws_feasible)) / _G0_MPS2,
                int(ws_feasible.size),
            )
            if ws_center_pa is not None and math.isfinite(float(ws_center_pa)):
                logger.info(
                    "Scanning order (%s pass %d): center-out around W/S=%.1f Pa",
                    stage,
                    pass_id,
                    float(ws_center_pa),
                )

            for idx in idxs_eval:
                dp = self._adr._design_point_from_curves(wingloading_pa=ws_grid, curves=curves, idx=int(idx))
                ws_key = round(float(dp.wing_loading_pa), 6)
                if ws_key in seen_ws_keys:
                    continue
                seen_ws_keys.add(ws_key)
                scanned_points += 1

                cfg_i = self._apply_design_point_to_cfg(cfg, dp)

                # Run mass-closure for this design point
                try:
                    design = HybridFuelCellAircraftDesign(cfg_i, out_dir=out_dir)
                    phases_i, mass_i = design.run(initial_mtom_kg=mtom_seed, initial_total_power_guess_w=p_seed)
                except Exception as e:
                    skipped_points += 1
                    logger.warning(
                        "Skipping W/S=%.1f Pa (%.2f kg/m^2) in %s pass %d: %s",
                        dp.wing_loading_pa,
                        dp.wing_loading_kg_per_m2,
                        stage,
                        pass_id,
                        str(e),
                    )
                    rows.append(
                        {
                            "scan_stage": stage,
                            "scan_pass": int(pass_id),
                            "wing_loading_pa": dp.wing_loading_pa,
                            "seed_distance_pa": abs(float(dp.wing_loading_pa) - float(ws_seed_pa)),
                            "wing_loading_kg_per_m2": dp.wing_loading_kg_per_m2,
                            "p_w_takeoff_kw_per_kg": dp.p_w_takeoff_kw_per_kg,
                            "p_w_climb_kw_per_kg": dp.p_w_climb_kw_per_kg,
                            "p_w_cruise_kw_per_kg": dp.p_w_cruise_kw_per_kg,
                            "mtom_kg": np.nan,
                            "p_total_climb_kw": np.nan,
                            "p_total_cruise_kw": np.nan,
                            "p_total_takeoff_kw": np.nan,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    continue

                # Warm-start next design-point solve.
                mtom_seed = float(mass_i.mtom_kg)
                p_seed = float(phases_i["climb"].p_total_w)

                rows.append(
                    {
                        "scan_stage": stage,
                        "scan_pass": int(pass_id),
                        "wing_loading_pa": dp.wing_loading_pa,
                        "seed_distance_pa": abs(float(dp.wing_loading_pa) - float(ws_seed_pa)),
                        "wing_loading_kg_per_m2": dp.wing_loading_kg_per_m2,
                        "p_w_takeoff_kw_per_kg": dp.p_w_takeoff_kw_per_kg,
                        "p_w_climb_kw_per_kg": dp.p_w_climb_kw_per_kg,
                        "p_w_cruise_kw_per_kg": dp.p_w_cruise_kw_per_kg,
                        "mtom_kg": float(mass_i.mtom_kg),
                        "p_total_climb_kw": float(phases_i["climb"].p_total_w) / 1000.0,
                        "p_total_cruise_kw": float(phases_i["cruise"].p_total_w) / 1000.0,
                        "p_total_takeoff_kw": float(phases_i["takeoff"].p_total_w) / 1000.0,
                        "status": "ok",
                        "error": "",
                    }
                )
                successful_samples.append((float(dp.wing_loading_pa), float(mass_i.mtom_kg)))
                _log_converged_state(
                    state=f"{stage} pass {pass_id} | W/S={dp.wing_loading_pa:,.1f} Pa ("
                    f"{dp.wing_loading_kg_per_m2:.2f} kg/m^2)",
                    phases=phases_i,
                    mass=mass_i,
                    cfg=cfg_i,
                )

                cand: BestScanResult = (float(mass_i.mtom_kg), cfg_i, phases_i, mass_i, dp)
                if best_global is None or _candidate_rank(cand) < _candidate_rank(best_global):
                    best_global = cand
                if _is_near_seed(float(dp.wing_loading_pa)):
                    if best_near_seed is None or _candidate_rank(cand) < _candidate_rank(best_near_seed):
                        best_near_seed = cand

            # Keep next pass warm-start near the near-seed best whenever available.
            warm_ref = best_near_seed if best_near_seed is not None else best_global
            if warm_ref is not None:
                mtom_seed = float(warm_ref[0])
                p_seed = float(warm_ref[2]["climb"].p_total_w)

        try:
            # Stage 0: evaluate ADRpy min_combined_pw seed explicitly first.
            _scan_ws_grid(
                ws_grid=np.asarray([float(ws_seed_pa)], dtype=float),
                pass_id=0,
                stage="seed",
                ws_center_pa=ws_seed_pa,
            )

            # Stage 1: global non-uniform seeds over the feasible W/S window.
            ws_candidates_global = ws_seed_feasible if ws_seed_feasible.size > 0 else ws_grid_focus
            global_budget = min(15, max(7, int(2.0 * math.sqrt(max(1, int(ws_candidates_global.size))) + 3)))
            ws_grid_global = _build_nonuniform_seed_grid(
                ws_candidates_pa=ws_candidates_global,
                ws_center_pa=float(ws_seed_pa),
                max_points=int(global_budget),
            )
            if ws_grid_global.size == 0:
                ws_grid_global = np.asarray([float(ws_seed_pa)], dtype=float)
            logger.info(
                "Global seed scan: %d non-uniform points over %.1f..%.1f Pa (seed %.1f Pa).",
                int(ws_grid_global.size),
                float(np.min(ws_grid_global)),
                float(np.max(ws_grid_global)),
                float(ws_seed_pa),
            )
            _scan_ws_grid(
                ws_grid=ws_grid_global,
                pass_id=1,
                stage="global",
                ws_center_pa=ws_seed_pa,
            )

            # Stage 2: decision-driven local seeds around promising region.
            ws_local_center_pa, ws_local_half_span_pa, ws_local_step_pa = _decide_local_seed_window()
            ws_grid_local_dense = _build_refine_grid(
                ws_center_pa=float(ws_local_center_pa),
                ws_half_span_pa=float(ws_local_half_span_pa),
                ws_step_pa=float(ws_local_step_pa),
            )
            local_budget = min(11, max(5, int(2.0 * math.sqrt(max(1, int(ws_grid_local_dense.size))) + 1)))
            ws_grid_local = _build_nonuniform_seed_grid(
                ws_candidates_pa=ws_grid_local_dense,
                ws_center_pa=float(ws_local_center_pa),
                max_points=int(local_budget),
            )
            if ws_grid_local.size == 0:
                ws_grid_local = np.asarray([float(ws_local_center_pa)], dtype=float)
            logger.info(
                "Local seed decision: center %.1f Pa, half-span %.1f Pa, step %.3f Pa -> %d non-uniform points.",
                float(ws_local_center_pa),
                float(ws_local_half_span_pa),
                float(ws_local_step_pa),
                int(ws_grid_local.size),
            )
            _scan_ws_grid(
                ws_grid=ws_grid_local,
                pass_id=2,
                stage="local",
                ws_center_pa=float(ws_local_center_pa),
            )

            # Optional refinement passes (capped non-uniform sub-sampling for low compute effort).
            for p in range(refine_passes):
                refine_ref = best_near_seed if best_near_seed is not None else best_global
                if refine_ref is None:
                    break
                best_dp = refine_ref[4]
                ws_half_span_pa = ws_full_span_pa * refine_span_fraction * (0.5 ** p)
                ws_step_refine_pa = float(cs.ws_step_pa) * (0.5 ** (p + 1))
                ws_grid_refine_dense = _build_refine_grid(
                    ws_center_pa=float(best_dp.wing_loading_pa),
                    ws_half_span_pa=float(ws_half_span_pa),
                    ws_step_pa=float(ws_step_refine_pa),
                )
                refine_budget = min(9, max(5, int(2.0 * math.sqrt(max(1, int(ws_grid_refine_dense.size))))))
                ws_grid_refine = _build_nonuniform_seed_grid(
                    ws_candidates_pa=ws_grid_refine_dense,
                    ws_center_pa=float(best_dp.wing_loading_pa),
                    max_points=int(refine_budget),
                )
                logger.info(
                    "Refine pass %d/%d: center %.1f Pa, half-span %.1f Pa, step %.3f Pa -> %d/%d points",
                    p + 1,
                    refine_passes,
                    float(best_dp.wing_loading_pa),
                    float(ws_half_span_pa),
                    float(ws_step_refine_pa),
                    int(ws_grid_refine.size),
                    int(ws_grid_refine_dense.size),
                )
                _scan_ws_grid(
                    ws_grid=ws_grid_refine,
                    pass_id=p + 3,
                    stage="refine",
                    ws_center_pa=float(best_dp.wing_loading_pa),
                )

        finally:
            if cs.scan_quiet:
                logger.setLevel(old_level)

        if best_global is None:
            if feasible_points == 0:
                raise RuntimeError(
                    "No feasible W/S points for selection='min_mtom' "
                    "(check ADRpy constraints and stall limit settings)."
                )
            raise RuntimeError(
                "No feasible designs found during selection='min_mtom' scan "
                f"(all {scanned_points} scanned points failed in mass-closure)."
            )

        selected_best = best_near_seed if best_near_seed is not None else best_global
        assert selected_best is not None
        best_mtom, best_cfg, best_phases, best_mass, best_dp = selected_best
        seed_delta_pa = abs(float(best_dp.wing_loading_pa) - float(ws_seed_pa))
        if best_near_seed is None:
            logger.warning(
                "No converged near-seed candidate found within %.1f Pa of ADRpy min combined P/W seed "
                "(W/S_seed=%.1f Pa). Falling back to global MTOM minimum at W/S=%.1f Pa (distance %.1f Pa).",
                ws_seed_neighborhood_pa,
                ws_seed_pa,
                best_dp.wing_loading_pa,
                seed_delta_pa,
            )
        elif best_global is not None and best_global is not best_near_seed:
            global_dp = best_global[4]
            logger.info(
                "Using near-seed MTOM minimum at W/S=%.1f Pa (distance %.1f Pa from seed %.1f Pa). "
                "Global MTOM minimum was at W/S=%.1f Pa.",
                best_dp.wing_loading_pa,
                seed_delta_pa,
                ws_seed_pa,
                global_dp.wing_loading_pa,
            )

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
        if skipped_points > 0:
            logger.warning(
                "Constraint scan skipped %d/%d W/S points due to mass-closure failures; selected best from remaining points.",
                skipped_points,
                scanned_points,
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
            fc = FuelCellSystemModel(
                self._cfg.fuel_cell_arch,
                comp_stl_path=out_dir / "media" / "comp.stl",
                comp_mass_mode=self._cfg.solver.compressor_mass_mode,
            )
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
            self._write_plotly_image_subprocess(fig=fig, save_path=save_path, timeout_s=timeout_s)
        except Exception as e:
            logger.warning("Could not generate PEMFC figure: %s", e)

    @staticmethod
    def _write_plotly_image_subprocess(*, fig: object, save_path: Path, timeout_s: float) -> None:
        """Write Plotly image in a subprocess to isolate hard crashes (e.g., Qt/Kaleido aborts)."""

        fig_json = fig.to_json()
        with tempfile.TemporaryDirectory(prefix="hfcad-kaleido-") as td:
            json_path = Path(td) / "figure.json"
            json_path.write_text(fig_json, encoding="utf-8")

            code = (
                "import json, sys\n"
                "import plotly.graph_objects as go\n"
                "with open(sys.argv[1], 'r', encoding='utf-8') as f:\n"
                "    fig_data = json.load(f)\n"
                "fig = go.Figure(fig_data)\n"
                "fig.write_image(sys.argv[2])\n"
            )

            env = os.environ.copy()
            # Headless-safe default for Qt-backed image exporters.
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
            proc = subprocess.run(
                [sys.executable, "-c", code, str(json_path), str(save_path)],
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
                env=env,
            )

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                raise RuntimeError(
                    f"subprocess image export failed (exit={proc.returncode})"
                    + (f": {stderr}" if stderr else "")
                )

    def write_mission_profile_outputs(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
        show_plot: bool = False,
    ) -> None:
        """Write mission power plot/export using the phase-based mission timeline."""

        if show_plot:
            import matplotlib.pyplot as plt
        else:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

        cfg = self._cfg
        brief = cfg.constraint_brief

        # Mission timeline (independent of cfg.mission.times_min):
        # ready(0), taxi(5), takeoff(1), climb(variable), cruise(variable), loiter(15), landing(1).
        climb_rate = Q_(float(brief.climbrate_fpm), "foot / minute").to("meter / second")
        if float(climb_rate.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.climbrate_fpm must be > 0, got {brief.climbrate_fpm}.")

        v_climb = Q_(float(brief.climbspeed_kias), "knot").to("meter / second")
        v_cruise = Q_(float(brief.cruisespeed_ktas), "knot").to("meter / second")
        loiter_speed_ratio = _resolve_loiter_speed_ratio(cfg.flight.loiter_speed_ratio_to_cruise)
        v_loiter = loiter_speed_ratio * v_cruise
        if float(v_climb.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.climbspeed_kias must be > 0, got {brief.climbspeed_kias}.")
        if float(v_cruise.to("meter / second").magnitude) <= 0.0:
            raise ValueError(f"constraint_brief.cruisespeed_ktas must be > 0, got {brief.cruisespeed_ktas}.")

        # Use runway elevation as mission climb start altitude; climbalt_m is the constraint evaluation altitude.
        h_delta = Q_(max(float(brief.cruisealt_m) - float(brief.rwyelevation_m), 0.0), "meter")
        t_climb = (h_delta / climb_rate).to("second")
        r_climb = (v_climb * t_climb).to("meter")

        t_ready = Q_(0.0, "minute").to("second")
        t_taxi = Q_(5.0, "minute").to("second")
        t_takeoff = Q_(1.0, "minute").to("second")
        t_loiter = Q_(15.0, "minute").to("second")
        t_landing = Q_(1.0, "minute").to("second")

        r_loiter = (v_loiter * t_loiter).to("meter")
        range_total = Q_(float(cfg.hydrogen.range_total_m), "meter")
        r_cruise = (range_total - r_climb - r_loiter).to("meter")
        if float(r_cruise.to("meter").magnitude) < 0.0:
            logger.warning(
                "Total range %.1f km is smaller than climb+loiter range %.1f km. "
                "Setting cruise segment to zero in mission profile output.",
                float(range_total.to("kilometer").magnitude),
                float((r_climb + r_loiter).to("kilometer").magnitude),
            )
            r_cruise = Q_(0.0, "meter")
        t_cruise = (r_cruise / v_cruise).to("second")

        pfc_ready = float(mass.p_fuelcell_engine_w)
        pfc_taxi = float(mass.p_fuelcell_taxing_w)
        pfc_takeoff = float(phases["takeoff"].p_total_w - phases["takeoff"].p_battery_w)
        pbat_takeoff = float(phases["takeoff"].p_battery_w)
        pfc_climb = float(phases["climb"].p_total_w - phases["climb"].p_battery_w)
        pbat_climb = float(phases["climb"].p_battery_w)
        pfc_cruise = float(phases["cruise"].p_total_w - phases["cruise"].p_battery_w)
        pbat_cruise = float(phases["cruise"].p_battery_w)
        pfc_loiter = float(phases["loiter"].p_total_w - phases["loiter"].p_battery_w)
        pbat_loiter = float(phases["loiter"].p_battery_w)
        pfc_landing = 0.0

        phase_names = ["ready", "taxi", "takeoff", "climb", "cruise", "loiter", "landing"]
        phase_durations_min = np.array(
            [
                float(t_ready.to("minute").magnitude),
                float(t_taxi.to("minute").magnitude),
                float(t_takeoff.to("minute").magnitude),
                float(t_climb.to("minute").magnitude),
                float(t_cruise.to("minute").magnitude),
                float(t_loiter.to("minute").magnitude),
                float(t_landing.to("minute").magnitude),
            ],
            dtype=float,
        )
        power_fc_w = np.array([pfc_ready, pfc_taxi, pfc_takeoff, pfc_climb, pfc_cruise, pfc_loiter, pfc_landing], dtype=float)
        power_bat_w = np.array([0.0, 0.0, pbat_takeoff, pbat_climb, pbat_cruise, pbat_loiter, 0.0], dtype=float)

        t_edges_min = np.concatenate(([0.0], np.cumsum(phase_durations_min)))
        y_fc_kw = power_fc_w / 1000.0
        y_bat_kw = power_bat_w / 1000.0
        y_total_kw = y_fc_kw + y_bat_kw
        y_fc_step_kw = np.append(y_fc_kw, y_fc_kw[-1])
        y_bat_step_kw = np.append(y_bat_kw, y_bat_kw[-1])
        y_total_step_kw = np.append(y_total_kw, y_total_kw[-1])

        # Plot
        plt.step(t_edges_min, y_bat_step_kw, where="post", linestyle="solid", color="gray", label="Battery")
        plt.step(t_edges_min, y_fc_step_kw, where="post", linestyle="dashed", color="orange", label="Fuel Cell")
        plt.step(t_edges_min, y_total_step_kw, where="post", linestyle="solid", color="blue", label="Total")

        plt.xlabel("Time(min)")
        plt.ylabel("Power(kW)")
        phase_midpoints_min = 0.5 * (t_edges_min[:-1] + t_edges_min[1:])
        phase_labels = ["Ready", "Taxing", "Takeoff", "Climb", "Loiter", "Landing"]
        phase_tick_positions = np.array(
            [
                phase_midpoints_min[0],
                phase_midpoints_min[1],
                phase_midpoints_min[2],
                phase_midpoints_min[3],
                phase_midpoints_min[5],
                phase_midpoints_min[6],
            ],
            dtype=float,
        )
        plt.xticks(phase_tick_positions, phase_labels, rotation=0, ha="center")
        x_max = max(float(t_edges_min[-1]), 1.0)
        y_max = max(10.0, float(np.max(np.maximum(y_total_step_kw, 0.0))) * 1.15)
        y_min = min(-200.0, float(np.min(y_total_step_kw)) * 1.15)
        plt.axis([0.0, x_max, y_min, y_max])
        plt.title("Mission Profile (Required Power vs Time)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)

        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_dir / "Power Mission Profile.png"), dpi=400)
        if show_plot:
            plt.show()
        plt.close()

        # Excel export
        # Excel export aligned with the new phase timeline.
        df = pd.DataFrame(
            {
                "Phase": phase_names,
                "Start_min": t_edges_min[:-1],
                "End_min": t_edges_min[1:],
                "Duration_min": phase_durations_min,
                "ReqPow_AC_kW": y_total_kw,
                "ReqPow_FC_kW": y_fc_kw,
                "ReqPow_Batt_kW": y_bat_kw,
            }
        )
        df.to_excel(str(out_dir / "ReqPowDATA.xlsx"), index=False)

    def write_converged_text(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
        execution_time_s: Optional[float] = None,
    ) -> None:
        """Write converged summary text to the output directory."""

        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "ConvergedData.txt"
        summary_text = _converged_summary_text(
            phases=phases,
            mass=mass,
            cfg=self._cfg,
            execution_time_s=execution_time_s,
        )
        summary_path.write_text(summary_text + "\n", encoding="utf-8")

    @staticmethod
    def _to_jsonable(value: object) -> object:
        if is_dataclass(value):
            return OutputWriter._to_jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return [OutputWriter._to_jsonable(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): OutputWriter._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [OutputWriter._to_jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _flatten_for_table(cls, value: object, prefix: str, out: Dict[str, object]) -> None:
        normalized = cls._to_jsonable(value)
        if isinstance(normalized, dict):
            for key, sub_value in normalized.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                cls._flatten_for_table(sub_value, child, out)
            return
        if isinstance(normalized, list):
            if not normalized:
                if prefix:
                    out[prefix] = ""
                return
            for i, sub_value in enumerate(normalized):
                child = f"{prefix}.{i}" if prefix else str(i)
                cls._flatten_for_table(sub_value, child, out)
            return
        if prefix:
            out[prefix] = normalized

    def write_case_variable_exports(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
        input_path: Optional[Path] = None,
        initial_cfg: Optional[DesignConfig] = None,
        execution_time_s: Optional[float] = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        phase_payload = {
            name: (asdict(res) if is_dataclass(res) else self._to_jsonable(res))
            for name, res in phases.items()
        }
        mass_payload = asdict(mass) if is_dataclass(mass) else self._to_jsonable(mass)

        payload = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_path": str(input_path) if input_path is not None else "",
                "output_path": str(out_dir),
                "execution_time_s": float(execution_time_s) if execution_time_s is not None else None,
            },
            "config_used": asdict(self._cfg),
            "config_initial": asdict(initial_cfg) if initial_cfg is not None else None,
            "mass": mass_payload,
            "phases": phase_payload,
        }

        json_payload = self._to_jsonable(payload)
        json_path = out_dir / "all_case_variables.json"
        json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        flat_row: Dict[str, object] = {}
        self._flatten_for_table(json_payload, "", flat_row)
        flat_csv_path = out_dir / "all_case_variables_flat.csv"
        pd.DataFrame([flat_row]).to_csv(flat_csv_path, index=False)

    @staticmethod
    def _nb_source_lines(text: str) -> List[str]:
        if not text.endswith("\n"):
            text += "\n"
        return text.splitlines(keepends=True)

    def write_constraint_diagram_notebook(
        self,
        *,
        phases: Dict[str, PhasePowerResult],
        mass: MassBreakdown,
        out_dir: Path,
        input_path: Optional[Path] = None,
        initial_cfg: Optional[DesignConfig] = None,
    ) -> None:
        """Write a per-case ADRpy constraint notebook when constraint sizing is enabled."""

        cfg = self._cfg
        if not cfg.constraint_sizing.enable:
            return

        b = cfg.constraint_brief
        p = cfg.constraint_performance
        g = cfg.constraint_geometry
        cs = cfg.constraint_sizing

        cfg_initial = initial_cfg or cfg
        tow_kg = float(mass.mtom_kg) if float(mass.mtom_kg) > 0.0 else float(cfg.initial_mtom_kg)
        ws_min_pa = float(cs.ws_min_pa)
        ws_max_pa = float(cs.ws_max_pa)
        ws_step_pa = float(cs.ws_step_pa) if float(cs.ws_step_pa) > 0.0 else 25.0
        takeoff_curve_key = str(cs.takeoff_constraint)
        climb_curve_key = str(cs.climb_constraint)
        cruise_curve_key = str(cs.cruise_constraint)
        ws_initial_pa = float(cfg_initial.wing.wing_loading_kg_per_m2) * _G0_MPS2
        if float(mass.wing_area_m2) > 0.0:
            ws_converged_pa = float(mass.mtom_kg / mass.wing_area_m2) * _G0_MPS2
        else:
            ws_converged_pa = float(cfg.wing.wing_loading_kg_per_m2) * _G0_MPS2
        p_w_initial_climb_kw_per_kg = float(cfg_initial.p_w.p_w_climb_kw_per_kg)
        if "climb" in phases and float(mass.mtom_kg) > 0.0:
            p_w_converged_climb_kw_per_kg = float(phases["climb"].p_total_w / mass.mtom_kg / 1000.0)
        else:
            p_w_converged_climb_kw_per_kg = float(cfg.p_w.p_w_climb_kw_per_kg)

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
            "weight_n": "co.kg2n(TOW_kg)",
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

        if input_path is not None:
            case_name = input_path.name
        else:
            case_name = "input file"

        dsdef_text = pformat(designdefinition, sort_dicts=False).replace("'co.kg2n(TOW_kg)'", "co.kg2n(TOW_kg)")
        code_setup = (
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from ADRpy import unitconversions as co\n"
            "from ADRpy import constraintanalysis as ca\n"
            "from ADRpy import atmospheres as at\n\n"
            f"designbrief = {pformat(designbrief, sort_dicts=False)}\n"
            "designatm = at.Atmosphere()\n"
            f"TOW_kg = {tow_kg:.3f}\n\n"
            f"designdefinition = {dsdef_text}\n\n"
            f"designperformance = {pformat(designperformance, sort_dicts=False)}\n\n"
            f"designpropulsion = {repr(str(cs.propulsion_type))}\n\n"
            "concept = ca.AircraftConcept(designbrief, designdefinition, designperformance, designatm, designpropulsion)\n"
        )
        code_ws_grid = (
            f"ws_min_pa = {ws_min_pa:.3f}\n"
            f"ws_max_pa = {ws_max_pa:.3f}\n"
            f"ws_step_pa = {ws_step_pa:.3f}\n\n"
            "ws_step_plot_pa = max(ws_step_pa / 10.0, 1.0)\n"
            "wslist_pa = np.arange(ws_min_pa, ws_max_pa + 0.5 * ws_step_plot_pa, ws_step_plot_pa)\n"
            "if wslist_pa.size < 2:\n"
            "    wslist_pa = np.array([ws_min_pa, ws_max_pa], dtype=float)\n\n"
            "preq_quick = concept.powerrequired(wslist_pa, tow_kg=TOW_kg, feasibleonly=False, map2sl=True)\n"
            "twreq_quick = concept.twrequired(wslist_pa, feasibleonly=False, map2sl=True)\n"
            "preq_quick_kw_combined = np.asarray(co.hp2kw(preq_quick['combined']), dtype=float)\n"
            "y_lim_power_kw = float(np.nanmax(preq_quick_kw_combined) * 2.0)\n"
            "y_lim_tw = float(np.nanmax(twreq_quick['combined']) * 2.0)\n\n"
            f"ws_initial_pa = {ws_initial_pa:.6f}\n"
            f"ws_converged_pa = {ws_converged_pa:.6f}\n"
            f"p_w_initial_climb_kw_per_kg = {p_w_initial_climb_kw_per_kg:.8f}\n"
            f"p_w_converged_climb_kw_per_kg = {p_w_converged_climb_kw_per_kg:.8f}\n\n"
            "g0_mps2 = 9.80665\n"
            "wslist_kg_per_m2 = wslist_pa / g0_mps2\n"
            "ws_initial_kg_per_m2 = ws_initial_pa / g0_mps2\n"
            "ws_converged_kg_per_m2 = ws_converged_pa / g0_mps2\n\n"
            "pwreq_quick_kw_per_kg = preq_quick_kw_combined / float(TOW_kg)\n"
            "y_lim_pw_kw_per_kg = float(np.nanmax(pwreq_quick_kw_per_kg) * 2.0)\n"
            "twreq_quick_combined = np.asarray(twreq_quick['combined'], dtype=float)\n\n"
            "pw_comb_initial_kw_per_kg = float(np.interp(ws_initial_pa, wslist_pa, pwreq_quick_kw_per_kg))\n"
            "pw_comb_converged_kw_per_kg = float(np.interp(ws_converged_pa, wslist_pa, pwreq_quick_kw_per_kg))\n"
            "tw_comb_initial = float(np.interp(ws_initial_pa, wslist_pa, twreq_quick_combined))\n"
            "tw_comb_converged = float(np.interp(ws_converged_pa, wslist_pa, twreq_quick_combined))\n\n"
            "try:\n"
            "    wsmax_cleanstall_pa = float(concept.wsmaxcleanstall_pa())\n"
            "except Exception:\n"
            "    wsmax_cleanstall_pa = np.nan\n"
            "valid_stall = np.isfinite(wslist_pa)\n"
            "if np.isfinite(wsmax_cleanstall_pa):\n"
            "    valid_stall &= wslist_pa <= wsmax_cleanstall_pa\n\n"
            "combined_front_kw_per_kg = np.asarray(co.hp2kw(preq_quick['combined']), dtype=float) / float(TOW_kg)\n"
            "valid_combined_front = np.isfinite(combined_front_kw_per_kg) & valid_stall\n"
            "idxs_combined_front = np.where(valid_combined_front)[0]\n"
            "print('--- Minimum ADRpy P/W on combined front line ---')\n"
            "if idxs_combined_front.size > 0:\n"
            "    i_min_combined_front = idxs_combined_front[int(np.nanargmin(combined_front_kw_per_kg[idxs_combined_front]))]\n"
            "    ws_min_combined_front_pa = float(wslist_pa[i_min_combined_front])\n"
            "    pw_min_combined_front_kw_per_kg = float(combined_front_kw_per_kg[i_min_combined_front])\n"
            "    print(\n"
            "        f\"combined front: W/S={ws_min_combined_front_pa:.1f} Pa \"\n"
            "        f\"({ws_min_combined_front_pa / g0_mps2:.2f} kg/m^2), \"\n"
            "        f\"P/W={pw_min_combined_front_kw_per_kg:.5f} kW/kg\"\n"
            "    )\n"
            "else:\n"
            "    print('combined front: no finite/stall-feasible points in scan window')\n\n"
            "print('--- Marked design points (INI vs converged) ---')\n"
            "print(f\"INI input:      W/S={ws_initial_pa:.1f} Pa | P/W(climb,input)={p_w_initial_climb_kw_per_kg:.5f} kW/kg | \"\n"
            "      f\"P/W(combined@W/S)={pw_comb_initial_kw_per_kg:.5f} kW/kg | T/W(combined@W/S)={tw_comb_initial:.5f}\")\n"
            "print(f\"Converged run:  W/S={ws_converged_pa:.1f} Pa | P/W(climb,conv)={p_w_converged_climb_kw_per_kg:.5f} kW/kg | \"\n"
            "      f\"P/W(combined@W/S)={pw_comb_converged_kw_per_kg:.5f} kW/kg | T/W(combined@W/S)={tw_comb_converged:.5f}\")\n\n"
        )
        code_quick_power = (
            "from matplotlib.ticker import FuncFormatter\n\n"
            "y_lim_p_hp = float(co.kw2hp(y_lim_pw_kw_per_kg * TOW_kg))\n"
            "_show_orig = plt.show\n"
            "_close_orig = plt.close\n"
            "try:\n"
            "    # Keep ADRpy's original combined-plot styling figure alive for post-formatting.\n"
            "    plt.show = lambda *args, **kwargs: None\n"
            "    plt.close = lambda *args, **kwargs: None\n"
            "    _ = concept.propulsionsensitivity_monothetic(\n"
            "        wingloading_pa=wslist_pa,\n"
            "        show='combined',\n"
            "        y_var='p_hp',\n"
            "        x_var='ws_pa',\n"
            "        y_lim=y_lim_p_hp,\n"
            "    )\n"
            "finally:\n"
            "    plt.show = _show_orig\n"
            "    plt.close = _close_orig\n"
            "ax = plt.gca()\n"
            "p_hp_initial_marker = float(co.kw2hp(pw_comb_initial_kw_per_kg * TOW_kg))\n"
            "p_hp_converged_marker = float(co.kw2hp(pw_comb_converged_kw_per_kg * TOW_kg))\n"
            "ax.scatter([ws_initial_pa], [p_hp_initial_marker], color='tab:orange', marker='o', s=40, label='INI point')\n"
            "ax.scatter([ws_converged_pa], [p_hp_converged_marker], color='tab:red', marker='X', s=65, label='Converged point')\n"
            "ax.annotate('INI', (ws_initial_pa, p_hp_initial_marker), textcoords='offset points', xytext=(6, 8), fontsize=9)\n"
            "ax.annotate('Converged', (ws_converged_pa, p_hp_converged_marker), textcoords='offset points', xytext=(6, -12), fontsize=9)\n"
            "ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f\"{x / g0_mps2:.0f}\"))\n"
            "ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f\"{co.hp2kw(y) / TOW_kg:.3f}\"))\n"
            "ax.set_xlabel('Wing loading [kg/m$^2$]')\n"
            "ax.set_ylabel('Power-to-weight [kW/kg]')\n"
            "ax.set_title('Constraint Diagram')\n"
            "ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)\n"
            # "plt.tight_layout()\n"
            "plt.show()\n"
        )
        code_quick_tw = (
            "_ = concept.propulsionsensitivity_monothetic(\n"
            "    wingloading_pa=wslist_pa,\n"
            "    show='combined',\n"
            "    y_var='tw',\n"
            "    x_var='ws_pa',\n"
            "    y_lim=y_lim_tw,\n"
            ")\n"
        )
        code_flexible = (
            "preq = concept.powerrequired(wslist_pa, tow_kg=TOW_kg, feasibleonly=False, map2sl=True)\n"
            "preq_kw = {k: np.asarray(co.hp2kw(v), dtype=float) for k, v in preq.items()}\n"
            "Smin_m2 = concept.smincleanstall_m2(TOW_kg)\n"
            "wingarea_m2 = co.kg2n(TOW_kg) / wslist_pa\n\n"
            "s_initial_m2 = float(co.kg2n(TOW_kg) / ws_initial_pa)\n"
            "s_converged_m2 = float(co.kg2n(TOW_kg) / ws_converged_pa)\n"
            "p_kw_initial = float(np.interp(ws_initial_pa, wslist_pa, preq_kw['combined']))\n"
            "p_kw_converged = float(np.interp(ws_converged_pa, wslist_pa, preq_kw['combined']))\n\n"
            "plt.rcParams['figure.figsize'] = [8, 6]\n"
            "plt.rcParams['figure.dpi'] = 160\n"
            "plt.plot(wingarea_m2, preq_kw['take-off'], label='Take-off')\n"
            "plt.plot(wingarea_m2, preq_kw['turn'], label='Turn')\n"
            "plt.plot(wingarea_m2, preq_kw['climb'], label='Climb')\n"
            "plt.plot(wingarea_m2, preq_kw['cruise'], label='Cruise')\n"
            "plt.plot(wingarea_m2, preq_kw['servceil'], label='Service ceiling')\n"
            "combplot = plt.plot(wingarea_m2, preq_kw['combined'], label='Combined')\n"
            "plt.setp(combplot, linewidth=4)\n"
            "y_max = float(np.nanmax(preq_kw['combined']) * 2.2)\n"
            "stall_label = f\"Clean stall at {designbrief['vstallclean_kcas']} KCAS\"\n"
            "plt.plot([Smin_m2, Smin_m2], [0, y_max], label=stall_label)\n"
            "plt.scatter([s_initial_m2], [p_kw_initial], color='tab:orange', marker='o', s=40, label='INI point')\n"
            "plt.scatter([s_converged_m2], [p_kw_converged], color='tab:red', marker='X', s=65, label='Converged point')\n"
            "plt.annotate('INI', (s_initial_m2, p_kw_initial), textcoords='offset points', xytext=(6, 8), fontsize=9)\n"
            "plt.annotate('Converged', (s_converged_m2, p_kw_converged), textcoords='offset points', xytext=(6, -12), fontsize=9)\n"
            "plt.legend(loc='upper left')\n"
            "plt.ylabel('Power required (kW)')\n"
            "plt.xlabel('S (m$^2$)')\n"
            "plt.title(f\"Minimum SL power required (MTOW = {round(TOW_kg)} kg)\")\n"
            "plt.xlim(min(wingarea_m2), max(wingarea_m2))\n"
            "plt.ylim(0, y_max)\n"
            "plt.grid(True)\n"
            "plt.tight_layout()\n"
            "plt.savefig('Constraint_Diagram.png', dpi=200)\n"
            "plt.show()\n"
        )
        code_marked_pw_tw = (
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)\n\n"
            "ax[0].plot(wslist_kg_per_m2, pwreq_quick_kw_per_kg, color='black', linewidth=2.0, label='Combined P/W')\n"
            "ax[0].scatter([ws_initial_kg_per_m2], [pw_comb_initial_kw_per_kg], color='tab:orange', marker='o', s=40, label='INI point')\n"
            "ax[0].scatter([ws_converged_kg_per_m2], [pw_comb_converged_kw_per_kg], color='tab:red', marker='X', s=65, label='Converged point')\n"
            "ax[0].annotate(f\"INI: {pw_comb_initial_kw_per_kg:.4f}\", (ws_initial_kg_per_m2, pw_comb_initial_kw_per_kg),\n"
            "               textcoords='offset points', xytext=(8, 8), fontsize=8)\n"
            "ax[0].annotate(f\"Conv: {pw_comb_converged_kw_per_kg:.4f}\", (ws_converged_kg_per_m2, pw_comb_converged_kw_per_kg),\n"
            "               textcoords='offset points', xytext=(8, -12), fontsize=8)\n"
            "ax[0].set_xlim(float(np.min(wslist_kg_per_m2)), float(np.max(wslist_kg_per_m2)))\n"
            "ax[0].set_ylim(0.0, float(np.nanmax(pwreq_quick_kw_per_kg) * 2.0))\n"
            "ax[0].set_xlabel('Wing loading [kg/m$^2$]')\n"
            "ax[0].set_ylabel('Power-to-weight [kW/kg]')\n"
            "ax[0].set_title('P/W vs W/S with INI and Converged points')\n"
            "ax[0].grid(True)\n"
            "ax[0].legend(loc='upper left', fontsize=8)\n\n"
            "ax[1].plot(wslist_kg_per_m2, twreq_quick_combined, color='black', linewidth=2.0, label='Combined T/W')\n"
            "ax[1].scatter([ws_initial_kg_per_m2], [tw_comb_initial], color='tab:orange', marker='o', s=40, label='INI point')\n"
            "ax[1].scatter([ws_converged_kg_per_m2], [tw_comb_converged], color='tab:red', marker='X', s=65, label='Converged point')\n"
            "ax[1].annotate(f\"INI: {tw_comb_initial:.4f}\", (ws_initial_kg_per_m2, tw_comb_initial),\n"
            "               textcoords='offset points', xytext=(8, 8), fontsize=8)\n"
            "ax[1].annotate(f\"Conv: {tw_comb_converged:.4f}\", (ws_converged_kg_per_m2, tw_comb_converged),\n"
            "               textcoords='offset points', xytext=(8, -12), fontsize=8)\n"
            "ax[1].set_xlim(float(np.min(wslist_kg_per_m2)), float(np.max(wslist_kg_per_m2)))\n"
            "ax[1].set_ylim(0.0, float(np.nanmax(twreq_quick_combined) * 2.0))\n"
            "ax[1].set_xlabel('Wing loading [kg/m$^2$]')\n"
            "ax[1].set_ylabel('Thrust-to-weight ratio [-]')\n"
            "ax[1].set_title('Combined T/W with INI and Converged points')\n"
            "ax[1].grid(True)\n"
            "ax[1].legend(loc='upper left', fontsize=8)\n\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
        code_sensitivity = (
            "def _span(v, frac=0.05):\n"
            "    v = float(v)\n"
            "    if v <= 0.0:\n"
            "        return v\n"
            "    return [v * (1.0 - frac), v * (1.0 + frac)]\n\n"
            "designbrief_sens = {\n"
            "    'rwyelevation_m': float(designbrief['rwyelevation_m']),\n"
            "    'groundrun_m': _span(designbrief['groundrun_m']),\n"
            "    'stloadfactor': _span(designbrief['stloadfactor']),\n"
            "    'turnalt_m': _span(designbrief['turnalt_m']),\n"
            "    'turnspeed_ktas': _span(designbrief['turnspeed_ktas']),\n"
            "    'climbalt_m': float(designbrief['climbalt_m']),\n"
            "    'climbspeed_kias': _span(designbrief['climbspeed_kias']),\n"
            "    'climbrate_fpm': _span(designbrief['climbrate_fpm']),\n"
            "    'cruisealt_m': _span(designbrief['cruisealt_m']),\n"
            "    'cruisespeed_ktas': _span(designbrief['cruisespeed_ktas']),\n"
            "    'cruisethrustfact': _span(designbrief['cruisethrustfact']),\n"
            "    'servceil_m': _span(designbrief['servceil_m']),\n"
            "    'secclimbspd_kias': _span(designbrief['secclimbspd_kias']),\n"
            "    'vstallclean_kcas': _span(designbrief['vstallclean_kcas']),\n"
            "}\n\n"
            "designdefinition_sens = {\n"
            "    'aspectratio': _span(designdefinition['aspectratio']),\n"
            "    'sweep_le_deg': designdefinition['sweep_le_deg'],\n"
            "    'sweep_mt_deg': designdefinition['sweep_mt_deg'],\n"
            "    'weightfractions': designdefinition['weightfractions'],\n"
            "    'weight_n': co.kg2n(TOW_kg),\n"
            "}\n\n"
            "designperformance_sens = {\n"
            "    'CDTO': _span(designperformance['CDTO']),\n"
            "    'CLTO': _span(designperformance['CLTO']),\n"
            "    'CLmaxTO': _span(designperformance['CLmaxTO']),\n"
            "    'CLmaxclean': _span(designperformance['CLmaxclean']),\n"
            "    'mu_R': designperformance['mu_R'],\n"
            "    'CDminclean': _span(designperformance['CDminclean']),\n"
            "    'etaprop': {\n"
            "        # Keep etaprop scalar for ADRpy compatibility in electric cases.\n"
            "        'take-off': designperformance['etaprop']['take-off'],\n"
            "        'climb': designperformance['etaprop']['climb'],\n"
            "        'cruise': designperformance['etaprop']['cruise'],\n"
            "        'turn': designperformance['etaprop']['turn'],\n"
            "        'servceil': designperformance['etaprop']['servceil'],\n"
            "    },\n"
            "}\n\n"
            "concept_sens = ca.AircraftConcept(\n"
            "    designbrief_sens,\n"
            "    designdefinition_sens,\n"
            "    designperformance_sens,\n"
            "    designatm,\n"
            "    designpropulsion,\n"
            ")\n\n"
            "from matplotlib.ticker import FuncFormatter\n\n"
            "y_lim_sens_p_hp = float(co.kw2hp(y_lim_power_kw))\n\n"
            "def _run_sensitivity_plot_with_kw_labels(show_arg, **kwargs):\n"
            "    _show_orig = plt.show\n"
            "    _close_orig = plt.close\n"
            "    try:\n"
            "        # Preserve ADRpy styling while allowing post-format axis relabeling.\n"
            "        plt.show = lambda *args, **kw: None\n"
            "        plt.close = lambda *args, **kw: None\n"
            "        concept_sens.propulsionsensitivity_monothetic(\n"
            "            wingloading_pa=wslist_pa,\n"
            "            show=show_arg,\n"
            "            y_var='p_hp',\n"
            "            x_var='s_m2',\n"
            "            y_lim=y_lim_sens_p_hp,\n"
            "            **kwargs,\n"
            "        )\n"
            "    finally:\n"
            "        plt.show = _show_orig\n"
            "        plt.close = _close_orig\n"
            "    fig = plt.gcf()\n"
            "    for ax in fig.axes:\n"
            "        if ax.get_ylabel().strip() == 'Power required [hp]':\n"
            "            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f\"{co.hp2kw(y):.0f}\"))\n"
            "            ax.set_ylabel('Power required [kW]')\n"
            "    plt.show()\n\n"
            "_run_sensitivity_plot_with_kw_labels(True)\n\n"
            "_run_sensitivity_plot_with_kw_labels('cruise', maskbool=False)\n"
        )

        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._nb_source_lines(
                        f"# Constraint Diagram\n\n"
                        f"Auto-generated from `{case_name}` by `app/main.py`."
                    ),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_setup),
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._nb_source_lines("## Quick Constraint Diagram Plots"),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_ws_grid),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_quick_power),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_quick_tw),
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._nb_source_lines("## Flexible Constraint Plot"),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_flexible),
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._nb_source_lines("## INI vs Converged Markers (P/W and T/W)"),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_marked_pw_tw),
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": self._nb_source_lines(
                        "## Sensitivity Plots\n\n"
                        "This section mirrors the sensitivity plotting workflow from `constraint_example.ipynb` "
                        "using +/-5% ranges around this INI case values."
                    ),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": self._nb_source_lines(code_sensitivity),
                },
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        notebook_path = out_dir / "Constraint_Diagram.ipynb"
        notebook_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


# ============================
# Entry point
# ============================


def _converged_summary_text(
    phases: Dict[str, PhasePowerResult],
    mass: MassBreakdown,
    cfg: DesignConfig,
    execution_time_s: Optional[float] = None,
) -> str:
    """Converged summary used for console output and text export."""
    loiter_p_w_kw_per_kg = _LOITER_PW_RATIO_TO_CRUISE * float(cfg.p_w.p_w_cruise_kw_per_kg)

    def _phase_pw_kw_per_kg(phase: Optional[PhasePowerResult]) -> float:
        if phase is None or mass.mtom_kg <= 0.0:
            return math.nan
        return float(phase.p_total_w / mass.mtom_kg / 1000.0)

    def _fmt_pw_kw_per_kg(value_kw_per_kg: float) -> str:
        if not math.isfinite(float(value_kw_per_kg)):
            return "n/a"
        return f"{float(value_kw_per_kg):.5f}"

    def _fmt_kw(value_kw: float, width: int = 14) -> str:
        if not math.isfinite(float(value_kw)):
            return f"{'n/a':>{width}}"
        return f"{float(value_kw):>{width},.1f}"

    def _fmt_gps(value_gps: float, width: int = 12) -> str:
        if not math.isfinite(float(value_gps)):
            return f"{'n/a':>{width}}"
        return f"{float(value_gps):>{width},.1f}"

    # Mission timing summary.
    brief = cfg.constraint_brief
    climb_rate = Q_(float(brief.climbrate_fpm), "foot / minute").to("meter / second")
    v_climb = Q_(float(brief.climbspeed_kias), "knot").to("meter / second")
    v_cruise = Q_(float(brief.cruisespeed_ktas), "knot").to("meter / second")
    loiter_speed_ratio = _resolve_loiter_speed_ratio(cfg.flight.loiter_speed_ratio_to_cruise)
    v_loiter = loiter_speed_ratio * v_cruise
    # Use runway elevation as mission climb start altitude; climbalt_m is the constraint evaluation altitude.
    h_delta = Q_(max(float(brief.cruisealt_m) - float(brief.rwyelevation_m), 0.0), "meter")

    t_climb = (h_delta / climb_rate).to("second")
    r_climb = (v_climb * t_climb).to("meter")
    t_ready = Q_(0.0, "minute").to("second")
    t_taxi = Q_(5.0, "minute").to("second")
    t_takeoff = Q_(1.0, "minute").to("second")
    t_loiter = Q_(15.0, "minute").to("second")
    t_landing = Q_(1.0, "minute").to("second")
    r_loiter = (v_loiter * t_loiter).to("meter")
    range_total = Q_(float(cfg.hydrogen.range_total_m), "meter")
    r_cruise = (range_total - r_climb - r_loiter).to("meter")
    if float(r_cruise.to("meter").magnitude) < 0.0:
        r_cruise = Q_(0.0, "meter")
    t_cruise = (r_cruise / v_cruise).to("second")

    phase_times_min = {
        "ready": float(t_ready.to("minute").magnitude),
        "taxi": float(t_taxi.to("minute").magnitude),
        "takeoff": float(t_takeoff.to("minute").magnitude),
        "climb": float(t_climb.to("minute").magnitude),
        "cruise": float(t_cruise.to("minute").magnitude),
        "loiter": float(t_loiter.to("minute").magnitude),
        "landing": float(t_landing.to("minute").magnitude),
    }
    climb_time_min = phase_times_min["climb"]
    cruise_time_min = phase_times_min["cruise"]
    total_flight_time_min = float(sum(phase_times_min.values()))

    pnet = phases["climb"].p_total_w * cfg.eff.eta_pdu * cfg.eff.eta_em
    sizing_phase_name, sizing_phase, _ = _max_phase_fc_sizing_power(
        {
            name: phases[name]
            for name in ("takeoff", "climb", "cruise", "loiter", "cruise_charger")
            if name in phases
        }
    )
    nac = sizing_phase.nacelle
    w_fcs_single_nacelle_kg = float(nac.m_stacks_kg + nac.m_comp_kg + nac.m_humid_kg + nac.m_hx_kg)
    w_fcs_total_kg = float(w_fcs_single_nacelle_kg * cfg.fuel_cell_arch.n_stacks_parallel)
    # w_motors_kg = float(mass.m_e_motor_kg * cfg.fuel_cell_arch.n_stacks_parallel)
    w_motors_kg = float(mass.m_e_motor_kg)
    w_powertrain_kg = float(w_fcs_total_kg + mass.m_pmad_kg + w_motors_kg)
    oem_grouped_kg = float(
        mass.w_wing_kg
        + mass.w_ht_kg
        + mass.w_vt_kg
        + mass.w_fus_kg
        + mass.w_lnd_main_kg
        + mass.w_lnd_nose_kg
        + mass.w_flight_control_kg
        + mass.w_els_kg
        + mass.w_iae_kg
        + mass.w_hydraulics_kg
        + mass.w_furnishings_kg
    )
    mtom_grouped_kg = float(
        oem_grouped_kg + mass.m_tank_kg + mass.m_fuel_kg + mass.m_battery_kg + w_powertrain_kg + mass.payload_kg
    )
    p_to_w_converged_w_per_kg = phases["climb"].p_total_w / mass.mtom_kg
    wing_loading_kg_per_m2 = mass.mtom_kg / mass.wing_area_m2
    max_p_comp_phase, max_p_comp_w = _max_phase_comp_power(phases)

    takeoff = phases.get("takeoff")
    climb = phases.get("climb")
    cruise = phases.get("cruise")
    loiter = phases.get("loiter")
    conv_p_w_takeoff_kw_per_kg = _phase_pw_kw_per_kg(takeoff)
    conv_p_w_climb_kw_per_kg = _phase_pw_kw_per_kg(climb)
    conv_p_w_cruise_kw_per_kg = _phase_pw_kw_per_kg(cruise)
    conv_p_w_loiter_kw_per_kg = _phase_pw_kw_per_kg(loiter)

    def _phase_row(
        phase_name: str,
        duration_min: float,
        p_total_w: float,
        p_fc_stack_w: float,
        p_batt_w: float,
        p_comp_w: float,
        p_cooling_w: float,
        mdot_h2_kgps: float,
    ) -> str:
        return (
            f"{phase_name:<10} {duration_min:>9.2f}"
            f"{_fmt_kw(p_total_w / 1000.0, 14)}"
            f"{_fmt_kw(p_fc_stack_w / 1000.0, 16)}"
            f"{_fmt_kw(p_batt_w / 1000.0, 14)}"
            f"{_fmt_kw(p_comp_w / 1000.0, 17)}"
            f"{_fmt_kw(p_cooling_w / 1000.0, 14)}"
            f"{_fmt_gps(mdot_h2_kgps * 1000.0, 12)}"
        )

    mdot_taxi_kgps = 0.10 * float(climb.mdot_h2_kgps) if climb is not None else math.nan
    phase_rows = [
        _phase_row("ready", phase_times_min["ready"], mass.p_fuelcell_engine_w, mass.p_fuelcell_engine_w, 0.0, 0.0, 0.0, 0.0),
        _phase_row(
            "taxi", phase_times_min["taxi"], mass.p_fuelcell_taxing_w, mass.p_fuelcell_taxing_w, 0.0, 0.0, 0.0, mdot_taxi_kgps
        ),
        _phase_row(
            "takeoff",
            phase_times_min["takeoff"],
            float(takeoff.p_total_w) if takeoff is not None else math.nan,
            float(takeoff.p_fuelcell_w) if takeoff is not None else math.nan,
            float(takeoff.p_battery_w) if takeoff is not None else math.nan,
            float(takeoff.p_comp_w) if takeoff is not None else math.nan,
            float(takeoff.p_cooling_w) if takeoff is not None else math.nan,
            float(takeoff.mdot_h2_kgps) if takeoff is not None else math.nan,
        ),
        _phase_row(
            "climb",
            phase_times_min["climb"],
            float(climb.p_total_w) if climb is not None else math.nan,
            float(climb.p_fuelcell_w) if climb is not None else math.nan,
            float(climb.p_battery_w) if climb is not None else math.nan,
            float(climb.p_comp_w) if climb is not None else math.nan,
            float(climb.p_cooling_w) if climb is not None else math.nan,
            float(climb.mdot_h2_kgps) if climb is not None else math.nan,
        ),
        _phase_row(
            "cruise",
            phase_times_min["cruise"],
            float(cruise.p_total_w) if cruise is not None else math.nan,
            float(cruise.p_fuelcell_w) if cruise is not None else math.nan,
            float(cruise.p_battery_w) if cruise is not None else math.nan,
            float(cruise.p_comp_w) if cruise is not None else math.nan,
            float(cruise.p_cooling_w) if cruise is not None else math.nan,
            float(cruise.mdot_h2_kgps) if cruise is not None else math.nan,
        ),
        _phase_row(
            "loiter",
            phase_times_min["loiter"],
            float(loiter.p_total_w) if loiter is not None else math.nan,
            float(loiter.p_fuelcell_w) if loiter is not None else math.nan,
            float(loiter.p_battery_w) if loiter is not None else math.nan,
            float(loiter.p_comp_w) if loiter is not None else math.nan,
            float(loiter.p_cooling_w) if loiter is not None else math.nan,
            float(loiter.mdot_h2_kgps) if loiter is not None else math.nan,
        ),
        _phase_row("landing", phase_times_min["landing"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    phase_header = (
        "Phase      Time[min]     Total[kW]    FC Stack[kW]   Battery[kW]   Compressor[kW]   Cooling[kW]   mdot_H2[g/s]"
    )
    phase_rule = "-" * len(phase_header)

    lines = [
        "=============================================================",
        "========================= CONVERGED =========================",
        "=============================================================",
        "",
        "# Mission Time",
        f"Climb time: {climb_time_min:,.2f} min ({climb_time_min/60.0:,.2f} h)",
        f"Cruise time: {cruise_time_min:,.2f} min ({cruise_time_min/60.0:,.2f} h)",
        f"Total flight time: {total_flight_time_min:,.2f} min ({total_flight_time_min/60.0:,.2f} h)",
        "",
        "# Phase Power Breakdown",
        phase_header,
        phase_rule,
        *phase_rows,
        "",
        f"Pcomp_max (solved phases): {max_p_comp_w/1000:,.1f} kW ({max_p_comp_phase})",
        f"Pnet (climb): {pnet/1000:,.0f} kW",
        f"Pelectricnet (climb): {phases['climb'].p_bus_required_w/1000:,.0f} kW",
        f"eta_pt: {cfg.eff.eta_em*cfg.eff.eta_pdu:,.4f}",
        "",
        "# Geometry",
        f"Main Wing Area: {mass.wing_area_m2:,.2f} m^2",
        f"Wingspan: {mass.wing_span_m:,.2f} m",
        f"Fuselage Length: {mass.fuselage_length_m:,.2f} m",
        f"Hydrogen Tank Length: {mass.tank_length_m:,.2f} m",
        f"Horizontal Tail Area: {mass.s_ht_m2:,.2f} m^2",
        f"Vertical Tail Area: {mass.s_vt_m2:,.2f} m^2",
        f"Horizontal Tail span: {mass.b_ht_m:,.2f} m",
        f"Vertical Tail span: {mass.b_vt_m:,.2f} m",
        f"Horizontal Tail X_Position: {mass.x_true_ht_m:,.2f} m",
        f"Vertical Tail X_Position: {mass.x_true_vt_m:,.2f} m",
        "",
        f"# Single Nacelle (1/{cfg.fuel_cell_arch.n_stacks_parallel})",
        (
            f"Stack(1/{cfg.fuel_cell_arch.n_stacks_parallel}): {nac.m_stacks_kg:,.0f} kg, "
            f"Compressor: {nac.m_comp_kg:,.0f} kg, Humidifier: {nac.m_humid_kg:,.0f} kg, HX: {nac.m_hx_kg:,.0f} kg"
        ),
        f"Fuelcell System(FCS; Stack+Compressor+Humidifier+Hx): {w_fcs_single_nacelle_kg:,.0f} kg",
        f"Representative FC phase: {sizing_phase_name}",
        f"Specific Power of Fuelcell System: {mass.nacelle_design_power_kw_per_kg:,.3f} kW/kg",
        f"Stack dimension: dX={mass.nacelle_stack_dim_m[2]:,.3f} m, dY={mass.nacelle_stack_dim_m[0]:,.3f} m, dZ={mass.nacelle_stack_dim_m[1]:,.3f} m",
        f"Heat Exchanger dimension: dX={mass.nacelle_hx_dim_m[0]:,.3f} m, dY={mass.nacelle_hx_dim_m[1]:,.3f} m, dZ={mass.nacelle_hx_dim_m[2]:,.3f} m",
        "",
        "# ALL Nacelles",
        f"FCS_total: {w_fcs_total_kg:,.0f} kg",
        f"PMAD: {mass.m_pmad_kg:,.0f} kg",
        f"Motors: {w_motors_kg:,.0f} kg",
        "",
        "# Total Mass Breakdown",
        "-----------------------",
        f"Powertrain (FCS_total+PMAD+Motors): {w_powertrain_kg:,.0f} kg",
        "-----------------------",
        f"Main Wing: {mass.w_wing_kg:,.0f} kg",
        f"Horizontal Tail: {mass.w_ht_kg:,.0f} kg",
        f"Vertical Tail: {mass.w_vt_kg:,.0f} kg",
        f"Fuselage: {mass.w_fus_kg:,.0f} kg",
        f"Landing gear(main): {mass.w_lnd_main_kg:,.0f} kg",
        f"Landing gear(nose): {mass.w_lnd_nose_kg:,.0f} kg",
        f"Flight Control System: {mass.w_flight_control_kg:,.0f} kg",
        f"Electric System: {mass.w_els_kg:,.0f} kg",
        f"Instruments: {mass.w_iae_kg:,.0f} kg",
        f"Hydraulics: {mass.w_hydraulics_kg:,.0f} kg",
        f"Furnishing: {mass.w_furnishings_kg:,.0f} kg",
        "-----------------------",
        f"Operating Empty Mass(OEM): {oem_grouped_kg:,.0f} kg",
        "-----------------------",
        f"Hydrogen Tank: {mass.m_tank_kg:,.0f} kg",
        f"Hydrogen Fuel: {mass.m_fuel_kg:,.1f} kg",
        f"Battery: {mass.m_battery_kg:,.1f} kg",
        "-----------------------",
        f"Payload: {mass.payload_kg:,.0f} kg",
        "-----------------------",
        # f"MTOM_grouped (OEM+tank+fuel+batt+Powertrain+payload): {mtom_grouped_kg:,.0f} kg",
        f"Maximum Takeoff Mass(MTOM): {mass.mtom_kg:,.0f} kg",
        "-----------------------",
        "",
        f"Wing loading (MTOM/S_wing): {wing_loading_kg_per_m2:,.2f} kg/m^2",
        f"Power-to-weight (maxPow/MTOM): {p_to_w_converged_w_per_kg:,.2f} W/kg",
        "",
        f"mdot_H2(cruise): {phases['cruise'].mdot_h2_kgps*1000:,.1f} g/s",
        f"Hydrogen Tank Volume: {mass.tank_volume_m3:,.1f} m^3",
        "========================== END ==============================",
    ]
    if cfg.constraint_sizing.enable:
        hdr = [
            f"Constraint sizing: ENABLED ({cfg.constraint_sizing.selection})",
            f"Input W/S: {cfg.wing.wing_loading_kg_per_m2:,.2f} kg/m^2 ({cfg.wing.wing_loading_kg_per_m2*_G0_MPS2:,.1f} Pa)",
            f"Input P/W [kW/kg]: takeoff {cfg.p_w.p_w_takeoff_kw_per_kg:.5f}, climb {cfg.p_w.p_w_climb_kw_per_kg:.5f}, "
            f"cruise {cfg.p_w.p_w_cruise_kw_per_kg:.5f}, loiter {loiter_p_w_kw_per_kg:.5f}",
            f"Converged P/W [kW/kg]: takeoff {_fmt_pw_kw_per_kg(conv_p_w_takeoff_kw_per_kg)}, "
            f"climb {_fmt_pw_kw_per_kg(conv_p_w_climb_kw_per_kg)}, "
            f"cruise {_fmt_pw_kw_per_kg(conv_p_w_cruise_kw_per_kg)}, "
            f"loiter {_fmt_pw_kw_per_kg(conv_p_w_loiter_kw_per_kg)}",
            "-------------------------------------------------------------",
        ]
        lines = lines[:3] + hdr + lines[3:]
    if execution_time_s is not None and math.isfinite(float(execution_time_s)):
        lines[-1] = f"{lines[-1]} | Execution time: {float(execution_time_s):.1f} seconds"
    return "\n".join(lines)


def _log_converged_state(
    *,
    state: str,
    phases: Dict[str, PhasePowerResult],
    mass: MassBreakdown,
    cfg: DesignConfig,
) -> None:
    """Log a compact but actionable snapshot for a converged solution state."""
    loiter_p_w_kw_per_kg = _LOITER_PW_RATIO_TO_CRUISE * float(cfg.p_w.p_w_cruise_kw_per_kg)

    climb = phases["climb"]
    cruise = phases["cruise"]
    loiter = phases.get("loiter")
    takeoff = phases.get("takeoff")
    cruise_charger = phases.get("cruise_charger")
    max_p_comp_phase, max_p_comp_w = _max_phase_comp_power(phases)
    fc_rep_phase_name, _, fc_rep_power_w = _max_phase_fc_sizing_power(
        {
            name: phases[name]
            for name in ("takeoff", "climb", "cruise", "loiter", "cruise_charger")
            if name in phases
        }
    )

    mtom = mass.mtom_kg
    wing_loading = mass.mtom_kg / mass.wing_area_m2 if mass.wing_area_m2 > 0.0 else math.nan
    p_to_w = climb.p_total_w / mtom if mtom > 0 else math.nan
    takeoff_kw = takeoff.p_total_w / 1000.0 if takeoff is not None else math.nan
    cruise_kw = cruise.p_total_w / 1000.0
    loiter_kw = loiter.p_total_w / 1000.0 if loiter is not None else math.nan
    climb_kw = climb.p_total_w / 1000.0
    p_charger_kw = cruise_charger.p_total_w / 1000.0 if cruise_charger is not None else math.nan

    logger.info("")
    logger.info("CONVERGED STATE [%s]", state)
    logger.info(
        f"  MTOM={mtom:,.0f} kg | Wing loading={wing_loading:,.2f} kg/m^2 | P/W(climb)={p_to_w:,.2f} W/kg",
    )
    logger.info(
        f"  Phase power: climb={climb_kw:,.1f} kW | cruise={cruise_kw:,.1f} kW | loiter={loiter_kw:,.1f} kW | "
        f"takeoff={takeoff_kw:,.1f} kW | cruise+charger={p_charger_kw:,.1f} kW",
    )
    logger.info(
        f"  Power split (climb): FC={climb.p_fuelcell_w / 1000.0:,.1f} kW | "
        f"Battery={climb.p_battery_w / 1000.0:,.1f} kW | Compressor={climb.p_comp_w / 1000.0:,.1f} kW | "
        f"Cooling={climb.p_cooling_w / 1000.0:,.1f} kW | H2={climb.mdot_h2_kgps * 1000.0:,.3f} g/s",
    )
    logger.info(
        f"  Compressor max across phases: {max_p_comp_w / 1000.0:,.1f} kW ({max_p_comp_phase})",
    )
    logger.info(
        f"  FC representative phase (mass sizing): {fc_rep_phase_name} ({fc_rep_power_w / 1000.0:,.1f} kW)",
    )
    logger.info(
        "  Mass summary: "
        f"MTOM={mtom:,.0f} kg | Powertrain={mass.m_powertrain_total_kg:,.0f} kg | "
        f"Tank={mass.m_tank_kg:,.1f} kg | Fuel={mass.m_fuel_kg:,.1f} kg | "
        f"Battery={mass.m_battery_kg:,.1f} kg | Nacelle power density={mass.nacelle_design_power_kw_per_kg:,.2f} kW/kg",
    )
    logger.info(
        "  Geometry: "
        f"Main wing area={mass.wing_area_m2:,.2f} m^2 | "
        f"Horizontal tail area={mass.s_ht_m2:,.3f} m^2 | "
        f"Horizontal tail location={mass.x_true_ht_m:,.3f} m | "
        f"Vertical tail area={mass.s_vt_m2:,.3f} m^2 | "
        f"Vertical tail location={mass.x_true_vt_m:,.3f} m",
    )
    logger.info(
        f"  Config: wing loading input={cfg.wing.wing_loading_kg_per_m2:,.2f} kg/m^2 | "
        f"P/W TO/Cl/Cr/Lo={cfg.p_w.p_w_takeoff_kw_per_kg:,.5f}/{cfg.p_w.p_w_climb_kw_per_kg:,.5f}/"
        f"{cfg.p_w.p_w_cruise_kw_per_kg:,.5f}/{loiter_p_w_kw_per_kg:,.5f} kW/kg",
    )


def _print_summary(phases: Dict[str, PhasePowerResult], mass: MassBreakdown, cfg: DesignConfig) -> None:
    """Console report similar to the legacy script."""

    print()
    print(_converged_summary_text(phases=phases, mass=mass, cfg=cfg))


def _output_subdir_from_input(input_path: Path) -> str:
    stem = input_path.stem
    if stem.startswith("input_"):
        return stem[len("input_"):]
    return stem


def _phase_fc_sizing_power_w(phase: PhasePowerResult) -> float:
    """Return FC net electrical power used for nacelle sizing in a phase."""

    return float(phase.p_total_w) - max(float(phase.p_battery_w), 0.0)


def _max_phase_fc_sizing_power(
    phases: Dict[str, PhasePowerResult],
) -> Tuple[str, PhasePowerResult, float]:
    """Return phase name, phase result, and FC sizing power (W) at the maximum."""

    if not phases:
        raise ValueError("At least one phase is required to select FC sizing representative phase.")
    phase_name, phase = max(phases.items(), key=lambda item: _phase_fc_sizing_power_w(item[1]))
    return phase_name, phase, _phase_fc_sizing_power_w(phase)


def _max_phase_comp_power(phases: Dict[str, PhasePowerResult]) -> Tuple[str, float]:
    """Return phase name and value for maximum compressor power."""

    if not phases:
        return "n/a", math.nan
    phase_name, phase = max(phases.items(), key=lambda item: float(item[1].p_comp_w))
    return phase_name, float(phase.p_comp_w)


def main(argv: Optional[List[str]] = None) -> Optional[float]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start_time = time.perf_counter()

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

    if not args.show_plot:
        # Avoid Qt backend/plugin crashes in headless runs.
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    input_path = Path(args.input).expanduser()
    if args.write_template:
        write_input_template(input_path, DesignConfig())
        logger.info("Wrote template input file: %s", input_path)
        return None

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
    elapsed_time = time.perf_counter() - start_time
    writer.write_converged_text(
        phases=phases,
        mass=mass,
        out_dir=out_dir,
        execution_time_s=elapsed_time,
    )
    writer.write_constraint_diagram_notebook(
        phases=phases,
        mass=mass,
        out_dir=out_dir,
        input_path=input_path,
        initial_cfg=cfg,
    )
    writer.write_case_variable_exports(
        phases=phases,
        mass=mass,
        out_dir=out_dir,
        input_path=input_path,
        initial_cfg=cfg,
        execution_time_s=elapsed_time,
    )
    return elapsed_time


if __name__ == "__main__":
    elapsed_time = main()
    if elapsed_time is not None:
        print("\n\n=============================================================")
        print(f"Execution time: {elapsed_time:.1f} seconds")
        print("=============================================================")
