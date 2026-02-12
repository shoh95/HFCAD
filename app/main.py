"""HFCBattACDesign_SH_260117 (OOP rewrite)

This module is an object-oriented refactor of the legacy script:
  - configuration is explicit (dataclasses)
  - responsibilities are separated (powertrain, FC system, cooling, mass estimation, solver)
  - the numerical approach and default constants are intentionally kept close to the legacy behaviour

Note:
- This is a refactor, not a physics/model correctness overhaul.
- Some legacy formulas contain unit ambiguities; these are preserved by default for backward compatibility.
"""

from __future__ import annotations

import argparse
import configparser
import shutil
import threading

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
    """Shaft power-to-weight ratios used in the legacy script.

    Units: W/N. (Because Pshaft = MTOM * g * (W/N)).
    """

    p_w_climb_w_per_n: float = 0.1139 * 1000 / 9.81
    p_w_cruise_w_per_n: float = 0.0876 * 1000 / 9.81
    p_w_takeoff_w_per_n: float = 0.0739 * 1000 / 9.81


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

    wing_loading_pa: float = 2830.24
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
      cooling, hydrogen, wing, fuselage, weights, solver, design
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

    mission = _update_dataclass_from_section(cfg_default.mission, "mission", section("mission"))
    flight = _update_dataclass_from_section(cfg_default.flight, "flight", section("flight"))

    fuel_cell_arch = _update_dataclass_from_section(cfg_default.fuel_cell_arch, "fuel_cell_arch", section("fuel_cell_arch"))
    fuel_cell_op = _update_dataclass_from_section(cfg_default.fuel_cell_op, "fuel_cell_op", section("fuel_cell_op"))

    hybrid = _update_dataclass_from_section(cfg_default.hybrid, "hybrid", section("hybrid"))
    eff = _update_dataclass_from_section(cfg_default.eff, "eff", section("eff"))
    densities = _update_dataclass_from_section(cfg_default.densities, "densities", section("densities"))
    p_w = _update_dataclass_from_section(cfg_default.p_w, "p_w", section("p_w"))

    cooling = _update_dataclass_from_section(cfg_default.cooling, "cooling", section("cooling"))
    hydrogen = _update_dataclass_from_section(cfg_default.hydrogen, "hydrogen", section("hydrogen"))

    wing = _update_dataclass_from_section(cfg_default.wing, "wing", section("wing"))
    fuselage = _update_dataclass_from_section(cfg_default.fuselage, "fuselage", section("fuselage"))
    weights = _update_dataclass_from_section(cfg_default.weights, "weights", section("weights"))

    solver = _update_dataclass_from_section(cfg_default.solver, "solver", section("solver"))

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

    def __init__(self, arch: FuelCellArchitecture):
        self._arch = arch

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

            m_comp = float(compressor_mass_model(geom_comp, power_comp)) if geom_comp is not None else 0.0
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
        p_w_w_per_n: float,
        flight_point: FlightPoint,
        psi: float,
        beta: float,
        initial_total_power_w: float,
        oversizing: float,
    ) -> PhasePowerResult:
        """Fixed-point iteration for total electrical power including auxiliaries."""

        g = 9.81
        p_total = float(initial_total_power_w)

        for inner_iter in range(1, self._cfg.solver.max_inner_iter + 1):
            p_shaft = float(mtom_kg * g * p_w_w_per_n)

            split = self._powertrain.split_shaft_power(p_shaft, psi)
            p_bus_required = split.p_bus_required_w

            # Allocate FC system net output used for per-nacelle sizing.
            # Legacy behaviour:
            #   - when battery discharges (p_battery > 0), subtract it from the total to get FC share
            #   - when battery charges (p_battery < 0), do not subtract (charging already appears as +abs(p_battery))
            p_fc_sys_total = p_total - max(split.p_battery_w, 0.0)

            # Guardrail: ensure sizing power stays positive.
            if p_fc_sys_total <= 0.0:
                # Prefer a consistent fallback over the legacy 'set to ptotal' (which mixes per-system/per-nacelle units).
                p_fc_sys_total = max(split.p_fuelcell_w, 1.0)

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

    def __init__(self, *, cfg: DesignConfig):
        self._cfg = cfg
        self._conv = Conversions()

        # Pre-compute cruise environment
        atm_cr = Atmosphere(cfg.flight.h_cr_m)
        self._c_cr_mps = float(atm_cr.speed_of_sound[0])
        self._v_cr_mps = float(cfg.flight.mach_cr * self._c_cr_mps)
        self._rho_cr = float(atm_cr.density[0])

        # Wing loading in psf (legacy stores it as "_imp" but it's really psf)
        self._w_s_psf = float(self._conv.pa_psf(cfg.wing.wing_loading_pa, "psf"))

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
        power_fc_sys_per_nacelle = climb.p_fuelcell_w / cfg.fuel_cell_arch.n_stacks_parallel

        nacelle_design = FuelCellSystemModel(cfg.fuel_cell_arch).size_nacelle(
            power_fc_sys_w=power_fc_sys_per_nacelle,
            flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
            beta=1.05,
            oversizing=cfg.fuel_cell_op.oversizing,
            comp_bool=True,
            make_fig=False,
            verbose=True,
        )

        # Mass per nacelle (legacy: stack + humidifier + compressor + HX)
        m_fc_system = cfg.fuel_cell_arch.n_stacks_parallel * (
            nacelle_design.m_stacks_kg
            + nacelle_design.m_humid_kg
            + nacelle_design.m_comp_kg
            + nacelle_design.m_hx_kg
        )

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
        mtom_lb = float(conv.kg_pound(mtom_guess_kg, "pound"))
        s_wing_ft2 = float(mtom_lb / self._w_s_psf)
        b_wing_ft = float(math.sqrt(cfg.wing.aspect_ratio * s_wing_ft2))

        wing_span_m = float(conv.meter_feet(b_wing_ft, "meter"))
        wing_area_m2 = float(conv.m2_ft2(s_wing_ft2, "m2"))

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

        nacelle_pd_kw_per_kg = float((power_fc_sys_per_nacelle / 1000.0) / nacelle_design.m_sys_kg)

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

    def __init__(self, cfg: DesignConfig):
        self._cfg = cfg

        self._powertrain = HybridPowertrain(cfg.eff)
        self._cooling = CoolingSystem(
            cruise_altitude_m=cfg.flight.h_cr_m,
            cruise_mach=cfg.flight.mach_cr,
            cfg=cfg.cooling,
        )
        self._fc_model = FuelCellSystemModel(cfg.fuel_cell_arch)
        self._phase_solver = PhasePowerSolver(
            config=cfg,
            powertrain=self._powertrain,
            fc_model=self._fc_model,
            cooling=self._cooling,
        )
        self._mass_estimator = MassEstimator(cfg=cfg)

    def run(self) -> Tuple[Dict[str, PhasePowerResult], MassBreakdown]:
        cfg = self._cfg

        # Initial conditions
        mtom = float(cfg.initial_mtom_kg)

        ptotal_guess = {
            "cruise": float(cfg.initial_total_power_guess_w),
            "cruise_charger": float(cfg.initial_total_power_guess_w),
            "takeoff": float(cfg.initial_total_power_guess_w),
            "climb": float(cfg.initial_total_power_guess_w),
        }

        phases: Dict[str, PhasePowerResult] = {}

        for outer_iter in range(1, cfg.solver.max_outer_iter + 1):
            logger.info("======================================================")
            logger.info(f"======================= ITER {outer_iter} =======================")

            # Cruise (FC only -> psi = 0)
            phases["cruise"] = self._phase_solver.solve(
                name="cruise",
                mtom_kg=mtom,
                p_w_w_per_n=cfg.p_w.p_w_cruise_w_per_n,
                flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
                psi=0.0,
                beta=1.05,
                initial_total_power_w=ptotal_guess["cruise"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["cruise"] = phases["cruise"].p_total_w

            # Cruise charger (psi < 0)
            phases["cruise_charger"] = self._phase_solver.solve(
                name="cruise_charger",
                mtom_kg=mtom,
                p_w_w_per_n=cfg.p_w.p_w_cruise_w_per_n,
                flight_point=FlightPoint(cfg.flight.h_cr_m, cfg.flight.mach_cr),
                psi=cfg.hybrid.psi_cruise_charger,
                beta=1.05,
                initial_total_power_w=ptotal_guess["cruise_charger"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["cruise_charger"] = phases["cruise_charger"].p_total_w

            # Takeoff
            phases["takeoff"] = self._phase_solver.solve(
                name="takeoff",
                mtom_kg=mtom,
                p_w_w_per_n=cfg.p_w.p_w_takeoff_w_per_n,
                flight_point=FlightPoint(cfg.flight.h_takeoff_m, cfg.flight.mach_takeoff),
                psi=cfg.hybrid.psi_takeoff,
                beta=1.05,
                initial_total_power_w=ptotal_guess["takeoff"],
                oversizing=cfg.fuel_cell_op.oversizing,
            )
            ptotal_guess["takeoff"] = phases["takeoff"].p_total_w

            # Climb
            phases["climb"] = self._phase_solver.solve(
                name="climb",
                mtom_kg=mtom,
                p_w_w_per_n=cfg.p_w.p_w_climb_w_per_n,
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

            logger.info("\n-----------------------")
            logger.info(f"ptotal_climb: {phases['climb'].p_total_w/1000:,.0f} kW, mtom: {mass.mtom_kg:,.0f} kg\n\n")

            if abs(mass.mtom_kg - mtom) <= cfg.solver.mtom_tol_kg:
                logger.info("\nCONVERGED")
                return phases, mass

            mtom = float(mass.mtom_kg)

        raise RuntimeError(f"MTOM did not converge within {cfg.solver.max_outer_iter} outer iterations")


# ============================
# Output utilities
# ============================


class OutputWriter:
    def __init__(self, cfg: DesignConfig):
        self._cfg = cfg

    def write_pemfc_figure(
        self,
        *,
        nacelle_power_w: float,
        out_dir: Path,
        timeout_s: float = 30.0,
    ) -> None:
        """Generate PEMFC polarization figure once (expensive)."""

        try:
            fc = FuelCellSystemModel(self._cfg.fuel_cell_arch)
            res = fc.size_nacelle(
                power_fc_sys_w=nacelle_power_w,
                flight_point=FlightPoint(self._cfg.flight.h_cr_m, self._cfg.flight.mach_cr),
                beta=1.05,
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
        plt.axis([0, 180, -200, 2500])
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
    p_to_w_converged_w_per_kg = phases["climb"].p_total_w / mass.mtom_kg
    wing_loading_tmtom_per_m2 = (mass.mtom_kg / 1000.0) / mass.wing_area_m2

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
            f"Compressor: {nac.m_comp_kg:,.0f} kg, Humidifier: {nac.m_humid_kg:,.0f} kg, HX: {nac.m_hx_kg:,.0f} kg"
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
        f"Wing loading (tMTOM/S_wing): {wing_loading_tmtom_per_m2:,.4f} t/m^2",
        "",
        f"Vtankex: {mass.tank_volume_m3:,.1f} m^3",
        "========================== END ==============================",
    ]
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

    design = HybridFuelCellAircraftDesign(cfg)
    phases, mass = design.run()

    _print_summary(phases, mass, cfg)

    out_root = Path(args.outdir).expanduser()
    out_dir = out_root / _output_subdir_from_input(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_copy_path = out_dir / input_path.name
    if input_copy_path.resolve() != input_path.resolve():
        shutil.copy2(input_path, input_copy_path)

    writer = OutputWriter(cfg)
    writer.write_converged_text(phases=phases, mass=mass, out_dir=out_dir)

    # Fuel cell figure (per nacelle at design point)
    nacelle_power_w = phases["climb"].p_total_w / cfg.fuel_cell_arch.n_stacks_parallel
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
