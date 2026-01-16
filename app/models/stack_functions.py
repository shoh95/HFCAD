import numpy as np
from functools import lru_cache
from math import log, sqrt, pi, isclose


def stack_model(n_stacks_series, volt_req, volt_cell, power_req, power_dens_cell):
    """Calculate mass of stack(s) in the FC system.

    :param n_stacks_series: Number of stacks in series in FC system
    :param volt_req: Voltage to be delivered by FC system in V
    :param volt_cell: Nominal cell voltage in V
    :param power_req: Electrical power to be delivered by stacks (bigger than propulsive output power of FC system)
    :param power_dens_cell: Nominal cell power density in W/m^2
    :return: mass of stack(s) in kg, dimensions, and resolution variables (n_cells, area_cell)
    """
    # constants
    # bipolar plate
    t_bp_dim = 2e-4  # m - thickness for determination of complete dimensions
    t_bp = 2e-4  # m
    rho_bp = 8e3  # kg/m3 - SS304L

    # endplate
    t_ep = 2.5e-2  # m
    rho_ep = 8e3  # kg/m3 - same as bp

    # bolts
    n_bolt = 10  # see Dey 2019
    rho_bolt = 8e3

    # MEA
    rho_mea = 0.2  # kg/m2 - Kadyk 2018

    # calculations - per stack
    n_cells = volt_req / volt_cell / n_stacks_series  # number of cells per stack
    area_cell = power_req / n_stacks_series / n_cells / power_dens_cell  # area of single cell

    m_bp = n_cells * t_bp * rho_bp * area_cell  # mass of bipolar plates of one stack
    m_ep = 2 * t_ep * rho_ep * area_cell  # mass of endplates of one stack
    m_mea = rho_mea * area_cell * n_cells  # mass of MEA of one stack

    d_bolt = sqrt(area_cell / 600)  # diameter of bolts - see Dey 2019
    l_bolt = 2 * t_ep + n_cells * t_bp  # length of bolts
    m_bolts = n_bolt * pi * (d_bolt / 2) ** 2 * l_bolt * rho_bolt  # mass of bolts of one stack

    m_tot = m_bp + m_ep + m_bolts + m_mea  # mass of a single stack

    dim = [np.sqrt(area_cell), np.sqrt(area_cell), l_bolt]

    return m_tot * n_stacks_series, dim, [n_cells, area_cell]


def mass_flow_stack(power_stack, volt_cell):
    """Compute mass flow of air required by fuel cell stack with given power output and cell voltage.

    :param power_stack: Electrical output power of stack in W
    :param volt_cell: Cell voltage in V
    :return: mass flow in kg/s
    """
    stoich = 2  # assumed stoichiometry TODO: consider this more in detail for air cooled
    return 3.58e-7 * stoich * power_stack / volt_cell


# -----------------------------
# Cell model (performance hotspot)
# -----------------------------


def _cell_voltage_arrays(js, pres_air, pres_h, cell_temp):
    """Vectorized computation of cell voltage for an array of current densities.

    This replaces the original per-element Python loop in `cell_model`, which was a major runtime hotspot.

    Returns:
        volt: corrected voltage array (includes altitude correction)
        volt_sl_loss: voltage array before altitude correction
        pot_rev0, pot_rev: reversible potentials
    """
    j = np.asarray(js, dtype=float)

    # constants - mostly based on textbook by O'Hayre - see report
    pot_rev0 = 1.229  # V - reversible potential at standard state for H2-O2 reaction - O'Hayre
    temp0 = 289.15  # K - reference temperature
    farad = 96485.3329  # C/mol - Faraday constant
    r_gas = 8.314  # J/K/mol - gas constant
    trans = 0.3  # transfer coefficient at cathode
    j_leak = 100.0  # A/m^2 - leakage current density
    res = 1e-6  # ohm/m^2 - area specific resistance
    mass_trans_c = 0.5  # V - mass transport loss constant
    j_lim = 2e4  # A/m^2 - limiting current density

    pres_air_atm = pres_air * 0.000009869233
    pres_h_atm = pres_h * 0.000009869233

    # Reversible potential at operating conditions (scalar)
    pot_rev = (
        pot_rev0
        - 44.34 / (2 * farad) * (cell_temp - temp0)
        + r_gas * cell_temp / (2 * farad) * np.log(pres_h_atm * np.sqrt(pres_air_atm * 0.21))
    )

    j_0 = 1.0  # O'Hayre (constant exchange current density)

    # Validity domain: j_lim - j - j_leak > 0
    denom = j_lim - j - j_leak
    valid = denom > 0

    volt_sl_loss = np.full_like(j, np.nan, dtype=float)
    if np.any(valid):
        jj = j[valid]
        denom_v = denom[valid]
        # Operational cell voltage including all losses (no altitude correction yet)
        a_act = r_gas * cell_temp / (2 * trans * farad)
        volt_v = (
            pot_rev
            - a_act * np.log((jj + j_leak) / j_0)
            - res * jj
            - mass_trans_c * np.log(j_lim / denom_v)
        )
        volt_v[volt_v < 0] = np.nan
        volt_sl_loss[valid] = volt_v

    # Apply altitude correction in case cathode inlet pressure is not sea level
    if not isclose(pres_air, 101325.0):
        pf = pres_air / 101325.0
        corr = -0.022830 * pf ** 4 + 0.230982 * pf ** 3 - 0.829603 * pf ** 2 + 1.291515 * pf + 0.329935
        volt = volt_sl_loss * corr
    else:
        volt = volt_sl_loss.copy()

    return volt, volt_sl_loss, pot_rev0, pot_rev


@lru_cache(maxsize=64)
def _cell_model_cached(pres_air, pres_h, cell_temp, oversizing):
    """Cached operating-point solution for `cell_model`.

    The operating point depends only on (pres_air, pres_h, cell_temp, oversizing). It is independent of stack power.
    Caching avoids recomputing the polarization curve scan in repeated system sizing calls.
    """
    js = np.arange(0, 20000, dtype=float)
    vs, vs_sl_loss, pot_rev0, pot_rev = _cell_voltage_arrays(js, pres_air, pres_h, cell_temp)
    ps = js * vs

    # Robust max search even with NaNs
    idx_max = int(np.nanargmax(ps))
    j_max = float(js[idx_max])
    j_op = (1.0 - float(oversizing)) * j_max

    # Compute voltage at chosen operating point using the scalar model (matches legacy behaviour)
    volt_cell, _, _, _ = cell_voltage(j_op, pres_air, pres_h, cell_temp)
    power_dens_cell = volt_cell * j_op
    eta_cell = volt_cell / 1.482  # HHV

    # Mark arrays as read-only to reduce risk of accidental in-place modification across cached calls
    js.setflags(write=False)
    vs.setflags(write=False)
    ps.setflags(write=False)
    vs_sl_loss.setflags(write=False)

    return volt_cell, power_dens_cell, eta_cell, js, vs, ps, j_op, pot_rev0, pot_rev, vs_sl_loss


def cell_model(pres_air, pres_h, cell_temp, oversizing=0.1, make_fig=True):
    """Find nominal operating point of cell.

    Compared to the original implementation, this version:
      - Uses a vectorized voltage evaluation (numpy) instead of 20k Python calls.
      - Caches results for repeated calls with identical operating conditions.
      - Allows disabling figure creation (Plotly) for speed.

    :param pres_air: Inlet air pressure in Pa
    :param pres_h: Inlet hydrogen pressure in Pa
    :param cell_temp: Operational temperature of cell in K
    :param oversizing: oversizing factor (e.g. 0.1 means max power is 10% above operating point)
    :param make_fig: if True, return a Plotly figure; otherwise return None for fig
    :returns:
        - volt_cell: cell voltage at chosen point in V
        - power_dens_cell: power density at chosen point in W/m^2
        - eta_cell: cell efficiency
        - fig: Plotly figure (or None if make_fig=False)
    """
    volt_cell, power_dens_cell, eta_cell, js, vs, ps, j_op, pot_rev0, pot_rev, vs_sl_loss = _cell_model_cached(
        float(pres_air), float(pres_h), float(cell_temp), float(oversizing)
    )

    fig = None
    if make_fig:
        fig = plot_polarization_curve(js, vs, ps, j_op, pot_rev0, pot_rev, vs_sl_loss)

    return volt_cell, power_dens_cell, eta_cell, fig


def cell_voltage(j, pres_air, pres_h, cell_temp):
    """Compute cell voltage from the current density and the pressure of the reactants.

    Kept as a scalar function for backwards compatibility.

    :param j: Current density in A/m^2
    :param pres_air: Inlet air pressure in Pa
    :param pres_h: Inlet hydrogen pressure in Pa
    :param cell_temp: Operational temperature of cell in K
    :returns:
        - volt: Cell voltage in V (after altitude correction)
        - pot_rev0: reversible potential at standard conditions
        - pot_rev: reversible potential at operating conditions
        - volt_sl_loss: cell voltage before altitude correction
    """
    # constants - mostly based on textbook by O'Hayre - see report
    pot_rev0 = 1.229  # V
    temp0 = 289.15  # K
    farad = 96485.3329  # C/mol
    r_gas = 8.314  # J/K/mol
    trans = 0.3
    j_leak = 100  # A/m^2
    res = 1e-6  # ohm/m^2
    mass_trans_c = 0.5  # V
    j_lim = 2e4  # A/m^2

    pres_air_atm, pres_h_atm = pres_air * 0.000009869233, pres_h * 0.000009869233

    pot_rev = pot_rev0 - 44.34 / (2 * farad) * (cell_temp - temp0) + r_gas * cell_temp / (2 * farad) * log(
        pres_h_atm * sqrt(pres_air_atm * 0.21)
    )

    j_0 = 1  # O'Hayre

    # Determine voltage
    if j_lim - j - j_leak > 0:
        volt = pot_rev - r_gas * cell_temp / (2 * trans * farad) * log((j + j_leak) / j_0) - res * j - \
               mass_trans_c * log(j_lim / (j_lim - j - j_leak))
    else:
        volt = np.nan

    if volt < 0:
        volt = np.nan

    volt_sl_loss = volt

    if not isclose(pres_air, 101325):
        pf = pres_air / 101325
        volt = volt * (
            -0.022830 * pf ** 4 + 0.230982 * pf ** 3 - 0.829603 * pf ** 2 + 1.291515 * pf + 0.329935
        )

    return volt, pot_rev0, pot_rev, volt_sl_loss


def plot_polarization_curve(js, vs, ps, j_op, pot_rev0, pot_rev, vs_sl_loss):
    """Plot polarization curve with design point."""
    import plotly.io as pio
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Avoid forcing a renderer here; let the calling environment decide.
    # pio.renderers.default = 'browser'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    j_max = js[np.nanargmax(ps)]

    fig.add_trace(go.Scatter(x=js, y=vs, name="Actual cell voltage"), secondary_y=False)
    fig.add_trace(go.Scatter(x=js, y=ps, name="Power density"), secondary_y=True)
    fig.add_trace(
        go.Scatter(
            x=js,
            y=vs_sl_loss,
            name="Cell voltage without effect of altitude on voltage losses",
            line=dict(color='blue', dash='dash'),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[j_op, j_op],
            y=[0.95 * np.nanmin(vs), 1.05 * pot_rev0],
            name="Nominal current density",
            mode='lines',
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[j_max, j_max],
            y=[0.95 * np.nanmin(vs), 1.05 * pot_rev0],
            name="Current density at max. power density",
            mode='lines',
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[float(np.nanmin(js)), float(np.nanmax(js))],
            y=[pot_rev0, pot_rev0],
            name="Reversible potential at standard conditions",
            mode='lines',
            line=dict(color='black', dash='dash'),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[float(np.nanmin(js)), float(np.nanmax(js))],
            y=[pot_rev, pot_rev],
            name="Reversible potential at operating conditions",
            mode='lines',
            line=dict(color='black'),
        ),
        secondary_y=False,
    )

    # Determine x-axis range (stop at first NaN if present)
    nan_idx = np.where(np.isnan(vs))[0]
    x_max = float(js[nan_idx[0]]) if nan_idx.size else float(np.nanmax(js))

    fig.update_xaxes(
        title_text="Current density in A/m<sup>2</sup>",
        tickformat=".5g",
        range=[0, x_max],
    )

    fig.update_yaxes(rangemode='tozero')
    fig.update_yaxes(
        range=[0, 1.05 * pot_rev0],
        title_text="Cell voltage in V",
        color="blue",
        title_font_color="blue",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=r"Power density in W/m<sup>2</sup>",
        color="red",
        title_font_color="red",
        secondary_y=True,
    )

    fig.update_layout(autosize=False, width=1600, height=800)
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        )
    )

    return fig


if __name__ == "__main__":
    # Minimal smoke test
    _, _, _, fig = cell_model(60000, 100000, 350)
    # fig.show()  # optional
