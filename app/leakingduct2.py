#!/usr/bin/env python3
"""

Inputs
------
* **x** (array) : 위치 [m] (단조 증가)
* **D** (array) : 각 위치의 원형 직경 [m]
* **u0** (float) : 입구 평균 유속 [m/s]
* **rho** (float, opt) : 밀도 [kg/m³] (기본 1.225)
* **P0** (float, opt)  : 입구 정압 [Pa] (기본 101 325)
* **leak_segments** (list, opt) : `{start,end,m_dot}` 누설 정의 (기존)
* **pressure_losses** (list, opt) : `{"x": pos, "dP": loss}` 리스트
  * **x** : 손실 발생 위치 [m] (반드시 `x` 배열 값 중 하나와 일치)
  * **dP**: 손실 크기 [Pa] (양수 ⇒ 정압 강하)
  * 여러 손실이 동일 위치에 있을 경우 합산

Outputs
-------
`numpy.recarray` 필드
`x, A, u, m_dot, P_static, q_dynamic, P_total`

CLI 예시
--------
```json
{
  "x": [0.0, 0.5, 1.0, 1.5],
  "D": [0.05, 0.05, 0.04, 0.03],
  "u0": 15,
  "rho": 1.18,
  "P0": 101325,
  "leak_segments": [
    {"start": 0.7, "end": 1.2, "m_dot": 0.02}
  ],
  "pressure_losses": [
    {"x": 1.0, "dP": 200.0},
    {"x": 1.5, "dP": 500.0}
  ]
}
```
Run:
```bash
python duct_leak_solver.py input.json --csv result.csv
```

"""

from __future__ import annotations
import argparse
import json
from typing import List, Dict

import numpy as np

try:
    import pandas as pd
except ImportError:  # optional
    pd = None

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _segment_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return length of overlap between [a0,a1] and [b0,b1] (0 if none)."""
    left = max(a0, b0)
    right = min(a1, b1)
    return max(right - left, 0.0)

# ------------------------------------------------------------------
# Core solver
# ------------------------------------------------------------------

def compute_flow(
    x: np.ndarray,
    D: np.ndarray,
    leak_segments: List[Dict[str, float]] | None = None,
    pressure_losses: List[Dict[str, float]] | None = None,
    *,
    u0: float,
    rho: float = 1.225,
    P0: float = 101_325.0,
) -> "np.recarray":
    """Compute velocity, pressure, and mass‑flow distributions.

    Parameters
    ----------
    x, D, u0, rho, P0 : see module docstring
    leak_segments     : list of leakage dicts (may be empty or None)
    pressure_losses   : list of `{x, dP}` dicts (Pa, positive = drop)

    Returns
    -------
    recarray with fields `x, A, u, m_dot, P_static, q_dynamic, P_total`
    """

    # Validation ------------------------------------------------------
    if not (np.all(np.diff(x) > 0)):
        raise ValueError("x must be strictly increasing.")
    if x.shape != D.shape:
        raise ValueError("x and D must have same length.")

    leak_segments = leak_segments or []
    pressure_losses = pressure_losses or []

    # Area, spacing ---------------------------------------------------
    A = np.pi * (D / 2) ** 2  # m²
    dx = np.diff(x)
    n = len(x)

    # --------------------------------------------------
    # 1) Mass‑flow accounting (leakage)
    # --------------------------------------------------
    m_dot = np.empty(n)
    m_dot[0] = rho * u0 * A[0]

    leak_info = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "rate_per_x": seg["m_dot"] / (seg["end"] - seg["start"]),
        }
        for seg in leak_segments
    ]

    cumulative_leak = 0.0
    for i in range(1, n):
        x_left, x_right = x[i - 1], x[i]
        leak_interval = sum(
            seg["rate_per_x"] * _segment_overlap(x_left, x_right, seg["start"], seg["end"])
            for seg in leak_info
        )
        cumulative_leak += leak_interval
        m_dot[i] = m_dot[0] - cumulative_leak

    # --------------------------------------------------
    # 2) Velocity field
    # --------------------------------------------------
    u = m_dot / (rho * A)

    # --------------------------------------------------
    # 3) Static‑pressure integration + local losses
    # --------------------------------------------------
    # Loss dict keyed by position (exact match within tol)
    loss_map: dict[float, float] = {}
    for pl in pressure_losses:
        pos = float(pl["x"])
        loss_map[pos] = loss_map.get(pos, 0.0) + float(pl["dP"])

    tol = 10**(-9)  # positional tolerance
    P = np.empty(n)
    P[0] = P0

    for i in range(1, n):
        du = u[i] - u[i - 1]
        P[i] = P[i - 1] - rho * u[i] * du  # momentum eq.
        # Check if current x matches a loss location
        for pos, dP in loss_map.items():
            if abs(x[i] - pos) < tol:
                P[i] -= dP  # apply drop

    # --------------------------------------------------
    # 4) Dynamic & total pressures
    # --------------------------------------------------
    q = 0.5 * rho * u ** 2
    P_total = P + q

    dtype = [
        ("x", float),
        ("A", float),
        ("u", float),
        ("m_dot", float),
        ("P_static", float),
        ("q_dynamic", float),
        ("P_total", float),
    ]
    return np.rec.fromarrays([x, A, u, m_dot, P, q, P_total], dtype=dtype)

# ------------------------------------------------------------------
# CLI wrapper
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="1‑D variable‑area duct flow with leakage, local pressure losses (incompressible)."
    )
    parser.add_argument("input_file", help="JSON input definition (see docs).")
    parser.add_argument("--csv", metavar="FILE", help="Save results to CSV.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf‑8") as f:
        data = json.load(f)

    x = np.asarray(data["x"], dtype=float)
    D = np.asarray(data["D"], dtype=float)
    u0 = float(data["u0"])
    rho = float(data.get("rho", 1.225))
    P0 = float(data.get("P0", 101_325.0))
    leak_segments = data.get("leak_segments", [])
    pressure_losses = data.get("pressure_losses", [])

    res = compute_flow(x, D, leak_segments, pressure_losses, u0=u0, rho=rho, P0=P0)

    # Pretty print ----------------------------------------------------
    hdr = (
        "    x [m]    |   u [m/s]  |  m_dot [kg/s] |  P_static [Pa] |"
        " q_dynamic [Pa] |  P_total [Pa] "
    )
    print(hdr)
    print("‑" * len(hdr))
    for r in res:
        print(
            f"{r.x:11.4f} | {r.u:10.4f} | {r.m_dot:13.5f} |"
            f" {r.P_static:14.2f} | {r.q_dynamic:14.2f} | {r.P_total:12.2f}"
        )

    if args.csv:
        if pd is None:
            raise RuntimeError("pandas required for CSV export.")
        pd.DataFrame.from_records(res).to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
