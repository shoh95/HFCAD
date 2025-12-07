#!/usr/bin/env python3
"""

----------------------------------------------------------------------
**입력**
```jsonc
{
  "x": [0.0, 0.4, 0.8, 1.2, 1.6],        // 위치 배열 (단조 증가)
  "D": [0.05, 0.05, 0.04, 0.04, 0.035],  // 직경 배열 (m)
  "u0": 12.0,                             // 입구 평균 유속 (m/s)
  "rho": 1.20,                            // 밀도 (kg/m³) – optional
  "P0": 101325,                           // 입구 정압 (Pa) – optional
  "leak_segments": [                      // ✔ 다중 구간 지원
    {"start": 0.5, "end": 0.9, "m_dot": 0.015},
    {"start": 1.1, "end": 1.4, "m_dot": 0.010}
  ],
  "pressure_losses": [                    // optional, 다중 손실 지원
    {"x": 0.8, "dP": 150},
    {"x": 1.6, "dP": 300}
  ]
}
```

----------------------------------------------------------------------
**출력** (numpy `recarray` + 터미널 + CSV)
`x, A, u, m_dot, P_static, q_dynamic, P_total`

"""
from __future__ import annotations
import argparse
import json
from typing import List, Dict

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def _segment_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return length of overlap between [a0,a1] and [b0,b1] (0‑length if none)."""
    left = max(a0, b0)
    right = min(a1, b1)
    return max(right - left, 0.0)

# -------------------------------------------------------------------
# Core solver
# -------------------------------------------------------------------

def compute_flow(
    x: np.ndarray,
    D: np.ndarray,
    *,
    u0: float,
    rho: float = 1.225,
    P0: float = 101_325.0,
    leak_segments: List[Dict[str, float]] | None = None,
    pressure_losses: List[Dict[str, float]] | None = None,
) -> "np.recarray":
    """1‑D incompressible solver with multiple leakages & local losses."""
    # --- validation -------------------------------------------------
    if not (np.all(np.diff(x) > 0)):
        raise ValueError("'x' must be strictly increasing.")
    if x.shape != D.shape:
        raise ValueError("'x' and 'D' lengths differ.")

    leak_segments = leak_segments or []
    pressure_losses = pressure_losses or []

    # Sanity‑check leak segments
    for seg in leak_segments:
        if seg["end"] <= seg["start"]:
            raise ValueError(f"Leak segment start >= end: {seg}")
        if seg["start"] < x[0] or seg["end"] > x[-1]:
            raise ValueError(f"Leak segment out of bounds: {seg}")

    # area & spacing --------------------------------------------------
    A = np.pi * (D / 2) ** 2
    dx = np.diff(x)
    n = len(x)

    # ------------------------------------------------ mass‑flow ------
    m_dot = np.empty(n)
    m_dot[0] = rho * u0 * A[0]

    # prepare leak‑rate per‑length list (multiple segments)
    leak_info = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "rate_per_x": float(seg["m_dot"]) / (float(seg["end"]) - float(seg["start"]))
        }
        for seg in leak_segments
    ]

    cumulative_leak = 0.0
    for i in range(1, n):
        x_left, x_right = x[i - 1], x[i]
        leak_this = sum(
            li["rate_per_x"] * _segment_overlap(x_left, x_right, li["start"], li["end"])
            for li in leak_info
        )
        cumulative_leak += leak_this
        m_dot[i] = m_dot[0] - cumulative_leak

    # ------------------------------------------------ velocity -------
    u = m_dot / (rho * A)

    # ------------------------------------------------ pressure -------
    loss_map: dict[float, float] = {}
    for pl in pressure_losses:
        loc = float(pl["x"])
        loss_map[loc] = loss_map.get(loc, 0.0) + float(pl["dP"])

    tol = 10**(-9)
    P = np.empty(n)
    P[0] = P0
    for i in range(1, n):
        du = u[i] - u[i - 1]
        P[i] = P[i - 1] - rho * u[i] * du  # momentum eq.
        if any(abs(x[i] - loc) < tol for loc in loss_map):
            P[i] -= loss_map[x[i]]  # apply aggregated loss

    q = 0.5 * rho * u**2
    P_total = P + q

    dtype = [
        ("x", float), ("A", float), ("u", float), ("m_dot", float),
        ("P_static", float), ("q_dynamic", float), ("P_total", float)
    ]
    return np.rec.fromarrays([x, A, u, m_dot, P, q, P_total], dtype=dtype)

# -------------------------------------------------------------------
# CLI helper
# -------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Duct Leak Solver (rev‑3)")
    p.add_argument("input", help="Input JSON file path")
    p.add_argument("--csv", metavar="FILE", help="Output CSV path (optional)")
    a = p.parse_args()

    with open(a.input, "r", encoding="utf‑8") as f:
        data = json.load(f)
    res = compute_flow(
        np.asarray(data["x"], float),
        np.asarray(data["D"], float),
        u0=float(data["u0"]),
        rho=float(data.get("rho", 1.225)),
        P0=float(data.get("P0", 101_325.0)),
        leak_segments=data.get("leak_segments", []),
        pressure_losses=data.get("pressure_losses", [])
    )

    hdr = (
        "    x [m]   |  u [m/s] | ṁ [kg/s] | P_static [Pa] | q_dyn [Pa] | P_total [Pa] "
    )
    print(hdr)
    print("-" * len(hdr))
    for r in res:
        print(f"{r.x:10.3f} | {r.u:9.3f} | {r.m_dot:9.5f} | {r.P_static:13.2f} | {r.q_dynamic:10.2f} | {r.P_total:12.2f}")

    if a.csv:
        if pd is None:
            raise RuntimeError("pandas required for CSV export")
        pd.DataFrame.from_records(res).to_csv(a.csv, index=False)
        print(f"\nCSV saved → {a.csv}")

if __name__ == "__main__":  # pragma: no cover
    _cli()
