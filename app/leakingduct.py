#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from typing import List, Dict

import numpy as np

try:
    import pandas as pd
except ImportError:  # keep pandas optional
    pd = None


def _segment_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Return length of overlap between [a0,a1] and [b0,b1] (0 if none)."""
    left = max(a0, b0)
    right = min(a1, b1)
    return max(right - left, 0.0)


def compute_flow(
    x: np.ndarray,
    D: np.ndarray,
    leak_segments: List[Dict[str, float]],
    u0: float,
    rho: float = 1.225,
    P0: float = 101_325.0,
) -> "np.recarray":
    """
    Compute velocity, pressure, and mass‑flow‑rate distributions.

    Returns
    -------
    recarray with fields: x, A, u, P_static, q_dynamic, P_total, m_dot
    """
    if not (np.all(np.diff(x) > 0)):
        raise ValueError("Array x must be strictly increasing.")

    if x.shape != D.shape:
        raise ValueError("x and D must have the same length.")

    A = np.pi * (D / 2) ** 2  # area [m²]
    dx = np.diff(x)
    n = len(x)

    # Initial mass flow rate at inlet
    m_dot0 = rho * u0 * A[0]

    # Compute mass flow rate m_dot[i] at each station
    m_dot = np.empty(n)
    m_dot[0] = m_dot0

    # Pre-compute leak per unit length for each segment
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
        leak_this_interval = 0.0
        x_left = x[i - 1]
        x_right = x[i]

        # sum contributions of every leak segment overlapping [x_left,x_right]
        for seg in leak_info:
            overlap = _segment_overlap(
                x_left, x_right, seg["start"], seg["end"]
            )
            if overlap > 0:
                leak_this_interval += seg["rate_per_x"] * overlap

        cumulative_leak += leak_this_interval
        m_dot[i] = m_dot0 - cumulative_leak

    # Velocity
    u = m_dot / (rho * A)

    # Static pressure via finite‑difference integration of dP = -ρ u du
    P = np.empty(n)
    P[0] = P0
    for i in range(1, n):
        du = u[i] - u[i - 1]
        P[i] = P[i - 1] - rho * u[i] * du  # midpoint approximation

    q = 0.5 * rho * u**2
    P_total = P + q

    dtype = [
        ("x", float),
        ("A", float),
        ("u", float),
        ("P_static", float),
        ("q_dynamic", float),
        ("P_total", float),
        ("m_dot", float),
    ]
    res = np.rec.fromarrays(
        [x, A, u, P, q, P_total, m_dot], dtype=dtype
    )
    return res


def main():
    parser = argparse.ArgumentParser(
        description="1‑D variable‑area duct flow with leakage (incompressible)."
    )
    parser.add_argument(
        "input_file", help="JSON file with input definition (see docstring)."
    )
    parser.add_argument(
        "--csv", metavar="FILE",
        help="Write results to CSV file (optional)."
    )

    args = parser.parse_args()
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    x = np.asarray(data["x"], dtype=float)
    D = np.asarray(data["D"], dtype=float)
    u0 = float(data["u0"])
    rho = float(data.get("rho", 1.225))
    P0 = float(data.get("P0", 101_325.0))
    leak_segments = data.get("leak_segments", [])

    result = compute_flow(x, D, leak_segments, u0, rho=rho, P0=P0)

    # Pretty print
    header = (
        "    x [m]    |    u [m/s]   | m_dot [kg/s] |  P_static [Pa] |"
        " q_dynamic [Pa] |  P_total [Pa] "
    )
    print(header)
    print("-" * len(header))
    for r in result:
        print(
            f"{r.x:11.4f} | {r.u:11.4f} | {r.m_dot:13.5f} |"
            f" {r.P_static:14.2f} | {r.q_dynamic:14.2f} | {r.P_total:12.2f}"
        )

    if args.csv:
        if pd is None:
            raise RuntimeError("pandas is required to export CSV.")
        df = pd.DataFrame.from_records(result)
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
