"""AD Inverse Input file"""

"""Aircraft Name: DURUMI"""

import math

# import XFOIL_SH
import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import xmltodict

import AD_inverse_Class

# from tabulate import tabulate

AD_velocity = AD_inverse_Class.Velocity()

"""nacafoil = '4415'
airfoilname = 'GA(W)-2'
airfoildat = 'GAW-2a.dat'
alpha_start = 0
alpha_end = 10
delta_alpha = 1
v_cruise_s = AD_velocity.km1h_m1s(90, 'm1s')
ReynoldsNum = 6.0e5

XFOIL_DATA = XFOIL_SH.XFOILDATA(alpha_start, alpha_end, delta_alpha, v_cruise_s, ReynoldsNum, nacafoil, airfoilname, airfoildat)
alpha_xfoil = XFOIL_DATA['alpha']
CL_xfoil = XFOIL_DATA['CL']
CD_xfoil = XFOIL_DATA['CD']
CM_xfoil = XFOIL_DATA['CM']

a_2d_degrees = (CL_xfoil[-1] - CL_xfoil[0]) / (alpha_xfoil[-1] - alpha_xfoil[0])
a_2d = math.degrees(a_2d_degrees)
alpha_0_degrees = -CL_xfoil[0]/a_2d_degrees

'''AD_xfoil = AD_inverse_Class.Xfoil()
alpha_idx = AD_xfoil.find_alpha_loc(XFOIL_DATA['alpha'], 0)
print(f"{CL_xfoil[alpha_idx]}")'''
"""

name = "RIMP-02"
print("\n\n\nCode Builder: [Seunghwan Oh]")
print("Aircraft Name: " + "[" + name + "]")


print("\n-----------Initial Sizing-----------")
print("************************************")
print("************************************")

WS_initial = 110  # g/dm2, Wing loading Guess
WCL_initial = 16  # kg/m3, WCL_metric = WCL_imperical * 1.00117
S_initial = (WCL_initial / (WS_initial / 10)) ** (-2)
W_initial = (WS_initial / 10) * (WCL_initial / (WS_initial / 10)) ** (-2)
A_initial = 7
b_initial = (A_initial * S_initial) ** 0.5
Cref_initial = S_initial / b_initial

print(f"\n\nW_guess: {W_initial*1000:,.0f} g")
print(f"b_guess: {b_initial*1000:,.0f} mm")
print(f"Cref_guess: {Cref_initial*1000:,.0f} mm")
print(f"\nS_guess: {S_initial:,.4f} m^2")
print(f"AR_guess: {A_initial:,.1f}")
print(f"W/S: {WS_initial:,.0f} g/dm2")
print(f"WCL: {WCL_initial:,.1f}")
print("\n************************************")
print("************************************")


W = 8600  # Aircraft Weight, kg
# S_initial = (W/WCL_initial)**(2/3)
b = 17.25  # Wing span, m
# A = b**2 / S_initial
A = 10  # Wing Aspect Ratio
lamda = 0.42  # Lambda
dihedral = 0.0  # deg, dihedral angle
Awetted = 2.1  # Swet = (Swet_w + Swet_fus + Swet_HT + Swet_VT) = 18.4776+13.556+3.2093+2.3497
L_FL = 21750/17250  # Fuselage length coefficient
Cx = 0  # Wing_root and Wing_tip Offset
wingpos_w = "rand"  # Wing Position
theta_LE = math.radians(5)  # deg, LE Sweep
wingpos_HT = "rand"
wingpos_VT = "rand"

CLWa = 5.122  # per rad, CLa of Wing by AVL probably, with HT effect included // Airfoil: E562


lamda_len = math.tan(theta_LE) * b/2


delta4_main_wing = AD_inverse_Class.Config(b, A, lamda)
[S, WS, WS_N, WS_imperical, WCL, WCL_imperical, S_short, WS_short, WS_imperical_short, WCL_imperical_short] = delta4_main_wing.wing(W, dihedral)
print(f"\nW: {W:,.1f} kg")
print("\n-----------Main Wing-----------")
print("S: " + f"{S:,.4f}" + " m^2")
print("W/S: " + f"{WS:,.2f}" + " kg/m^2")
print("W/S: " + f"{WS*10:,.0f}" + " g/dm^2")
# print("W/S_imp: " + f"{WS_imperical:,.2f}" + " psf")
print(f"W/S_imp: {WS_imperical*16:,.1f} oz/in^2")
print("WCL_imp: " + f"{WCL_imperical_short:,.1f}")

[Croot, Croot_short, Ctip, Ctip_short, Cbar, Cbar_short, Ybar, Ybar_short] = (
    delta4_main_wing.wingchord(S)
)
print("\nCroot: " + f"{Croot:,.3f}" + " m")
print("Ctip: " + f"{Ctip:,.3f}" + " m")
print(f"AR: {A:,.2f}")
print("Cbar: " + f"{Cbar:,.3f}" + " m")
print("Ybar: " + f"{Ybar:,.3f}" + " m")

[FL, FL_short] = delta4_main_wing.fuselage(L_FL)
print("\nFuselage Length: " + f"{FL:,.3f}" + " m")


LL_HT = 0.49380955          # L_HT/Fuselage_Length
LL_VT = 0.48267689          # L_VT/Fuselage_Length

# C_HT = 0.544					#Horizontal Tail Coefficient
C_HT = 0.9  # Horizontal Tail Coefficient
C_VT = 0.09  # Vertical Tail Coefficient

L_wing = 8.7  # Fuselage nose tip to wing root tip(sugested: FL*0.35 ,, FL/3)

A_HT = 2 * A / 3  # Horizontal Tail Aspect Ratio (proposed)
if A_HT < 3.5:
    A_HT = 3.5
elif A_HT > 4.5:
    A_HT = 4.5

A_HT = 5.0  # Horizontal Tail Aspect Ratio
lamda_HT = 0.4  # Horizontal Tail lambda
theta_LE_HT = math.radians(10)


A_VT = 1.92
lamda_VT = 0.4
theta_LE_VT = math.radians(25)

AR_Vtail = 4.4  # Vtail Aspect Ratio

X_Fuse_mm = 0

delta_HS_cap_length_mm = 300+300
delta_VS_cap_length_mm = 300
delta_Vtail_cap_length_mm = 50
X_true_Vtail_mm = 1300

# AC = 0.235
# AC = 1.3341734465E-1 + 7.21347520444E-3*math.log(4.430E+6, math.e)
# AC = 0.245  # NACA 4412
AC = 0.25
# AC = 0.239

delta4_tail = AD_inverse_Class.TailCompu(LL_HT, LL_VT, C_HT, C_VT, L_wing, A_HT, lamda_HT, A_VT, lamda_VT, X_Fuse_mm, AC, Ybar)

[L_HT, L_HT_short, L_VT, L_VT_short, L_wing_short, S_HT, S_HT_short, S_VT, S_VT_short, b_HT, b_HT_short, Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short, b_VT, b_VT_short, Croot_VT, Croot_VT_short, Ctip_VT, Ctip_VT_short, Cbar_VT, Cbar_VT_short, Ybar_VT, Ybar_VT_short, X_wing_mm, X_HT_mm, X_VT_mm, X_true_wing_mm, X_true_HT_mm, X_true_VT_mm, L_AC_wing, L_AC_wing_mm, L_AC_HT, L_AC_HT_mm, L_AC_VT, L_AC_VT_mm, X_AC_wing_mm, X_AC_HT_mm, X_AC_VT_mm, L_HT_act_mm, L_VT_act_mm, L_HT_act, L_VT_act, L_HT_act_short, L_VT_act_short, C_HT_act, C_HT_act_short, C_VT_act, C_VT_act_short, L_AC_HT_act, L_AC_VT_act, L_AC_HT_act_short, L_AC_VT_act_short, delta_HT, delta_VT, delta_HT_short, delta_VT_short, LL_HT_rcmd, LL_VT_rcmd, LL_HT_rcmd_short, LL_VT_rcmd_short, X_true_HT_mm_short, X_true_VT_mm_short] =\
      delta4_tail.tailsizingtwo(LL_HT, LL_VT, FL, S, b, Croot, Cbar, Ybar, Cx, wingpos_w, delta_HS_cap_length_mm, delta_VS_cap_length_mm, Ctip, A, lamda, theta_LE, theta_LE_HT, theta_LE_VT)
# print("\nS_HT: " + str(S_HT_short) + " m^2")
print(f"S_HT: {S_HT:,.4f} m^2")
# print("S_VT: " + str(S_VT_short) + " m^2")
print(f"S_HT: {S_VT:,.4f} m^2")
print("L_HT: " + str(L_HT_short) + " m")
print("L_VT: " + str(L_VT_short) + " m")
print("L_wing: " + str(L_wing_short) + " m")

print("\n------Horizontal Stabilizer------")
print("b_HT: " + str(b_HT_short) + " m")
print("Croot_HT: " + str(Croot_HT_short) + " m")
print("Ctip_HT: " + str(Ctip_HT_short) + " m")
print(f"AR_HT: {A_HT:,.3f}")
print("Cbar_HT: " + str(Cbar_HT_short) + " m")

print("\n------Vertical Stabilizer------")
print("b_VT: " + str(b_VT_short) + " m")
print("Croot_VT: " + str(Croot_VT_short) + " m")
print("Ctip_VT: " + str(Ctip_VT_short) + " m")
print("Cbar_VT: " + str(Cbar_VT_short) + " m")

print("\n------Tail Moment Arms------")
print("L_HT_act: " + str(L_HT_act_short) + " m")
print("L_VT_act: " + str(L_VT_act) + " m")
print("C_HT_act: " + str(C_HT_act_short))
print("C_VT_act: " + str(C_VT_act_short))

print("\n------Tail Actual Position-----")
print("X_true_HT: " + str(X_true_HT_mm_short) + " mm")
print("X_true_VT: " + str(X_true_VT_mm_short) + " mm")

print("\n-----Tail Actual Moment Arms-----")
print("L_AC_HT_act: " + str(L_AC_HT_act_short) + " m")
print("L_AC_VT_act: " + str(L_AC_VT_act_short) + " m")
print("delta_HT_short: " + str(delta_HT_short))
print("delta_VT_short: " + str(delta_VT_short))
print("Recommanded_LL_HT: " + f"{LL_HT_rcmd:,.8f}")
print("Recommanded_LL_VT: " + f"{LL_VT_rcmd:,.8f}")


lamda_len_HT = math.tan(theta_LE_HT)*b_HT/2
lamda_len_VT = math.tan(theta_LE_VT)*b_VT/2

# [
#     S_Vtail,
#     phi_Vtail,
#     b_Vtail,
#     D_Vtail,
#     C_Vtail,
#     E_Vtail,
#     C_vtail_h,
#     C_vtail_v,
#     X_ac_vtail_ht,
#     S_vtail_ht,
#     Croot_vtail_ht,
#     L_AC_vtail_HT_act,
# ] = delta4_tail.vtail(
#     AR_Vtail, S_HT, S_VT, X_true_Vtail_mm, Croot, AC, L_wing, S, Cbar, b
# )
#
# print("\n================  V-Tail  ================")
# print(f"S_Vtail: {S_Vtail:,.4f} m^2")
# print(f"phi_Vtail: {math.degrees(phi_Vtail):,.1f} ")
# print(f"b_Vtail/2: {b_Vtail/2:,.6f} m   << AVL Ycomp")
# print(f"D_Vtail: {D_Vtail:,.6f} m     << AVL Zcomp")
# print(f"\nC_Vtail: {C_Vtail:,.3f} m   << FLOW5 chord")
# print(f"E_Vtail: {E_Vtail:,.3f} m   << FLOW5  Ypos")
# print(f"\nC_vtail_H: {C_vtail_h:,.3f}")
# print(f"C_vtail_V: {C_vtail_v:,.3f}")
# print("================  ======  ================")


delta4_atm = AD_inverse_Class.Atmosphere()
AD_aero = AD_inverse_Class.Aerodynamics()
delta4_perf = AD_inverse_Class.Performance()

"""Performance DATA"""
rho_a = 1.225  # Atmospheric Density, kg/m^3
g = 9.80665
tw = 0.254  # Thrust to Weight Ratio

# Stall Condition(stall)
h_stall_roskam = 0  # Test:: 375m ~ 2000m Range
# h_stall_roskam = 2000              #Test:: 375m ~ 2000m Range
h_stall_TO = 0
h_stall_cruise = 20
h_stall_cruise2 = 40

# CN_max_roskam = 1.1*1.977                            #Jan Roskam "AD Part5, p.32": CN_max = CL_max*1.1
CN_max_roskam = 1.1 * (1.53 + 0.637)  # (CLmax + CL_flap_increment) = 2.167
CL_stall_TO = 1.2
# CL_stall_cruise = 0.285
CL_stall_cruise = 2.0
CL_stall_cruise2 = 2.0
CL_stall_landing = 2.2

W_landing_stall = 13  # Wo - W_fuel_est + W_reserve

CD_stall_TO = 0.65
CD_stall_cruise = 0.045
CD_stall_cruise2 = 0.04
# v_knot_stall = 42
# v_stall = v_knot_stall*0.5144				#Flight vehicle stall speed, m/s /22m/s, CL 1.93, CD 0.65

[
    Ta_stall_roskam,
    Pa_stall_roskam,
    rho_a_stall_roskam,
    Va_stall_roskam,
    g_stall_roskam,
] = delta4_atm.atmdata(h_stall_roskam)
[Ta_stall_TO, Pa_stall_TO, rho_a_stall_TO, Va_stall_TO, g_stall_TO] = (
    delta4_atm.atmdata(h_stall_TO)
)
[
    Ta_stall_cruise,
    Pa_stall_cruise,
    rho_a_stall_cruise,
    Va_stall_cruise,
    g_stall_cruise,
] = delta4_atm.atmdata(h_stall_cruise)
[
    Ta_stall_cruise2,
    Pa_stall_cruise2,
    rho_a_stall_cruise2,
    Va_stall_cruise2,
    g_stall_cruise2,
] = delta4_atm.atmdata(h_stall_cruise2)
[v_stall_roskam, v_stall_roskam_knot, v_stall_roskam_kmh] = delta4_perf.stallcond(
    W, rho_a_stall_roskam, S, CN_max_roskam, g_stall_roskam
)
[v_stall_TO, v_stall_TO_knot, v_stall_TO_kmh] = delta4_perf.stallcond(
    W, rho_a_stall_TO, S, CL_stall_TO, g_stall_TO
)
[v_stall_cruise, v_stall_cruise_knot, v_stall_cruise_kmh] = delta4_perf.stallcond(
    W, rho_a_stall_cruise, S, CL_stall_cruise, g_stall_cruise
)
[v_stall_cruise2, v_stall_cruise2_knot, v_stall_cruise2_kmh] = delta4_perf.stallcond(
    W, rho_a_stall_cruise2, S, CL_stall_cruise2, g_stall_cruise2
)
[v_stall_landing, v_stall_landing_knot, v_stall_landing_kmh] = delta4_perf.stallcond(
    W_landing_stall, rho_a_stall_TO, S, CL_stall_TO, g_stall_TO
)

# @ 7 deg
h_TO = 375
CL_TO = 1.84
CD_TO = 0.65
# CD_v_TO = 0.01890
# CD_i_TO = 0.05031
v_knot_TO = 45
v_TO = AD_velocity.m1s_kn(v_knot_TO, "m1s")  # Flight vehicle Take-off speed, m/s,
CDo = 0.03125
# CDo = 0.06

# Take Off with flaps(TOF)
alpha_TOF = 4.4
h_TOF = 0
# CL_TOF = 1.85
CL_TOF = 1.2
CD_TOF = 0.0557 + 0.0398 * CL_TOF**2
# v_knot_TOF = AD_velocity.m1s_kn(AD_velocity.km1h_m1s(37, 'm1s'), 'kn')
v_knot_TOF = AD_velocity.km1h_kn(45, "kn")
v_TOF = AD_velocity.m1s_kn(
    v_knot_TOF, "m1s"
)  # Flight vehicle Take-off speed, m/s \\22m/s, CL 1.93, CD 0.65

# Cruise
alpha_cruise = 6.0
h_cruise = 25
# CL_cruise = 0.4
CL_cruise = AD_aero.LLTgeneral(S, b, Croot, Ctip, alpha_cruise) * 0.82
# CL_cruise = XFOIL_DATA["CL"][3]/2
# CL_cruise = AD_aero.liftinglinetheory(lamda, S, A, a_2d, alpha_0_degrees, alpha_cruise) * 0.88
# e = 1.78*(1 - 0.045*A**0.68) - 0.64
e = 0.7919
K = 1 / (np.pi * e * A)
CD_cruise = CDo + K * CL_cruise**2
# CD_cruise = 0.057
# CD_cruise = 0.0237+0.0374*CL_cruise**2
# CD_cruise = 0.057
v_knot_cruise = AD_velocity.m1s_kn(AD_velocity.km1h_m1s(400, "m1s"), "kn")
v_cruise = AD_velocity.m1s_kn(v_knot_cruise, "m1s")  # 60 knot
Pav_cruise = 2.41  # available Power, 550hp

# Cruise2
alpha_cruise2 = 2.5
h_cruise2 = 50  # 3000m
# CL_cruise2 = 0.3
CL_cruise2 = AD_aero.LLTgeneral(S, b, Croot, Ctip, alpha_cruise2) * 0.82
# CL_cruise2 = XFOIL_DATA["CL"][2]/2
# CL_cruise2 = AD_aero.liftinglinetheory(lamda, S, A, a_2d, alpha_0_degrees, alpha_cruise2) * 0.8
CD_cruise2 = 0.0237 + 0.0374 * CL_cruise2**2
# CD_cruise2 = 0.05
v_knot_cruise2 = AD_velocity.m1s_kn(AD_velocity.km1h_m1s(400, "m1s"), "kn")
v_cruise2 = AD_velocity.m1s_kn(
    v_knot_cruise2, "m1s"
)  # 130 knot//||\\Never Exceed Speed: 135 knot

# Landing
h_landing = 0
# CL_landing = 1.85
CL_landing = 1.80
# CL_landing = XFOIL_DATA["CL"][8]
CD_landing = 0.1007 + 0.0424 * CL_landing**2
v_knot_landing = AD_velocity.m1s_kn(AD_velocity.km1h_m1s(180, "m1s"), "kn")
v_landing = AD_velocity.m1s_kn(
    v_knot_landing, "m1s"
)  # Flight vehicle Take-off speed, m/s \\22m/s, CL 1.93, CD 0.65

CD_min = 0.06

delta4_atm = AD_inverse_Class.Atmosphere()
[Ta_TO, Pa_TO, rho_a_TO, Va_TO, g_TO] = delta4_atm.atmdata(h_TO)
[Ta_TOF, Pa_TOF, rho_a_TOF, Va_TOF, g_TOF] = delta4_atm.atmdata(h_TOF)
[Ta_cruise, Pa_cruise, rho_a_cruise, Va_cruise, g_cruise] = delta4_atm.atmdata(h_cruise)
[Ta_cruise2, Pa_cruise2, rho_a_cruise2, Va_cruise2, g_cruise2] = delta4_atm.atmdata(
    h_cruise2
)
[Ta_landing, Pa_landing, rho_a_landing, Va_landing, g_landing] = delta4_atm.atmdata(
    h_landing
)

delta4_perf = AD_inverse_Class.Performance()

[
    L_TO,
    L_TO_kg,
    L_TO_kg_short,
    D_TO,
    D_TO_kg,
    D_TO_kg_short,
    v_TO_kmh,
    v_TO_kmh_short,
    v_kn_TO,
    v_kn_TO_short,
    Mach_TO,
    Mach_TO_short,
    mu_TO,
    Re_W_TO,
    Re_W_TO_short,
] = delta4_perf.liftanddrag(S, CL_TO, CD_TO, v_TO, rho_a_TO, g_TO, Va_TO, Ta_TO, Cbar)
[
    L_TOF,
    L_TOF_kg,
    L_TOF_kg_short,
    D_TOF,
    D_TOF_kg,
    D_TOF_kg_short,
    v_TOF_kmh,
    v_TOF_kmh_short,
    v_kn_TOF,
    v_kn_TOF_short,
    Mach_TOF,
    Mach_TOF_short,
    mu_TOF,
    Re_W_TOF,
    Re_W_TOF_short,
] = delta4_perf.liftanddrag(
    S, CL_TOF, CD_TOF, v_TOF, rho_a_TOF, g_TOF, Va_TOF, Ta_TOF, Cbar
)
[
    L_cruise,
    L_cruise_kg,
    L_cruise_kg_short,
    D_cruise,
    D_cruise_kg,
    D_cruise_kg_short,
    v_cruise_kmh,
    v_cruise_kmh_short,
    v_kn_cruise,
    v_kn_cruise_short,
    Mach_cruise,
    Mach_cruise_short,
    mu_cruise,
    Re_W_cruise,
    Re_W_cruise_short,
] = delta4_perf.liftanddrag(
    S,
    CL_cruise,
    CD_cruise,
    v_cruise,
    rho_a_cruise,
    g_cruise,
    Va_cruise,
    Ta_cruise,
    Cbar,
)
# [L_cruise, L_cruise_kg, L_cruise_kg_short, D_cruise, D_cruise_kg, D_cruise_kg_short, v_cruise_kmh, v_cruise_kmh_short] = delta4_perf.liftanddrag(S, CL_cruise, CD_cruise, v_cruise, rho_a_cruise, g)
[
    L_cruise2,
    L_cruise2_kg,
    L_cruise2_kg_short,
    D_cruise2,
    D_cruise2_kg,
    D_cruise2_kg_short,
    v_cruise2_kmh,
    v_cruise2_kmh_short,
    v_kn_cruise2,
    v_kn_cruise2_short,
    Mach_cruise2,
    Mach_cruise2_short,
    mu_cruise2,
    Re_W_cruise2,
    Re_W_cruise2_short,
] = delta4_perf.liftanddrag(
    S,
    CL_cruise2,
    CD_cruise2,
    v_cruise2,
    rho_a_cruise2,
    g_cruise2,
    Va_cruise2,
    Ta_cruise2,
    Cbar,
)
[
    L_landing,
    L_landing_kg,
    L_landing_kg_short,
    D_landing,
    D_landing_kg,
    D_landing_kg_short,
    v_landing_kmh,
    v_landing_kmh_short,
    v_kn_landing,
    v_kn_landing_short,
    Mach_landing,
    Mach_landing_short,
    mu_landing,
    Re_W_landing,
    Re_W_landing_short,
] = delta4_perf.liftanddrag(
    S,
    CL_landing,
    CD_landing,
    v_landing,
    rho_a_landing,
    g_landing,
    Va_landing,
    Ta_landing,
    Cbar,
)

[T_TO_kg, T_TO_kg_short, T_TO_lbs, T_TO_lbs_short] = delta4_perf.t2w(W, tw)
[T_TOF_kg, T_TOF_kg_short, T_TOF_lbs, T_TOF_lbs_short] = delta4_perf.t2w(W, tw)
[T_cruise_kg, T_cruise_kg_short, T_cruise_lbs, T_cruise_lbs_short] = delta4_perf.t2w(
    W, tw
)
[T_cruise2_kg, T_cruise2_kg_short, T_cruise2_lbs, T_cruise2_lbs_short] = (
    delta4_perf.t2w(W, tw)
)
[T_landing_kg, T_landing_kg_short, T_landing_lbs, T_landing_lbs_short] = (
    delta4_perf.t2w(W, tw)
)

[
    Preq_TO,
    Preq_TO_short,
    Preq_kW_TO,
    Preq_kW_TO_short,
    Preq_550hp_TO,
    Preq_550hp_TO_short,
] = delta4_perf.powreq(D_TO, v_TO)
[
    Preq_TOF,
    Preq_TOF_short,
    Preq_kW_TOF,
    Preq_kW_TOF_short,
    Preq_550hp_TOF,
    Preq_550hp_TOF_short,
] = delta4_perf.powreq(D_TOF, v_TOF)
[
    Preq_cruise,
    Preq_cruise_short,
    Preq_kW_cruise,
    Preq_kW_cruise_short,
    Preq_550hp_cruise,
    Preq_550hp_cruise_short,
] = delta4_perf.powreq(D_cruise, v_cruise)
[
    Preq_cruise2,
    Preq_cruise2_short,
    Preq_kW_cruise2,
    Preq_kW_cruise2_short,
    Preq_550hp_cruise2,
    Preq_550hp_cruise2_short,
] = delta4_perf.powreq(D_cruise2, v_cruise2)
[
    Preq_landing,
    Preq_landing_short,
    Preq_kW_landing,
    Preq_kW_landing_short,
    Preq_550hp_landing,
    Preq_550hp_landing_short,
] = delta4_perf.powreq(D_landing, v_landing)

print("\n------STALL Condition------")
print(f"v_stall_roskam_knot: {v_stall_roskam_knot:,.0f} knot")
print(f"v_stall_TO_knot: {v_stall_TO_knot:,.0f} knot")
# print("v_stall_TO: " + f"{v_stall_TO:,.0f}" + ' m/s')
print(f"v_stall_cruise_knot: {v_stall_cruise_knot:,.0f} knot")
print(f"v_stall_cruise2_knot: {v_stall_cruise2_knot:,.0f} knot")
print(f"v_stall_landing_knot: {v_stall_landing_knot:,.0f} knot")

print("\n------Take-off FLAPS------")
print(f"v_TOF: {v_TOF_kmh:,.1f} km/h")
print(f"v_TOF: {v_TOF:,.1f} m/s")
print(f"Mach number_TOF: {Mach_TOF:,.3f} Mach")
print(f"Re_W_Take-off: {Re_W_TOF:.1E}")
print(f"Lift @ Take-off: {L_TOF_kg:,.1f} kg")
print(f"Drag @ Take-off: {D_TOF_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_TOF_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_TOF_lbs:,.1f} lbs")
print(f"Required Power @ Take-off: {Preq_kW_TOF:,.2f} kW")
print(f"Required Power: {Preq_550hp_TOF:,.2f} (550)hp")

print("\n------Cruise Condition------")
print(f"v_TOF: {v_cruise_kmh:,.1f} km/h")
print(f"v_TOF: {v_cruise:,.1f} m/s")
print(f"Mach number_TOF: {Mach_cruise:,.3f} Mach")
print(f"Re_W_Take-off: {Re_W_cruise:.1E}")
print(f"CL_cruise: {CL_cruise:,.3f}")
print(f"Lift @ Take-off: {L_cruise_kg:,.1f} kg")
print(f"Drag @ Take-off: {D_cruise_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_cruise_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_cruise_lbs:,.1f} lbs")
print(f"Required Power @ Take-off: {Preq_kW_cruise:,.2f} kW")
print(f"Required Power: {Preq_550hp_cruise:,.2f} (550)hp")

print("\n------Cruise2 Condition------")
print(f"v_TOF: {v_cruise2_kmh:,.1f} km/h")
print(f"v_TOF: {v_cruise2:,.1f} m/s")
print(f"Mach number_TOF: {Mach_cruise2:,.3f} Mach")
print(f"Re_W_Take-off: {Re_W_cruise2:.1E}")
print(f"CL_cruise2: {CL_cruise2:,.3f}")
print(f"Lift @ Take-off: {L_cruise2_kg:,.1f} kg")
print(f"Drag @ Take-off: {D_cruise2_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_cruise2_kg:,.1f} kg")
print(f"Required Thrust@ Take-off: {T_cruise2_lbs:,.1f} lbs")
print(f"Required Power @ Take-off: {Preq_kW_cruise2:,.2f} kW")
print(f"Required Power: {Preq_550hp_cruise2:,.2f} (550)hp")

print("\n------Landing------")
print(f"v_landing: {v_landing_kmh:,.1f} km/h")
print(f"v_landing: {v_landing:,.1f} m/s")
print(f"Mach number_landing: {Mach_landing:,.3f} Mach")
print(f"Re_W_landing: {Re_W_landing:.1E}")
print(f"Lift @ landing: {L_landing_kg:,.1f} kg")
print(f"Drag @ landing: {D_landing_kg:,.1f} kg")
print(f"Required Thrust@ landing: {T_landing_kg:,.1f} kg")
print(f"Required Thrust@ landing: {T_landing_lbs:,.2f} lbs")
print(f"Required Power @ landing: {Preq_kW_landing:,.2f} kW")
print(f"Required Power: {Preq_550hp_landing:,.2f} (550)hp")

"""For Neutral Point Calculations"""
# Initial Condition
c = Va_cruise  # Speed of sound, m/s

# CG position
XCG_mm = 9500  # XCG, mm    625mm <= XCG <= 641mm  ||  XNP ~ 665mm  \\//  Fuslage influence: -18mm
# Required Static Margin
req_sm = 20  # (%)

# Fuselage Geometry
scaleratio = 1.0

# fuselage_xml = f"xfl_fuse_2.xml"
with open("HFCT_fuse.xml", "r", encoding="utf-8") as file:
    fuselage_xml = file.read()


fuse_coord = {
    "X_pos": [],
    "Y_pos": [],
    "Z_pos": [],
    "x_fuse": [],
    "y_fuse": [],
    "z_fuse": [],
}

# fuse_coord_dict = xmltodict.parse(fuselage_xml, process_namespaces=True)
fuse_coord_dict = xmltodict.parse(fuselage_xml)
fuse_coord_dict = fuse_coord_dict["xflfuse"]["body"]["NURBS"]["frame"]
# print(fuse_coord_dict)

for i in range(19):
    k = fuse_coord_dict[i]
    k_1 = list(map(float, k["Position"].split(",")))
    fuse_coord["X_pos"].append(k_1[0])
    fuse_coord["Y_pos"].append(k_1[1])
    fuse_coord["Z_pos"].append(k_1[2])
    # print(k_1)

for i in range(19):
    k = fuse_coord_dict[i]
    k_31 = []
    k_32 = []
    k_33 = []
    for j in range(6):
        k_2 = k["point"][j]
        k_2 = list(map(float, k_2.split(",")))
        k_31.append(k_2[0])
        k_32.append(k_2[1])
        k_33.append(k_2[2])
    fuse_coord["x_fuse"].append(k_31)
    fuse_coord["y_fuse"].append(k_32)
    fuse_coord["z_fuse"].append(k_33)

# print(fuse_coord)


# print(fuse_coord)

# x_sta_mm = np.array([0, 75, 200, 400, 550, 750, 900, 1100, 1250, 1550, 1600])

x_sta_mm = np.asarray(fuse_coord["X_pos"])

x_sta_mm = x_sta_mm * 1000 * scaleratio

# print(x_sta_mm)

# ssta1 = [0, 0]
# sta0_mm = np.array([[0, 0],
#                  [0, 0],
#                  [0, 0],
#                  [0, 0],
#                  [0, 0]])
sta_mm = []
for i in range(19):
    # j_x = fuse_coord['x_fuse'][i]
    j_y = fuse_coord["y_fuse"][i]
    j_z = fuse_coord["z_fuse"][i]
    coord_yz = []
    for k in range(6):
        coord_yz.append([j_y[k], j_z[k]])
    sta_mm.append(coord_yz)

sta_mm = np.asarray(sta_mm)
sta_mm = sta_mm * scaleratio
# print(sta_mm)

sta0_mm = sta_mm[0] * 1000 * scaleratio
sta1_mm = sta_mm[1] * 1000 * scaleratio
sta2_mm = sta_mm[2] * 1000 * scaleratio
sta3_mm = sta_mm[3] * 1000 * scaleratio
sta4_mm = sta_mm[4] * 1000 * scaleratio
sta5_mm = sta_mm[5] * 1000 * scaleratio
sta6_mm = sta_mm[6] * 1000 * scaleratio
sta7_mm = sta_mm[7] * 1000 * scaleratio
sta8_mm = sta_mm[8] * 1000 * scaleratio
sta9_mm = sta_mm[9] * 1000 * scaleratio
sta10_mm = sta_mm[10] * 1000 * scaleratio
sta11_mm = sta_mm[11] * 1000 * scaleratio
sta12_mm = sta_mm[12] * 1000 * scaleratio
sta13_mm = sta_mm[13] * 1000 * scaleratio
sta14_mm = sta_mm[14] * 1000 * scaleratio
sta15_mm = sta_mm[15] * 1000 * scaleratio
sta16_mm = sta_mm[16] * 1000 * scaleratio
sta17_mm = sta_mm[17] * 1000 * scaleratio
sta18_mm = sta_mm[18] * 1000 * scaleratio

stations_yz_mm = [sta0_mm, sta1_mm, sta2_mm, sta3_mm, sta4_mm, sta5_mm, sta6_mm, sta7_mm, sta8_mm, sta9_mm, sta10_mm, sta11_mm, sta12_mm, sta13_mm, sta14_mm, sta15_mm, sta16_mm, sta17_mm, sta18_mm]

stations_yz_mm = np.asarray(stations_yz_mm)
# print(stations_yz_mm[:, :, 0])
stations_y_mm = stations_yz_mm[:, :, 0]

print(stations_y_mm)

stations_ymax_mm = []
for i, ymax_i in enumerate(stations_y_mm):
    y_max_i = max(stations_y_mm[i])
    stations_ymax_mm.append(y_max_i)

stations_ymax_mm = np.asarray(stations_ymax_mm)

print(f"\n {x_sta_mm}")
print(f"\n {stations_ymax_mm}")

# print("#####################")
# print("#####################")

# print(stations_y_mm[:, 3])

# stations_ymax_mm = stations_y_mm[:, 3]

stations_z_mm = stations_yz_mm[:, :, 1]
stations_zmax_mm = []
stations_zmin_mm = []
for i, zmax_i in enumerate(stations_z_mm):
    z_max_i = max(stations_z_mm[i])
    stations_zmax_mm.append(z_max_i)
    z_min_i = min(stations_z_mm[i])
    stations_zmin_mm.append(z_min_i)

stations_zmax_mm = np.asarray(stations_zmax_mm)
stations_zmin_mm = np.asarray(stations_zmin_mm)

print(f"\n {stations_zmax_mm}")
print(f"\n {stations_zmin_mm}")

stations_hf_mm = stations_zmax_mm - stations_zmin_mm

# Fuselage DATA
Wf_mm = max(stations_ymax_mm) * 2  # Fuselage maximum width, mm
hf_mm = max(stations_hf_mm)  # mm, maximum fuselage height
# Wf_mm = 1.85*1000
# hf_mm = 1.85*1000
l_n_mm = 3.5*1000  # length of nose
h_t_mm = 1.35*1000  # Horizontal Tail height - Wing height, mm
# delta_xi = 0.44
# Wf = [0.33341538, 0.38350769, 0.5248, 0.6996, 0.85285714, 0.5, 0.5, 0.69827457, 0.54954047, 0.40017002, 0.3, 0.3, 0.3]
# xi = [0.44*4 + 0.181, 0.44*3 + 0.214, 0.44*2 + 0.208, 0.44*1 + 0.211, 0.215, 0.46545*2 + 0.372, 0.46545*1 + 0.393, 0.241, 0.46545*1 + 0.243, 0.46545*2 + 0.247, 0.46545*3 + 0.235, 0.46545*4 + 0.233, 0.46545*5 + 0.233]
# xi = [0.44*4 + 0.181, 0.44*3 + 0.214, 0.44*2 + 0.208, 0.44*1 + 0.211, 0.215, 0.372, 0.393, 0.241, 0.46545*1 + 0.243, 0.46545*2 + 0.247, 0.46545*3 + 0.235, 0.46545*4 + 0.233, 0.46545*5 + 0.233]
# xi_67 = [0.372, 0.393]
# xi_aft = [0.241, 0.46545*1 + 0.243, 0.46545*2 + 0.247, 0.46545*3 + 0.235, 0.46545*4 + 0.233, 0.46545*5 + 0.233]
# delta_x5 = 0.458874172
# x_h = X_AC_HT_mm / 1000 - (L_wing + Croot)  # x_h for conventional crusiform Aircraft
x_h = L_HT_act_mm/1000
# x_h = X_ac_vtail_ht/1000 - (L_wing + Croot)     # x_h for V-tail Aircraft
# x_h = 2.167726
# Cf = 1.50728
# sum_x8_13 = 2.787

# Thrust Effect
eta_h_T0 = 0.90  # eta_h_(T=0): No thurst effect eta_h
T_kgf = W * tw  # Thrust, kgf
# Dp_in = 70							#Propeller Diameter, inch
Dp_in = 87

delta4_sta = AD_inverse_Class.Stability(
    c, A, v_cruise, XCG_mm, Wf_mm, h_t_mm, eta_h_T0, T_kgf, Dp_in
)

[wf, xi_1to5, xi_6to13, xi, delta_xi_1to5, delta_xi_8to13] = delta4_sta.bodygeo2(
    x_sta_mm, stations_ymax_mm, stations_yz_mm, L_wing, Croot, FL
)

# [wf, xi_1to5, xi_6to13, xi, delta_xi_1to5, delta_xi_8to13] = delta4_sta.bodygeo3(
#     x_sta_mm, stations_ymax_mm, stations_zmax_mm, L_wing, Croot, FL
# )

[
    theta_LE,
    theta_LE_deg,
    theta_c2,
    theta_c2_deg,
    theta_c4,
    theta_c4_deg,
    theta_c3,
    theta_c3_deg,
] = delta4_sta.sweep(Croot_short, Ctip_short, b, A, lamda, wingpos_w, lamda_len)
[
    theta_LE_HT,
    theta_LE_deg_HT,
    theta_c2_HT,
    theta_c2_deg_HT,
    theta_c4_HT,
    theta_c4_deg_HT,
    theta_c3_HT,
    theta_c3_deg_HT,
] = delta4_sta.sweep(Croot_HT, Ctip_HT, b_HT, A_HT, lamda_HT, wingpos_HT, lamda_len_HT)
# [theta_LE_VT, theta_LE_deg_VT, theta_c2_VT, theta_c2_deg_VT, theta_c4_VT, theta_c4_deg_VT] = delta4_sta.sweep(Croot_VT_short, Ctip_VT_short, b_VT_short, A_VT, lamda_VT)
# [theta_LE_VT, theta_LE_deg_VT, theta_c4_VT, theta_c4_deg_VT] = delta4_sta.sweepvt(Croot_VT_short, Ctip_VT_short, b_VT_short, A_VT, lamda_VT)
[theta_LE_VT, theta_LE_deg_VT, theta_c4_VT, theta_c4_deg_VT] = delta4_sta.sweepvt(
    Croot_VT, Ctip_VT, b_VT, A_VT, lamda_VT
)
[
    theta_LE_VT,
    theta_LE_deg_VT,
    theta_c2_VT,
    theta_c2_deg_VT,
    theta_c4_VT,
    theta_c4_deg_VT,
    theta_c3_VT,
    theta_c3_deg_VT,
] = delta4_sta.sweep(Croot_VT, Ctip_VT, b_VT, A_VT, lamda_VT, wingpos_VT, lamda_len_VT)
print("\n---------Sweep Angles---------")
print("Wing LE Sweep: " + f"{theta_LE_deg:,.10f}" + "\N{DEGREE SIGN}")
print("Wing C/4 Sweep: " + f"{theta_c4_deg:,.10f}" + "\N{DEGREE SIGN}")
print("HT LE Sweep:   " + f"{theta_LE_deg_HT:,.10f}" + "\N{DEGREE SIGN}")
print("VT LE Sweep:   " + f"{theta_LE_deg_VT:,.10f}" + "\N{DEGREE SIGN}")
print("VT C/4 Sweep:   " + f"{theta_c4_deg_VT:,.10f}" + "\N{DEGREE SIGN}")

[
    M,
    CLa_W,
    CLa_W_M0,
    CLa_W_short,
    CLa_W_deg,
    CLa_W_deg_short,
    Re_w,
    mu_w,
    Re_w_short,
] = delta4_sta.cl_slope(A, theta_c4, Ta_cruise, rho_a_cruise, Cbar, v_cruise)
[
    M_h,
    CLa_HT,
    CLa_HT_M0,
    CLa_HT_short,
    CLa_HT_deg,
    CLa_HT_deg_short,
    Re_HT,
    mu_HT,
    Re_HT_short,
] = delta4_sta.cl_slope(A_HT, theta_c4_HT, Ta_cruise, rho_a_cruise, Cbar_HT, v_cruise)
[
    M_v,
    CLa_VT,
    CLa_VT_M0,
    CLa_VT_short,
    CLa_VT_deg,
    CLa_VT_deg_short,
    Re_VT,
    mu_VT,
    Re_VT_short,
] = delta4_sta.cl_slope(A_VT, theta_c4_VT, Ta_cruise, rho_a_cruise, Cbar_VT, v_cruise)
print("\n------CL slope Wing and Tail------")
print("CLa_W: " + str(CLa_W_short) + " per rad")
print("CLa_W_deg: " + str(CLa_W_deg_short) + " per degree")
print("Re_w @ Cruise: " + f"{Re_w_short:,.0f}")
print("\nCLa_HT: " + str(CLa_HT_short) + " per rad")
print("CLa_HT_deg: " + str(CLa_HT_deg_short) + " per degree")

[l_h, h_t, Lf, K_A, K_lamda, K_h, dedt_deg, dedt, dedt_0_deg] = delta4_sta.downwash(
    L_AC_HT_act, FL_short, A, lamda, b, theta_c4, CLa_W, CLa_W_M0
)
print(f"L_AC_HT_act: {L_AC_HT_act}")

[
    Wf_meter,
    X_Cr4_mm,
    X_Cr4,
    X_bar_Cr4,
    Kf,
    Cma_fus_deg,
    Cma_fus,
    dahda_deg,
    dahda,
    X_AC_wing,
    X_AC_HT,
    X_bar_ac_w,
    X_bar_ac_h,
    X_bar_NP,
    XNP,
    XNP_mm,
    XNP_mm_short,
    XCG,
    SM,
    SM_short,
    prp_xcg,
    X_bar_NP_thrusteffect,
    XNP_thrusteffect,
    XNP_thrusteffect_mm,
    eta_h,
    SM_thrusteffect,
] = delta4_sta.neutralpoint(
    X_AC_wing_mm,
    L_AC_wing,
    Lf,
    Cbar,
    S,
    dedt_deg,
    X_AC_HT_mm,
    CLa_W,
    S_HT,
    CLa_HT,
    L_wing,
    Croot,
    req_sm,
    Croot_HT,
    Ctip_HT,
    rho_a_cruise,
    v_cruise,
    Preq_550hp_cruise,
    Dp_in,
    A,
    L_HT_act_mm,
    lamda,
    b
)

print(f"\n----------POSITIONS----------")
print(f"X_wing_AC: {X_AC_wing_mm/1000:,.3f} m")
print(f"X_HT_AC: {X_AC_HT_mm/1000:,.2f} m")
print(f"X_VT_AC: {X_AC_VT_mm/1000:,.2f} m")
print(f"X_wing_AC - X_HT_AC: {L_HT_act_mm/1000:,.2f} m")



print("\n------XNP NO Thrust effect------")
print("XNP_No_Thrust_effect: " + f"{XNP_mm:,.0f}" + " mm")
# print('\nStatic_Margin: ' + f"{SM:,.0f}" + ' %')
print(f"Static_Margin: {SM:,.0f} % (@ xCG = {XCG_mm:,.0f} mm)")
print("\nProposed_xCG" + "(" + str(req_sm) + "%SM" + "): " f"{prp_xcg:,.0f}" + " mm")
print("")


"""X_ac_vtail_ht_mm = X_ac_vtail_ht * 1000
[l_h, h_t, Lf, K_A, K_lamda, K_h, dedt_deg, dedt, dedt_0_deg] = delta4_sta.downwash(L_AC_vtail_HT_act, FL_short, A, lamda, b, theta_c4, CLa_W, CLa_W_M0)
[Wf_meter, X_Cr4_mm, X_Cr4, X_bar_Cr4, Kf, Cma_fus_deg, Cma_fus, dahda_deg, dahda, X_AC_wing, X_AC_HT, X_bar_ac_w, X_bar_ac_h, X_bar_NP, XNP, XNP_mm, XNP_mm_short, XCG, SM, SM_short, prp_xcg, X_bar_NP_thrusteffect, XNP_thrusteffect, XNP_thrusteffect_mm, eta_h, SM_thrusteffect] = delta4_sta.neutralpoint(X_AC_wing_mm, L_AC_wing, Lf, Cbar, S, dedt_deg, X_ac_vtail_ht_mm, CLa_W, S_vtail_ht, CLa_HT, L_wing, Croot, req_sm, Croot_vtail_ht, Croot_vtail_ht, rho_a_cruise, v_cruise, Preq_550hp_cruise, Dp_in)
print("\n------------ V-Tail ------------")
print('XNP_No_Thrust_effect: ' + f"{XNP_mm:,.0f}" + ' mm')
#print('\nStatic_Margin: ' + f"{SM:,.0f}" + ' %')
print(f"Static_Margin: {SM:,.0f} % (@ xCG = {XCG_mm:,.0f} mm)")
print('\nProposed_xCG' + '(' + str(req_sm) + '%SM' + '): ' f"{prp_xcg:,.0f}" + ' mm')

print("\n------XNP Thrust effect------")
print("xNP_thrusteffect: " + f"{XNP_thrusteffect_mm:,.0f}" + ' mm')
print('Static_Margin: ' + f"{SM_thrusteffect:,.0f}" + ' %')"""

[
    dMda_im,
    deltaXac_f_im,
    K_wf,
    CLa_WF_deg,
    XNPtwo_im,
    XNPtwo,
    XNPtwo_mm,
    sumWf,
    X_bar_NPtwo_im,
    SM_roskam,
    prp_xcgtwo,
] = delta4_sta.neutralpointfour(wf, xi, Croot, dedt_0_deg, x_h, delta_xi_1to5, delta_xi_8to13, rho_a_cruise, v_cruise, S, Cbar, CLa_W_deg, Wf_mm, b, X_AC_wing_mm, S_HT, CLa_HT_deg, dahda_deg, X_AC_HT_mm, eta_h, XCG_mm, req_sm, theta_LE, dedt_deg, lamda, L_wing, X_true_HT_mm_short)
print("\n------XNP Roskam Method------")
print("XNP_two: " + f"{XNPtwo_mm:,.0f}" + " mm")
print("Static_Margin: " + f"{SM_roskam:,.0f}" + f" % (@ xCG = {XCG_mm:,.0f} mm)")
print(f"\nProposed_xCG({req_sm}%SM): {prp_xcgtwo:,.0f} mm")
print(f"deltaXac_f_im: {deltaXac_f_im}")

Xac_cbar__WF = delta4_sta.neutralpoint_torenbeek(
    CLWa, Wf_mm, AC, Cbar, L_wing, S, lamda, b, theta_c4, l_n_mm, hf_mm, Croot
)

print("\n------XNP Torenbeek Method------")
print(f"(Xac/Cbar)_WF: {Xac_cbar__WF:,.3f}")

#  Root - Locus Plot
# filepath = "C:/Users/shoh9/Desktop/SH/Programming/Aircraft_Design"
# filepath = "D:/ACML_SH/Programming/ADFC"
filepath = "."
# filepath = "/home/shoh/Programming/ADFC"
dynstab_folder = "dynstab_RIMP-02"
imgfolder = "savedimg"
num_line = []
# f = open("input/durumi_eigenvalue_20240202-2.txt", 'r')
eigenvalue_filename = "RIMP-02_nofus_eigenvalue_250208.txt"
system_matrix_filename = "RIMP-02_sys_matrix_case1.txt"
system_matrix_file_path = f"{filepath}/input/{system_matrix_filename}"
# system_matrix_file_path = f'{filepath}/input/{durumiV_sys_matrix_case[i]}'+
# find:::: _sys_matrix_case
closeit = 0  # closeit = 1 --> save stability figures

eigenvalue_file_path = f"{filepath}/input/{eigenvalue_filename}"
f = open(eigenvalue_file_path, "r")
data = f.readlines()
f.close()
casenums = []
flts = []
iiii = 1
while True:
    try:
        if iiii > 2:
            for inte in data[iiii]:
                if inte.isdigit() == True:
                    # flts.append([int(inte)])
                    numbs_maxx = int(inte)
                    break
            match_number = re.compile("-?\\ *[0-9]+\\.?[0-9]*(?:[Ee]\\ *-?\\ *[0-9]+)?")
            final_list = [float(x) for x in re.findall(match_number, data[iiii])]
            flts.append(final_list)
        iiii += 1
    except IndexError:
        break
f.close()
flts = np.array(flts)
eigs_real, eigs_img = [], []
eigs_real_n, eigs_img_n = [], []
numcase = flts[:, 0]
numcase_max = int(numcase[-1])
for i in range(numcase_max + 1):
    eigs_real_n = []
    eigs_img_n = []
    for j in range(len(flts)):
        if flts[j, 0] == i:
            eigs_real_n.append(flts[j, 1])
            eigs_img_n.append(flts[j, 2])
    eigs_real.append(eigs_real_n)
    eigs_img.append(eigs_img_n)

casenumbers = np.arange(1, numcase_max + 1)
eigs_real_flt = eigs_real.copy()
eigs_real_flt.pop(0)
# print(eigs_real_flt)
xpmax = max(map(max, eigs_real_flt))
xpmin = min(map(min, eigs_real_flt))
eigs_img_flt = eigs_img.copy()
eigs_img_flt.pop(0)
ypmax = max(map(max, eigs_img_flt))
ypmin = min(map(min, eigs_img_flt))


# Switching Positions for case >= 3 ; because of AVL program eigenvalue output mode order change, changes every 8 cases
"""laps = 0
while True:
    if laps > 2:
        break
    else:
        laps += 1
    try:
        for c in range(3, 5):
            eigs_real[c][1], eigs_real[c][3] = eigs_real[c][3], eigs_real[c][1]
            eigs_real[c][2], eigs_real[c][4] = eigs_real[c][4], eigs_real[c][2]
            spiralmode_real = eigs_real[c].pop(-1)
            eigs_real[c].insert(3, spiralmode_real)
            eigs_img[c][1], eigs_img[c][3] = eigs_img[c][3], eigs_img[c][1]
            eigs_img[c][2], eigs_img[c][4] = eigs_img[c][4], eigs_img[c][2]
            spiralmode_img = eigs_img[c].pop(-1)
            eigs_img[c].insert(3, spiralmode_img)
    except IndexError:
        break"""

"""for cc in range(3, 5+1):
    if cc == 6:
        continue
    else:
        eigs_real[cc][1], eigs_real[cc][3] = eigs_real[cc][3], eigs_real[cc][1]
        eigs_real[cc][2], eigs_real[cc][4] = eigs_real[cc][4], eigs_real[cc][2]
        spiralmode_real = eigs_real[cc].pop(-1)
        eigs_real[cc].insert(3, spiralmode_real)
        eigs_img[cc][1], eigs_img[cc][3] = eigs_img[cc][3], eigs_img[cc][1]
        eigs_img[cc][2], eigs_img[cc][4] = eigs_img[cc][4], eigs_img[cc][2]
        spiralmode_img = eigs_img[cc].pop(-1)
        eigs_img[cc].insert(3, spiralmode_img)"""
"""
if numcase_max > 6:
    spiralmode_real = eigs_real[6].pop(2)
    eigs_real[6].insert(0, spiralmode_real)
    spiralmode_img = eigs_img[6].pop(2)
    eigs_img[6].insert(0, spiralmode_img)
"""

ary_real = []
ary_img = []

for i in range(1, 1 + numcase_max):
    roll_spiral_r = []
    roll_r = []
    sprial_r = []
    dutch_r = []
    shortperiod_r = []
    phugoid_r = []
    roll_spiral_i = []
    roll_i = []
    sprial_i = []
    dutch_i = []
    shortperiod_i = []
    phugoid_i = []
    dutch_short_phugoid_r = []
    dutch_short_phugoid_i = []
    eigs_real_mm = eigs_real[i]
    eigs_img_mm = eigs_img[i]

    for j in range(8):
        if eigs_img_mm[j] == 0:
            roll_spiral_r.append(eigs_real_mm[j])
            roll_spiral_i.append(eigs_img_mm[j])

    spiral_r_idx = roll_spiral_r.index(max(roll_spiral_r))
    sprial_r.append(roll_spiral_r.pop(spiral_r_idx))
    sprial_i.append(roll_spiral_i.pop(spiral_r_idx))
    roll_r.append(roll_spiral_r.pop(-1))
    roll_i.append(roll_spiral_i.pop(-1))

    for idx, aa in enumerate(eigs_real_mm):
        ab = eigs_real_mm[idx - 1]
        if aa == ab:
            dutch_short_phugoid_r.append(ab)
            dutch_short_phugoid_r.append(aa)
            dutch_short_phugoid_i.append(eigs_img_mm[idx - 1])
            dutch_short_phugoid_i.append(eigs_img_mm[idx])

    valmax = min(dutch_short_phugoid_r)
    valmax_idx = dutch_short_phugoid_r.index(valmax)

    shortperiod_r.append(valmax)
    if valmax == dutch_short_phugoid_r[valmax_idx + 1]:
        shortperiod_r.append(dutch_short_phugoid_r[valmax_idx + 1])
        shortperiod_i.append(dutch_short_phugoid_i[valmax_idx])
        shortperiod_i.append(dutch_short_phugoid_i[valmax_idx + 1])
        dutch_short_phugoid_r.pop(valmax_idx + 1)
        dutch_short_phugoid_i.pop(valmax_idx + 1)
    elif valmax == dutch_short_phugoid_r[valmax_idx - 1]:
        shortperiod_r.append(dutch_short_phugoid_r[valmax_idx - 1])
        shortperiod_i.append(dutch_short_phugoid_i[valmax_idx])
        shortperiod_i.append(dutch_short_phugoid_i[valmax_idx - 1])
        dutch_short_phugoid_r.pop(valmax_idx - 1)
        dutch_short_phugoid_i.pop(valmax_idx - 1)

    kdx = dutch_short_phugoid_r.index(valmax)
    dutch_short_phugoid_r.pop(kdx)
    dutch_short_phugoid_i.pop(kdx)

    dutch_phugoid_r = dutch_short_phugoid_r
    dutch_phugoid_i = dutch_short_phugoid_i

    dutch_r_val = min(dutch_phugoid_r)
    dutch_r_idx = dutch_phugoid_r.index(dutch_r_val)

    dutch_r.append(dutch_r_val)
    dutch_i.append(dutch_phugoid_i[dutch_r_idx])
    if dutch_r_val == dutch_phugoid_r[dutch_r_idx + 1]:
        dutch_r.append(dutch_phugoid_r[dutch_r_idx + 1])
        dutch_i.append(dutch_phugoid_i[dutch_r_idx + 1])
        dutch_phugoid_r.pop(dutch_r_idx + 1)
        dutch_phugoid_i.pop(dutch_r_idx + 1)
    elif dutch_r_val == dutch_phugoid_r[dutch_r_idx - 1]:
        dutch_r.append(dutch_phugoid_r[dutch_r_idx - 1])
        dutch_i.append(dutch_phugoid_i[dutch_r_idx - 1])
        dutch_phugoid_r.pop(dutch_r_idx - 1)
        dutch_phugoid_i.pop(dutch_r_idx - 1)

    kjdx = dutch_phugoid_r.index(dutch_r_val)
    dutch_phugoid_r.pop(kjdx)
    dutch_phugoid_i.pop(kjdx)
    phugoid_r = dutch_phugoid_r
    phugoid_i = dutch_phugoid_i

    flag_dutch_phugoid_swap = 0
    if max(phugoid_i) > max(dutch_i):
        phugoid_r_copy = phugoid_r.copy()
        phugoid_i_copy = phugoid_i.copy()
        phugoid_r = dutch_r.copy()
        phugoid_i = dutch_i.copy()
        dutch_r = phugoid_r_copy.copy()
        dutch_i = phugoid_i_copy.copy()
        flag_dutch_phugoid_swap = 1

    ary_real_i = [
        roll_r[0],
        dutch_r[0],
        dutch_r[1],
        sprial_r[0],
        shortperiod_r[0],
        shortperiod_r[1],
        phugoid_r[0],
        phugoid_r[1],
    ]
    ary_img_i = [
        roll_i[0],
        dutch_i[0],
        dutch_i[1],
        sprial_i[0],
        shortperiod_i[0],
        shortperiod_i[1],
        phugoid_i[0],
        phugoid_i[1],
    ]
    ary_real.append(ary_real_i)
    ary_img.append(ary_img_i)


ary_real.insert(0, [])
ary_img.insert(0, [])
eigs_real = np.array(ary_real, dtype=object)
eigs_img = np.array(ary_img, dtype=object)



pngpath = f"{filepath}/{imgfolder}/{dynstab_folder}"
isExist = os.path.exists(pngpath)
if not isExist:
    os.makedirs(pngpath)

cases = []
for i in range(1, 1 + max(casenumbers)):
    cases.append(f"CASE {i}")
cases = np.array(cases)
colors = plt.cm.rainbow(np.linspace(0, 1, 1 + max(casenumbers)))
plt.figure(figsize=(9, 5))
for i in range(1, 1 + max(casenumbers)):
    plt.scatter(eigs_real[i], eigs_img[i], marker="x", s=30, color=colors[i])
plt.legend(cases)
plt.grid(True, color="k")
plt.xlim((math.floor(xpmin - 0.5), round(xpmax) + 1))
plt.ylim((math.floor(ypmin) - 1, round(ypmax + 0.6) + 1))
plt.axhline(y=0, xmin=0, xmax=1, c="k")
plt.axvline(x=0, ymin=0, ymax=1, c="k")
plt.title("Dynamic Stability Analysis\nEigenmode Analysis | Root Locus Plot")
params = {"mathtext.default": "regular"}
plt.rcParams.update(params)
# plt.xlabel("$ζ ω_{n}$   $[s^{-1}]$")
# plt.ylabel("$ω_{d}$   $[s^{-1}]$")
plt.xlabel("$ζ ω_{n}$   [rad/s]")
plt.ylabel("$ω_{d}$   [rad/s]")
# plt.savefig('savedimg/eigenimg/eigRL-allcase.png', format = 'png')   # save the figure to file
plt.savefig(f"{pngpath}/eigRL-allcase.png", format="png")  # save the figure to file
# plt.savefig('eigenimg/eigRL-allcase.eps', format = 'eps')   # for LATEX
# plt.show()
plt.close()


tau_tconst = np.zeros((numcase_max, 8))
doubleamp_time = np.zeros((numcase_max, 8))
halfdampcycle = np.zeros((numcase_max, 8))
zeta = np.zeros((numcase_max, 8))
sigma_damped_rad = np.zeros((numcase_max, 8))
dampresponse = np.zeros((numcase_max, 8))
w_n = np.zeros((numcase_max, 8))
wn_zeta = np.zeros((numcase_max, 8))
wnd_zeta = np.zeros((numcase_max, 8))
for i in range(1, 1 + max(casenumbers)):
    eigs_real_ii = eigs_real[i]
    eigs_img_ii = eigs_img[i]
    for j in range(8):
        tau_tconst[i - 1, j] = abs(1 / (eigs_real_ii[j]))
        doubleamp_time[i - 1, j] = tau_tconst[i - 1, j] * np.log(2)
        zeta[i - 1, j] = abs(
            eigs_real_ii[j] / (np.sqrt(eigs_real_ii[j] ** 2 + eigs_img_ii[j] ** 2))
        )
        halfdampcycle[i - 1, j] = (
            np.log(2) / (2 * np.pi) * np.sqrt(1 - zeta[i - 1, j] ** 2) / zeta[i - 1, j]
        )
        dampresponse[i - 1, j] = eigs_real_ii[j]
        sigma_damped_rad[i - 1, j] = eigs_img_ii[j]
        w_n[i - 1, j] = np.sqrt(
            eigs_real_ii[j] ** 2 + eigs_img_ii[j] ** 2
        )  # Natural Frequency
        wn_zeta[i - 1, j] = w_n[i - 1, j] / (2 * np.pi) * zeta[i - 1, j]
        wnd_zeta[i - 1, j] = eigs_img_ii[j] * zeta[i - 1, j]

# print(f"\n\n========================================")
print("\n\n")
print(f"{'=':=^40}")
print(f"{'Dynamic Stability Analysis':+^40}")
# print(f"=======Dynamic Stability Analysis=======")
print(f"{'=':=^40}")
# print(f"========================================")
flag = 0
if flag_dutch_phugoid_swap == 1:
    print("    -------------------------------")
    print("    |  Dutch Roll / Phugoid Swap  |")
    print("    |           !Check!           |")
    print("    -------------------------------")


# casetoanalyze = np.array([1, 2, 3, 5, 7]) - 1
with open(filepath + f"/savedimg/{dynstab_folder}/OUTPUT.txt", "w") as fwrite:
    for i in range(0, numcase_max):
        # for i in casetoanalyze:
        nflg1, nflg2, nflg3, nflg4, nflg5 = 1, 1, 1, 1, 1
        eigs_img_case = eigs_img[i + 1]
        # print(f"\n++++++++++++++++CASE {i+1}++++++++++++++++")
        heading = f"CASE {i+1}"
        print(f"{heading:+^41}")
        print("    -------------------------------")
        print("    |  Lateral-Directional Modes  |")
        print("    -------------------------------")
        fwrite.write(f"{heading:+^41}\n")
        fwrite.write("    -------------------------------\n")
        fwrite.write("    |  Lateral-Directional Modes  |\n")
        fwrite.write("    -------------------------------\n")
        # print(f"***********Roll Damping***********")
        print(f"{'Roll Damping':*^40}")
        fwrite.write(f"{'Roll Damping':*^40}\n")
        if dampresponse[i][0] > 0:
            flag = 1
            nflg1 = -1
            print("\t########  Unstable  ########")
            fwrite.write("\t########  Unstable  ########\n")
        # print(f" Damping ratio(ζ): {nflg1*zeta[i, 0]:,.3f}")
        print(" Damping ratio(ζ): --")
        # print(f" Half damp Cycles: {halfdampcycle[i, 0]:,.2f} cycles")
        print(f" Time to half amplitude:(T2) {doubleamp_time[i, 0]:,.2f} s")
        print(f" Natural frequency(ωn): {nflg1*w_n[i][0]:,.3f} rad/s")
        fwrite.write(" Damping ratio(ζ): --\n")
        # fwrite.write(f" Half damp Cycles: {tau_tconst[i, 0]:,.2f} s\n")
        fwrite.write(f" Time to half amplitude:(T2) {doubleamp_time[i, 0]:,.2f} s\n")
        fwrite.write(f" Natural frequency(ωn): {nflg1*w_n[i][0]:,.3f} rad/s\n")

        # print(f"\n************Dutch Roll************")
        print(f"{'Dutch Roll':*^40}")
        fwrite.write(f"{'Dutch Roll':*^40}\n")
        if dampresponse[i][1] > 0:
            flag = 1
            nflg2 = -1
            print("\t########  Unstable  ########")
            fwrite.write("\t########  Unstable  ########\n")
        print(f" Damping ratio(ζ): {nflg2*zeta[i, 1]:,.3f}")
        print(f" ζ·ωnd: {nflg5*abs(wnd_zeta[i][1]):,.3f}")
        print(
            f" Damped Natural frequency(ωn_d): {nflg2*abs(sigma_damped_rad[i][1]):,.3f} rad/s\n"
        )
        print(f" Half damp Cycles: {halfdampcycle[i, 1]:,.2f} cycles")
        print(f" Time to half amplitude:(T2) {doubleamp_time[i, 1]:,.2f} s")
        # print(f" Natural frequency(ωn): {nflg2*abs(eigs_img_case[1]):,.3f} rad/s")
        print(f" Natural frequency(ωn): {nflg2*w_n[i][1]:,.3f} rad/s")
        fwrite.write(f" Damping ratio(ζ): {nflg2*zeta[i, 1]:,.3f}\n")
        fwrite.write(f" ζ·ωnd: {nflg5*abs(wnd_zeta[i][1]):,.3f}\n")
        fwrite.write(
            f" Damped Natural frequency(ωn_d): {nflg2*abs(sigma_damped_rad[i][1]):,.3f} rad/s\n\n"
        )
        fwrite.write(f" Half damp Cycles: {halfdampcycle[i, 1]:,.2f} cycles\n")
        fwrite.write(f" Time to half amplitude:(T2) {doubleamp_time[i, 1]:,.2f} s\n")
        fwrite.write(f" Natural frequency(ωn): {nflg2*w_n[i][1]:,.3f} rad/s\n")

        # print(f"\n**************Spiral*************")
        print(f"{'Spiral':*^40}")
        fwrite.write(f"{'Spiral':*^40}\n")
        if dampresponse[i][3] > 0:
            flag = 1
            nflg3 = -1
            print("\t##########  Unstable  ##########")
            fwrite.write("\t##########  Unstable  ##########\n")
        # print(f"Damping ratio(ζ): {zeta[i, 3]:,.2f}")
        print(" Damping ratio(ζ): --")
        # print(f" Half damp Cycles: {halfdampcycle[i, 3]:,.2f} cycles")
        print(f" Time to half amplitude:(T2) {doubleamp_time[i, 3]:,.2f} s")
        fwrite.write(" Damping ratio(ζ): --\n")
        # fwrite.write(f" Half damp Cycles: {halfdampcycle[i, 3]:,.2f} cycles\n")
        fwrite.write(f" Time to half amplitude:(T2) {doubleamp_time[i, 3]:,.2f} s\n")

        print("\n        ------------------------")
        print("        |  Longitudinal Modes  |")
        print("        ------------------------")
        print(f"{'Short period':*^40}")
        fwrite.write("\n        ------------------------\n")
        fwrite.write("        |  Longitudinal Modes  |\n")
        fwrite.write("        ------------------------\n")
        fwrite.write(f"{'Short period':*^40}\n")
        if dampresponse[i][4] > 0:
            flag = 1
            nflg4 = -1
            print("\t########  Unstable  ########")
            fwrite.write("\t########  Unstable  ########\n")
        print(f" Damping ratio(ζ): {nflg4*zeta[i, 4]:,.3f}")
        print(f" ζ·ωn: {nflg5*abs(wn_zeta[i][4]):,.3f}")
        print(
            f" Damped Natural frequency(ωn_d): {nflg4*abs(sigma_damped_rad[i][4]):,.2f} rad/s\n"
        )
        print(f" Half damp Cycles: {halfdampcycle[i, 4]:,.2f} cycles")
        print(f" Time to half amplitude:(T2) {doubleamp_time[i, 4]:,.2f} s")
        # print(f" Natural frequency(ωn): {nflg4*abs(eigs_img_case[4]):,.3f} rad/s")
        print(f" Natural frequency(ωn): {nflg2*w_n[i][4]:,.3f} rad/s")
        fwrite.write(f" Damping ratio(ζ): {nflg4*zeta[i, 4]:,.3f}\n")
        fwrite.write(f" ζ·ωn: {nflg5*abs(wn_zeta[i][4]):,.3f}\n")
        fwrite.write(
            f" Damped Natural frequency(ωn_d): {nflg4*abs(sigma_damped_rad[i][4]):,.2f} rad/s\n\n"
        )
        fwrite.write(f" Half damp Cycles: {halfdampcycle[i, 4]:,.2f} cycles\n")
        fwrite.write(f" Time to half amplitude:(T2) {doubleamp_time[i, 3]:,.2f} s\n")
        fwrite.write(f" Natural frequency(ωn): {nflg2*w_n[i][4]:,.3f} rad/s\n")

        # print(f"\n**************Phugoid**************")
        print(f"{'Phugoid':*^40}")
        fwrite.write(f"{'Phugoid':*^40}\n")
        if dampresponse[i][6] > 0:
            flag = 1
            nflg5 = -1
            print("\t########  Unstable  ########")
            fwrite.write("\t########  Unstable  ########\n")
        print(f" Damping ratio(ζ): {nflg5*zeta[i, 6]:,.3f}")
        print(f" ζ·ωn: {nflg5*abs(wn_zeta[i][6]):,.3f}")
        print(
            f" Damped Natural frequency(ωn_d): {nflg5*abs(sigma_damped_rad[i][6]):,.3f} rad/s\n"
        )
        print(f" Half damp Cycles: {halfdampcycle[i, 6]:,.2f} cycles")
        print(f" Time to half amplitude:(T2) {doubleamp_time[i, 6]:,.2f} s")
        fwrite.write(f" Damping ratio(ζ): {nflg5*zeta[i, 6]:,.3f}\n")
        fwrite.write(f" ζ·ωn: {nflg5*abs(wn_zeta[i][6]):,.3f}\n")
        fwrite.write(
            f" Damped Natural frequency(ωn_d): {nflg5*abs(sigma_damped_rad[i][6]):,.3f} rad/s\n\n"
        )
        fwrite.write(f" Half damp Cycles: {halfdampcycle[i, 6]:,.2f} cycles\n")
        fwrite.write(f" Time to half amplitude:(T2) {doubleamp_time[i, 6]:,.2f} s\n")

        if flag == 1:
            # print(f"\n  ###################################")
            print(f"\n  {'#':#^35}")
            fwrite.write(f"\n  {'#':#^35}\n")
            # print(f"  + UNSTABLE Mode found: < CASE {i+1} > +")
            heading = f" UNSTABLE Mode found: < CASE {i+1} > "
            print(f"  {heading:+^35}")
            print(f"  {'#':#^35}")
            fwrite.write(f"  {heading:+^35}\n")
            fwrite.write(f"  {'#':#^35}\n")
            # print(f"  ###################################")
            flag = 0
        # print(f"\n======================================\n")
        print(f"\n{'=':=^41}\n")
        fwrite.write(f"\n{'=':=^41}\n\n")
fwrite.close()


flightmode = np.array(
    [
        "Roll Damping",
        "Dutch Roll_1",
        "Dutch Roll_2",
        "Spiral",
        "Short period_1",
        "Short period_2",
        "Phugoid_1",
        "Phugoid_2",
    ]
)
modemarker = np.array(["o", "v", "^", "*", "P", "X", "s", "D"])
flightmode_lat = flightmode[0:4]
flightmode_lon = flightmode[4:8]
# colors2 = cm.rainbow(np.linspace(0, 1, 8))
colors2 = ["orange", "green", "green", "red", "blue", "blue", "purple", "purple"]
for i in range(1, numcase_max + 1):
    eigs_real_nn = eigs_real[i]
    eigs_img_nn = eigs_img[i]
    xpmax0 = max(eigs_real_nn[0:4])
    xpmin0 = min(eigs_real_nn[0:4])
    ypmax0 = max(eigs_img_nn[0:4])
    ypmin0 = min(eigs_img_nn[0:4])
    xpmax1 = max(eigs_real_nn[4:8])
    xpmin1 = min(eigs_real_nn[4:8])
    ypmax1 = max(eigs_img_nn[4:8])
    ypmin1 = min(eigs_img_nn[4:8])
    f, axes = plt.subplots(1, 2, figsize=(9, 5))
    for j in range(4):
        axes[0].scatter(
            eigs_real_nn[j],
            eigs_img_nn[j],
            s=50,
            marker=modemarker[j],
            color=colors2[j],
        )
        axes[1].scatter(
            eigs_real_nn[j + 4],
            eigs_img_nn[j + 4],
            s=50,
            marker=modemarker[j + 4],
            color=colors2[j + 4],
        )
    axes[0].legend(flightmode_lat)
    axes[1].legend(flightmode_lon)
    axes[0].grid(True, color="k")
    axes[1].grid(True, color="k")
    axes[0].set_xlim((math.floor(xpmin0 - 1), round(xpmax0 + 0.5) + 0.1))
    axes[0].set_ylim((math.floor(ypmin0), round(ypmax0 + 0.6)))
    axes[1].set_xlim((math.floor(xpmin1 - 0.5), round(xpmax1 + 0.5) + 0.1))
    axes[1].set_ylim((math.floor(ypmin1), round(ypmax1 + 0.6)))
    axes[0].axhline(y=0, xmin=0, xmax=1, c="k")
    axes[0].axvline(x=0, ymin=0, ymax=1, c="k")
    axes[1].axhline(y=0, xmin=0, xmax=1, c="k")
    axes[1].axvline(x=0, ymin=0, ymax=1, c="k")
    axes[0].title.set_text("[Lateral-Directional Modes]")
    axes[1].title.set_text("[Longitudinal Modes]")
    f.suptitle(f"Root Locus Plot\n<{cases[i-1]}>")
    axes[0].set_xlabel("$ζ ω_{n}$   [rad/s]")
    axes[1].set_xlabel("$ζ ω_{n}$   [rad/s]")
    axes[0].set_ylabel("$ω_{d}$   [rad/s]")
    axes[1].set_ylabel("$ω_{d}$   [rad/s]")
    # figname_png = f'savedimg/eigenimg/eigRL-case{i}.png'
    # plt.savefig(figname_png, format = 'png')   # save the figure to file
    # figname_eps = f'savedimg/eigenimg/eigRL-case{i}.eps'

    if closeit == 1:
        figname_png = f"{pngpath}/eigRL-case{i}.png"
        plt.savefig(figname_png, format="png")  # save the figure to file
        # plt.savefig(figname_eps, format = 'eps')   # for LATEX
        plt.show(block=False)
        plt.close()
    else:
        continue


# omega_n_dutchroll1 = w_n[0, 1]
# zeta_dutchroll = zeta[0, 1]

selectedcases = []
t_k = [1, 10, 10, 500, 3.0, 3.0, 500, 500]
for i in range(numcase_max):  # numcase_max
    # for i in selectedcases:
    system_matrix = np.loadtxt(
        f"{filepath}/input/RIMP-02_sys_matrix_case{i+1}.txt",
        skiprows=0,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        dtype=str,
    )
    # print(system_matrix)
    system_matrix = np.delete(system_matrix, 0, 0)
    system_matrix = system_matrix.astype(float)
    # system_matrix = system_matrix.T
    eigs_real_ds = eigs_real[i + 1]
    eigs_img_ds = eigs_img[i + 1]
    # print(eigs_real_ds)
    # print(f"\n")
    # print(eigs_img_ds)
    eigenvalue = np.array((eigs_real_ds, eigs_img_ds), dtype=complex)
    eigenvalue.imag = eigs_img_ds
    eigenvalue = eigenvalue[0]
    # print(eigenvalue)
    for k in range(8):
        system_matrix_k = system_matrix[k]
        # system_matrix_k = system_matrix_k[0:8]
        print(system_matrix_k)
        # print(system_matrix_k)
        eigenvalue_k = eigenvalue[k]
        # print(f"{eigenvalue_k} \n")
        time_ds = np.arange(0, t_k[k], 0.01)
        resp = []
        for ss in time_ds:
            resp_ss = system_matrix_k * np.exp(eigenvalue_k * ss)
            resp.append(resp_ss)
        resp = np.asarray(resp)
        # plt.title(f'CASE{i+1}\n{flightmode[k]}')
        # plt.plot(time_ds, resp)
        # plt.xlabel("Time [s]")
        if k == 0:  # Roll Damping
            plt.title(f"CASE{i+1}\n{flightmode[k]}")
            # plt.plot(time_ds, resp[:, 0:4], label=["v", "p", "r", "Φ"])
            plt.plot(time_ds, resp[:, 0:4], label=["Sideslip", "Roll rate", "Yaw rate", "Roll angle"])
            plt.xlabel("Time [s]")
            # plt.ylabel("Phi [°]")
            plt.legend()
        elif k == 1 or k == 2:  # Dutch Roll
            plt.title(f"CASE{i+1}\n{flightmode[k]}")
            # plt.plot(time_ds, resp[:, 0:4], label=["v", "p", "r", "Φ"])
            plt.plot(time_ds, resp[:, 0:4], label=["Sideslip", "Roll rate", "Yaw rate", "Roll angle"])
            plt.xlabel("Time [s]")
            # plt.ylabel("Phi [°]")
            plt.legend()
        elif k == 3:  # Spiral
            plt.title(f"CASE{i+1}\n{flightmode[k]}")
            # plt.plot(time_ds, resp[:, 0:4], label=["v", "p", "r", "Φ"])
            plt.plot(time_ds, resp[:, 0:4], label=["Sideslip", "Roll rate", "Yaw rate", "Roll angle"])
            plt.xlabel("Time [s]")
            # plt.ylabel("Phi [°]")
            plt.legend()
        elif k == 4 or k == 5:  # Short period
            plt.title(f"CASE{i+1}\n{flightmode[k]}")
            # plt.plot(time_ds, resp[:, 4:8], label=["u", "w", "q", "θ"])
            plt.plot(time_ds, resp[:, 4:8], label=["Speed u/u0", "Angle of Attack", "Pitch rate", "Pitch angle"])
            plt.xlabel("Time [s]")
            # plt.ylabel("u [m/s]")
            plt.legend()
        elif k == 6 or k == 7:  # Phugoid
            plt.title(f"CASE{i+1}\n{flightmode[k]}")
            # plt.plot(time_ds, resp[:, 4:8], label=["u", "w", "q", "θ"])
            plt.plot(time_ds, resp[:, 4:8], label=["Speed u/u0", "Angle of Attack", "Pitch rate", "Pitch angle"])
            plt.xlabel("Time [s]")
            # plt.ylabel("u [m/s]")
            plt.legend()
        plt.grid(True, color="k")

        if closeit == 1:
            figname_png = f"{filepath}/savedimg/{dynstab_folder}/case{i+1}-timeresponse_{flightmode[k]}.png"
            plt.savefig(figname_png, format="png")
            plt.show(block=False)
            plt.close()
        else:
            plt.close()
            continue

# print(f"\n")
# print(system_matrix)
# displacement_dutchroll = np.exp(-1*zeta_dutchroll*omega_n_dutchroll1*time_ds)*(0*np.cos(omega_n_dutchroll1*np.sqrt(1-zeta_dutchroll**2)*time_ds) + 1*np.sin(omega_n_dutchroll1*np.sqrt(1-zeta_dutchroll**2)*time_ds))
# plt.plot(time_ds, displacement_dutchroll)
# plt.show() 'Roll Damping', 'Dutch Roll_1', 'Dutch Roll_2', 'Spiral', 'Short period_1', 'Short period_2', 'Phugoid_1', 'Phugoid_2'
"""
t_j = [1, 15, 15, 500, 1.75, 1.75, 160, 160]
for i in range(numcase_max): #numcase_max
    eigs_real_ds = eigs_real[i+1]
    eigs_img_ds = eigs_img[i+1]
    for j in range(8):
        time_ds = np.arange(0, t_j[j], 0.01)
        omega_n_fm = w_n[i, j]
        zeta_fm = zeta[i, j]
        eigs_sq = np.sqrt(eigs_real_ds[j]**2 + eigs_img_ds[j]**2)
        if j == 0 or j == 3:
            resp = np.exp(-1*zeta_fm*omega_n_fm*time_ds)*(eigs_real_ds[j]/eigs_sq*np.cos(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds) + eigs_img_ds[j]/eigs_sq*np.sin(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds))
        elif j == 4 or j ==5:
            resp = np.exp(-1*zeta_fm*omega_n_fm*time_ds)*(eigs_real_ds[j]/eigs_sq*np.cos(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds) + eigs_img_ds[j]/eigs_sq*np.sin(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds))
        else:
            resp = np.exp(-1*zeta_fm*omega_n_fm*time_ds)*(eigs_img_ds[j]/eigs_sq*np.cos(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds) + eigs_real_ds[j]/eigs_sq*np.sin(omega_n_fm*np.sqrt(1-zeta_fm**2)*time_ds))
        plt.title(f'CASE{i+1}\n{flightmode[j]}')
        plt.plot(time_ds, resp)
        plt.xlabel("Time [s]")
        if j == 0: # Roll Damping
            plt.ylabel("Phi [°]")
        elif j == 1 or j == 2: # Dutch Roll
            plt.ylabel("Phi [°]")
        elif j == 3: # Spiral
            plt.ylabel("Phi [°]")
        elif j == 4 or j == 5: # Short period
            plt.ylabel("u [m/s]")
        elif j == 6 or j == 7: # Phugoid
            plt.ylabel("u [m/s]")
        plt.grid(True, color='k')

        if closeit == 1:
            figname_png = f'{filepath}\savedimg\{dynstab_folder}\case{i+1}-timeresponse_{flightmode[j]}.png'
            plt.savefig(figname_png, format = 'png')
            plt.show(block=False)
            plt.close()
        else:
            continue
"""

AD_length = AD_inverse_Class.Length()

loadfactor = 4.4                               #Positive load factor
safac = 1.25                                     #Safety Factor
n_ult = loadfactor*safac                          #Ultimate load factor
W_fw = 0                                      #kg, Weight of the fuel in wing
tc = 0.12                                       #Airfoil Thickness / Chord
tc_r = 0.12                                     #tc @ root
tc_t = 0.12                                     #tc @ tip
tc_HT = 0.09
tc_VT = 0.09
t_r_HT = Croot_HT * tc
t_r_VT = Croot_VT * tc
N_pax = 5                                      #Number of Passengers
N_row = 2                                      #Number of Passenger Seat Rows
W_crew = 75*N_pax                                    #kg, Crew mass
W_avionic = 34
W_payload = 15*5
hf_mm = 1200                                  #mm, maximum fuselage height
Pmax = AD_length.meter_feet(2*math.pi*(math.sqrt(((Wf_mm/1000)**2 + (hf_mm/1000)**2)/8)), 'ft')       #ft, Maximum fuselage perimeter
W_press = 0                                    #kg, Pressurization Weight, See raymer

W_misc = 200                                     #kg, misc. Weight: for future design changes

LD = 14                                        #L/D_Cruise
l_n_mm = l_n_mm                                  #length of nose

W_pwr = 117*4                                     #kg, 6kW/kg Electric Motor, Dry mass

delta4_weight = AD_inverse_Class.WeightEst()

[W_imp, W_fw_imp, S_imp, b_imp, q, q_imp, S_HT_imp, S_VT_imp, t_r_w, t_r_w_imp, t_r_HT_imp, t_r_VT_imp, L_HT_act_imp, b_HT_imp, b_VT_imp, FL_imp, Wf_imp, hf_imp, Swet_fus_imp, Swet_fus, W_press_imp] = delta4_weight.imperial(W, W_fw, S, b, rho_a_cruise, v_cruise, t_r_HT, S_HT, S_VT, t_r_VT, L_HT_act, b_HT, b_VT, FL, Wf_mm, hf_mm, W_press, FL_short, l_n_mm, Croot, tc_r)
[W_w_imp_cessna, W_w_cessna, W_h_imp_cessna, W_v_imp_cessna, W_h_cessna, W_v_cessna, W_f_imp_cessna, W_f_cessna, W_fc_cessna, W_els_cessna, W_fur_cessna, W_fc_imp_cessna, W_els_imp_cessna, W_fur_imp_cessna] = delta4_weight.cessnamethod(W_imp, S_imp, n_ult, A, S_HT_imp, A_HT, t_r_HT_imp, S_VT_imp, A_VT, t_r_VT_imp, theta_c4_VT, Pmax, FL_imp, N_pax)
[W_w_imp_usaf, W_w_usaf, W_h_imp_usaf, W_h_usaf, W_v_imp_usaf, W_v_usaf, W_f_imp_usaf, W_f_usaf, W_fc_imp_usaf, W_fc_usaf] = delta4_weight.usafmethod(W_imp, n_ult, A, theta_c4, S_imp, lamda, tc, v_kn_cruise, S_HT_imp, L_HT_act_imp, b_HT_imp, t_r_HT_imp, S_VT_imp, b_VT_imp, t_r_VT_imp, FL_imp, Wf_imp, hf_imp)
[W_w_imp_raymer, W_w_raymer, W_h_imp_raymer, W_h_raymer, W_v_imp_raymer, W_v_raymer, W_f_imp_raymer, W_f_raymer] = delta4_weight.raymermethod(S_imp, W_fw_imp, A, theta_c4, q_imp, lamda, tc, n_ult, W_imp, S_HT_imp, theta_c4_HT, lamda_HT, S_VT_imp, theta_c4_VT, lamda_VT, Swet_fus_imp, L_HT_act_imp, LD, W_press_imp)
[W_w_torenbeek_imp, W_w_torenbeek, W_iae_imp_torenbeek, W_iae_torenbeek, W_fur_torenbeek] = delta4_weight.torenbeekmethod(W_imp, b_imp, theta_c2, n_ult, S_imp, t_r_w_imp, N_pax, N_row)
[W_w_vtol_imp, W_w_vtol, W_landingskid_imp, W_landingskid] = delta4_weight.vtolweightest(W_imp, S_imp, lamda, A, theta_LE, tc)
[W_w_mean, W_w_mean_short, W_h_mean, W_h_mean_short, W_v_mean, W_v_mean_short, W_f_mean, W_f_mean_short, W_fc_mean, W_fur_mean] = delta4_weight.meanval(W_w_cessna, W_w_usaf, W_w_raymer, W_w_torenbeek, W_h_cessna, W_h_usaf, W_v_cessna, W_v_usaf, W_f_cessna, W_f_usaf, W_h_raymer, W_v_raymer, W_f_raymer, W_fc_cessna, W_fc_usaf, W_fur_cessna, W_fur_torenbeek)
[W_tot_mean, W_tot_eVTOL_mean, W_total_mean, W_total_eVTOL_mean] = delta4_weight.totalweight(W_crew, W_w_mean, W_h_mean, W_v_mean, W_f_mean, W_avionic, W_payload, W_pwr, W_fc_mean, W_fur_mean, W_landingskid, W_misc)

print("\n------Weight Estimation------")
# print('W_w_mean: ' + f"{W_w_mean:,.0f}" + ' kg')
# print('W_w_vtol: ' + f"{W_w_vtol:,.0f}", ' kg')
# print('W_h_mean: ' + f"{W_h_mean:,.1f}" + ' kg')
# print('W_v_mean: ' + f"{W_v_mean:,.1f}" + ' kg')
# print('W_f_mean: ' + f"{W_f_mean:,.0f}" + ' kg')
# print('W_misc_mean: ' + f"{W_misc:,.0f}" + ' kg')
# print('\nW_empty_mean: ' + f"{W_tot_mean:,.0f}" + ' kg')
# print('W_dry_mean: ' + f"{W_total_mean:,.0f}" + ' kg')
# print('\nW_empty_eVTOL_mean: ' + f"{W_tot_eVTOL_mean:,.0f}" + ' kg')
# print('W_empty+pax_eVTOL_mean: ' + f"{W_total_eVTOL_mean:,.0f}" + ' kg')


W1_WTO = 0.998
W2_W1 = 0.998
W3_W2 = 0.998

W7_W6 = 0.995
W8_W7 = 0.995

E_climb = 0.19                 #hours, time of climb to altitude
v_climb_kn = 45                #climb velocity
Cp_climb = 0.454
eta_p_climb = 0.8
LD_climb = 9

R_cruise = 30                 #km, Range
Cp_cruise = 0.454
eta_p_cruise = 0.7
LD_cruise = 11.5

E_loiter = 5/60
v_loiter_kn = 50
Cp_loiter = 0.454
eta_p_loiter = 0.6
LD_loiter = 12

rho_fuel = 970             #kg/m3, Typical Automotive Gasoline Density
W_fuel_res = 12             #kg, reserve fuel

ADconceptual_conceptweight = AD_inverse_Class.conceptWeight()
[W4_W3, W5_W4, W6_W5, Mff, W_f_used_lbs, W_f_lbs, W_f, W_empty, V_fuel, V_fuel_liters] = ADconceptual_conceptweight.prefuel(W1_WTO, W2_W1, W3_W2, W7_W6, W8_W7, Cp_climb, Cp_cruise, Cp_loiter, eta_p_climb, eta_p_cruise, eta_p_loiter, LD_climb, LD_cruise, LD_loiter, R_cruise, E_climb, E_loiter, v_climb_kn, v_loiter_kn, W, W_fuel_res, rho_fuel)
[V_WF_ft3, V_WF, V_WF_min, V_WF_max, V_WF_liters, V_WF_liters_min, V_WF_liters_max, V_WF_liters_diff] = ADconceptual_conceptweight.wingfuelvol(S_imp, b, lamda, tc_r, tc_t)

print("\n------Conceptual Weight------")
print('W_dry_est: ' + f"{W_empty:,.0f}" + ' kg')
print('W_fuel_est:  ' + f"{W_f:,.0f}" + ' kg')
print('Required_Fuel_Volume: ' + f"{V_fuel_liters:,.0f}" + ' L')
print('(Wing_tank_Vol: ' + f"{V_WF_liters:,.0f}" + 'L ± ' + f"{V_WF_liters_diff:,.0f}" + 'L)')

W_tot_eVTOL_mean = 10.2
"""eVTOL Battery Calculations"""
W_P_eVTOL = 2.8                                # kg/kW, Power loading (for tilt-rotor: 2.4 ~ 3.1) | 2200/650
LD_cruise_eVTOL = 11                                # L/D @ Cruise
LD_loiter_eVTOL = LD_cruise_eVTOL*0.866                                # L/D @ Loiter
LD_climb_eVTOL = 12                                 # L/D @ Climb
E_loiter_eVTOL = 5/60                              # hour, Loiter Time
v_loiter = v_loiter_kn*1.85184                    # km/h, Loiter Velocity
v_climb = v_knot_cruise*0.76*1.85184                # km/h, Climb Velocity
h_climb_alt_eVTOL = 150/1000                             # km, Climb Altitude (Max ALT = 3km, Cruise ALT = 0.3 ~ 0.5km)
R_cruise_eVTOL = 30                                # km, Cruise Range
# W_payload_eVTOL = W_crew + W_payload                # Payload Mass (Passenger + Baggage)
W_payload_eVTOL = 0.1                # Payload Mass (Passenger + Baggage)
We_W0_eVTOL = W_tot_eVTOL_mean/W                    # empty weight / gross weight
Esb = 170                                           # Wh/kg, Battery Specific Energy ; Li-ion(100~265), LiS(400)
eta_b2s = 0.9                                       # Total system efficiency from battery to motor output shaft
eta_p_eVTOL = 0.83                                  # Propeller Efficiency
eta_power_cruise = 0.75                             # Power use @ cruise
eta_power_loiter = 0.70                             # Power use @ loiter
eta_power_climb = 0.85                              # Power use @ climb
Pused_cruise = (W/W_P_eVTOL)*eta_power_cruise
Pused_loiter = (W/W_P_eVTOL)*eta_power_loiter
Pused_climb = (W/W_P_eVTOL)*eta_power_climb

m_b_guess = 4.2                                 #kg, initial Battery mass Guess
ADconceptual_eVTOL_BMF = AD_inverse_Class.eVTOL_BMF(LD_loiter_eVTOL, LD_cruise_eVTOL, LD_climb_eVTOL, Esb, eta_b2s, eta_p_eVTOL, g, W, Pused_climb, v_climb)
[E_eVTOL, R_eVTOL, V_v_eVTOL] = ADconceptual_eVTOL_BMF.endurance_range(v_cruise_kmh, m_b_guess)
[BMF_loiter, BMF_cruise, BMF_climb, BMF_sum, Battery_available, W0_eVTOL, W0_eVTOL2] = ADconceptual_eVTOL_BMF.bmf(E_loiter_eVTOL, R_cruise_eVTOL, h_climb_alt_eVTOL, v_loiter, V_v_eVTOL, Pused_climb, W_tot_eVTOL_mean, W_payload_eVTOL, We_W0_eVTOL, m_b_guess)
# print("\n++++++Battery Mass Fraction++++++")
# print("=================================")
# print(f'Range_eVTOL: {R_cruise_eVTOL:,.0f} km')
# print(f'Loiter_Time_eVTOL: {E_loiter_eVTOL*60:,.0f} min')
# print(f'm_TO_eVTOL: {W0_eVTOL:,.0f} kg')
# print(f'Fill-ratio: {((W0_eVTOL - W)/W*100):,.2f}% < 0')
# print("---------------------------------")
# print(f'Curr. Battery Capacity: {m_b_guess*Esb/1000:,.0f} kWh')
# print(f'Battery_available: {Battery_available:,.0f} kg')
# print("---------------------------------")
# # print("Endurance_eVTOL: " + f"{E_eVTOL:,.3f}" + ' hour')
# print(f'Endurance_eVTOL: {E_eVTOL:,.3f} hour')
# # print("Range_eVTOL: " + f"{R_eVTOL:,.0f}" + ' km')
# print(f'Range_eVTOL: {R_eVTOL:,.0f} km')
# # print("Battery_available: " + f"{Battery_available:,.0f}" + ' kg')
# # print("m_TO_eVTOL: " + f"{W0_eVTOL:,.0f}" + ' kg')
# # print("m_TO_eVTOL_updated: " + f"{W0_eVTOL2:,.0f}" + ' kg')
# print(f'm_TO_eVTOL_Potential: {W0_eVTOL2:,.0f} kg')
# print(f'Fill-ratio: {((W0_eVTOL2 - W)/W*100):,.2f}% < 0')
# print("=================================")
# print(f'\nBMFp_climb: {BMF_climb*100:,.1f} %')
# print(f'BMFp_cruise: {BMF_cruise*100:,.0f} %')
# print(f'BMFp_loiter: {BMF_loiter*100:,.1f} %')
# print("===================")
# print(f'BMFp_sum: {BMF_sum*100:,.0f} %')

print("\n+++++++++++++++++++++++++++++++++")
merit = 0.75
eta_mech = 0.93
f_adj = 1.03
E_vertiTO = 9/60                # 3 minutes
E_loiter_eVTOL_init = 5/60
E_vertiland = 2/60              # 3 minutes
R_cruise_eVTOL_g = 30
[W0, Wf_W0, Wbat] = ADconceptual_eVTOL_BMF.BMFsizing(W_payload_eVTOL, E_loiter_eVTOL_init, R_cruise_eVTOL_g, V_v_eVTOL, h_climb_alt_eVTOL, Pused_climb, rho_a_cruise, WS, merit, eta_mech, f_adj, v_loiter, E_vertiTO, E_vertiland)
# print(f"range: {R_cruise_eVTOL_g:,.0f} km")
# print(f"Endurance_TO: {E_vertiTO*60:,.0f} min")
# print(f"Endurance_Landing: {E_vertiland*60:,.0f} min")
# print(f"W0: {W0:,.0f} kg")
# print(f"Wbat/W0: {Wf_W0:,.3f}")
# print(f"Wbat: {Wbat:,.0f} kg")
print("+++++++++++++++++++++++++++++++++")

theta_tc_max_deg = theta_c3_deg               #Wing sweep angle of maximum airfoil thickness
theta_tc_max_deg_HT = theta_c3_deg_HT
theta_tc_max_deg_VT = theta_c3_deg_VT
S_exp_plf = Croot*Wf_mm/1000
# Lprime = 2                    # chord pos @ max t/c < 30% of chord
Lprime = 1.2                   # chord pos @ max t/c >= 30% of chord
Lprime_HT = 1.2
Lprime_VT = 1.2
# R_LS = 1.07

inc_agl_w_deg = 0                  #Angle of Incident of Wing
inc_agl_HT_deg = -3              #Angle of Incident of Horizontal Tail
eplison_t_deg = -3
# l_LER = 0.0248                 #Airfoil Leading Edge Radius: NACA_23015
l_LER = 1.58/100
aoa_deg = alpha_cruise                        #Angle of Attack of Aircraft
R = 0.95                       #leading edge suction parameter
#d_b = Wf_mm/1000
#d_f_x_mm = 200                 #Fuselage Base Surface X axis length
#d_f_y_mm = 140                 #Fuselage Base Surface Y axis length

Swet_fus = math.pi*Wf_mm/1000*FL*(0.5+0.135*(l_n_mm/1000/FL))**(2/3) * (1.015+0.3/((FL/(Wf_mm/1000)**(3/2))))
print(f'Swet_fus_est.: {Swet_fus:,.2f} m^2')

Swet_fus = 0.9              #Fuslage Wetted Area
fuselage_length = FL        #Fuslage Length

AD_drag = AD_inverse_Class.Drag()

[mu_fus, Re_fus, Rwf, Cfw, Cf_fus, Cfw_HT, Cfw_VT] = AD_drag.curvefit(Ta_cruise, rho_a_cruise, v_cruise, FL_short, Mach_cruise, Re_W_cruise, Re_HT, Re_VT)
print("\n------Drag_Estimates------")
print(f"Re_fus: {Re_fus:,.2E}")
print("Rwf: " + f"{Rwf:,.3e}")
print("Cfw: " + f"{Cfw:,.3e}")
print("Cf_fus: " + f"{Cf_fus:,.3e}")


[CD_wing, CD_O_W, CD_O_HT, CD_O_VT] = AD_drag.wingcd(W, S_HT, tc, tc_r, tc_t, S, Croot, Ctip, Wf_mm, lamda, Cfw, Lprime, theta_tc_max_deg, theta_LE, Re_fus,
                                                     tc_HT, tc_VT, lamda_HT, lamda_VT, S_VT, S_exp_plf, l_LER, CLa_HT_deg, aoa_deg, inc_agl_HT_deg, q,
                                                     rho_a_cruise, v_cruise, mu_cruise, Mach_cruise, CLa_W, A, R, Cfw_HT, Cfw_VT, Lprime_HT, Lprime_VT,
                                                     theta_tc_max_deg_HT, theta_tc_max_deg_VT)
CD_O_fus = AD_drag.fuscd(Croot, Wf_mm, hf_mm, FL, S, Re_fus, Cf_fus, Swet_fus)

[vel_knot, vel_km1h, D_tot, D_tot_kgf, D_induced, D_induced_kgf, D_parasitic, D_parasitic_kgf, T_req0, T_req1, T_req2, T_req0_kgf, T_req1_kgf, T_req2_kgf, pwr_req, pwr_req0, pwr_req1, pwr_req2, pwr_req_climb, CD_O_tot, v_drag_min, drag_min, power_min] = \
    AD_drag.dragpolar(S, CD_O_W, CD_O_HT, CD_O_VT, fuselage_length, Swet_fus, Wf_mm, Cf_fus, A, e, W, g, rho_a_cruise, v_stall_TO, v_cruise, v_cruise2, v_climb, CD_O_fus)

print(f"CD_O_fus: {CD_O_fus:,.4f}")
print(f'\nCD_O_tot: {CD_O_tot:,.5f}')
print(f"\nThrust Req. @ {v_knot_cruise/1.9438:,.0f} m/s: {T_req1_kgf:,.0f} kgf")
print(f"Thrust Req. @ {v_knot_cruise2/1.9438:,.0f} m/s: {T_req2_kgf:,.0f} kgf")

print(f"\nPower Req._climb @ {v_climb/3.6:,.0f} m/s: {pwr_req_climb/1000:,.0f} kW")
print(f"Power Req. @ {v_knot_cruise/1.9438:,.0f} m/s: {pwr_req1/1000:,.0f} kW")
print(f"Power Req2. @ {v_knot_cruise2/1.9438:,.0f} m/s: {pwr_req2/1000:,.0f} kW")

plt.title("Drag vs Velocity")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Drag (N)")
plt.grid(True, color='k')
plt.plot(vel_knot/1.9438, D_parasitic, 'g', linewidth=1.5, label="Total Parasitic Drag (N)")
plt.plot(vel_knot/1.9438, D_induced, 'b', linewidth=1.5, label="Total Induced Drag (N)")
plt.plot(vel_knot/1.9438, D_tot, 'r', linewidth=2.5, label="Total Drag (N)")
plt.plot(v_stall_TO_knot/1.9438, T_req0, 'm-o', linewidth=2.5)
plt.plot(v_kn_cruise/1.9438, T_req1, 'r-o', linewidth=2.5)
plt.plot(v_kn_cruise2/1.9438, T_req2, 'r-o', linewidth=2.5)
plt.scatter(v_drag_min/1.9438, drag_min, c='k', marker='o')
plt.legend()
# plt.show(block=False)
plt.close()

plt.title("Req. Power vs Velocity")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Req. Power (kW)")
plt.grid(True, color='k')
plt.plot(vel_knot/1.9438, pwr_req/1000, 'r', linewidth = 2.5, label='Required Power (kW)')
plt.plot(v_stall_TO_knot/1.9438, pwr_req0/1000, 'r-o', linewidth=2.5)
plt.plot(v_kn_cruise/1.9438, pwr_req1/1000, 'r-o', linewidth=2.5)
plt.plot(v_kn_cruise2/1.9438, pwr_req2/1000, 'r-o', linewidth=2.5)
# plt.scatter(v_drag_min, power_min/1000, c='k', marker='o')
# plt.show(block=False)
plt.close()




'''
loadfactor = 6                               #Positive load factor
safac = 1.5                                     #Safety Factor
n_ult = loadfactor*safac                          #Ultimate load factor
W_fw = 0                                      #kg, Weight of the fuel in wing
tc = 0.12                                       #Airfoil Thickness / Chord
tc_r = 0.12                                     #tc @ root
tc_t = 0.12                                     #tc @ tip
tc_HT = 0.09
tc_VT = 0.09
t_r_HT = Croot_HT * tc
t_r_VT = Croot_VT * tc
Pmax = 1
N_pax = 1                                      #Number of Passengers
N_row = 1                                      #Number of Passenger Seat Rows
W_crew = 75                                    #kg, Crew mass
W_avionic = 20
W_payload = 50
hf_mm = Wf_mm                                  #mm, maximum fuselage height
W_press = 0                                    #kg, Pressurization Weight, See raymer
LD = 10                                        #L/D_Cruise
l_n_mm = 400                                  #length of nose

W_pwr = 60                                     #kg, Rotax 912 Dry mass
"""
delta4_weight = AD_inverse_Class.WeightEst()

[W_imp, W_fw_imp, S_imp, b_imp, q, q_imp, S_HT_imp, S_VT_imp, t_r_w, t_r_w_imp, t_r_HT_imp, t_r_VT_imp, L_HT_act_imp, b_HT_imp, b_VT_imp, FL_imp, Wf_imp, hf_imp, Swet_fus_imp, Swet_fus, W_press_imp] = delta4_weight.imperial(W, W_fw, S, b, rho_a_cruise, v_cruise, t_r_HT, S_HT, S_VT, t_r_VT, L_HT_act, b_HT, b_VT, FL, Wf_mm, hf_mm, W_press, FL_short, l_n_mm, Croot, tc_r)
[W_w_imp_cessna, W_w_cessna, W_h_imp_cessna, W_v_imp_cessna, W_h_cessna, W_v_cessna, W_f_imp_cessna, W_f_cessna, W_fc_cessna, W_els_cessna, W_fur_cessna, W_fc_imp_cessna, W_els_imp_cessna, W_fur_imp_cessna] = delta4_weight.cessnamethod(W_imp, S_imp, n_ult, A, S_HT_imp, A_HT, t_r_HT_imp, S_VT_imp, A_VT, t_r_VT_imp, theta_c4_VT, Pmax, FL_imp, N_pax)
[W_w_imp_usaf, W_w_usaf, W_h_imp_usaf, W_h_usaf, W_v_imp_usaf, W_v_usaf, W_f_imp_usaf, W_f_usaf, W_fc_imp_usaf, W_fc_usaf] = delta4_weight.usafmethod(W_imp, n_ult, A, theta_c4, S_imp, lamda, tc, v_kn_cruise, S_HT_imp, L_HT_act_imp, b_HT_imp, t_r_HT_imp, S_VT_imp, b_VT_imp, t_r_VT_imp, FL_imp, Wf_imp, hf_imp)
[W_w_imp_raymer, W_w_raymer, W_h_imp_raymer, W_h_raymer, W_v_imp_raymer, W_v_raymer, W_f_imp_raymer, W_f_raymer] = delta4_weight.raymermethod(S_imp, W_fw_imp, A, theta_c4, q_imp, lamda, tc, n_ult, W_imp, S_HT_imp, theta_c4_HT, lamda_HT, S_VT_imp, theta_c4_VT, lamda_VT, Swet_fus_imp, L_HT_act_imp, LD, W_press_imp)
[W_w_torenbeek_imp, W_w_torenbeek, W_iae_imp_torenbeek, W_iae_torenbeek, W_fur_torenbeek] = delta4_weight.torenbeekmethod(W_imp, b_imp, theta_c2, n_ult, S_imp, t_r_w_imp, N_pax, N_row)
[W_w_vtol_imp, W_w_vtol, W_landingskid_imp, W_landingskid] = delta4_weight.vtolweightest(W_imp, S_imp, lamda, A, theta_LE, tc)
[W_w_mean, W_w_mean_short, W_h_mean, W_h_mean_short, W_v_mean, W_v_mean_short, W_f_mean, W_f_mean_short, W_fc_mean, W_fur_mean] = delta4_weight.meanval(W_w_cessna, W_w_usaf, W_w_raymer, W_w_torenbeek, W_h_cessna, W_h_usaf, W_v_cessna, W_v_usaf, W_f_cessna, W_f_usaf, W_h_raymer, W_v_raymer, W_f_raymer, W_fc_cessna, W_fc_usaf, W_fur_cessna, W_fur_torenbeek)
[W_tot_mean, W_tot_eVTOL_mean, W_total_mean, W_total_eVTOL_mean] = delta4_weight.totalweight(W_crew, W_w_mean, W_h_mean, W_v_mean, W_f_mean, W_avionic, W_payload, W_pwr, W_fc_mean, W_fur_mean, W_landingskid)

print("\n------Weight Estimation------")
print('W_w_mean: ' + f"{W_w_mean:,.0f}" + ' kg')
print('W_h_mean: ' + f"{W_h_mean:,.1f}" + ' kg')
print('W_v_mean: ' + f"{W_v_mean:,.1f}" + ' kg')
print('W_f_mean: ' + f"{W_f_mean:,.0f}" + ' kg')
print('\nW_empty_mean: ' + f"{W_tot_mean:,.0f}" + ' kg')
print('W_dry_mean: ' + f"{W_total_mean:,.0f}" + ' kg')
#print(W_h_raymer)
#print(W_v_raymer)
#print(W_f_raymer)
#print(S_f_imp)
#print(q_imp)

W1_WTO = 0.998
W2_W1 = 0.998
W3_W2 = 0.998

W7_W6 = 0.995
W8_W7 = 0.995

E_climb = 0.19                 #hours, time of climb to altitude
v_climb_kn = 100                #climb velocity
Cp_climb = 0.454
eta_p_climb = 0.8
LD_climb = 9

R_cruise = 1500                 #km, Range
Cp_cruise = 0.454
eta_p_cruise = 0.7
LD_cruise = 11.5

E_loiter = 1.2
v_loiter_kn = 110
Cp_loiter = 0.454
eta_p_loiter = 0.6
LD_loiter = 12

rho_fuel = 710             #kg/m3, Typical Automotive Gasoline Density
W_fuel_res = 12             #kg, reserve fuel

ADconceptual_conceptweight = AD_inverse_Class.conceptWeight()
[W4_W3, W5_W4, W6_W5, Mff, W_f_used_lbs, W_f_lbs, W_f, W_empty, V_fuel, V_fuel_liters] = ADconceptual_conceptweight.prefuel(W1_WTO, W2_W1, W3_W2, W7_W6, W8_W7, Cp_climb, Cp_cruise, Cp_loiter, eta_p_climb, eta_p_cruise, eta_p_loiter, LD_climb, LD_cruise, LD_loiter, R_cruise, E_climb, E_loiter, v_climb_kn, v_loiter_kn, W, W_fuel_res, rho_fuel)
[V_WF_ft3, V_WF, V_WF_min, V_WF_max, V_WF_liters, V_WF_liters_min, V_WF_liters_max, V_WF_liters_diff] = ADconceptual_conceptweight.wingfuelvol(S_imp, b, lamda, tc_r, tc_t)

print("\n------Conceptual Weight------")
print('W_dry_est: ' + f"{W_empty:,.0f}" + ' kg')
print('W_fuel_est:  ' + f"{W_f:,.0f}" + ' kg')
print('Required_Fuel_Volume: ' + f"{V_fuel_liters:,.0f}" + ' L')
print('(Wing_tank_Vol: ' + f"{V_WF_liters:,.0f}" + 'L ± ' + f"{V_WF_liters_diff:,.0f}" + 'L)')
"""
q = 0.5*rho_a*v_cruise**2
v_climb = 10

theta_tc_max_deg = theta_c3_deg               #Wing sweep angle of maximum airfoil thickness
theta_tc_max_deg_HT = theta_c3_deg_HT
theta_tc_max_deg_VT = theta_c3_deg_VT
S_exp_plf = Croot*Wf_mm/1000
# Lprime = 2                    # chord pos @ max t/c < 30% of chord
Lprime = 1.2                   # chord pos @ max t/c >= 30% of chord
Lprime_HT = 1.2
Lprime_VT = 1.2
# R_LS = 1.07

inc_agl_w_deg = 0                  #Angle of Incident of Wing
inc_agl_HT_deg = -3              #Angle of Incident of Horizontal Tail
eplison_t_deg = -3
# l_LER = 0.0248                 #Airfoil Leading Edge Radius: NACA_23015
l_LER = 1.58/100
aoa_deg = alpha_cruise                        #Angle of Attack of Aircraft
R = 0.95                       #leading edge suction parameter
#d_b = Wf_mm/1000
#d_f_x_mm = 200                 #Fuselage Base Surface X axis length
#d_f_y_mm = 140                 #Fuselage Base Surface Y axis length

Swet_fus = math.pi*Wf_mm/1000*FL*(0.5+0.135*(l_n_mm/1000/FL))**(2/3) * (1.015+0.3/((FL/(Wf_mm/1000)**(3/2))))
print(f'Swet_fus_est.: {Swet_fus:,.2f} m^2')

Swet_fus = 24.902              #Fuslage Wetted Area
fuselage_length = FL        #Fuslage Length

AD_drag = AD_inverse_Class.Drag()

[mu_fus, Re_fus, Rwf, Cfw, Cf_fus, Cfw_HT, Cfw_VT] = AD_drag.curvefit(Ta_cruise, rho_a_cruise, v_cruise, FL_short, Mach_cruise, Re_W_cruise, Re_HT, Re_VT)
print("\n------Drag_Estimates------")
print(f"Re_fus: {Re_fus:,.2E}")
print("Rwf: " + f"{Rwf:,.3e}")
print("Cfw: " + f"{Cfw:,.3e}")
print("Cf_fus: " + f"{Cf_fus:,.3e}")


[CD_wing, CD_O_W, CD_O_HT, CD_O_VT] = AD_drag.wingcd(W, S_HT, tc, tc_r, tc_t, S, Croot, Ctip, Wf_mm, lamda, Cfw, Lprime, theta_tc_max_deg, theta_LE, Re_fus,
                                                     tc_HT, tc_VT, lamda_HT, lamda_VT, S_VT, S_exp_plf, l_LER, CLa_HT_deg, aoa_deg, inc_agl_HT_deg, q,
                                                     rho_a_cruise, v_cruise, mu_cruise, Mach_cruise, CLa_W, A, R, Cfw_HT, Cfw_VT, Lprime_HT, Lprime_VT,
                                                     theta_tc_max_deg_HT, theta_tc_max_deg_VT)
CD_O_fus = AD_drag.fuscd(Croot, Wf_mm, hf_mm, FL, S, Re_fus, Cf_fus, Swet_fus)

[vel_knot, vel_km1h, D_tot, D_tot_kgf, D_induced, D_induced_kgf, D_parasitic, D_parasitic_kgf, T_req0, T_req1, T_req2, T_req0_kgf, T_req1_kgf, T_req2_kgf, pwr_req, pwr_req0, pwr_req1, pwr_req2, pwr_req_climb, CD_O_tot] = \
    AD_drag.dragpolar(S, CD_O_W, CD_O_HT, CD_O_VT, fuselage_length, Swet_fus, Wf_mm, Cf_fus, A, e, W, g, rho_a_cruise, v_stall_TO, v_cruise, v_cruise2, v_climb, CD_O_fus)

print(f"\nThrust Req. @ {v_cruise_kmh:,.0f} km/h: {T_req1_kgf:,.2f} kgf")
print(f"Thrust Req. @ {v_cruise2_kmh:,.0f} km/h: {T_req2_kgf:,.2f} kgf")
print(f"\nPower Req. @ {v_cruise_kmh:,.0f} km/h: {pwr_req1:,.0f} W")
print(f"Power Req. @ {v_cruise2_kmh:,.0f} km/h: {pwr_req2:,.0f} W")


print(f"{X_bar_ac_w}")
print(f"{CLa_W}")
print(f"CLa_WF_deg: {math.degrees(CLa_WF_deg)}")

plt.title("Drag vs Velocity")
plt.xlabel("Velocity (km/h)")
plt.ylabel("Drag (kgf)")
plt.plot(vel_km1h, D_tot_kgf, 'k', linewidth=2)
plt.plot(v_stall_TO_kmh, T_req0_kgf, 'k-o', linewidth=2)
plt.plot(v_cruise_kmh, T_req1_kgf, 'k-o', linewidth=2)
plt.plot(v_cruise2_kmh, T_req2_kgf, 'k-o', linewidth=2)
plt.show()

plt.title("Req. Power vs Velocity")
plt.xlabel("Velocity (km/h)")
plt.ylabel("Req. Power (W)")
plt.plot(vel_km1h, pwr_req, 'r', linewidth = 2.5, label='Required Power (W)')
plt.plot(v_stall_TO_kmh, pwr_req0, 'r-o', linewidth=2.5)
plt.plot(v_cruise_kmh, pwr_req1, 'r-o', linewidth=2.5)
plt.plot(v_cruise2_kmh, pwr_req2, 'r-o', linewidth=2.5)
plt.show()
'''
"""
###ONTROL SURFACE
b_perflap = 50                #[%], flap span-wise percentage- FAA:"flap should be over 50% of wing span"
C_perflap = 30                #[%], Aileron Chord-wise percentage- Should be wing chord's 25% ~ 35% range

b_peraileron = 50             #[%], Aileron span-wise percentage- Aileron span about 50% of flap span"
C_peraileron = 25             #[%], Aileron Chord-wise percentage- Should be wing chord's 25% ~ 35% range; Consider Drag

[bflap, bflap_mm, Cflap, Cflap_mm, Swf_S] = delta4_main_wing.controlsurfw(Croot, b_perflap, C_perflap, Wf_mm, lamda)

print("\n======Control Surfaces======")
print("----------  Wing  ----------")
print("==FLAP==")
print('bflap/2: ' + f"{bflap_mm:,.0f}" + ' mm')
print('Cflap: ' + f"{Cflap_mm:,.0f}" + ' mm')
print('Swf/S: ' + f"{Swf_S:,.3f}")
print("==AILERON==")
print("\n------Horizontal Tail------")
print("==FLAP==")
print("==ELERVATOR==")
print("\n-------Vertical Tail-------")
print("==FLAP==")
print("==RUDDER==")
"""

# Carpet Plot
