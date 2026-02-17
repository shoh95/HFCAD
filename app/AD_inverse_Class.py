
"""
FILE NAME: AD_inverse_Class.py
Code Builder: [Seunghwan Oh]
"""


class Sciexp:
    def __init__(self):
        pass

    def exponent(self, val, n):
        if n > 0:
            newvalpos = val * 10**n
            return newvalpos
        elif n < 0:
            newvalneg = val * 10**n
            return newvalneg


class Mass:
    def __init__(self):
        pass

    def kg_pound(self, mass, conv_to):
        if conv_to == "kg":
            kg = mass * 0.45359237
            return kg
        elif conv_to == "lbs":
            lbs = mass / 0.45359237
            return lbs


class Length:
    def __init__(self):
        pass

    def meter_feet(self, length, conv_to):
        if conv_to == "meter":
            meter = length * 0.3048
            return meter
        elif conv_to == "ft":
            feet = length / 0.3048
            return feet

    def meter_inch(self, length, conv_to):
        if conv_to == "meter":
            meter = length * 25.4 / 1000
            return meter
        elif conv_to == "inch":
            inch = length / 25.4 * 1000
            return inch

    def meter_yard(self, length, conv_to):
        if conv_to == "meter":
            meter = length * 0.9144
            return meter
        elif conv_to == "yard":
            yard = length / 0.9144
            return yard

    def km_nmi(self, length, conv_to):
        if conv_to == "km":
            km = length * 1.852
            return km
        elif conv_to == "nmi":
            nmi = length / 1.852
            return nmi

    def km_statmile(self, length, conv_to):
        if conv_to == "statmile":
            statmile = length / 1.609344
            return statmile
        elif conv_to == "km":
            km = length * 1.609344
            return km


class Area:
    def __init__(self):
        pass

    def m2_ft2(self, area, conv_to):
        if conv_to == "ft2":
            ft2 = area * 10.76391042
            return ft2
        elif conv_to == "m2":
            m2 = area / 10.76391042
            return m2


class Volume:
    def __init__(self):
        pass

    def m3_ft3(self, volume, conv_to):
        if conv_to == "m3":
            m3 = volume / 35.31466672
            return m3
        elif conv_to == "ft3":
            ft3 = volume * 35.31466672


class Velocity:
    def __init__(self):
        pass

    def m1s_kn(self, vel, conv_to):
        if conv_to == "kn":
            kn = vel / 0.5144
            return kn
        elif conv_to == "m1s":
            m1s = vel * 0.5144
            return m1s

    def km1h_m1s(self, vel, conv_to):
        if conv_to == "km1h":
            km1h = vel * 3.6
            return km1h
        elif conv_to == "m1s":
            m1s = vel / 3.6
            return m1s

    def km1h_kn(self, vel, conv_to):
        if conv_to == "kn":
            kn = vel * 0.540003
            return kn
        elif conv_to == "km1h":
            km1h = vel / 0.540003
            return km1h

    def kn_mph(self, vel, conv_to):
        if conv_to == "mph":
            mph = vel * 1.150779448
            return mph
        elif conv_to == "kn":
            kn = vel / 1.150779448
            return kn

    def m1s_ft1s(self, vel, conv_to):
        if conv_to == "ft1s":
            ft1s = vel * 3.28084
            return ft1s
        elif conv_to == "m1s":
            m1s = vel / 3.28084
            return m1s


class Density:
    def __init__(self):
        pass

    def kg1m3_slug1ft3(self, density, conv_to):
        if conv_to == "slug1ft3":
            slug1ft3 = density / 515.379
            return slug1ft3
        elif conv_to == "kg1m3":
            kg1m3 = density * 515.379
            return kg1m3


class Pressure:
    def __init__(self):
        pass

    def pa_psf(self, pressure, conv_to):
        if conv_to == "psf":
            psf = pressure / 47.88025951
            return psf
        elif conv_to == "pa":
            pa = pressure * 47.88025951
            return pa


class Xfoil:
    def __init__(self, alpha_list):
        self.alpha_list = alpha_list

    def find_alpha_loc(self, alpha):
        return self.alpha_list.tolist().index(alpha)


class Config:
    def __init__(self, b, A, lamda):
        self.b = b
        self.A = A
        self.lamda = lamda

    def wing(self, W, dihedral):
        import math

        S = (self.b**2) / self.A  # Wing Reference Area
        S_short = round(S, 3)
        WS = W / S / math.cos(math.radians(dihedral))  # Wing loading
        WS_N = WS * 9.81
        WS_short = round(WS, 1)
        # WS_imperical = WS * 3.27706
        WS_imperical = WS * 0.204816141
        WS_imperical_short = round(WS_imperical, 1)
        WCL = W / (S**1.5)  # Wing Cubic Loading
        WCL_imperical = WCL / 1.00115395646  # WCL in imperical unit oz/(ft^2)^1.5
        WCL_imperical_short = round(WCL_imperical, 1)

        return (
            S,
            WS,
            WS_N,
            WS_imperical,
            WCL,
            WCL_imperical,
            S_short,
            WS_short,
            WS_imperical_short,
            WCL_imperical_short,
        )

    def vtailconfig(self, AR_Vtail, S_HT, S_VT):
        import math

        phi_Vtail = math.atan2(math.sqrt(S_VT), math.sqrt(S_HT))
        S_Vtail = S_HT + S_VT
        b_Vtail = math.cos(phi_Vtail) * math.sqrt(S_Vtail * AR_Vtail)
        D_Vtail = 0.5 * math.sin(phi_Vtail) * math.sqrt(S_Vtail * AR_Vtail)
        C_Vtail = S_Vtail / math.sqrt(S_Vtail * AR_Vtail)
        E_Vtail = 0.5 * math.sqrt(S_Vtail * AR_Vtail)

        return S_Vtail, phi_Vtail, b_Vtail, D_Vtail, C_Vtail, E_Vtail

    def wingchord(self, S):
        Croot = 2 * S / (self.b * (1 + self.lamda))  # Wing Refenece Root
        Croot_short = round(Croot, 3)
        Ctip = self.lamda * Croot  # Wing Tip
        Ctip_short = round(Ctip, 3)
        Cbar = 2 / 3 * Croot * ((1 + self.lamda + self.lamda**2) / (1 + self.lamda))
        Cbar_short = round(Cbar, 3)
        Ybar = self.b / 6 * ((1 + 2 * self.lamda) / (1 + self.lamda))
        Ybar_short = round(Ybar, 3)

        return Croot, Croot_short, Ctip, Ctip_short, Cbar, Cbar_short, Ybar, Ybar_short

    def fuselage(self, L_FL):
        FL = self.b * L_FL
        FL_short = round(FL, 3)  # m, Fuselage Length

        return FL, FL_short

    def controlsurfw(self, Croot, b_perflap, C_perflap, Wf_mm, lamda):
        bflap = (
            0.5 * self.b * b_perflap / 100
        )  # semi span(bflap = length-span- of left or right side wing's flap)
        bflap_mm = Sciexp.exponent(self, bflap, 3)
        y = bflap
        Cflap = C_perflap / 100 * Croot
        Cflap_mm = Sciexp.exponent(self, Cflap, 3)
        eta_o = (Wf_mm / 2 / 1000) / (self.b / 2)
        eta_i = b_perflap / 100
        Swf_S = (eta_i - eta_o) * (2 - (1 - lamda) * (eta_i + eta_o)) / (1 + lamda)
        return bflap, bflap_mm, Cflap, Cflap_mm, Swf_S


class TailCompu:
    AC = 0.25  # Aerodynamic Center percent Position, For Elevator and Vertical Tail

    def __init__(
        self,
        LL_HT,
        LL_VT,
        C_HT,
        C_VT,
        L_wing,
        A_HT,
        lamda_HT,
        A_VT,
        lamda_VT,
        X_Fuse_mm,
        AC,
        Ybar,
    ):
        # self.LL_HT = LL_HT
        # self.LL_VT = LL_VT
        self.C_HT = C_HT
        self.C_VT = C_VT
        self.L_wing = L_wing
        self.A_HT = A_HT
        self.lamda_HT = lamda_HT
        self.A_VT = A_VT
        self.lamda_VT = lamda_VT
        self.X_Fuse_mm = X_Fuse_mm
        self.AC = AC
        self.Ybar = Ybar

    def tailsizing(self, LL_HT, LL_VT, FL, S, b, Croot, Cbar, Ybar, Cx, wingpos, delta_HS_cap_length_mm, delta_VS_cap_length_mm, Ctip, A, lamda, theta_LE, theta_LE_HT, theta_LE_VT):
        import math

        k = 1
        # [theta_LE, theta_LE_deg, theta_c2, theta_c2_deg, theta_c4, theta_c4_deg, theta_c3, theta_c3_deg] = Stability.sweep(self, Croot, Ctip, b, A, lamda, wingpos, lamda_len)

        theta_c0 = theta_LE
        print(f"theta_c0: {theta_c0*180/math.pi}")
        l_wing_ac = (self.Ybar * math.tan(theta_c0) + Cbar * self.AC)  # Wing_root_pos_x - Wing_ac_pos_x
        # x_wing_ac = self.L_wing + l_wing_ac
        while True:
            print("\nitr: " + str(k))
            L_HT = LL_HT * FL  # Horizontal Tail Moment Arm
            L_HT_short = round(L_HT, 3)
            L_VT = LL_VT * FL  # Vertical Tail Moment Arm
            L_VT_short = round(L_VT, 3)
            # L_wing = FL*(1/3)					#Wing Moment Arm
            L_wing_short = round(self.L_wing, 3)

            S_HT = self.C_HT * Cbar * S / L_HT  # Horizontal Tail Area
            S_HT_short = round(S_HT, 3)
            S_VT = self.C_VT * b * S / L_VT  # Vertical Tail Area
            S_VT_short = round(S_VT, 3)

            # Horizontal Tail
            b_HT = math.sqrt(self.A_HT * S_HT)
            b_HT_short = round(b_HT, 3)
            horizontal = Config(b_HT, self.A_HT, self.lamda_HT)
            [Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short] = horizontal.wingchord(S_HT)

            # Vertical Tail
            b_VT = math.sqrt(self.A_VT * S_VT)
            b_VT_short = round(b_VT, 3)
            vertical = Config(b_VT, self.A_VT, self.lamda_VT)
            [Croot_VT,Croot_VT_short,Ctip_VT,Ctip_VT_short,Cbar_VT,Cbar_VT_short,Ybar_VT,Ybar_VT_short] = vertical.wingchord(S_VT)

            # Actual Moment Arm Length
            X_wing_mm = self.L_wing * 1000
            X_HT_mm = (FL - Croot_HT - delta_HS_cap_length_mm / 1000) * 1000
            X_VT_mm = (FL - Croot_VT - delta_VS_cap_length_mm / 1000) * 1000

            # X_true_Fuse_mm =
            X_true_wing_mm = X_wing_mm - self.X_Fuse_mm
            X_true_HT_mm = X_HT_mm - self.X_Fuse_mm
            X_true_HT_mm_short = round(X_true_HT_mm, 1)
            X_true_VT_mm = X_VT_mm - self.X_Fuse_mm
            X_true_VT_mm_short = round(X_true_VT_mm, 1)

            # if wingpos == 'fwd':
            #    L_AC_wing = Cbar*self.AC
            # elif wingpos == 'aft':
            # L_AC_wing = Croot - Cbar + Cbar*self.AC
            # L_AC_wing = Croot - Cbar + Cbar * self.AC

            # L_AC_wing = Cx - (2*Cx*Ybar/b - Cbar*self.AC)
            L_AC_wing = l_wing_ac

            L_AC_wing_mm = L_AC_wing * 1000
            L_AC_HT = Croot_HT - Cbar_HT + Cbar_HT * TailCompu.AC
            L_AC_HT_mm = L_AC_HT * 1000
            L_AC_VT = Croot_VT - Cbar_VT + Cbar_VT * TailCompu.AC
            L_AC_VT_mm = L_AC_VT * 1000
            X_AC_wing_mm = X_true_wing_mm + L_AC_wing_mm
            print(X_AC_wing_mm)
            X_AC_HT_mm = X_true_HT_mm + L_AC_HT_mm
            X_AC_VT_mm = X_true_VT_mm + L_AC_VT_mm

            L_HT_act_mm = X_AC_HT_mm - X_AC_wing_mm
            L_VT_act_mm = X_AC_VT_mm - X_AC_wing_mm
            L_HT_act = L_HT_act_mm / 1000
            L_VT_act = L_VT_act_mm / 1000

            L_HT_act_short = round(L_HT_act, 2)
            L_VT_act_short = round(L_VT_act, 2)

            C_HT_act = S_HT / (Cbar * S / L_HT_act)  # Horizontal Tail Area
            C_HT_act_short = round(C_HT_act, 3)
            C_VT_act = S_VT / (b * S / L_VT_act)  # Vertical Tail Area
            C_VT_act_short = round(C_VT_act, 3)

            """Real value of Tail moment Arm"""
            L_AC_HT_act = (X_AC_HT_mm - X_AC_wing_mm) / 1000
            L_AC_VT_act = (X_AC_VT_mm - X_AC_wing_mm) / 1000
            L_AC_HT_act_short = round(L_AC_HT_act, 3)
            L_AC_VT_act_short = round(L_AC_VT_act, 3)

            delta_HT = abs(L_AC_HT_act - L_HT)
            delta_VT = abs(L_AC_VT_act - L_VT)
            delta_HT_short = round(delta_HT, 3)
            delta_VT_short = round(delta_VT, 3)
            LL_HT_rcmd = C_HT_act * Cbar * S / S_HT / FL
            LL_VT_rcmd = C_VT_act * b * S / S_VT / FL
            LL_HT_rcmd_short = round(LL_HT_rcmd, 6)
            LL_VT_rcmd_short = round(LL_VT_rcmd, 6)

            if delta_HT <= 1e-5 and delta_VT <= 1e-5:
                LL_HT = LL_HT_rcmd
                LL_VT = LL_VT_rcmd
                print(f"============== iter converged ==============\n")
                return (L_HT, L_HT_short, L_VT, L_VT_short, L_wing_short, S_HT, S_HT_short, S_VT, S_VT_short, b_HT, b_HT_short, Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short, b_VT, b_VT_short, Croot_VT, Croot_VT_short, Ctip_VT, Ctip_VT_short, Cbar_VT, Cbar_VT_short, Ybar_VT, Ybar_VT_short, X_wing_mm, X_HT_mm, X_VT_mm, X_true_wing_mm, X_true_HT_mm, X_true_VT_mm, L_AC_wing, L_AC_wing_mm, L_AC_HT, L_AC_HT_mm, L_AC_VT, L_AC_VT_mm, X_AC_wing_mm, X_AC_HT_mm, X_AC_VT_mm, L_HT_act_mm, L_VT_act_mm, L_HT_act, L_VT_act, L_HT_act_short, L_VT_act_short, C_HT_act, C_HT_act_short, C_VT_act, C_VT_act_short, L_AC_HT_act, L_AC_VT_act, L_AC_HT_act_short, L_AC_VT_act_short, delta_HT, delta_VT, delta_HT_short, delta_VT_short, LL_HT_rcmd, LL_VT_rcmd, LL_HT_rcmd_short, LL_VT_rcmd_short, X_true_HT_mm_short, X_true_VT_mm_short)
                # break
            else:
                LL_HT = LL_HT_rcmd
                LL_VT = LL_VT_rcmd
                k = k + 1

        # return L_HT, L_HT_short, L_VT, L_VT_short, L_wing_short, S_HT, S_HT_short, S_VT, S_VT_short, b_HT, b_HT_short, Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short, b_VT, b_VT_short, Croot_VT, Croot_VT_short, Ctip_VT, Ctip_VT_short, Cbar_VT, Cbar_VT_short, Ybar_VT, Ybar_VT_short, X_wing_mm, X_HT_mm, X_VT_mm, X_true_wing_mm, X_true_HT_mm, X_true_VT_mm, L_AC_wing, L_AC_wing_mm, L_AC_HT, L_AC_HT_mm, L_AC_VT, L_AC_VT_mm, X_AC_wing_mm, X_AC_HT_mm, X_AC_VT_mm, L_HT_act_mm, L_VT_act_mm, L_HT_act, L_VT_act, L_HT_act_short, L_VT_act_short, C_HT_act, C_HT_act_short, C_VT_act, C_VT_act_short, L_AC_HT_act, L_AC_VT_act, L_AC_HT_act_short, L_AC_VT_act_short, delta_HT, delta_VT, delta_HT_short, delta_VT_short, LL_HT_rcmd, LL_VT_rcmd, LL_HT_rcmd_short, LL_VT_rcmd_short, X_true_HT_mm_short, X_true_VT_mm_short

    
    def tailsizingtwo(self, LL_HT, LL_VT, FL, S, b, Croot, Cbar, Ybar, Cx, wingpos, delta_HS_cap_length_mm, delta_VS_cap_length_mm, Ctip, A, lamda, theta_LE, theta_LE_HT, theta_LE_VT):
        import math

        k = 1
        # [theta_LE, theta_LE_deg, theta_c2, theta_c2_deg, theta_c4, theta_c4_deg, theta_c3, theta_c3_deg] = Stability.sweep(self, Croot, Ctip, b, A, lamda, wingpos, lamda_len)

        theta_c0 = theta_LE
        print(f"theta_c0: {theta_c0*180/math.pi}")
        l_wing_ac = (self.Ybar * math.tan(theta_c0) + Cbar * self.AC)  # Wing_root_pos_x - Wing_ac_pos_x
        # x_wing_ac = self.L_wing + l_wing_ac
        while True:
            print("\nitr: " + str(k))
            L_HT = LL_HT * FL  # Horizontal Tail Moment Arm
            L_HT_short = round(L_HT, 3)
            L_VT = LL_VT * FL  # Vertical Tail Moment Arm
            L_VT_short = round(L_VT, 3)
            # L_wing = FL*(1/3)					#Wing Moment Arm
            L_wing_short = round(self.L_wing, 3)

            S_HT = self.C_HT * Cbar * S / L_HT  # Horizontal Tail Area
            S_HT_short = round(S_HT, 3)
            S_VT = self.C_VT * b * S / L_VT  # Vertical Tail Area
            S_VT_short = round(S_VT, 3)

            # Horizontal Tail
            b_HT = math.sqrt(self.A_HT * S_HT)
            b_HT_short = round(b_HT, 3)
            horizontal = Config(b_HT, self.A_HT, self.lamda_HT)
            [Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short] = horizontal.wingchord(S_HT)

            # Vertical Tail
            b_VT = math.sqrt(self.A_VT * S_VT)
            b_VT_short = round(b_VT, 3)
            vertical = Config(b_VT, self.A_VT, self.lamda_VT)
            [Croot_VT,Croot_VT_short,Ctip_VT,Ctip_VT_short,Cbar_VT,Cbar_VT_short,Ybar_VT,Ybar_VT_short] = vertical.wingchord(S_VT)

            # Actual Moment Arm Length
            X_wing_mm = self.L_wing * 1000
            X_HT_mm = (FL - Croot_HT - delta_HS_cap_length_mm / 1000) * 1000
            X_VT_mm = (FL - Croot_VT - delta_VS_cap_length_mm / 1000) * 1000

            # X_true_Fuse_mm =
            X_true_wing_mm = X_wing_mm - self.X_Fuse_mm
            X_true_HT_mm = X_HT_mm - self.X_Fuse_mm
            X_true_HT_mm_short = round(X_true_HT_mm, 1)
            X_true_VT_mm = X_VT_mm - self.X_Fuse_mm
            X_true_VT_mm_short = round(X_true_VT_mm, 1)

            # if wingpos == 'fwd':
            #    L_AC_wing = Cbar*self.AC
            # elif wingpos == 'aft':
            # L_AC_wing = Croot - Cbar + Cbar*self.AC
            # L_AC_wing = Croot - Cbar + Cbar * self.AC

            # L_AC_wing = Cx - (2*Cx*Ybar/b - Cbar*self.AC)
            L_AC_wing = l_wing_ac

            L_AC_wing_mm = L_AC_wing * 1000
            L_AC_HT = Ybar_HT*math.tan(theta_LE_HT) + 0.25*Cbar_HT
            L_AC_HT_mm = L_AC_HT * 1000
            L_AC_VT = Ybar_VT*math.tan(theta_LE_VT) + 0.25*Cbar_VT
            L_AC_VT_mm = L_AC_VT * 1000
            X_AC_wing_mm = X_true_wing_mm + L_AC_wing_mm
            print(X_AC_wing_mm)
            X_AC_HT_mm = X_true_HT_mm + L_AC_HT_mm
            X_AC_VT_mm = X_true_VT_mm + L_AC_VT_mm

            L_HT_act_mm = X_AC_HT_mm - X_AC_wing_mm
            L_VT_act_mm = X_AC_VT_mm - X_AC_wing_mm
            L_HT_act = L_HT_act_mm / 1000
            L_VT_act = L_VT_act_mm / 1000

            L_HT_act_short = round(L_HT_act, 2)
            L_VT_act_short = round(L_VT_act, 2)

            C_HT_act = S_HT / (Cbar * S / L_HT_act)  # Horizontal Tail Area
            C_HT_act_short = round(C_HT_act, 3)
            C_VT_act = S_VT / (b * S / L_VT_act)  # Vertical Tail Area
            C_VT_act_short = round(C_VT_act, 3)

            """Real value of Tail moment Arm"""
            L_AC_HT_act = (X_AC_HT_mm - X_AC_wing_mm) / 1000
            L_AC_VT_act = (X_AC_VT_mm - X_AC_wing_mm) / 1000
            L_AC_HT_act_short = round(L_AC_HT_act, 3)
            L_AC_VT_act_short = round(L_AC_VT_act, 3)

            delta_HT = abs(L_AC_HT_act - L_HT)
            delta_VT = abs(L_AC_VT_act - L_VT)
            delta_HT_short = round(delta_HT, 3)
            delta_VT_short = round(delta_VT, 3)
            LL_HT_rcmd = C_HT_act * Cbar * S / S_HT / FL
            LL_VT_rcmd = C_VT_act * b * S / S_VT / FL
            LL_HT_rcmd_short = round(LL_HT_rcmd, 6)
            LL_VT_rcmd_short = round(LL_VT_rcmd, 6)

            if delta_HT <= 1e-5 and delta_VT <= 1e-5:
                LL_HT = LL_HT_rcmd
                LL_VT = LL_VT_rcmd
                print(f"============== iter converged ==============\n")
                return (L_HT, L_HT_short, L_VT, L_VT_short, L_wing_short, S_HT, S_HT_short, S_VT, S_VT_short, b_HT, b_HT_short, Croot_HT, Croot_HT_short, Ctip_HT, Ctip_HT_short, Cbar_HT, Cbar_HT_short, Ybar_HT, Ybar_HT_short, b_VT, b_VT_short, Croot_VT, Croot_VT_short, Ctip_VT, Ctip_VT_short, Cbar_VT, Cbar_VT_short, Ybar_VT, Ybar_VT_short, X_wing_mm, X_HT_mm, X_VT_mm, X_true_wing_mm, X_true_HT_mm, X_true_VT_mm, L_AC_wing, L_AC_wing_mm, L_AC_HT, L_AC_HT_mm, L_AC_VT, L_AC_VT_mm, X_AC_wing_mm, X_AC_HT_mm, X_AC_VT_mm, L_HT_act_mm, L_VT_act_mm, L_HT_act, L_VT_act, L_HT_act_short, L_VT_act_short, C_HT_act, C_HT_act_short, C_VT_act, C_VT_act_short, L_AC_HT_act, L_AC_VT_act, L_AC_HT_act_short, L_AC_VT_act_short, delta_HT, delta_VT, delta_HT_short, delta_VT_short, LL_HT_rcmd, LL_VT_rcmd, LL_HT_rcmd_short, LL_VT_rcmd_short, X_true_HT_mm_short, X_true_VT_mm_short)
                # break
            else:
                LL_HT = LL_HT_rcmd
                LL_VT = LL_VT_rcmd
                k = k + 1



    def vtail(
        self, AR_Vtail, S_HT, S_VT, X_true_Vtail_mm, Croot, AC, L_wing, S, Cbar, b
    ):
        import math

        phi_Vtail = math.atan2(math.sqrt(S_VT), math.sqrt(S_HT))
        S_Vtail = S_HT + S_VT
        b_Vtail = math.cos(phi_Vtail) * math.sqrt(S_Vtail * AR_Vtail)
        D_Vtail = 0.5 * math.sin(phi_Vtail) * math.sqrt(S_Vtail * AR_Vtail)
        C_Vtail = S_Vtail / math.sqrt(S_Vtail * AR_Vtail)
        E_Vtail = 0.5 * math.sqrt(S_Vtail * AR_Vtail)

        Croot_vtail_ht = C_Vtail
        b_vtail_ht = b_Vtail
        Croot_vtail_vt = C_Vtail
        b_vtail_vt = D_Vtail
        S_vtail_ht = S_Vtail * (math.cos(phi_Vtail)) ** 2
        S_vtail_vt = S_Vtail * (math.sin(phi_Vtail)) ** 2

        X_true_Vtail = X_true_Vtail_mm / 1000
        X_ac_wing = Croot * AC + L_wing
        X_ac_vtail_ht = X_true_Vtail + Croot_vtail_ht * 0.25
        X_ac_vtail_vt = X_true_Vtail + Croot_vtail_vt * 0.25
        L_vtail_ht = X_ac_vtail_ht - X_ac_wing
        L_vtail_vt = X_ac_vtail_vt - X_ac_wing

        C_vtail_h = S_vtail_ht * L_vtail_ht / (S * Cbar)
        C_vtail_v = S_vtail_vt * L_vtail_vt / (S * b)

        return (
            S_Vtail,
            phi_Vtail,
            b_Vtail,
            D_Vtail,
            C_Vtail,
            E_Vtail,
            C_vtail_h,
            C_vtail_v,
            X_ac_vtail_ht,
            S_vtail_ht,
            Croot_vtail_ht,
            X_ac_vtail_ht,
        )


class Stability:
    def __init__(self, c, A, v_cruise, XCG_mm, Wf_mm, h_t_mm, eta_h_T0, T_kgf, Dp_in):
        self.c = c
        self.v_cruise = v_cruise
        self.XCG_mm = XCG_mm
        self.Wf_mm = Wf_mm
        self.h_t_mm = h_t_mm
        self.eta_h_T0 = eta_h_T0
        self.T_kgf = T_kgf
        self.Dp_in = Dp_in
        self.A = A

    def bodygeo3(self, x_sta_mm, stations_ymax_mm, stations_zmax_mm, L_wing, Croot, FL, NumSlice=110):
        ## IN-PROGRESS ##
        import numpy as np

        x_sta_mm_sliced = np.linspace(0, x_sta_mm, NumSlice)
        delta_xi = L_wing * 1000 / 50
        delta_xi_aft = (FL * 1000 - (L_wing + Croot) * 1000) / 60

        xi = []
        wf = []
        xpos = []
        xi_i = 0
        for i in range(len(x_sta_mm_sliced)):
            if i >= 50:
                xi_i = delta_xi_aft / 2 + (i - 50) * delta_xi_aft
                xpos_i = -1 * delta_xi_aft / 2 + i * delta_xi_aft
                xi.append(xi_i)
                xpos.append(xpos_i)
            else:
                xi_i = delta_xi / 2 + i * delta_xi
                xpos_i = delta_xi / 2 + i * delta_xi
                xi.append(xi_i)
                xpos.append(xpos_i)
        
            for j, xx_sta in enumerate(x_sta_mm):
                if xpos[i] > xx_sta:
                    t = (xpos[i] - x_sta_mm[j]) / (x_sta_mm[j+1] - x_sta_mm[j])
                    y1, y2 = stations_ymax_mm[j], stations_ymax_mm[j+1]
                    z1, z2 = stations_zmax_mm[j], stations_zmax_mm[j+1]
                    ypos = y1 + t*(y2 - y1)
                    zpos = z1 + t*(z2 - z1)
                    wf.append(ypos*2)
                    break

        xpos = np.array(xpos)
        wf = np.array(wf)
        # print(wf)
        wf = wf / 1000
        xi = np.array(xi)
        xi = xi / 1000
        delta_xi_1to5 = delta_xi / 1000
        delta_xi_6to13 = delta_xi_aft / 1000
        xi_1to5 = []
        xi_6to13 = []
        for i in range(110):
            if i < 50:
                xi_1to5.append(xi[i] / 1000)
            elif i >= 50:
                xi_6to13.append(xi[i] / 1000)

        np.set_printoptions(precision=3)
        print(f"\n xi: {xi}")
        print(f"\n wf: {wf}")
        np.set_printoptions(precision=1)
        print(f"\n xpos: {xpos}")

        return wf, xi_1to5, xi_6to13, xi, delta_xi_1to5, delta_xi_6to13




    def bodygeo2(self, x_sta_mm, stations_ymax_mm, stations_yz_mm, L_wing, Croot, FL):
        import numpy as np

        bodylinefunc_slope = []
        bodylinefunc_intcpt = []
        for i in range(len(x_sta_mm) - 1):  # Body line function, y = mx+b
            x1, x2 = x_sta_mm[i], x_sta_mm[i + 1]
            y1, y2 = stations_ymax_mm[i], stations_ymax_mm[i + 1]
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            bodylinefunc_slope.append(m)
            bodylinefunc_intcpt.append(b)

        x_sta_mm60 = np.linspace(0, max(x_sta_mm), 110)

        delta_xi = L_wing * 1000 / 50
        delta_xi_behind = (FL * 1000 - (L_wing + Croot) * 1000) / 60
        xi = []
        wf = []
        xpos = []
        xi_i = 0
        for i in range(len(x_sta_mm60)):
            if i >= 50:
                xi_i = delta_xi_behind / 2 + (i - 50) * delta_xi_behind
                # xpos_i = -1 * delta_xi_behind / 2 + i * delta_xi_behind
                xpos_i = ((L_wing + Croot) * 1000) + (i - 50) * delta_xi_behind
                xi.append(xi_i)
                xpos.append(xpos_i)
            # if i == 50:
            #     xi_i += Croot*1000/2
            #     xi.append(xi_i)
            # elif i == 51:
            #     xi_i += Croot*1000
            #     xi.append(xi_i)
            # elif i == 52:
            #     xi_i = delta_xi_behind/2 + (L_wing + Croot)*1000
            #     xi.append(xi_i)
            # elif i > 52:
            #     xi_i += delta_xi_behind
            #     xi.append(xi_i)
            else:
                xi_i = delta_xi / 2 + (50 - i) * delta_xi
                xpos_i = delta_xi / 2 + i * delta_xi
                xi.append(xi_i)
                xpos.append(xpos_i)

            # xi.append(xi_i)   #list(xi) = [delta_xi/2, delta_xi + delta_xi/2, 2*delta_xi + delta_xi/2, ...]
            for j, xx_sta in enumerate(x_sta_mm):
                if xpos[i] < xx_sta:
                    wf_i = 2 * (
                        bodylinefunc_slope[j - 1] * xpos_i + bodylinefunc_intcpt[j - 1]
                    )
                    wf.append(wf_i)
                    break

        # print('\n')
        # print(wf)
        # print('\n')
        xpos = np.array(xpos)
        wf = np.array(wf)
        # print(wf)
        wf = wf / 1000
        xi = np.array(xi)
        xi = xi / 1000
        delta_xi_1to5 = delta_xi / 1000
        delta_xi_6to13 = delta_xi_behind / 1000
        xi_1to5 = []
        xi_6to13 = []
        for i in range(110):
            if i < 50:
                xi_1to5.append(xi[i] / 1000)
            elif i >= 50:
                xi_6to13.append(xi[i] / 1000)

        np.set_printoptions(precision=3)
        print(f"\n xi: {xi}")
        print(f"\n wf: {wf}")
        np.set_printoptions(precision=1)
        print(f"\n xpos: {xpos}")

        return wf, xi_1to5, xi_6to13, xi, delta_xi_1to5, delta_xi_6to13

    def bodygeo(self, x_sta_mm, stations_ymax_mm, stations_yz_mm, L_wing, Croot, FL):
        import numpy as np

        def linefunc(stations_xy, x):
            for i in range(0, len(stations_xy)):
                if stations_xy[i][0] < x:
                    continue
                else:
                    x1 = stations_xy[i - 1][0]
                    # print(f'x1: {x1}')
                    x2 = stations_xy[i][0]
                    # print(f'x2: {x2}')
                    y1 = stations_xy[i - 1][1]
                    # print(f'y1: {y1}')
                    y2 = stations_xy[i][1]
                    # print(f'y2: {y2}')
                    # print(f'x: {x}')
                    slope = (y1 - y2) / (x1 - x2)
                    # print(f'slope: {slope}')
                    yf = slope * (x - x1) + y1
                    # print(f'yf: {yf}')
                    wf = 2 * yf
                    return wf

        x_sta = x_sta_mm / 1000
        stations_yz = stations_yz_mm / 1000
        stations_xy = []
        # y_max = []

        for i in range(len(stations_yz)):
            ys = []
            for j in range(len(stations_yz[0])):
                ys_i = stations_yz[i][j][0]
                ys.append(ys_i)
            y_max_i = max(ys)
            # y_max.append(y_max_i)
            stations_xy.append([x_sta[i], y_max_i])

        stations_xy = np.array(stations_xy)
        # print(f'stations_xy: {stations_xy}')
        wf = []

        delta_xi_1to5 = L_wing / 5
        x_1to5 = np.arange(0, L_wing, delta_xi_1to5)
        # print(x_1to5)
        x_1to5 += delta_xi_1to5 / 2
        # print(x_1to5)
        for x in x_1to5:
            wf_i = linefunc(stations_xy, x)
            # print(f'wf_i: {wf_i}')
            wf.append(wf_i)

        delta_xi_6to7 = Croot / 2
        x_6to7 = np.arange(L_wing, L_wing + Croot, delta_xi_6to7)
        # print(x_6to7)
        x_6to7 += delta_xi_6to7 / 2
        # print(x_6to7)
        for x in x_6to7:
            wf_i = linefunc(stations_xy, x)
            wf.append(wf_i)

        delta_xi_6to13 = (FL - (L_wing + Croot)) / 6
        x_6to13 = np.arange(L_wing + Croot, FL, delta_xi_6to13)
        # print(x_8to13) x_6to13
        x_6to13 += delta_xi_6to13 / 2
        # print(x_8to13)
        for x in x_6to13:
            wf_i = linefunc(stations_xy, x)
            wf.append(wf_i)

        wf = np.array(wf)
        # print(f"wf: {wf}")
        # print(f'wf: {wf}')
        # x_1to13 = np.array([x_1to5, x_6to7, x_8to13])
        # xi_1to5 = np.empty(len(x_1to5))
        xi_1to5 = L_wing - x_1to5
        xi_6to7 = x_6to7 - L_wing
        xi_6to7[1] = xi_6to7[0]

        xi_8to13 = x_6to13 - (L_wing + Croot)
        xi = np.concatenate((xi_1to5, xi_6to7, xi_8to13))
        # print(xi)

        """
        for i in range(len(x_sta)):
            stations_xy_i = []
            for j in range(len(stations_yz[i])):
                x_i = x_sta[i]
                y_i = stations_yz[i][j][0]
                stations_xy_ij = [x_i, y_i]
                stations_xy_i.append(stations_xy_ij)
                staions_xy_ij_neg = [x_i, -1*y_i]
                stations_xy_i.append(staions_xy_ij_neg)
            stations_xy.append(stations_xy_i)
        
        staions_xy = np.array(stations_xy)

        # print(f'\n{stations_xy[1]}')

        A_sta = []
        for ii in range(len(x_sta)):
            A_sta_i = []
            for jj in range(len(staions_xy[ii][0])):
                A_sta_ij = staions_xy[ii][jj][0] * staions_xy[ii][jj + 1][1] - staions_xy[ii][jj + 1][0] * staions_xy[ii][jj][1]
                A_sta_i.append(A_sta_ij)
            A_sta_i_val = 0.5*sum(A_sta_i)
            A_sta.append(A_sta_i_val)

        A_sta = np.array(A_sta)

        C_sta = []
        for iii in range(len(x_sta)):
            Cx_sta_i = []
            Cy_sta_i = []
            for jjj in range(len(staions_xy[iii][0])):
                Cx_sta_ij = (staions_xy[iii][jjj][0] + staions_xy[iii][jjj + 1][0]) * (staions_xy[iii][jjj][0] * staions_xy[iii][jjj + 1][1] - staions_xy[iii][jjj + 1][0] * staions_xy[iii][jjj][1])
                Cy_sta_ij = (staions_xy[iii][jjj][1] + staions_xy[iii][jjj + 1][1]) * (staions_xy[iii][jjj][0] * staions_xy[iii][jjj + 1][1] - staions_xy[iii][jjj + 1][0] * staions_xy[iii][jjj][1])
                Cx_sta_i.append(Cx_sta_ij)
                Cy_sta_i.append(Cy_sta_ij)
            if A_sta[iii] == 0:
                    C_sta.append([0, 0])
            else:
                Cx_sta_i_val = (1/6)*(1/A_sta[iii])*sum(Cx_sta_i)
                Cy_sta_i_val = (1/6)*(1/A_sta[iii])*sum(Cy_sta_i)
                C_sta_i_val = [Cx_sta_i_val, Cy_sta_i_val]
                if np.isnan(C_sta_i_val[0]) or np.isnan(C_sta_i_val[1]):
                    C_sta_i_val = [0, 0]
                C_sta.append(C_sta_i_val)
        
        C_sta = np.array(C_sta)
        """
        # print(f'{C_sta}')

        return wf, xi_1to5, xi_8to13, xi, delta_xi_1to5, delta_xi_6to13

    def neutralpoint_torenbeek(
        self, CLWa, Wf_mm, AC, Cbar, L_wing, S, lamda, b, theta_c4, l_n_mm, hf_mm, Croot
    ):
        import numpy as np
        import math

        b_f = Wf_mm / 1000
        Xac_cbar__W = AC
        l_fn = l_n_mm / 1000
        h_f = hf_mm / 1000
        S_net = S - (Croot * b_f)

        K_I = (1 + 2.15 * b_f / b) * S_net / S + math.pi / (2 * CLWa) * b_f**2 / S
        CLaWF = K_I * CLWa
        corr1 = -1.8 / CLaWF * b_f * h_f * l_fn / (S * Cbar)

        cg = Cbar * (1 + lamda) / 2  # meter, cg: wing mean geometric chord length
        corr2 = (
            0.273
            / (1 + lamda)
            * (b_f * cg * (b - b_f))
            / (Cbar**2 * (b + 2.15 * b_f))
            * math.tan(theta_c4)
        )
        Xac_cbar__WF = Xac_cbar__W + corr1 + corr2

        return Xac_cbar__WF

    def neutralpointfour(self, wf, xi, Croot, dedt_0_deg, x_h, delta_xi_1to5, delta_xi_6to13, rho_a_cruise, v_cruise, S, Cbar, CLa_W_deg, Wf_mm, b, X_bar_ac_w, S_HT, CLa_HT_deg, dahda_deg, X_bar_ac_h, eta_h, XCG_mm, req_sm, theta_LE, dedt_deg, lamda, L_wing, X_true_HT_mm_short):
        import numpy as np
        import math

        def dEbarda1(x):
            #return (-0.415006 * x**5 + 2.64181 * x**4 - 6.48731 * x**3 + 7.76831 * x**2 - 4.78746 * x + 2.49001)
            return (0.14376218*x**6 + -1.12564368* x**5 +  3.58314542* x**4 + -6.00339362* x**3 + 5.79311192* x**2 + -3.41069309* x + 2.18900488)
        def dEbarda2(x):
            #return (-14.5661 * x**5 + 45.6298 * x**4 - 57.321 * x**3 + 39.0083 * x**2 - 16.6653 * x + 5.72027)
            return (74.03889688 * x**6 - 268.28816824 * x**5 + 395.12726474 * x**4 + -305.24502622* x**3 + 134.77612071* x**2 + -35.81781812*x + 7.20974534)
        
        # print(X_bar_ac_w)
        # K1 = 0.501881*lamda**3 - 0.891293*lamda**2 - 0.105592*lamda + 1.50017
        # X_bar_ac_w = (X_bar_ac_w - L_wing)/Croot/1000 * K1
        # X_bar_ac_h = (X_bar_ac_h - L_wing)/Croot/1000 * K1
        X_bar_ac_w = (X_bar_ac_w - L_wing)/Cbar/1000
        X_bar_ac_h = (X_bar_ac_h - L_wing)/Cbar/1000
        # X_bar_ac_w = X_bar_ac_w + (0.218 * self.A * math.tan(theta_LE) * S / b)/Cbar
        # print(X_bar_ac_w)
        # X_bar_ac_h += (0.218 * self.A * math.tan(theta_LE) * S / b)/Cbar
        Wf = np.copy(wf)
        Cf = Croot
        sumWf = []
        n = 0
        for val in range(len(Wf)):
            if 0 <= val < 50:
                Wf_i = Wf[val]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[val]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi_1to5, "ft")

                dEbarda = dEbarda1(xi_i_im / Cf_im)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)

            elif val == -99999:
                Wf_i = Wf[val]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[val]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi_1to5, "ft")

                dEbarda = dEbarda2(xi_i_im / Cf_im)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)

            elif val >= 50:
                # delta_xi = delta_xi_8to13/6
                delta_xi_im = Length.meter_feet(self, delta_xi_6to13, "ft")

                Wf_i = Wf[val]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[val]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                x_h_im = Length.meter_feet(self, x_h, "ft")
                dEbarda = xi_i_im / x_h_im * (1 - dedt_0_deg)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)
        # for val in range(1, 14):
        #     if 1 <= val <= 5:
        #         Wf_i = Wf[n]
        #         #Wf.insert(n - 1, Wf_i)

        #         xi_i = xi[n]
        #         #xi.insert(n - 1, xi_i)
        #         #print(xi_i)

        #         Wf_i_im = Length.meter_feet(self, Wf_i, 'ft')
        #         xi_i_im = Length.meter_feet(self, xi_i, 'ft')
        #         Cf_im = Length.meter_feet(self, Cf, 'ft')
        #         delta_xi_im = Length.meter_feet(self, delta_xi_1to5, 'ft')

        #         dEbarda = dEbarda1(xi_i_im/Cf_im)
        #         sumWf_im = (Wf_i_im**2)*dEbarda*delta_xi_im
        #         sumWf.append(sumWf_im)

        #     elif val == 500:
        #         Wf_i = Wf[n]
        #         #Wf.insert(n - 1, Wf_i)

        #         xi_i = xi[n]
        #         #xi.insert(n - 1, xi_i)
        #         #print(xi_i)

        #         Wf_i_im = Length.meter_feet(self, Wf_i, 'ft')
        #         xi_i_im = Length.meter_feet(self, xi_i, 'ft')
        #         Cf_im = Length.meter_feet(self, Cf, 'ft')
        #         delta_xi_im = Length.meter_feet(self, delta_xi_1to5, 'ft')

        #         dEbarda = dEbarda2(xi_i_im/Cf_im)
        #         sumWf_im = (Wf_i_im**2)*dEbarda*delta_xi_im
        #         sumWf.append(sumWf_im)

        #     elif 6 <= val <= 13:
        #         # delta_xi = delta_xi_8to13/6
        #         delta_xi_im = Length.meter_feet(self, delta_xi_8to13, 'ft')

        #         Wf_i = Wf[n]
        #         #Wf.insert(n - 1, Wf_i)

        #         xi_i = xi[n]
        #         #xi.insert(n - 1, xi_i)
        #         #print(xi_i)

        #         Wf_i_im = Length.meter_feet(self, Wf_i, 'ft')
        #         xi_i_im = Length.meter_feet(self, xi_i, 'ft')
        #         x_h_im = Length.meter_feet(self, x_h, 'ft')
        #         dEbarda = xi_i_im/x_h_im*(1 - dedt_0_deg)
        #         sumWf_im = (Wf_i_im**2)*dEbarda*delta_xi_im
        #         sumWf.append(sumWf_im)

        """sumofWf_im = 0
        for zal in range(0, 13):
            sumWf_i = sumWf[zal]
            sumofWf_im += sumWf_i"""
        sumofWf_im = sum(sumWf)
        print(f"length of sumWf is {len(sumWf)}")

        q = 0.5 * rho_a_cruise * v_cruise**2
        q_im = Pressure.pa_psf(self, q, "psf")
        dMda_im = (q_im / 36.5) * (CLa_W_deg / 0.08) * sumofWf_im

        S_im = Area.m2_ft2(self, S, "ft2")
        S_HT_im = Area.m2_ft2(self, S_HT, "ft2")
        Cbar_im = Length.meter_feet(self, Cbar, "ft")

        # dedt_deg = 1 - dahda_deg
        

        deltaXac_f_im = -1 * dMda_im / (q_im * S_im * Cbar_im * CLa_W_deg)
        # print(deltaXac_f)
        Wf_kf_im = Length.meter_feet(self, Wf_mm / 1000, "ft")
        b_im = Length.meter_feet(self, b, "ft")

        CLa_W = math.degrees(CLa_W_deg)

        K_wf = 1 + 0.025 * (Wf_kf_im / b_im) - 0.25 * (Wf_kf_im / b_im) ** 2
        CLa_WF = K_wf * CLa_W
        CLa_WF_deg = math.radians(CLa_WF)

        CLa_WF = math.degrees(CLa_WF_deg)
        CLa_HT = math.degrees(CLa_HT_deg)
        
        dedt = math.degrees(dedt_deg)

        # X_bar_NPtwo_im = (
        #     (X_bar_ac_w + deltaXac_f_im) * CLa_WF_deg
        #     + eta_h * (S_HT_im / S_im) * CLa_HT_deg * (1 - dedt_0_deg) * X_bar_ac_h
        # ) / (CLa_WF_deg + eta_h * S_HT_im / S_im * CLa_HT_deg * (1 - dedt_0_deg))
        X_bar_NPtwo_im = ((X_bar_ac_w + deltaXac_f_im) + eta_h*S_HT_im/S_im*CLa_HT/CLa_WF*(1-dedt_deg)*X_bar_ac_h)/(1+CLa_HT/CLa_W*eta_h*S_HT_im/S_im*(1-dedt_deg))
        XNPtwo_im = X_bar_NPtwo_im * Cbar_im
        XNPtwo = Length.meter_feet(self, XNPtwo_im, "meter")
        XNPtwo_mm = XNPtwo * 1000
        SM_roskam = (XNPtwo_mm - XCG_mm) / (Cbar * 1000) * 100
        prp_xcgtwo = XNPtwo * 1000 - req_sm / 100 * Cbar * 1000

        return (
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
        )

    def sweep(self, Croot_short, Ctip_short, b_short, A, lamda, wingpos, lamda_len):
        import math

        if wingpos == "fwd":
            theta_LE = 0
            theta_LE_deg = math.degrees(theta_LE)
            theta_c2 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 50) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c2_deg = math.degrees(theta_c2)
            theta_c4 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 25) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c4_deg = math.degrees(theta_c4)
            theta_c3 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 30) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c3_deg = math.degrees(theta_c3)
        elif wingpos == "aft":
            theta_LE = math.atan((Croot_short - Ctip_short) / (b_short / 2))
            theta_LE_deg = math.degrees(theta_LE)
            theta_c2 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 50) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c2_deg = math.degrees(theta_c2)
            theta_c4 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 25) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c4_deg = math.degrees(theta_c4)
            theta_c3 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 30) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c3_deg = math.degrees(theta_c3)
        elif wingpos == "rand":
            theta_LE = math.atan2((lamda_len), (b_short / 2))
            theta_LE_deg = math.degrees(theta_LE)
            theta_c2 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 50) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c2_deg = math.degrees(theta_c2)
            theta_c4 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 25) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c4_deg = math.degrees(theta_c4)
            theta_c3 = math.atan(
                math.tan(theta_LE)
                - (4 * (0 - 30) / A / 100) * ((1 - lamda) / (1 + lamda))
            )
            theta_c3_deg = math.degrees(theta_c3)

        return (
            theta_LE,
            theta_LE_deg,
            theta_c2,
            theta_c2_deg,
            theta_c4,
            theta_c4_deg,
            theta_c3,
            theta_c3_deg,
        )

    def sweepvt(self, Croot, Ctip, b, A, lamda):
        import math

        theta_LE = math.atan((Croot - Ctip) / b)
        theta_LE_deg = math.degrees(theta_LE)
        theta_c4 = math.atan(math.tan(theta_LE) - (1 - lamda) / (2 * A * (1 + lamda)))
        theta_c4_deg = math.degrees(theta_c4)

        return theta_LE, theta_LE_deg, theta_c4, theta_c4_deg

    def cl_slope(self, A, theta_c4, Ta, rho_a, Cbar, v):
        import math

        M = self.v_cruise / self.c  # Mach number
        CLa_W = (
            math.pi
            * A
            / (
                1
                + math.sqrt(
                    1
                    + (1 - (M * math.cos(theta_c4)) ** 2)
                    * (A / (2 * math.cos(theta_c4))) ** 2
                )
            )
        )
        CLa_W_M0 = (
            math.pi
            * A
            / (
                1
                + math.sqrt(
                    1
                    + (1 - (0 * math.cos(theta_c4)) ** 2)
                    * (A / (2 * math.cos(theta_c4))) ** 2
                )
            )
        )
        CLa_W_short = round(CLa_W, 4)
        CLa_W_deg = math.radians(CLa_W)
        CLa_W_deg_short = round(CLa_W_deg, 6)
        mu = 1.458e-6 * math.sqrt(Ta) / (1 + 110.4 / Ta)
        Re = rho_a * v * Cbar / mu
        Re_short = round(Re, -4)

        return (
            M,
            CLa_W,
            CLa_W_M0,
            CLa_W_short,
            CLa_W_deg,
            CLa_W_deg_short,
            Re,
            mu,
            Re_short,
        )

    def downwash(
        self, L_AC_HT_act, FL_short, A, lamda, b_short, theta_c4, CLa_W, CLa_W_M0
    ):
        import math

        l_h = L_AC_HT_act  # Tail True Moment Arm
        h_t = self.h_t_mm / 1000
        Lf = FL_short

        K_A = 1 / A - 1 / (1 + A**1.7)
        K_lamda = (10 - 3 * lamda) / 7
        K_h = (1 - abs(h_t / b_short)) / ((2 * L_AC_HT_act / b_short) ** (1 / 3))
        K_h_0 = (1 - 0 / b_short) / ((2 * L_AC_HT_act / b_short) ** (1 / 3))
        # dedt_deg = (21*CLa_W_deg/A**0.725)*(Cbar/l_h)*((10 - 3*lamda)/7)*(1 - h_t/b)
        dedt_deg = (
            4.44
            * (K_A * K_lamda * K_h * (math.cos(theta_c4)) ** 0.5) ** 1.19
            * CLa_W
            / CLa_W_M0
        )
        dedt_0_deg = (
            4.44
            * (K_A * K_lamda * K_h_0 * (math.cos(theta_c4)) ** 0.5) ** 1.19
            * CLa_W
            / CLa_W_M0
        )
        print(f"dedt_0_deg:{dedt_0_deg:,.5f}")
        dedt = math.degrees(dedt_deg)

        return l_h, h_t, Lf, K_A, K_lamda, K_h, dedt_deg, dedt, dedt_0_deg

    def neutralpoint(
        self,
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
    ):
        import math

        Wf = self.Wf_mm / 1000
        X_Cr4_mm = L_wing + L_AC_wing
        X_Cr4_mm = L_wing * 1000 + Croot * 1000 * 0.25
        X_Cr4_mm = X_AC_wing_mm
        X_Cr4 = X_Cr4_mm / 1000

        X_bar_Cr4 = X_Cr4 / Lf
        Kf = (
            0.0006
            - 0.01713 * X_bar_Cr4
            + 0.06655 * X_bar_Cr4**2
            + 0.1251 * X_bar_Cr4**3
        )

        Cma_fus_deg = Kf * (Wf**2) * Lf / Cbar / S  # per deg
        Cma_fus = math.degrees(Cma_fus_deg)  # per rad = m.degrees(per deg)

        L_HT_act = L_HT_act_mm/1000
        h_t = self.h_t_mm/1000
        CLa_W_deg = math.radians(CLa_W)
        # dedt_deg = 21*CLa_W_deg/(A**0.725) * Cbar/L_HT_act * ((10-3*lamda)/7) * (1 - h_t/b)
        dahda_deg = 1 - dedt_deg  # Horizontal Tail Down-wash
        dahda = math.degrees(dahda_deg)

        X_AC_wing = X_AC_wing_mm / 1000
        X_AC_HT = X_AC_HT_mm / 1000

        X_bar_ac_w = X_AC_wing / Cbar
        X_bar_ac_h = X_AC_HT / Cbar

        CLa_HT_deg = math.radians(CLa_HT)
        # Cma_fus_deg = math.radians(Cma_fus)

        X_bar_NP = (
            CLa_W * X_bar_ac_w
            - Cma_fus
            + self.eta_h_T0 * S_HT / S * CLa_HT * dahda_deg * X_bar_ac_h
        ) / (CLa_W + self.eta_h_T0 * S_HT / S * CLa_HT * dahda_deg)

        XNP = X_bar_NP * Cbar
        XNP_mm = XNP * 1000

        XNP_mm_short = round(XNP_mm, 0)

        XCG = self.XCG_mm / 1000

        SM = (XNP - XCG) / Cbar * 100
        SM_short = round(SM, 0)

        prp_xcg = XNP * 1000 - req_sm / 100 * Cbar * 1000

        S_HT_im = Area.m2_ft2(self, S_HT, "ft2")
        Croot_HT_im = Length.meter_feet(self, Croot_HT, "ft")
        Ctip_HT_im = Length.meter_feet(self, Ctip_HT, "ft")
        rho_a_cruise_im = Density.kg1m3_slug1ft3(self, rho_a_cruise, "slug1ft3")
        v_cruise_im = Velocity.m1s_ft1s(self, v_cruise, "ft1s")

        eta_h = 1 + (
            S_HT_im - (Croot_HT_im + Ctip_HT_im) * 0.5 * Dp_in / 1000
        ) / S_HT * 2200 * Preq_550hp_cruise / (
            0.5 * rho_a_cruise_im * v_cruise_im**2 * v_cruise_im * math.pi * Dp_in**2
        )

        X_bar_NP_thrusteffect = (
            CLa_W * X_bar_ac_w
            - Cma_fus
            + eta_h * S_HT / S * CLa_HT * dahda_deg * X_bar_ac_h
        ) / (CLa_W + eta_h * S_HT / S * CLa_HT * dahda_deg)
        XNP_thrusteffect = X_bar_NP_thrusteffect * Cbar
        XNP_thrusteffect_mm = XNP_thrusteffect * 1000
        SM_thrusteffect = (XNP_thrusteffect - XCG) / Cbar * 100

        return (Wf,X_Cr4_mm,X_Cr4,X_bar_Cr4,Kf,Cma_fus_deg,Cma_fus,dahda_deg,dahda,X_AC_wing,X_AC_HT,X_bar_ac_w,X_bar_ac_h,X_bar_NP,XNP,XNP_mm,XNP_mm_short,XCG,SM,SM_short,prp_xcg,X_bar_NP_thrusteffect,XNP_thrusteffect,XNP_thrusteffect_mm,eta_h,SM_thrusteffect)

    def neutralpointtwo(
        self,
        Wf_i,
        xi_fwd,
        Cf,
        delta_x5,
        dedt_0_deg,
        rho_a_cruise,
        v_cruise,
        S,
        Cbar,
        CLa_W_deg,
        Wf_mm,
        b,
        X_bar_ac_w,
        S_HT,
        CLa_HT_deg,
        dahda_deg,
        X_bar_ac_h,
        xi_aft,
        x_h,
        delta_xi,
        sum_x8_13,
        eta_h,
    ):
        def pointarea(sta1, sta2, sta3, sta4, sta5, sta6, sta7, sta8, sta9):
            import numpy as np
            import math

        def dEbarda1(x):
            a = -0.024114402286
            b = 0.245444318066
            c = -0.662338592797
            d = 1.59332757139
            return a * x**3 + b * x**2 + c * x + d
            # return -0.0241144 * x ** 3 + 0.245444 * x ** 2 + -0.6623386 * x + 1.59332757

        def dEbarda2(y):
            a = 8.93153846101
            b = -25.9516929839
            c = 29.1581791998
            d = -16.1850084707
            e = 5.79427066137
            return a * y**4 + b * y**3 + c * y**2 + d * y + e
            # return 8.931538461 * y ** 4 + -25.9517 * y ** 3 + 29.1582 * y ** 2 + -16.185 * y + 5.794271

        sumWf_new_im = 0
        sumWf_i = []
        n = 1
        for val in range(0, 13):
            if 1 <= val <= 4:
                Wf_i = Wf.pop(n - 1)
                Wf_i.insert(n - 1, Wf)

                Wf_im = Length.meter_feet(self, Wf, "ft")
                xi_fwd_im = Length.meter_feet(self, xi_fwd, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")

                # dEbarda = dEbarda1(xi_fwd_im / Cf_im)
                dEbarda = dEbarda1(Wf_im / Cf_im)
                sumWf_im = Wf_im**2 * dEbarda * delta_xi_im
                sumWf_i.append(sumWf_im)
                # sumWf_new_im = sumWf_im

            elif val == 5:
                delta_xi = delta_x5
                Wf = Wf_i.pop(n - 1)
                Wf_i.insert(n - 1, Wf)

                Wf_im = Length.meter_feet(self, Wf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")

                dEbarda = dEbarda2(delta_xi_im / Cf_im)
                sumWf_im = Wf_im**2 * dEbarda * delta_xi_im
                sumWf_i.append(sumWf_im)
                # sumWf_new_im = sumWf_im

            # if 6 <= n <= 13:
            elif val == 6 or val == 7:
                delta_xi = Cf / 2
                Wf = Wf_i.pop(n - 1)
                Wf_i.insert(n - 1, Wf)

                Wf_im = Length.meter_feet(self, Wf, "ft")
                xi_aft_im = Length.meter_feet(self, xi_aft, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")
                x_h_im = Length.meter_feet(self, x_h, "ft")

                # dEbarda = xi_aft_im / x_h_im * (1 - dedt_0_deg)
                dEbarda = Wf_im / x_h_im * (1 - dedt_0_deg)
                sumWf_im = Wf_im**2 * dEbarda * delta_xi_im
                sumWf_i.append(sumWf_im)
                # sumWf_new_im = sumWf_im

            elif 8 <= val <= 13:
                delta_xi = sum_x8_13 / 6

                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")
                Wf = Wf_i.pop(n - 1)
                Wf_i.insert(n - 1, Wf)

                Wf_im = Length.meter_feet(self, Wf, "ft")
                xi_aft_im = Length.meter_feet(self, xi_aft, "ft")
                x_h_im = Length.meter_feet(self, x_h, "ft")

                # dEbarda = xi_aft_im / x_h_im * (1 - dedt_0_deg)
                dEbarda = Wf_im / x_h_im * (1 - dedt_0_deg)
                sumWf_im = Wf_im**2 * dEbarda * delta_xi_im
                sumWf_i.append(sumWf_im)
                # sumWf_new_im = sumWf_im

            # sumWf_new_im = sumWf_im
            n = n + 1
            # if n > 13:
            # break
        sx_sum = 0
        for sx in sumWf_i:
            sx_sum = sx + sx_sum
        sumWf_new_im = sx_sum

        q = 0.5 * rho_a_cruise * v_cruise**2
        q_im = Pressure.pa_psf(self, q, "psf")
        dMda_im = (q_im / 36.5) * (CLa_W_deg / 0.08) * sumWf_new_im

        S_im = Area.m2_ft2(self, S, "ft2")
        S_HT_im = Area.m2_ft2(self, S_HT, "ft2")
        X_bar_ac_w_im = Length.meter_feet(self, X_bar_ac_w, "ft")
        X_bar_ac_h_im = Length.meter_feet(self, X_bar_ac_h, "ft")
        Cbar_im = Length.meter_feet(self, Cbar, "ft")

        deltaXac_f_im = -dMda_im / (q_im * S_im * Cbar_im * CLa_W_deg)
        # print(deltaXac_f)
        Wf_kf_im = Length.meter_feet(self, Wf_mm / 1000, "ft")
        b_im = Length.meter_feet(self, b, "ft")

        K_wf = (
            1 + 0.025 * (Wf_kf_im / 1000 / b_im) - 0.25 * (Wf_kf_im / 1000 / b_im) ** 2
        )
        CLa_WF_deg = K_wf * CLa_W_deg

        X_bar_NPtwo_im = (
            (X_bar_ac_w_im + deltaXac_f_im) * CLa_WF_deg
            + eta_h * S_HT_im / S_im * CLa_HT_deg * dahda_deg * X_bar_ac_h_im
        ) / (CLa_WF_deg + eta_h * S_HT_im / S_im * CLa_HT_deg * dahda_deg)
        XNPtwo_im = X_bar_NPtwo_im * Cbar_im
        XNPtwo = Length.meter_feet(self, XNPtwo_im, "meter")
        XNPtwo_mm = XNPtwo * 1000

        return (
            sumWf_new_im,
            n,
            dMda_im,
            deltaXac_f_im,
            K_wf,
            CLa_WF_deg,
            X_bar_NPtwo_im,
            XNPtwo,
            XNPtwo_mm,
            sumWf_i,
            sumWf_new_im,
        )

    def neutralpointthree(
        self,
        Wf,
        xi,
        Cf,
        dedt_0_deg,
        rho_a_cruise,
        v_cruise,
        S,
        Cbar,
        CLa_W_deg,
        Wf_mm,
        b,
        X_bar_ac_w,
        S_HT,
        CLa_HT_deg,
        dahda_deg,
        X_bar_ac_h,
        x_h,
        delta_xi,
        sum_x8_13,
        eta_h,
        XCG_mm,
    ):
        """
        def dEbarda1(x):
            a = -0.024114402286
            b = 0.245444318066
            c = -0.662338592797
            d = 1.59332757139
            return a*x**3 + b*x**2 + c*x + d
            # return -0.0241144 * x ** 3 + 0.245444 * x ** 2 + -0.6623386 * x + 1.59332757

        def dEbarda2(y):
            a = 8.93153846101
            b = -25.9516929839
            c = 29.1581791998
            d = -16.1850084707
            e = 5.79427066137
            return a*y**4 + b*y**3 + c*y**2 + d*y + e
            # return 8.931538461 * y ** 4 + -25.9517 * y ** 3 + 29.1582 * y ** 2 + -16.185 * y + 5.794271
        """

        def dEbarda1(x):
            return (
                -0.415006 * x**5
                + 2.64181 * x**4
                - 6.48731 * x**3
                + 7.76831 * x**2
                - 4.78746 * x
                + 2.49001
            )

        def dEbarda2(x):
            return (
                -14.5661 * x**5
                + 45.6298 * x**4
                - 57.321 * x**3
                + 39.0083 * x**2
                - 16.6653 * x
                + 5.72027
            )

        sumWf = []
        n = 1
        for val in range(1, 14):
            if 1 <= val <= 4:
                Wf_i = Wf[n - 1]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[n - 1]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")

                dEbarda = dEbarda1(xi_i_im / Cf_im)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)

            elif val == 5:
                Wf_i = Wf[n - 1]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[n - 1]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                Cf_im = Length.meter_feet(self, Cf, "ft")
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")

                dEbarda = dEbarda2(xi_i_im / Cf_im)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)

            elif 6 <= val <= 13:
                delta_xi = sum_x8_13 / 6
                delta_xi_im = Length.meter_feet(self, delta_xi, "ft")

                Wf_i = Wf[n - 1]
                # Wf.insert(n - 1, Wf_i)

                xi_i = xi[n - 1]
                # xi.insert(n - 1, xi_i)
                # print(xi_i)

                Wf_i_im = Length.meter_feet(self, Wf_i, "ft")
                xi_i_im = Length.meter_feet(self, xi_i, "ft")
                x_h_im = Length.meter_feet(self, x_h, "ft")

                dEbarda = xi_i_im / x_h_im * (1 - dedt_0_deg)
                sumWf_im = (Wf_i_im**2) * dEbarda * delta_xi_im
                sumWf.append(sumWf_im)
            n += 1

        sumofWf_im = 0
        for zal in range(0, 13):
            sumWf_i = sumWf[zal]
            sumofWf_im += sumWf_i

        q = 0.5 * rho_a_cruise * v_cruise**2
        q_im = Pressure.pa_psf(self, q, "psf")
        dMda_im = (q_im / 36.5) * (CLa_W_deg / 0.08) * sumofWf_im

        S_im = Area.m2_ft2(self, S, "ft2")
        S_HT_im = Area.m2_ft2(self, S_HT, "ft2")
        Cbar_im = Length.meter_feet(self, Cbar, "ft")

        deltaXac_f_im = -1 * dMda_im / (q_im * S_im * Cbar_im * CLa_W_deg)
        # print(deltaXac_f)
        Wf_kf_im = Length.meter_feet(self, Wf_mm / 1000, "ft")
        b_im = Length.meter_feet(self, b, "ft")

        K_wf = 1 + 0.025 * (Wf_kf_im / b_im) - 0.25 * (Wf_kf_im / b_im) ** 2
        CLa_WF_deg = K_wf * CLa_W_deg

        X_bar_NPtwo_im = (
            (X_bar_ac_w + deltaXac_f_im) * CLa_WF_deg
            + eta_h * S_HT_im / S_im * CLa_HT_deg * dahda_deg * X_bar_ac_h
        ) / (CLa_WF_deg + eta_h * S_HT_im / S_im * CLa_HT_deg * dahda_deg)
        XNPtwo_im = X_bar_NPtwo_im * Cbar_im
        XNPtwo = Length.meter_feet(self, XNPtwo_im, "meter")
        XNPtwo_mm = XNPtwo * 1000
        SM_roskam = (XNPtwo_mm - XCG_mm) / (Cbar * 1000) * 100

        return (
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
        )


class Aerodynamics:
    def __init__(self):
        pass

    def LLTgeneral(self, S:float, b:float, Croot:float, Ctip:float, AOA:float, alpha_L0=-4.3):
        import numpy as np

        
        AR = (b**2) / S
        N = 20  # Number of spanwise position
        # n = np.arange(1, N + 1)
        y = np.linspace(-b / 2, b / 2, N)
        # print(y)
        lamda = Ctip / Croot
        Chord = Croot * (1 + 2 * (lamda - 1) * abs(y) / b)

        # print(f"\nChord: {Chord}")

        alpha_L0 = alpha_L0 * (np.pi / 180)
        theta_0 = np.arccos(-2 * y / b)#.reshape(-1, 1)
        # theta_0[0] = theta_0[1]  # replace 0.0
        theta_0[0] = theta_0[1] / 10000  # replace 0.0

        # print(f"\ntheta_0: \n{theta_0}")

        alpha = AOA * np.radians(np.ones(N))
        alpha_prime = alpha - alpha_L0
        # print(f"\nalpha_prime: \n{alpha_prime}")

        # RHS1, RHS2 = 0, 0
        RHS = []

        for i, Chord_ in enumerate(Chord):
            matA, matB = [], []
            RHS_ = []
            for n in range(N):
                matA.append( 2*b/(np.pi*Chord_) * np.sin((n+1)*theta_0[i]) )
                matB.append( (n+1)*np.sin((n+1)*theta_0[i])/np.sin(theta_0[i]) )
            matA = np.asarray(matA)
            matB = np.asarray(matB)
            RHS_ = matA + matB
            RHS.append(RHS_)

        RHS = np.asarray(RHS)
        # RHS[np.isnan(RHS)] = 1.0  # , beta2[np.isnan(beta2)] = 1.0, 1.0
        # beta1[np.isinf(beta1)], beta2[np.isinf(beta2)] = 0.0, 0.0

        # print(f"\n\nRHS: \n{RHS}")
        # print(f"\n\nRHS: \n{RHS.shape}")
        # print(f"\n\nRHS: \n{RHS[:,:,0]}")

        A = np.linalg.solve(RHS, alpha_prime)

        # print(f"\nA: {A}")

        CL = A[0] * np.pi * AR

        Gamma_Vinf = []                             # Gamma_Vinf = Gamma / Vinf
        for i, theta_0_ in enumerate(theta_0):
            Gamma_Vinf_ = 0
            for n in range(N):
                Gamma_Vinf_ += A[n]*np.sin((n+1)*theta_0_)
            Gamma_Vinf_ = 2*b*Gamma_Vinf_
            Gamma_Vinf.append(Gamma_Vinf_)

        Gamma_Vinf = np.asarray(Gamma_Vinf)
        Cl = 2*Gamma_Vinf/Chord
        # print(f"Local Cl: {Cl}")

        return CL

    def liftinglinetheory(self, lamda, S, AR, a_2d, alpha_0_degrees, alpha):
        import numpy as np  # type: ignore
        import math
        import matplotlib.pylab as plt  # type: ignore

        N: int = 9  # (number of segments - 1)
        # S: float = 1.041  # wing area m^2
        # AR: float = 7.0  # Aspect ratio
        # taper: float = 1.0  # Taper ratio
        alpha_twist: float = 0.0  # Twist angle (deg)
        i_w: float = alpha  # wing setting angle (deg)
        # a_2d: float = 6.8754  # lift curve slope (1/rad)
        # alpha_0: float = -4.2  # zero-lift angle of attack (deg)

        b = math.sqrt(AR * S)  # wing span (m)
        MAC = S / b  # Mean Aerodynamic Chord (m)
        Croot = (1.5 * (1 + lamda) * MAC) / (1 + lamda + lamda**2)  # root chord (m)

        # theta = np.arange(math.pi/(2*N), math.pi/2, math.pi/(2*(N)))
        theta = np.linspace((math.pi / (2 * N)), (math.pi / 2), N, endpoint=True)
        # alpha =np.arange(i_w+alpha_twist,i_w ,-alpha_twist/(N))
        alpha = np.linspace(i_w + alpha_twist, i_w, N)
        z = (b / 2) * np.cos(theta)
        c = Croot * (1 - (1 - lamda) * np.cos(theta))  # Mean Aerodynamics
        mu = c * a_2d / (4 * b)

        LHS = (
            mu * (np.array(alpha) - alpha_0_degrees) / 57.3
        )  # .reshape((N-1),1)# Left Hand Side

        RHS = []
        for i in range(1, 2 * N + 1, 2):
            RHS_iter = np.sin(i * theta) * (
                1 + (mu * i) / (np.sin(list(theta)))
            )  # .reshape(1,N)
            # print(RHS_iter,"RHS_iter shape")
            RHS.append(RHS_iter)

        test = np.asarray(RHS)
        x = np.transpose(test)
        inv_RHS = np.linalg.inv(x)

        ans = np.matmul(inv_RHS, LHS)
        """
        mynum = np.divide((4 * b), c)

        test = (np.sin((1) * theta)) * ans[0] * mynum
        test1 = (np.sin((3) * theta)) * ans[1] * mynum
        test2 = (np.sin((5) * theta)) * ans[2] * mynum
        test3 = (np.sin((7) * theta)) * ans[3] * mynum
        test4 = (np.sin((9) * theta)) * ans[4] * mynum
        test5 = (np.sin((11) * theta)) * ans[5] * mynum
        test6 = (np.sin((13) * theta)) * ans[6] * mynum
        test7 = (np.sin((15) * theta)) * ans[7] * mynum
        test8 = (np.sin((17) * theta)) * ans[8] * mynum

        CL = test + test1 + test2 + test3 + test4 + test5 + test6 + test7 + test8
        
        CL1 = np.append(0, CL)
        y_s = [b / 2, z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8]]
        plt.plot(y_s, CL1, marker="o")
        plt.title("Lifting Line Theory\n Elliptical Lift distribution")
        plt.xlabel("Semi-span location (m)")
        plt.ylabel("Lift coefficient")
        plt.grid()
        plt.show()
        """
        CL_wing = (
            math.pi * AR * ans[0]
        )  # USE THIS CL WITH CRUISE SPEED TO CALCULATE THE ACCURATE LIFT!!!!!!!!!!
        # print(CL_wing, "CL_wing")

        return CL_wing


class flap:
    def __init__(self, flpos1, flpos2, dflang, cf_c):
        self.flpos1 = flpos1
        self.flpos2 = flpos2
        self.deltaeta = abs(flpos2 - flpos1)
        self.dflang = dflang
        self.cf_c = cf_c

    def kb_eta_curve(self, eta):
        return (
            -0.366918 * eta**4
            + 0.344043 * eta**3
            - 0.23606 * eta**2
            + 1.2596 * eta
            - 0.00272394
        )

    def flpkprime(self):
        if 10 < self.dflang <= 25:
            return (
                -1.50362175e-06 * self.dflang**6
                + 1.56725007e-04 * self.dflang**5
                - 6.65311877e-03 * self.dflang**4
                + 1.47279221e-01 * self.dflang**3
                - 1.79634510e00 * self.dflang**2
                + 1.14572023e01 * self.dflang
                - 2.88656973e01
            )
        elif 25 < self.dflang < 60:
            return (
                -3.48075148e-06 * self.dflang**3
                + 5.52494059e-04 * self.dflang**2
                - 3.24255644e-02 * self.dflang
                + 1.14776538e00
            )

    def lifteff(self):
        return (
            -7.5528 * self.cf_c**4
            + 30.632 * self.cf_c**3
            + -34.8172 * self.cf_c**2
            + 23.863 * self.cf_c
            - 0.657831
        )

    def plainflp(self):
        pass
        # deltacl = dflang *

    def flapperf(self):
        Kb = abs(Aerodynamics.kb_eta(self.flpos2) - Aerodynamics.kb_eta(self.flpos1))


class Performance:
    def __init__(self):
        pass
        # self.rho_a = rho_a
        # self.g = g

    def stallcond(self, W, rho, S, CL_stall, g):
        import math

        v_stall = math.sqrt(2 * W * g / rho / S / CL_stall)
        v_stall_knot = v_stall / 0.5144
        v_stall_kmh = v_stall * 3.6

        return v_stall, v_stall_knot, v_stall_kmh

    def liftanddrag(self, S, CL, CD, v, rho_a, g, Va, Ta, Cbar):
        import math

        v_kmh = v * 3.6
        v_kmh_short = round(v_kmh, 0)
        # v_kn = v * 1.943844492
        v_kn = Velocity.m1s_kn(self, v, "kn")
        v_kn_short = round(v_kn, 0)
        Mach = v / Va
        Mach_short = round(Mach, 3)
        L = 0.5 * rho_a * S * CL * (v**2)
        L_kg = L / g  # Lift, kg
        L_kg_short = round(L_kg, 0)
        D = 0.5 * rho_a * S * CD * (v**2)
        D_kg = D / g  # Drag, kg
        D_kg_short = round(D_kg, 0)
        mu = 1.458e-6 * math.sqrt(Ta) / (1 + 110.4 / Ta)
        Re = rho_a * v * Cbar / mu
        Re_short = round(Re, -4)

        return (
            L,
            L_kg,
            L_kg_short,
            D,
            D_kg,
            D_kg_short,
            v_kmh,
            v_kmh_short,
            v_kn,
            v_kn_short,
            Mach,
            Mach_short,
            mu,
            Re,
            Re_short,
        )

    def t2w(self, W, tw):
        T_kg = W * tw
        T_kg_short = round(T_kg, 0)
        # T_lbs = T_kg * 2.204622622
        T_lbs = Mass.kg_pound(self, T_kg, "lbs")
        T_lbs_short = round(T_lbs, 0)

        return T_kg, T_kg_short, T_lbs, T_lbs_short

    def powreq(self, D, v):
        Preq = D * v
        Preq_short = round(Preq, 0)
        Preq_kW = Preq / 1000
        Preq_kW_short = round(Preq_kW, 1)
        Preq_550hp = Preq * 0.001341022
        Preq_550hp_short = round(Preq_550hp, 0)

        return Preq, Preq_short, Preq_kW, Preq_kW_short, Preq_550hp, Preq_550hp_short


class Atmosphere:
    def __init__(self):
        pass

    def atmdata(self, h):
        import math

        To = 288.1667
        Po = 101314.628
        R = 6371315
        phi = 37 * math.pi / 180
        g_o = 9.780490 * (
            1 + 0.0052884 * (math.sin(phi)) ** 2 - 0.0000059 * (math.sin(2 * phi)) ** 2
        )
        Rprime = 286.99236

        if h <= 11000:
            Ta = To - (0.006499708) * h
            Pa = Po * (1 - 2.255692257e-5 * h) ** 5.2561
            rho_a = Pa / (Rprime * Ta)
            Va = 20.037673 * math.sqrt(Ta)
            g = g_o * (R / (R + h)) ** 2

        elif h > 11000 and h <= 25000:
            Ta = 216.66666667
            Pa = Po * (0.223358) * (math.exp((-1.576883202e-4) * (h - 11000)))
            rho_a = Pa / (Rprime * Ta)
            Va = 20.037673 * math.sqrt(Ta)
            g = g_o * (R / (R + h)) ** 2
        elif h > 25000:
            Ta = 216.66666667 + (3.000145816e-3) * (h - 25000)
            Pa = 2489.773467 * (math.exp((-1.576883202e-4) * (h - 25000)))
            rho_a = Pa / (Rprime * Ta)
            Va = 20.037673 * math.sqrt(Ta)
            g = g_o * (R / (R + h)) ** 2

        return Ta, Pa, rho_a, Va, g


class conceptWeight:
    def __init__(self):
        pass

    def prefuel(
        self,
        W1_WTO,
        W2_W1,
        W3_W2,
        W7_W6,
        W8_W7,
        Cp_climb,
        Cp_cruise,
        Cp_loiter,
        eta_p_climb,
        eta_p_cruise,
        eta_p_loiter,
        LD_climb,
        LD_cruise,
        LD_loiter,
        R_cruise,
        E_climb,
        E_loiter,
        v_climb_kn,
        v_loiter_kn,
        W,
        W_fuel_res,
        rho_fuel,
    ):
        import math

        W_lbs = Mass.kg_pound(self, W, "lbs")
        W_fuel_res_lbs = Mass.kg_pound(self, W_fuel_res, "lbs")
        R_cruise_imp = Length.km_statmile(self, R_cruise, "statmile")
        v_climb_imp = Velocity.kn_mph(self, v_climb_kn, "mph")
        v_loiter_imp = Velocity.kn_mph(self, v_loiter_kn, "mph")

        W4_W3 = math.exp(
            -Cp_climb * E_climb * v_climb_imp / 375 / LD_climb / eta_p_climb
        )
        W5_W4 = math.exp(-R_cruise_imp * Cp_cruise / 375 / LD_cruise / eta_p_cruise)
        W6_W5 = math.exp(
            -Cp_loiter * E_loiter * v_loiter_imp / 375 / LD_loiter / eta_p_loiter
        )

        Mff = W1_WTO * W2_W1 * W3_W2 * W4_W3 * W5_W4 * W6_W5 * W7_W6 * W8_W7
        W_f_used_lbs = (1 - Mff) * W_lbs
        W_f_lbs = W_f_used_lbs + W_fuel_res_lbs
        W_f = Mass.kg_pound(self, W_f_lbs, "kg")
        V_f = W_f / rho_fuel
        V_f_liters = Sciexp.exponent(self, V_f, 3)

        W_empty = W - W_f

        return (
            W4_W3,
            W5_W4,
            W6_W5,
            Mff,
            W_f_used_lbs,
            W_f_lbs,
            W_f,
            W_empty,
            V_f,
            V_f_liters,
        )

    def wingfuelvol(self, S_imp, b, lamda, tc_r, tc_t):
        import math

        b_imp = Length.meter_feet(self, b, "ft")

        V_WF_ft3 = (
            0.54
            * S_imp**2
            / b_imp
            * tc_r
            * (1 + lamda * (tc_t / tc_r) ** 0.5 + lamda**2 * (tc_t / tc_r))
            / (1 + lamda) ** 2
        )
        V_WF = Volume.m3_ft3(self, V_WF_ft3, "m3")
        V_WF_min = V_WF * 0.9
        V_WF_max = V_WF * 1.1
        V_WF_diff = V_WF * 0.1
        V_WF_liters = Sciexp.exponent(self, V_WF, 3)
        V_WF_liters_min = Sciexp.exponent(self, V_WF_min, 3)
        V_WF_liters_max = Sciexp.exponent(self, V_WF_max, 3)
        V_WF_liters_diff = Sciexp.exponent(self, V_WF_diff, 3)

        return (
            V_WF_ft3,
            V_WF,
            V_WF_min,
            V_WF_max,
            V_WF_liters,
            V_WF_liters_min,
            V_WF_liters_max,
            V_WF_liters_diff,
        )


class eVTOL_BMF:
    def __init__(
        self,
        LD_loiter_eVTOL,
        LD_cruise_eVTOL,
        LD_climb_eVTOL,
        Esb,
        eta_b2s,
        eta_p_eVTOL,
        g,
        W,
        Pused_climb,
        v_climb,
    ):
        self.LD_loiter_eVTOL = LD_loiter_eVTOL
        self.LD_cruise_eVTOL = LD_cruise_eVTOL
        self.LD_climb_eVTOL = LD_climb_eVTOL
        self.Esb = Esb
        self.eta_b2s = eta_b2s
        self.eta_p_eVTOL = eta_p_eVTOL
        self.g = g
        self.W = W
        self.Pused_climb = Pused_climb
        self.v_climb = v_climb

    def endurance_range(self, v_cruise, m_b):
        E_eVTOL = (
            3.6
            * self.LD_cruise_eVTOL
            * self.Esb
            * self.eta_b2s
            * self.eta_p_eVTOL
            / self.g
            / v_cruise
            * m_b
            / self.W
        )
        R_eVTOL = (
            3.6
            * self.LD_cruise_eVTOL
            * self.Esb
            * self.eta_b2s
            * self.eta_p_eVTOL
            / self.g
            * m_b
            / self.W
        )
        V_v_eVTOL = (
            1000 * self.eta_p_eVTOL / self.g * self.Pused_climb / self.W
            - self.v_climb / 3.6 / self.LD_climb_eVTOL
        )
        return E_eVTOL, R_eVTOL, V_v_eVTOL

    def bmf(
        self,
        E_loiter_eVTOL,
        R_cruise_eVTOL,
        h_climb_alt_eVTOL,
        v_loiter,
        V_v_eVTOL,
        Pused_climb,
        W_tot_mean,
        W_payload_eVTOL,
        We_W0_eVTOL,
        m_b_guess,
    ):
        BMF_loiter = (
            E_loiter_eVTOL
            * v_loiter
            * self.g
            / 3.6
            / self.Esb
            / self.eta_b2s
            / self.eta_p_eVTOL
            / self.LD_loiter_eVTOL
        )
        BMF_cruise = (
            R_cruise_eVTOL
            * self.g
            / 3.6
            / self.Esb
            / self.eta_b2s
            / self.eta_p_eVTOL
            / self.LD_cruise_eVTOL
        )
        BMF_climb = (
            h_climb_alt_eVTOL
            / (3.6 * V_v_eVTOL * self.Esb * self.eta_b2s)
            * (1000 * Pused_climb / self.W)
        )
        BMF_sum = BMF_loiter + BMF_cruise + BMF_climb
        Battery_available = self.W - W_tot_mean - W_payload_eVTOL
        W0_eVTOL = W_payload_eVTOL / (1 - BMF_sum - We_W0_eVTOL)
        W0_eVTOL2 = W_payload_eVTOL / (1 - m_b_guess / self.W - We_W0_eVTOL)
        return (
            BMF_loiter,
            BMF_cruise,
            BMF_climb,
            BMF_sum,
            Battery_available,
            W0_eVTOL,
            W0_eVTOL2,
        )

    def BMFsizing(
        self,
        W_payload_eVTOL,
        E_loiter_eVTOL,
        R_cruise_eVTOL,
        V_v_eVTOL,
        h_climb_alt_eVTOL,
        Pused_climb,
        rho_a_cruise,
        WS,
        merit,
        eta_mech,
        f_adj,
        v_loiter,
        E_vertiTO,
        E_vertiland,
    ):
        import numpy as np

        P_hovclimb_div = (
            1 / eta_mech * f_adj / merit * (f_adj * WS / (2 * rho_a_cruise)) ** 0.5
        )
        BMF_hovascend = E_vertiTO / (self.Esb * self.eta_b2s) * P_hovclimb_div
        BMF_climb = (
            h_climb_alt_eVTOL
            / (3.6 * V_v_eVTOL * self.Esb * self.eta_b2s)
            * (1000 * Pused_climb / self.W)
        )
        BMF_cruise = (
            R_cruise_eVTOL
            * self.g
            / 3.6
            / self.Esb
            / self.eta_b2s
            / self.eta_p_eVTOL
            / self.LD_cruise_eVTOL
        )
        BMF_loiter = (
            E_loiter_eVTOL
            * v_loiter
            * self.g
            / 3.6
            / self.Esb
            / self.eta_b2s
            / self.eta_p_eVTOL
            / self.LD_loiter_eVTOL
        )
        P_hovdescend_div = (
            1 / eta_mech * f_adj / merit * (f_adj * WS / (2 * rho_a_cruise)) ** 0.5
        )  # originaly f_adj * W0 / merit but W5/W4 = ...*P_hovclimb/W0
        BMF_hovdescend = (
            E_vertiland / (self.Esb * self.eta_b2s) * P_hovdescend_div
        )  # originaly P_hovclimb/W0
        Wf_W0 = BMF_hovascend + (BMF_climb + BMF_cruise + BMF_loiter) + BMF_hovdescend
        W0 = self.W  # initial W0 guess
        for i in range(10000):
            W0_guess = W0
            Wes_Wo = 2.9566 * W0 ** (-0.24) * 0.95
            # Wes_Wo = (0.82826 - 1.6436/(W0 * 9.81)) / 9.81        # For eVTOL UAV
            W0 = W_payload_eVTOL / (1 - Wf_W0 - Wes_Wo)
            if abs(W0 - W0_guess) / W0_guess < 0.001:
                Wbat = Wf_W0 * W0
                break
        return W0, Wf_W0, Wbat


class WeightEst:
    def __init__(self):
        pass

    def imperial(
        self,
        W,
        W_fw,
        S,
        b,
        rho_a_cruise,
        v_cruise,
        t_r_HT,
        S_HT,
        S_VT,
        t_r_VT,
        L_HT_act,
        b_HT,
        b_VT,
        FL,
        Wf_mm,
        hf_mm,
        W_press,
        FL_short,
        l_n_mm,
        Croot,
        tc_r,
    ):
        import math

        W_imp = Mass.kg_pound(self, W, "lbs")
        W_fw_imp = Mass.kg_pound(self, W_fw, "lbs")
        S_imp = Area.m2_ft2(self, S, "ft2")
        b_imp = Length.meter_feet(self, b, "ft")
        q = 0.5 * rho_a_cruise * v_cruise**2
        q_imp = Pressure.pa_psf(self, q, "psf")
        S_HT_imp = Area.m2_ft2(self, S_HT, "ft2")
        S_VT_imp = Area.m2_ft2(self, S_VT, "ft2")
        t_r_HT_imp = Length.meter_feet(self, t_r_HT, "ft")
        t_r_VT_imp = Length.meter_feet(self, t_r_VT, "ft")
        t_r_w = tc_r * Croot
        t_r_w_imp = Length.meter_feet(self, t_r_w, "ft")
        L_HT_act_imp = Length.meter_feet(self, L_HT_act, "ft")
        b_HT_imp = Length.meter_feet(self, b_HT, "ft")
        b_VT_imp = Length.meter_feet(self, b_VT, "ft")
        FL_imp = Length.meter_feet(self, FL, "ft")
        Wf_imp = Length.meter_feet(self, Wf_mm / 1000, "ft")
        hf_imp = Length.meter_feet(self, hf_mm / 1000, "ft")
        Swet_fus = (
            math.pi
            * Wf_mm
            / 1000
            * FL_short
            * (0.5 + 0.135 * (l_n_mm / 1000) / FL_short) ** (2 / 3)
            * (1.015 + 0.3 / (FL_short / (Wf_mm / 1000)) ** 1.5)
        )
        Swet_fus_imp = Area.m2_ft2(self, Swet_fus, "ft2")
        W_press_imp = Mass.kg_pound(self, W_press, "lbs")

        return (
            W_imp,
            W_fw_imp,
            S_imp,
            b_imp,
            q,
            q_imp,
            S_HT_imp,
            S_VT_imp,
            t_r_w,
            t_r_w_imp,
            t_r_HT_imp,
            t_r_VT_imp,
            L_HT_act_imp,
            b_HT_imp,
            b_VT_imp,
            FL_imp,
            Wf_imp,
            hf_imp,
            Swet_fus_imp,
            Swet_fus,
            W_press_imp,
        )

    def cessnamethod(
        self,
        W_imp,
        S_imp,
        n_ult,
        A,
        S_HT_imp,
        A_HT,
        t_r_HT_imp,
        S_VT_imp,
        A_VT,
        t_r_VT_imp,
        theta_c4_VT,
        Pmax,
        FL_imp,
        N_pax,
    ):
        import math

        W_w_imp = 0.04674 * W_imp**0.397 * S_imp**0.36 * n_ult**0.397 * A**1.712
        W_w = Mass.kg_pound(self, W_w_imp, "kg")
        W_h_imp = (
            3.184
            * W_imp**0.887
            * S_HT_imp**0.101
            * A_HT**0.138
            / (174.04 * (t_r_HT_imp) ** 0.223)
        )
        W_h = Mass.kg_pound(self, W_h_imp, "kg")
        W_v_imp = (
            1.68
            * W_imp**0.567
            * S_VT_imp**1.249
            * A_VT**0.482
            / (639.95 * t_r_VT_imp**0.747 * (math.cos(theta_c4_VT)) ** 0.882)
        )
        W_v = Mass.kg_pound(self, W_v_imp, "kg")
        W_f_imp = 0.04682 * W_imp**0.692 * Pmax**0.374 * FL_imp**0.59
        W_f = Mass.kg_pound(self, W_f_imp, "kg")
        W_fc_imp = 0.0168 * W_imp
        W_fc = Mass.kg_pound(self, W_fc_imp, "kg")
        W_els_imp = 0.0268 * W_imp
        W_els = Mass.kg_pound(self, W_els_imp, "kg")
        W_fur_imp = 0.412 * (N_pax) ** 1.145 * (W_imp) ** 0.489
        W_fur = Mass.kg_pound(self, W_fur_imp, "kg")

        return (
            W_w_imp,
            W_w,
            W_h_imp,
            W_v_imp,
            W_h,
            W_v,
            W_f_imp,
            W_f,
            W_fc,
            W_els,
            W_fur,
            W_fc_imp,
            W_els_imp,
            W_fur_imp,
        )

    def usafmethod(
        self,
        W_imp,
        n_ult,
        A,
        theta_c4,
        S_imp,
        lamda,
        tc,
        v_kn_cruise,
        S_HT_imp,
        L_HT_act_imp,
        b_HT_imp,
        t_r_HT_imp,
        S_VT_imp,
        b_VT_imp,
        t_r_VT_imp,
        FL_imp,
        Wf_imp,
        hf_imp,
    ):
        import math

        W_w_imp = (
            96.948
            * (
                (W_imp * n_ult / 10**5) ** 0.65
                * (A / math.cos(theta_c4)) ** 0.57
                * (S_imp / 100) ** 0.61
                * ((1 + lamda) / 2 / tc) ** 0.36
                * (1 + v_kn_cruise / 500) ** 0.5
            )
            ** 0.993
        )
        W_w = W_w_imp / 2.204622622
        W_h_imp = (
            127
            * (
                (W_imp * n_ult / 10**5) ** 0.87
                * (S_HT_imp / 100) ** 1.2
                * 0.289
                * (L_HT_act_imp / 10) ** 0.483
                * (b_HT_imp / t_r_HT_imp) ** 0.5
            )
            ** 0.458
        )
        W_h = W_h_imp / 2.204622622
        W_v_imp = (
            98.5
            * (
                (W_imp * n_ult / 10**5) ** 0.87
                * (S_VT_imp / 100) ** 1.2
                * 0.289
                * (b_VT_imp / t_r_VT_imp) ** 0.5
            )
            ** 0.458
        )
        W_v = W_v_imp / 2.204622622
        W_f_imp = (
            200
            * (
                (W_imp * n_ult / 10**5) ** 0.286
                * (FL_imp / 10) ** 0.857
                * ((Wf_imp + hf_imp) / 10)
                * (v_kn_cruise / 100) ** 0.338
            )
            ** 1.1
        )
        W_f = W_f_imp / 2.204622622
        W_fc_imp = 1.08 * (W_imp) ** 0.7
        W_fc = Mass.kg_pound(self, W_fc_imp, "kg")

        return W_w_imp, W_w, W_h_imp, W_h, W_v_imp, W_v, W_f_imp, W_f, W_fc_imp, W_fc

    def raymermethod(
        self,
        S_imp,
        W_fw_imp,
        A,
        theta_c4,
        q_imp,
        lamda,
        tc,
        n_ult,
        W_imp,
        S_HT_imp,
        theta_c4_HT,
        lamda_HT,
        S_VT_imp,
        theta_c4_VT,
        lamda_VT,
        Swet_fus_imp,
        L_HT_act_imp,
        LD,
        W_press_imp,
    ):
        import math

        if W_fw_imp == 0:
            W_w_imp = (
                0.036
                * S_imp**0.758
                * (A / (math.cos(theta_c4)) ** 2) ** 0.6
                * q_imp**0.006
                * lamda**0.04
                * (100 * tc / math.cos(theta_c4)) ** (-0.3)
                * (n_ult * W_imp) ** 0.49
            )
        else:
            W_w_imp = (
                0.036
                * S_imp**0.758
                * W_fw_imp**0.0035
                * (A / (math.cos(theta_c4)) ** 2) ** 0.6
                * q_imp**0.006
                * lamda**0.04
                * (100 * tc / math.cos(theta_c4)) ** (-0.3)
                * (n_ult * W_imp) ** 0.49
            )
        W_w = W_w_imp / 2.204622622
        W_h_imp = (
            0.016
            * (n_ult * W_imp) ** 0.414
            * q_imp**0.168
            * S_HT_imp**0.896
            * (100 * tc / math.cos(theta_c4)) ** (-0.12)
            * (A / (math.cos(theta_c4_HT)) ** 2) ** 0.043
            * lamda_HT ** (-0.02)
        )
        W_h = W_h_imp / 2.204622622
        W_v_imp = (
            0.073
            * (n_ult * W_imp) ** 0.376
            * q_imp**0.122
            * S_VT_imp**0.873
            * (100 * tc / math.cos(theta_c4_VT)) ** (-0.49)
            * (A / (math.cos(theta_c4_VT)) ** 2) ** 0.357
            * lamda_VT**0.039
        )
        W_v = W_v_imp / 2.204622622
        W_f_imp = (
            0.052
            * Swet_fus_imp**1.086
            * (n_ult * W_imp) ** 0.177
            * L_HT_act_imp ** (-0.051)
            * LD ** (-0.072)
            * q_imp**0.241
            + W_press_imp
        )
        W_f = W_f_imp / 2.204622622

        return W_w_imp, W_w, W_h_imp, W_h, W_v_imp, W_v, W_f_imp, W_f

    def torenbeekmethod(
        self, W_imp, b_imp, theta_c2, n_ult, S_imp, t_r_w_imp, N_pax, N_row
    ):
        import math

        W_w_imp = (
            0.00125
            * W_imp
            * (b_imp / math.cos(theta_c2)) ** 0.75
            * (1 + 6.3 * (math.cos(theta_c2) / b_imp) ** 0.5)
            * n_ult**0.55
            * (b_imp * S_imp / t_r_w_imp / W_imp / math.cos(theta_c2)) ** 0.3
        )
        W_w = Mass.kg_pound(self, W_w_imp, "kg")
        W_iae_imp = 33 * N_pax
        W_iae = Mass.kg_pound(self, W_iae_imp, "kg")
        W_fur_imp = 5 + 13 * N_pax + 25 * N_row
        W_fur = Mass.kg_pound(self, W_fur_imp, "kg")

        return W_w_imp, W_w, W_iae_imp, W_iae, W_fur

    def vtolweightest(self, W_imp, S_imp, lamda, A, theta_LE, tc):
        import math

        W_w_vtol_imp = (
            0.032
            * S_imp**0.76
            * lamda**0.04
            * (1.5 * W_imp) ** 0.49
            * (A / (math.cos(theta_LE) ** 2)) ** 0.6
            * (100 * tc / math.cos(theta_LE)) ** (-0.3)
        )
        W_w_vtol = Mass.kg_pound(self, W_w_vtol_imp, "kg")
        W_landingskid_imp = 0.44 * 0.8 * W_imp**0.63
        W_landingskid = Mass.kg_pound(self, W_landingskid_imp, "kg")

        return W_w_vtol_imp, W_w_vtol, W_landingskid_imp, W_landingskid

    def meanval(
        self,
        W_w_cessna,
        W_w_usaf,
        W_w_raymer,
        W_w_torenbeek,
        W_h_cessna,
        W_h_usaf,
        W_v_cessna,
        W_v_usaf,
        W_f_cessna,
        W_f_usaf,
        W_h_raymer,
        W_v_raymer,
        W_f_raymer,
        W_fc_cessna,
        W_fc_usaf,
        W_fur_cessna,
        W_fur_torenbeek,
    ):
        W_w_mean = (W_w_cessna + W_w_usaf + W_w_raymer + W_w_torenbeek) / 4
        W_w_mean_short = round(W_w_mean, 0)
        W_h_mean = (W_h_cessna + W_h_usaf + W_h_raymer) / 3
        W_h_mean_short = round(W_h_mean, 0)
        W_v_mean = (W_v_cessna + W_v_usaf + W_v_raymer) / 3
        W_v_mean_short = round(W_v_mean, 0)
        W_f_mean = (W_f_cessna + W_f_usaf + W_f_raymer) / 3
        W_f_mean_short = round(W_f_mean, 0)
        W_fc_mean = (W_fc_cessna + W_fc_usaf) / 2
        W_fur_mean = (W_fur_cessna + W_fur_torenbeek) / 2

        return (
            W_w_mean,
            W_w_mean_short,
            W_h_mean,
            W_h_mean_short,
            W_v_mean,
            W_v_mean_short,
            W_f_mean,
            W_f_mean_short,
            W_fc_mean,
            W_fur_mean,
        )

    def totalweight(
        self,
        W_crew,
        W_w_mean,
        W_h_mean,
        W_v_mean,
        W_f_mean,
        W_avionic,
        W_payload,
        W_pwr,
        W_fc_mean,
        W_fur_mean,
        W_landingskid,
        W_misc,
    ):
        W_tot_mean = (
            W_w_mean
            + W_h_mean
            + W_v_mean
            + W_f_mean
            + W_pwr
            + W_fc_mean
            + W_fur_mean
            + W_avionic
            + W_landingskid
            + W_misc
        )
        W_tot_eVTOL_mean = W_tot_mean - W_fc_mean
        W_total_mean = W_tot_mean + W_crew + W_payload
        W_total_eVTOL_mean = W_tot_eVTOL_mean + W_crew + W_payload
        return W_tot_mean, W_tot_eVTOL_mean, W_total_mean, W_total_eVTOL_mean


class Drag:
    def __init__(self):
        pass

    def curvefit(self, Ta, rho_a, v, FL_short, Mach, Re_W_cruise, Re_HT, Re_VT):
        import math

        # Body Reynolds number
        Re_fus_i = [
            3e6,
            4e6,
            5e6,
            6e6,
            7e6,
            8e6,
            9e6,
            1e7,
            2e7,
            3e7,
            4e7,
            5e7,
            6e7,
            7e7,
            8e7,
            9e7,
            1e8,
            2e8,
            3e8,
        ]

        # Wing-Fuselage interference factor
        Rwf_i = [
            1.061,
            1.067,
            1.07,
            1.072,
            1.073,
            1.074,
            1.075,
            1.075,
            1.065,
            1.042,
            0.988,
            0.951,
            0.94,
            0.935,
            0.932,
            0.931,
            0.929,
            0.924,
            0.923,
        ]

        num_T = len(Re_fus_i)
        num_Rwf = len(Rwf_i)

        mu_fus = 1.458e-6 * math.sqrt(Ta) / (1 + 110.4 / Ta)

        Re_fus = rho_a * v * FL_short / mu_fus

        k = 1
        y_a = []

        count = len(Re_fus_i)

        while k <= count:
            if k == count:
                Rwf = 0.923
                # print(Rwf)
                break
            else:
                gamma = Re_fus_i.pop(k - 1)
                Re_fus_i.insert(k - 1, gamma)
                delta = Re_fus_i.pop(k)
                Re_fus_i.insert(k, delta)
                if Re_fus >= gamma and Re_fus < delta:
                    alpha = Rwf_i.pop(k - 1)
                    Rwf_i.insert(k - 1, alpha)
                    beta = Rwf_i.pop(k)
                    Rwf_i.insert(k, beta)

                    a = (beta - alpha) / (delta - gamma)
                    # b = alpha - a*beta

                    y_a.insert(k - 1, a)
                    # y_b.insert(k-1, b)

                    Rwf = a * (Re_fus - gamma) + alpha
                    # print(V)
                    break
                else:
                    k = k + 1
        # print(Rwf)

        if Mach >= 0 and Mach < 0.15:
            A = 0.0391
            B = -0.157
        elif Mach >= 0.15 and Mach < 0.5:
            A = 0.0399
            B = -0.159
        elif Mach >= 0.5 and Mach < 0.8:
            A = 0.0392
            B = -0.16
        elif Mach >= 0.8 and Mach < 0.95:
            A = 0.0376
            B = -0.159
        elif Mach >= 0.95 and Mach < 1.25:
            A = 0.0381
            B = -0.161
        elif Mach >= 1.25 and Mach < 1.75:
            A = 0.0371
            B = -0.164
        elif Mach >= 1.75 and Mach < 2.25:
            A = 0.0329
            B = -0.162
        elif Mach >= 2.25 and Mach < 2.75:
            A = 0.0286
            B = -0.161
        elif Mach >= 2.75 and Mach < 3.5:
            A = 0.0261
            B = -0.161

        Cfw = A * Re_W_cruise**B
        Cfw_HT = A * Re_HT**B
        Cfw_VT = A * Re_VT**B
        Cf_fus = A * Re_fus**B

        return mu_fus, Re_fus, Rwf, Cfw, Cf_fus, Cfw_HT, Cfw_VT

    def wingfusintf(self, Re_fus):
        import math

        if Re_fus <= 3.86e7:
            Re_fus = math.log(math.log(Re_fus))
            Re_fus = (Re_fus - 2.796) / 0.05329
            y1 = (
                -0.003403 * Re_fus**9
                - 0.006558 * Re_fus**8
                + 0.008138 * Re_fus**7
                + 0.01214 * Re_fus**6
                - 0.01764 * Re_fus**5
                - 0.01375 * Re_fus**4
                + 0.007883 * Re_fus**3
                - 0.01137 * Re_fus**2
                - 0.01131 * Re_fus
                + 1.0735
            )
            return y1
        elif 3.86e7 < Re_fus <= 7.01e8:
            Re_fus = math.log(math.log(Re_fus))
            Re_fus = (Re_fus - 2.92) / 0.04914
            y2 = (
                0.0004778 * Re_fus**8
                - 0.002573 * Re_fus**7
                + 0.004867 * Re_fus**6
                - 0.004452 * Re_fus**5
                + 0.003175 * Re_fus**4
                - 0.00326 * Re_fus**3
                + 0.007128 * Re_fus**2
                - 0.00894 * Re_fus
                + 0.9254
            )
            return y2

    def wingcd(
        self,
        W,
        S_HT,
        tc,
        tc_r,
        tc_t,
        S,
        Croot,
        Ctip,
        Wf_mm,
        lamda,
        Cfw,
        Lprime,
        theta_tc_max_deg,
        theta_LE,
        Re_fus,
        tc_HT,
        tc_VT,
        lamda_HT,
        lamda_VT,
        S_VT,
        S_exp_plf,
        l_LER,
        CLa_HT_deg,
        aoa_deg,
        inc_agl_HT_deg,
        q,
        rho_a_cruise,
        v_cruise,
        mu_cruise,
        M_cruise,
        CLa_W,
        A,
        R,
        Cfw_HT,
        Cfw_VT,
        Lprime_HT,
        Lprime_VT,
        theta_tc_max_deg_HT,
        theta_tc_max_deg_VT,
    ):
        import math

        # Rls = -5.2658*(math.cos(math.radians(theta_tc_max_deg)))**6 + 21.729*(math.cos(math.radians(theta_tc_max_deg)))**5 \
        #      - 34.248*(math.cos(math.radians(theta_tc_max_deg)))**4 + 25.419*(math.cos(math.radians(theta_tc_max_deg)))**3 \
        #      - 9.3788*(math.cos(math.radians(theta_tc_max_deg)))**2 + 2.4357*(math.cos(math.radians(theta_tc_max_deg))) + 0.1279
        # Rls = -2.9170587200E-11*x**6 + 7.3759476790E-09*x**5 - 6.2844423615E-07*x**4 + 2.0845733859E-05*x**3 - 2.9510563508E-04*x**2 + 1.4571027862E-03*x + 1.0630288141
        Rls = (
            26.0679 * math.cos(math.radians(theta_tc_max_deg)) ** 5
            - 93.9127 * math.cos(math.radians(theta_tc_max_deg)) ** 4
            + 131.227 * math.cos(math.radians(theta_tc_max_deg)) ** 3
            - 89.4831 * math.cos(math.radians(theta_tc_max_deg)) ** 2
            + 30.6855 * math.cos(math.radians(theta_tc_max_deg))
            - 3.51374
        )  # from "Engauge Digitizer"
        Rls_HT = (
            26.0679 * math.cos(math.radians(theta_tc_max_deg_HT)) ** 5
            - 93.9127 * math.cos(math.radians(theta_tc_max_deg_HT)) ** 4
            + 131.227 * math.cos(math.radians(theta_tc_max_deg_HT)) ** 3
            - 89.4831 * math.cos(math.radians(theta_tc_max_deg_HT)) ** 2
            + 30.6855 * math.cos(math.radians(theta_tc_max_deg_HT))
            - 3.51374
        )
        Rls_VT = (
            26.0679 * math.cos(math.radians(theta_tc_max_deg_VT)) ** 5
            - 93.9127 * math.cos(math.radians(theta_tc_max_deg_VT)) ** 4
            + 131.227 * math.cos(math.radians(theta_tc_max_deg_VT)) ** 3
            - 89.4831 * math.cos(math.radians(theta_tc_max_deg_VT)) ** 2
            + 30.6855 * math.cos(math.radians(theta_tc_max_deg_VT))
            - 3.51374
        )
        # ^ x = wing sweep angle
        Rwf = Drag.wingfusintf(self, Re_fus)
        tau = tc_r / tc_t
        Swet_w = 2 * S_exp_plf * (1 + 0.25 * tc_r * (1 + tau * lamda) / (1 + lamda))
        tau = tc_r / tc_t
        tau_HT = 1
        tau_VT = 1
        S_exp_plf = S - 0.5 * (Croot + Ctip) * Wf_mm / 1000
        S_exp_plf_HT = S_HT
        S_exp_plf_VT = S_VT
        Swet_w = 2 * S_exp_plf * (1 + 0.25 * tc_r) * (1 + tau * lamda) / (1 + lamda)
        Swet_HT = (
            2
            * S_exp_plf_HT
            * (1 + 0.25 * tc_HT)
            * (1 + tau_HT * lamda_HT)
            / (1 + lamda_HT)
        )
        Swet_VT = (
            2
            * S_exp_plf_VT
            * (1 + 0.25 * tc_VT)
            * (1 + tau_VT * lamda_VT)
            / (1 + lamda_VT)
        )
        CD_O_W = Rwf * Rls * Cfw * (1 + Lprime * tc + 100 * tc**4) * Swet_w / S
        CL = W / q / S
        CL_HT = CLa_HT_deg * (aoa_deg + inc_agl_HT_deg)
        CLw = CL - CL_HT * S_HT / S
        Rl_LER = rho_a_cruise * v_cruise * l_LER / mu_cruise
        """control = Rl_LER*1/math.tan(theta_LE)*math.sqrt(1 - M_cruise**2*(math.cos(theta_LE))**2)
        if control > 1.3E+5:
            x = A*lamda/math.cos(math.cos(theta_LE))
        else:
            print('Enter value of R: Roskam VI page29 Figure4.7')"""
        e = 1.1 * (CLa_W / A) / (R * (CLa_W / A) + (1 - R) * math.pi)
        CD_L_W = CLw**2 / math.pi / A / e

        CD_wing = CD_O_W + CD_L_W
        CD_O_HT = (
            Rwf
            * Rls_HT
            * Cfw_HT
            * (1 + Lprime_HT * tc_HT + 100 * tc_HT**4)
            * Swet_HT
            / S
        )
        CD_O_VT = (
            Rwf
            * Rls_VT
            * Cfw_VT
            * (1 + Lprime_VT * tc_VT + 100 * tc_VT**4)
            * Swet_VT
            / S
        )

        return CD_wing, CD_O_W, CD_O_HT, CD_O_VT

    def fuscd(self, Croot, Wf_mm, hf_mm, FL, S, Re_fus, Cf_fus, Swet_fus):
        import math

        """
        Sfus = d_b ** 2
        Sb_fus = d_f_x_mm*d_f_y_mm/1000/1000
        alpha = (W/q/S - CLo)/CLa

        CD_o_fus_base = Cf_fus * (1 + 60 / (FL_short / d_f) ** 3 + 0.0025 * (FL_short / d_f)) * Swet_fus / S
        CD_b_fus = 0.029 * (d_b / d_f) ** 3 / (CD_o_fus_base * S / fus) ** 0.5 * Sfus / S
        CD_o_fus = Rwf * Cf_fus * (1 + 60 / (FL_short / d_f) ** 3 + 0.0025 * (FL_short / d_f)) * Swet_fus / S + CD_b_fus
        #CD_L_fus = 2*alpha*Sb_fus/S +
        """
        Rwf = Drag.wingfusintf(self, Re_fus)
        # Swet_fus = Croot * Wf_mm/1000
        # Sfus = (Wf_mm/1000) ** 2
        # d_f = 2*math.pi*(math.sqrt(abs(((Wf_mm/1000)**2 - (hf_mm/1000)**2)/16)))
        d_f = (Wf_mm + hf_mm) / 1000 / 2
        print(f"d_f: {d_f:,.2f} m")
        d_b = d_f * 0.1
        Sfus = math.pi / 4 * (d_f**2)
        CD_o_fus_base = (
            Cf_fus * (1 + 60 / (FL / d_f) ** 3 + 0.0025 * (FL / d_f)) * Swet_fus / S
        )
        CD_b_fus = (0.029 * (d_b / d_f) ** 3 / (CD_o_fus_base * (S / Sfus)) ** 0.5) * (
            Sfus / S
        )
        CD_O_fus = (
            Rwf * Cf_fus * (1 + 60 / (FL / d_f) ** 3 + 0.0025 * FL / d_f) * Swet_fus / S
        ) + CD_b_fus

        return CD_O_fus

    def dragpolar(
        self,
        S,
        CD_O_W,
        CD_O_HT,
        CD_O_VT,
        fuselage_length,
        Swet_fus,
        Wf_mm,
        Cf_fus,
        A,
        e,
        W,
        g,
        rho_a_cruise,
        v_stall_TO,
        v_cruise,
        v_cruise2,
        v_climb,
        CD_O_fus,
    ):
        import math
        import numpy as np

        Kc = 1.20  # Drag Factor for general aviation
        FF = (
            1
            + 3 / 2 * ((Wf_mm / 1000) / fuselage_length) ** (3 / 2)
            + 50 * ((Wf_mm / 1000) / fuselage_length) ** 3
        )  # Fuselage form factor
        # CD_fus = Cf_fus*FF*Swet_fus

        CD_O_tot = (
            Kc * (CD_O_W + CD_O_HT + CD_O_VT + CD_O_fus) * 2.5
        )  # *2.5 is multiplied to increase CD_o
        n = 1
        n_climb = 1.5
        # vel = np.arange(v_stall_TO - 20, v_cruise2 + 8, 0.01)
        vel = np.arange(v_stall_TO - 10, v_cruise2 + 5, 0.01)
        vel_knot = np.empty(len(vel))
        vel_km1h = np.empty(len(vel))
        for i, vel_i in enumerate(vel):
            vel_knot[i] = Velocity.m1s_kn(self, vel_i, "kn")
            vel_km1h[i] = Velocity.km1h_m1s(self, vel_i, "km1h")
        q = 0.5 * rho_a_cruise * (vel) ** 2
        CL = 2 * W * n * g / (rho_a_cruise * S * vel**2)
        CL_climb = 2 * W * n_climb * g / (rho_a_cruise * S * vel**2)
        CDi_W = (1.05 * CL) ** 2 / np.pi / A / e
        CDi_W_climb = (1.05 * CL_climb) ** 2 / np.pi / A / e
        D_induced = S * CDi_W * q
        D_induced_kgf = D_induced / g
        D_parasitic = S * CD_O_tot * q
        D_parasitic_kgf = D_parasitic / g

        for i, induceddrag_i in enumerate(D_induced):
            if abs(induceddrag_i - D_parasitic[i]) / induceddrag_i < 0.005:
                v_drag_min = vel_knot[i]
                drag_min = induceddrag_i + D_parasitic[i] #* 2
                power_min = drag_min * v_drag_min
                break

        CD_tot = CD_O_tot + CDi_W
        CD_tot_climb = CD_O_tot + CDi_W_climb
        D_tot = S * CD_tot * q
        D_tot_climb = S * CD_tot_climb * q
        D_tot_kgf = D_tot / g
        D_tot_climb_kgf = D_tot_climb / g

        for i, v in enumerate(vel):
            if v >= v_stall_TO:
                v_0 = v
                T_req0_kgf = D_tot_kgf[i]
                T_req0 = D_tot[i]
                break

        for k, v in enumerate(vel):
            if v >= v_cruise:
                v_1 = v
                T_req1_kgf = D_tot_kgf[k]
                T_req1 = D_tot[k]
                break

        for j, v in enumerate(vel):
            if v >= v_cruise2:
                v_2 = v
                T_req2_kgf = D_tot_kgf[j]
                T_req2 = D_tot[j]
                break

        for jj, v in enumerate(vel):
            if v >= v_climb / 3.6:  # here, v_climb is km/h
                v_3 = v_climb / 3.6
                T_req_climb_kgf = D_tot_climb_kgf[jj] * 1.2
                T_req_climb = D_tot_climb[jj] * 1.2
                break

        pwr_req = D_tot * vel
        pwr_req0 = T_req0 * v_0
        pwr_req1 = T_req1 * v_1
        pwr_req2 = T_req2 * v_2
        pwr_req_climb = T_req_climb * v_3

        return (
            vel_knot,
            vel_km1h,
            D_tot,
            D_tot_kgf,
            D_induced,
            D_induced_kgf,
            D_parasitic,
            D_parasitic_kgf,
            T_req0,
            T_req1,
            T_req2,
            T_req0_kgf,
            T_req1_kgf,
            T_req2_kgf,
            pwr_req,
            pwr_req0,
            pwr_req1,
            pwr_req2,
            pwr_req_climb,
            CD_O_tot,
            v_drag_min,
            drag_min,
            power_min,
        )
