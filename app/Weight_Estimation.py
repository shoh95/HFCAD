import math
from Conversions import Conversions


class WeightEst:
    def __init__(self, W, W_fw, S, b, rho_a_cruise, v_cruise, t_r_HT, S_HT, S_VT, t_r_VT, L_HT_act, b_HT, b_VT, FL, Wf_mm, hf_mm, W_press, l_n_mm, Croot, tc_r, n_ult, Swet_fus):
        self.W_imp = Conversions.kg_pound(self, W, "pound")
        self.W_fw_imp = Conversions.kg_pound(self, W_fw, "pound")
        self.S_imp = Conversions.m2_ft2(self, S, "ft2")
        self.b_imp = Conversions.meter_feet(self, b, "ft")
        self.q = 0.5 * rho_a_cruise * v_cruise**2
        self.q_imp = Conversions.pa_psf(self, self.q, "psf")
        self.S_HT_imp = Conversions.m2_ft2(self, S_HT, "ft2")
        self.S_VT_imp = Conversions.m2_ft2(self, S_VT, "ft2")
        self.t_r_HT_imp = Conversions.meter_feet(self, t_r_HT, "ft")
        self.t_r_VT_imp = Conversions.meter_feet(self, t_r_VT, "ft")
        self.t_r_w = tc_r * Croot
        self.t_r_w_imp = Conversions.meter_feet(self, self.t_r_w, "ft")
        self.L_HT_act_imp = Conversions.meter_feet(self, L_HT_act, "ft")
        self.b_HT_imp = Conversions.meter_feet(self, b_HT, "ft")
        self.b_VT_imp = Conversions.meter_feet(self, b_VT, "ft")
        self.FL_imp = Conversions.meter_feet(self, FL, "ft")
        self.Wf_imp = Conversions.meter_feet(self, Wf_mm / 1000, "ft")
        self.hf_imp = Conversions.meter_feet(self, hf_mm / 1000, "ft")
        # self.Swet_fus = (math.pi * Wf_mm / 1000 * FL * (0.5 + 0.135 * (l_n_mm / 1000) / FL) ** (2 / 3) * (1.015 + 0.3 / (FL / (Wf_mm / 1000)) ** 1.5))
        self.Swet_fus_imp = Conversions.m2_ft2(self, Swet_fus, "ft2")
        self.W_press_imp = Conversions.kg_pound(self, W_press, "pound")
        self.n_ult = n_ult



    def cessnamethod(self, A, A_HT, A_VT, theta_c4_VT, Pmax, N_pax):
        W_w_imp = 0.04674 * self.W_imp**0.397 * self.S_imp**0.36 * self.n_ult**0.397 * A**1.712
        W_w = Conversions.kg_pound(self, W_w_imp, "kg")
        W_h_imp = (3.184 * self.W_imp**0.887 * self.S_HT_imp**0.101 * A_HT**0.138 / (174.04 * (self.t_r_HT_imp) ** 0.223))
        W_h = Conversions.kg_pound(self, W_h_imp, "kg")
        W_v_imp = (1.68 * self.W_imp**0.567 * self.S_VT_imp**1.249 * A_VT**0.482 / (639.95 * self.t_r_VT_imp**0.747 * (math.cos(theta_c4_VT)) ** 0.882))
        W_v = Conversions.kg_pound(self, W_v_imp, "kg")
        W_f_imp = 0.04682 * self.W_imp**0.692 * Pmax**0.374 * self.FL_imp**0.59
        W_f = Conversions.kg_pound(self, W_f_imp, "kg")
        W_fc_imp = 0.0168 * self.W_imp
        W_fc = Conversions.kg_pound(self, W_fc_imp, "kg")
        W_els_imp = 0.0268 * self.W_imp
        W_els = Conversions.kg_pound(self, W_els_imp, "kg")
        W_fur_imp = 0.412 * (N_pax) ** 1.145 * (self.W_imp) ** 0.489
        W_fur = Conversions.kg_pound(self, W_fur_imp, "kg")

        return W_w_imp, W_w, W_h_imp, W_v_imp, W_h, W_v, W_f_imp, W_f, W_fc, W_els, W_fur, W_fc_imp, W_els_imp, W_fur_imp

    def usafmethod(self, n_ult, A, theta_c4, lamda, tc, v_kn_cruise):
        W_w_imp = (96.948 * ((self.W_imp * n_ult / 10**5) ** 0.65 * (A / math.cos(theta_c4)) ** 0.57 * (self.S_imp / 100) ** 0.61 * ((1 + lamda) / 2 / tc) ** 0.36 * (1 + v_kn_cruise / 500) ** 0.5) ** 0.993)
        W_w = Conversions.kg_pound(self, W_w_imp, 'kg')
        W_h_imp = (127 * ((self.W_imp * n_ult / 10**5) ** 0.87 * (self.S_HT_imp / 100) ** 1.2 * 0.289 * (self.L_HT_act_imp / 10) ** 0.483 * (self.b_HT_imp / self.t_r_HT_imp) ** 0.5) ** 0.458)
        W_h = Conversions.kg_pound(self, W_h_imp, 'kg')
        W_v_imp = (98.5 * ((self.W_imp * n_ult / 10**5) ** 0.87 * (self.S_VT_imp / 100) ** 1.2 * 0.289 * (self.b_VT_imp / self.t_r_VT_imp) ** 0.5) ** 0.458)
        W_v = Conversions.kg_pound(self, W_v_imp, 'kg')
        W_f_imp = (200* ((self.W_imp * n_ult / 10**5) ** 0.286 * (self.FL_imp / 10) ** 0.857 * ((self.Wf_imp + self.hf_imp) / 10) * (v_kn_cruise / 100) ** 0.338)** 1.1)
        W_f = Conversions.kg_pound(self, W_f_imp, 'kg')
        W_fc_imp = 1.08 * (self.W_imp) ** 0.7
        W_fc = Conversions.kg_pound(self, W_fc_imp, "kg")

        return W_w_imp, W_w, W_h_imp, W_h, W_v_imp, W_v, W_f_imp, W_f, W_fc_imp, W_fc

    def raymermethod(self, A, theta_c4, lamda, tc, theta_c4_HT, lamda_HT, theta_c4_VT, lamda_VT, LD, Kmp, W_l, N_l, L_m, N_mw, N_mss, V_stall, Knp, L_n, N_nw, K_uht, F_w, L_t, K_y, A_ht, S_e, H_t__H_v, K_z, t_c):
        if self.W_fw_imp == 0:
            W_w_imp = (0.036 * self.S_imp**0.758 * (A / (math.cos(theta_c4)) ** 2) ** 0.6 * self.q_imp**0.006 * lamda**0.04 * (100 * tc / math.cos(theta_c4)) ** (-0.3) * (self.n_ult * self.W_imp) ** 0.49)
        else:
            W_w_imp = ( 0.036 * self.S_imp**0.758 * self.W_fw_imp**0.0035 * (A / (math.cos(theta_c4)) ** 2) ** 0.6 * self.q_imp**0.006 * lamda**0.04 * (100 * tc / math.cos(theta_c4)) ** (-0.3) * (self.n_ult * self.W_imp) ** 0.49)
        W_w = Conversions.kg_pound(self, W_w_imp, 'kg')
        # W_h_imp = ( 0.016 * (self.n_ult * self.W_imp) ** 0.414 * self.q_imp**0.168 * self.S_HT_imp**0.896 * (100 * tc / math.cos(theta_c4)) ** (-0.12) * (A / (math.cos(theta_c4_HT)) ** 2) ** 0.043 * lamda_HT ** (-0.02))
        W_h_imp = 0.0379*K_uht*(1+ F_w/self.b_imp)**(-0.25)*self.W_imp**0.639*self.n_ult**(0.1) * self.S_HT_imp**0.75 * L_t**(-1.0)*K_y**0.704 * (math.cos(theta_c4_HT))**(-1.0)*A_ht**0.166 * (1+ S_e/self.S_HT_imp)**0.1
        W_h = Conversions.kg_pound(self, W_h_imp, 'kg')
        # W_v_imp = (0.073 * (self.n_ult * self.W_imp) ** 0.376 * self.q_imp**0.122 * self.S_VT_imp**0.873 * (100 * tc / math.cos(theta_c4_VT)) ** (-0.49) * (A / (math.cos(theta_c4_VT)) ** 2) ** 0.357 * lamda_VT**0.039)
        W_v_imp = 0.0026*(1+H_t__H_v)**0.225*self.W_imp**0.556*self.n_ult**0.536*L_t**(-0.5)*self.S_VT_imp**0.5*K_z**0.875*(math.cos(theta_c4_VT))**(-1) * t_c**(-0.5)
        W_v = Conversions.kg_pound(self, W_v_imp, 'kg')
        W_f_imp = (0.052 * self.Swet_fus_imp**1.086 * (self.n_ult * self.W_imp) ** 0.177 * self.L_HT_act_imp ** (-0.051) * LD ** (-0.072) * self.q_imp**0.241 + self.W_press_imp)
        W_f = Conversions.kg_pound(self, W_f_imp, 'kg')
        W_lndgearmain_imp = 0.0106*Kmp * W_l**0.888 * N_l**0.25 * L_m**0.4 * N_mw**0.321 * N_mss**(-0.5) * V_stall**0.1
        W_lndgearmain = Conversions.kg_pound(self, W_lndgearmain_imp, 'kg')
        W_lndgearnose_imp = 0.032*Knp * W_l**0.646 * N_l**0.2 * L_n**0.5 * N_nw**0.45
        W_lndgearnose = Conversions.kg_pound(self, W_lndgearnose_imp, 'kg')
        
        return W_w_imp, W_w, W_h_imp, W_h, W_v_imp, W_v, W_f_imp, W_f, W_lndgearmain_imp, W_lndgearmain, W_lndgearnose_imp, W_lndgearnose

    def torenbeekmethod(self, theta_c2, N_pax, N_row):
        W_w_imp = (0.00125 * self.W_imp * (self.b_imp / math.cos(theta_c2)) ** 0.75 * (1 + 6.3 * (math.cos(theta_c2) / self.b_imp) ** 0.5) * self.n_ult**0.55 * (self.b_imp * self.S_imp / self.t_r_w_imp / self.W_imp / math.cos(theta_c2)) ** 0.3)
        W_w = Conversions.kg_pound(self, W_w_imp, "kg")
        W_iae_imp = 33 * N_pax
        W_iae = Conversions.kg_pound(self, W_iae_imp, "kg")
        W_fur_imp = 5 + 13 * N_pax + 25 * N_row
        W_fur = Conversions.kg_pound(self, W_fur_imp, "kg")

        return W_w_imp, W_w, W_iae_imp, W_iae, W_fur

    def vtolweightest(self, lamda, A, theta_LE, tc):
        W_w_vtol_imp = (0.032 * self.S_imp**0.76 * lamda**0.04 * (1.5 * self.W_imp) ** 0.49 * (A / (math.cos(theta_LE) ** 2)) ** 0.6 * (100 * tc / math.cos(theta_LE)) ** (-0.3))
        W_w_vtol = Conversions.kg_pound(self, W_w_vtol_imp, "kg")
        W_landingskid_imp = 0.44 * 0.8 * self.W_imp**0.63
        W_landingskid = Conversions.kg_pound(self, W_landingskid_imp, "kg")

        return W_w_vtol_imp, W_w_vtol, W_landingskid_imp, W_landingskid
    

    def miscweightest(self, pow_max, N_f, N_m, S_cs, Iyaw, R_kva, L_a, N_gen, N_e, W_TO, N_c, W_c, S_f, N_pil):
        W_motor = (0.288*pow_max - 1.49) * N_e      #already metric unit!
        W_flight_control_imp = 145.9*N_f**0.554 * (1 + N_m/N_f)**(-1.0) * S_cs**(0.2)*(Iyaw * 10**(-6))**(0.07)
        W_flight_control = Conversions.kg_pound(self, W_flight_control_imp, 'kg')
        W_els_imp = 7.291 * R_kva**0.782 * L_a**0.346 * N_gen**0.1
        W_els = Conversions.kg_pound(self, W_els_imp, 'kg')
        W_iae_imp = N_pil * (15 + 0.032*self.W_imp/1000) + 0.15*(self.W_imp/1000) + 0.012*self.W_imp
        W_iae = Conversions.kg_pound(self, W_iae_imp, 'kg')
        W_fur_imp = 0.0577*N_c**0.1 * W_c**0.393 * S_f**0.75
        W_fur = Conversions.kg_pound(self, W_fur_imp, 'kg')
        W_hydraulics_imp = 0.012 * self.W_imp
        W_hydraulics = Conversions.kg_pound(self, W_hydraulics_imp, 'kg')


        return W_motor, W_flight_control, W_els, W_iae, W_fur, W_hydraulics

    def meanweights(self, W_w_cessna, W_w_usaf, W_w_raymer, W_w_torenbeek, W_h_cessna, W_h_usaf, W_v_cessna, W_v_usaf, W_f_cessna, W_f_usaf, W_h_raymer, W_v_raymer, W_f_raymer, W_fc_cessna, W_fc_usaf, W_fur_cessna, W_fur_torenbeek):
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

        return W_w_mean, W_w_mean_short, W_h_mean, W_h_mean_short, W_v_mean, W_v_mean_short, W_f_mean, W_f_mean_short, W_fc_mean, W_fur_mean

    def totalweight(self, W_crew, W_w_mean, W_h_mean, W_v_mean, W_f_mean, W_avionic, W_payload, W_pwr, W_fc_mean, 
                    W_fur_mean, W_landingskid, W_misc):
        W_tot_mean = ( W_w_mean + W_h_mean + W_v_mean + W_f_mean + W_pwr + W_fc_mean + W_fur_mean + W_avionic + W_landingskid + W_misc)
        W_tot_eVTOL_mean = W_tot_mean - W_fc_mean
        W_total_mean = W_tot_mean + W_crew + W_payload
        W_total_eVTOL_mean = W_tot_eVTOL_mean + W_crew + W_payload
        return W_tot_mean, W_tot_eVTOL_mean, W_total_mean, W_total_eVTOL_mean
