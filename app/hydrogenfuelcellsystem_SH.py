import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl

from Weight_Estimation import WeightEst
from Conversions import Conversions

import sys
from ambiance import Atmosphere
from models.stack_functions import cell_model, stack_model, mass_flow_stack
from models.compressor_performance import compressor_performance_model
from models.compressor_mass import compressor_mass_model
from models.humidifier import humidifier_model
from models.heat_exchanger import heat_exchanger_model




class FuelCellSizing:
    def __init__(self):
        # self.mtom_a = 8600
        # self.mtom_b = 0
        self.MTOM = 8500
        self.PFC1 = 200000 # unit(W)
        self.ptotal_a = 0
        self.ptotal_b = 0
        self.ptotal_c = 0
        self.ptotal_d = 0
        self.mfuel = 0
        self.OEM = 0
        self.W_fus = 0
        self.W_wing = 0
        self.mtank = 0
        self.mem = 0
        self.Pcomp = 0
        self.Pcooling_system = 0
        self.mPMAD = 0
        self.mFC = 0
        self.mcomp = 0
        self.mcoolingsystem = 0
        self.Pheat_rejected = 0
        # self.f_dT = 0
        self.Mfftotal = 0 # TEST
        self.eta_pt = 0.5 # TEST
        self.mpt = 0
        self.Mffextra = 0

        #Powertrain parameter
        self.rhofc = 3000 #2000 W/kg
        self.rhopmad = 15000 #W/kg
        self.rhoem = 7000 ####### W/kg
        self.eta_FC = 0.5 
        self.eta_em = 0.9215 # Mission profile Powertrain -> Motor, Inverter efficiency
        self.eta_storage = 0.4 ############
        self.eta_vol = 0.5 #0.5
        self.rho_H2 = 70
        self.LHVfuel = 120 #120e6 MJ/kg

        #Pelectric calculation
        self.P_W_climb = 14.89 #15.97  # Power to Weight case1(climb),2(cruise),3(descent)
        self.P_W_cruise = 10.03 #10.53
        self.P_W_takeoff = 11.21 #8.6

        #Pcomp
        self.gamma_air = 1.4  # specific heat ratio
        self.Psl = 101325 # P(N/m^2) 0 m
        self.Palt = 70121 # 3000 m
        self.eta_comp = 0.8
        self.lamda_O2 = 1.5  # stoichiometric ratio
        self.Cp_air = 1005 # temperature(0'C), density, specific heat

        #Pcooling
        self.Tt1 = 268.67 # 3000 m
        self.Tair = 268.67 # 3000 m
        self.dT = 60
        self.f_dT = 0.0038 * ((self.Tair/self.dT)**2) + 0.0352 * (self.Tair/self.dT) + 0.1817

        # Calculate temperature rise for compressor
        self.PRcomp = (self.Psl/self.Palt) * 1.05 # compressor ratio
        self.Tt2 = self.Tt1 * (1 + (1 / self.eta_comp) * ((self.PRcomp)**((self.gamma_air-1)/self.gamma_air)- 1))

        #Fuel cell mass
        self.rhocomp = 2000

        #Hydrogen fuel mass
        self.Rtotal = 600000
        self.hcruise = 3000
        self.hTO = 0
        self.Vv = 7
        self.Vclimb = 100
        self.eta_prop = 0.8
        self.L_D1 = 14.5 # cruise Lift_to_Drag case1(climb),2(cruise),3(descent)
        self.L_D2 = 0
        self.g = 9.81
        self.Vtankex = 0.


        #Hydrogen tank geometry
        self.Coversize = 4.5

        #Wing sizing(kg -> lb) *2.205
        # self.W_S_imp = 2830.24 # FPS unit>>>>>>>>>>>>>>>># SH:~$ ??? psf ? N/m2 ???
        self.W_S_imp = Conversions.pa_psf(self, 2830.24, "psf")               ## SH:~$ Wing loading CHANGED(2830.24 -> 2755) to match origianl A/C Specification
        self.AR_wing = 10 
        self.t_c = 0.2
        self.nz = 3*1.5  ##3.8                           ################ SH:~$ 1.5 is TOO LOW -> n_ult = SF*(loadfactor) = 1.25*3
        self.lc4 = 0
        self.lam = 0.42
        self.Vcruise = 110 # m/s
        self.rho_cruise = 0.9093
        self.q_cruise = 0.5 * self.rho_cruise * self.Vcruise**2
        self.q_cruise_imp = self.q_cruise * (0.020885) #SH: slug/fts

        #Fuselage sizing
        #self.Lnose = 1.85*1.5*3.281 #ft
        #self.Lcabin = 7*3.281 #ft
        #self.Ltail = 1.85*2*3.281 #ft
        #self.Dfus = 1.85*3.281 #ft
        self.finerat_fus_nose = 0.6
        self.finerat_fus_tail = 0.8
        self.k_w_nose = -0.603291 * self.finerat_fus_nose**2 + 2.17154 * self.finerat_fus_nose + -0.425122
        self.k_w_tail = -0.603291 * self.finerat_fus_tail**2 + 2.17154 * self.finerat_fus_tail + -0.425122
        self.k_c_nose = -1.72626 * self.finerat_fus_nose**3 + 4.43622 * self.finerat_fus_nose**2 + -3.05539 * self.finerat_fus_nose + 1.3414
        self.k_c_nose = -1.72626 * self.finerat_fus_tail**3 + 4.43622 * self.finerat_fus_tail**2 + -3.05539 * self.finerat_fus_tail + 1.3414
        self.Dfus_imp = Conversions.meter_feet(self, 2.4, 'ft')         # 1.85 * 3.281 #ft
        self.hfus_imp = self.Dfus_imp           # ft, Fuselage max Height
        self.circum_fus_imp = 2*self.k_c_nose * (self.Dfus_imp + self.hfus_imp)
        self.Fnose = 1 # Fineness ratio for nose
        self.Ftail = 2 # Fineness ratio for tail
        self.Lseat = 0.8 # Sea          t pitch [m]
        self.Npass = 18 # Number of passengers
        self.Nseat_abreast = 2 # Number of seats abreast
        self.Ldoor = 1 # Length of door [m]
        self.Lnose_imp = self.Dfus_imp * self.Fnose #ft
        self.Ltail_imp = self.Dfus_imp * self.Ftail #ft
        self.Lcabin_imp = ((self.Npass / self.Nseat_abreast) * self.Lseat + self.Ldoor) * 3.281 #ft


        self.lHT_imp = 27
        self.dFS_imp = 6.07
        self.lFS_imp = 43.46

        self.S_wing = 0
        self.b_wing = 0


        #Aircraft sizing
        self.OEMmisc = 0 #2086.39 #2574.09
        self.OEMmisc_ = 0 #2000 # 1000 # 100

        self.mpayload = 2200
        self.rhobatt = 0.3 #[kWh/kg]

        #hybrid parameter
        self.psi_takeoff = 0.25 #0.176 #0.251
        self.psi_climb = 0.25 #0.176 #0.251
        self.psi_cruise = -0.053 #-0.065
        self.eta_fuelcell = 0.6 #연료전지 효율은 가정
        self.eta_Converter = 0.97
        self.eta_PDU = 0.98
        self.eta_Inverter = 0.97
        self.eta_Motor = 0.95

        self.Pbat_climb = 0.    # SH:~$ place holder
        self.mair1 = 2.856e-7 * self.lamda_O2 * ((self.ptotal_a - self.Pbat_climb)/ self.eta_FC)        # SH:~$ place holder, no difference if it's zero
        self.Pcomp1 = self.mair1 * self.Cp_air * (self.Tt2-self.Tt1) / self.eta_em  # SH:~$ place holder, no difference if it's zero

        
        self.Pheat_rejected1 = ((1 / self.eta_FC) - 1) * (self.ptotal_a - self.Pbat_climb) / 1000       # SH:~$ place holder, no difference if it's zero
        self.Pcooling_system1_fc = (0.371 * self.Pheat_rejected1 + 1.33) * self.f_dT * 1000             # SH:~$ place holder, no difference if it's zero

        self.Pfuelcell_climb = 0.   # SH:~$ place holder
        self.Pfuelcell_cruise = 0.  # SH:~$ place holder
        self.Pcomp2 = 0.    # SH:~$ place holder
        self.Pcooling_system2_fc = 0.   # SH:~$ place holder

        self.Pbat_cruise_charger = 0    # SH:~$ place holder
        self.Pfuelcell_cruise_charger = 0   # SH:~$ place holder

        self.Pcomp3 = 0.    # SH:~$ place holder
        self.mbatt = 0  # SH:~$ place holder
        self.Ptotal_climb = 0.  # SH:~$ place holder

        self.lndgear = 0
        self.W_HT = 0
        self.W_VT = 0
        self.W_motor = 0
        self.W_flight_control = 0
        self.W_els = 0
        self.W_iae = 0
        self.W_fur = 0
        self.W_hydraulics = 0

    def cal_Ptotal_climb(self, PFC1):
        self.ptotal_a = PFC1
        iter = 1

        while True:
            # Calculate shaft power for flight based on MTOM and power-to-weight ratio
            Pshaft1 = self.MTOM * 9.81 * self.P_W_climb
            
            
            # Calculate net electrical output power of fuel cell
            self.Pfuelcell_climb = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*((1-self.psi_climb)/(self.psi_climb+self.eta_Converter*(1-self.psi_climb))) * Pshaft1
            self.Pbat_climb = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*(self.psi_climb/(self.psi_climb+self.eta_Converter*(1-self.psi_climb))) * Pshaft1
            
            Pelectricnet1 = self.Pfuelcell_climb + self.Pbat_climb
            
            # Calculate air mass flow for compressor
            self.mair1 = 2.856e-7 * self.lamda_O2 * ((self.ptotal_a - self.Pbat_climb)/ self.eta_FC)
            # Calculate fuel cell power system power including compressor
            self.Pcomp1 = self.mair1 * self.Cp_air * (self.Tt2-self.Tt1) / self.eta_em  # Placeholder value for compressor power

            # Calculate fuel cell power system power including cooling
            self.Pheat_rejected1 = ((1 / self.eta_FC) - 1) * (self.ptotal_a - self.Pbat_climb) / 1000
            self.Pcooling_system1_fc = (0.371 * self.Pheat_rejected1 + 1.33) * self.f_dT * 1000
            
            self.ptotal_b = Pelectricnet1 + self.Pcomp1 + self.Pcooling_system1_fc
            # print('Pshaft1: %.2f Pcomp1_fc: %.2f Pcooling_system1_fc: %.2f Pfuelcell_climb: %.2f Pbat_climb: %.2f Ptotal_climb: %.2f' %(Pshaft1, self.Pcomp1, self.Pcooling_system1_fc, self.Pfuelcell_climb, self.Pbat_climb, self.ptotal_b))
            # print(f"Pshaft1: {Pshaft1:,.2f}, Pcomp1_fc: {self.Pcomp1:,.2f}, Pcooling_system1_fc: {self.Pcooling_system1_fc:,.2f}, Pfuelcell_climb: {self.Pfuelcell_climb:,.2f}, Pbat_climb: {self.Pbat_climb:,.2f}, Ptotal_climb: {self.ptotal_b:,.2f}")
            if abs(self.ptotal_b - self.ptotal_a) <= 100:
                print(f"------------------------------------------------------")
                print(f"----------------------- iter {iter} -----------------------")
                print(f"Pshaft1: {Pshaft1/1000:,.0f} kW, Pcomp1_fc: {self.Pcomp1/1000:,.0f} kW, Pcooling_system1_fc: {self.Pcooling_system1_fc/1000:,.0f} kW, Pfuelcell_climb: {self.Pfuelcell_climb/1000:,.0f} kW, Pbat_climb: {self.Pbat_climb/1000:,.0f} kW, Ptotal_climb: {self.ptotal_b/1000:,.0f} kW")
                break
            else: 
                self.ptotal_a = self.ptotal_b
                iter += 1
                
        return Pshaft1

    def cal_Ptotal_cruise(self, PFC2):
        self.ptotal_c = PFC2 
        while True:
            #Calculate shaft power for flight based on MTOM and power-to-weight ratio
            Pshaft2 = self.MTOM * 9.81 * self.P_W_cruise 
            
            # Calculate net electrical output power of fuel cell
            self.Pfuelcell_cruise = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*((1/self.eta_Converter)) * Pshaft2
            
            Pelectricnet2 = self.Pfuelcell_cruise
            
            # Calculate air mass flow for compressor
            mair2 = 2.856e-7 * self.lamda_O2 * (self.ptotal_c/ self.eta_FC)
            # Calculate fuel cell power system power including compressor
            self.Pcomp2 = mair2 * self.Cp_air * (self.Tt2-self.Tt1) / self.eta_em  # Placeholder value for compressor power

            # Calculate fuel cell power system power including cooling
            Pheat_rejected2 = ((1 / self.eta_FC) - 1) * self.ptotal_c/ 1000
            self.Pcooling_system2_fc = (0.371 * Pheat_rejected2 + 1.33) * self.f_dT * 1000
            
            self.ptotal_d = Pelectricnet2 + self.Pcomp2 + self.Pcooling_system2_fc


            if abs(self.ptotal_d - self.ptotal_c) <= 100:
                break
            else: 
                self.ptotal_c = self.ptotal_d
            

        return self.ptotal_d


    def cal_Ptotal_cruise_charger(self, PFC3):
        ptotal_e = PFC3
        ptotal_f = 0.
        while True:
            #Calculate shaft power for flight based on MTOM and power-to-weight ratio
            Pshaft3 = self.MTOM * 9.81 * self.P_W_cruise 
            
            # Calculate net electrical output power of fuel cell
            self.Pfuelcell_cruise_charger = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*((1-self.psi_cruise)/(self.psi_cruise+self.eta_Converter*(1-self.psi_cruise))) * Pshaft3
            self.Pbat_cruise_charger = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*(self.psi_cruise/(self.psi_cruise+self.eta_Converter*(1-self.psi_cruise))) * Pshaft3
            
            Pelectricnet3 = self.Pfuelcell_cruise_charger + self.Pbat_cruise_charger
            
            # Calculate air mass flow for compressor
            mair3 = 2.856e-7 * self.lamda_O2 * ((ptotal_e - self.Pbat_cruise_charger) / self.eta_FC)
            # Calculate fuel cell power system power including compressor
            self.Pcomp3 = mair3 * self.Cp_air * (self.Tt2-self.Tt1) / self.eta_em  # Placeholder value for compressor power

            # Calculate fuel cell power system power including cooling
            Pheat_rejected3 = ((1 / self.eta_FC) - 1) * (ptotal_e - self.Pbat_cruise_charger) / 1000
            Pcooling_system3_fc = (0.371 * Pheat_rejected3 + 1.33) * self.f_dT * 1000
            
            ptotal_f = Pelectricnet3 + self.Pcomp3 + Pcooling_system3_fc


            if abs(ptotal_f - ptotal_e) <= 100:
                break
            else: 
                ptotal_e = ptotal_f
                
        return ptotal_f, Pcooling_system3_fc


    def cal_Ptotal_takeoff(self, PFC4):
        ptotal_g = PFC4
        ptotal_h = 0.
        while True:
            #Calculate shaft power for flight based on MTOM and power-to-weight ratio
            Pshaft4 = self.MTOM * 9.81 * self.P_W_takeoff 
            
            # Calculate net electrical output power of fuel cell
            #To account for the difference in net and gross the fuel cell
            Pfuelcell_takeoff = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*((1-self.psi_takeoff)/(self.psi_takeoff+self.eta_Converter*(1-self.psi_takeoff))) * Pshaft4
            Pbat_takeoff = (1/(self.eta_Inverter*self.eta_Motor*self.eta_PDU*self.eta_prop))*(self.psi_takeoff/(self.psi_takeoff+self.eta_Converter*(1-self.psi_takeoff))) * Pshaft4
            
            Pelectricnet4 = Pfuelcell_takeoff + Pbat_takeoff
            
            # Calculate air mass flow for compressor
            mair4 = 2.856e-7 * self.lamda_O2 * (ptotal_g - Pbat_takeoff/ self.eta_FC)
            # Calculate fuel cell power system power including compressor
            Pcomp4 = mair4 * self.Cp_air * (self.Tt2-self.Tt1) / self.eta_em  # Placeholder value for compressor power

            # Calculate fuel cell power system power including cooling
            Pheat_rejected4 = ((1 / self.eta_FC) - 1) * (ptotal_g - Pbat_takeoff)/ 1000
            Pcooling_system4_fc = (0.371 * Pheat_rejected4 + 1.33) * self.f_dT * 1000
            
            ptotal_h = Pelectricnet4 + Pcomp4 + Pcooling_system4_fc
            
            if abs(ptotal_h - ptotal_g) <= 100:
                break
            else: 
                ptotal_g = ptotal_h

        return ptotal_h


    def cal_MTOM(self):
        MTOM = self.MTOM
        # Fuel cell powertrain mass

        # #PMAD and electric motor power
        # PPDU = (self.Pfuelcell_climb + self.Pheat_rejected1 + self.Pcooling_system1_fc) * self.eta_Converter + self.Pbat_climb
        
        # Pem = ((self.Pfuelcell_climb + self.Pheat_rejected1 + self.Pcooling_system1_fc) * self.eta_Converter + self.Pbat_climb) * self.eta_PDU * self.eta_Inverter
        

        n_stacks_series = 1
        n_stacks_parallel = 4
        power_fc_sys_parallel = self.Pfuelcell_climb/n_stacks_parallel         # + self.Pheat_rejected1 + self.Pcooling_system1_fc
        volt_req = 1000  # voltage to be produced by a fuel cell system
        h_cr = 6000
        mach_cr = 0.5
        oversizing = 0.15
        beta = 1.05
        comp_bool = 1
        pemfcsys_sizing_results = FuelCellSystem.size_system(power_fc_sys_parallel, volt_req, h_cr, mach_cr, oversizing, beta, comp_bool, n_stacks_series)

        print(f"Power density of Nacelle System: {power_fc_sys_parallel/1000/pemfcsys_sizing_results['m_sys']:,.3f} kW/kg")

        print(f"dim_hx: dX = {pemfcsys_sizing_results['dim_hx'][0]:,.3f} m, dY = {pemfcsys_sizing_results['dim_hx'][1]:,.3f} m, dZ = {pemfcsys_sizing_results['dim_hx'][2]:,.3f} m")
        print(f"dim_stack: dX = {pemfcsys_sizing_results['dim_stack'][2]:,.3f} m, dY = {pemfcsys_sizing_results['dim_stack'][0]:,.3f} m, dZ = {pemfcsys_sizing_results['dim_stack'][1]:,.3f} m")
        
        
        # Calculate powertrain component masses
        # self.mFC = (self.Pfuelcell_climb + self.Pcomp1 + self.Pcooling_system1_fc) / self.rhofc
        self.mFC = n_stacks_parallel * (pemfcsys_sizing_results['m_stacks'] + pemfcsys_sizing_results['m_humid'] + pemfcsys_sizing_results['m_comp'])
        mbatt_old = (self.Pbat_climb * 0.234) / (self.rhobatt * 1000)
        self.mbatt = mbatt_old/(1 - 0.25) # 100은 여분
        # self.mcomp = self.Pcomp1 / self.rhocomp
        self.mcomp = (n_stacks_parallel)*pemfcsys_sizing_results['m_comp']
        
        # self.mcoolingsystem = (0.194 * self.Pheat_rejected1 + 1.39) * self.f_dT
        self.mcoolingsystem = (n_stacks_parallel)*pemfcsys_sizing_results['m_hx']
        # print('mcoolingsystem: %f' %self.mcoolingsystem)
        print(f"mcoolingsystem: {self.mcoolingsystem:,.0f} kg")
        

        self.Pcomp1 = n_stacks_parallel*pemfcsys_sizing_results["power_comp"]
        self.Pcomp2 = n_stacks_parallel*pemfcsys_sizing_results["power_comp"]
        self.Pcomp3 = n_stacks_parallel*pemfcsys_sizing_results["power_comp"]

        #PMAD and electric motor power

        self.Pheat_rejected1 = n_stacks_parallel*pemfcsys_sizing_results['q_all']/1000


        PPDU = (self.Pfuelcell_climb + self.Pheat_rejected1 + self.Pcooling_system1_fc) * self.eta_Converter + self.Pbat_climb
        
        Pem = ((self.Pfuelcell_climb + self.Pheat_rejected1 + self.Pcooling_system1_fc) * self.eta_Converter + self.Pbat_climb) * self.eta_PDU * self.eta_Inverter
        
        self.mem = Pem / self.rhoem
        self.mPMAD = PPDU / self.rhopmad


        # Calculate total powertrain mass
        # self.mpt = (self.mFC + self.mcomp + self.mcoolingsystem + self.mPMAD + self.mem) * 1.1 # 1.2
        self.mpt = (self.mFC + self.mcoolingsystem + self.mPMAD + self.mem)

        # Calculate hydrogen fuel mass

        tclimb = (self.hcruise - self.hTO) / self.Vv
        Rclimb = self.Vclimb * tclimb
        Rdescent = Rclimb  # Assuming descent range equals climb range
        Rcruise = self.Rtotal - Rclimb - Rdescent

        # self.Ptotal_climb = self.Pfuelcell_climb + self.Pbat_climb + self.Pcomp1 + self.Pcooling_system1_fc
        # self.eta_fuelcell = pemfcsys_sizing_results['eta_fcsys']
        self.Ptotal_climb = self.Pfuelcell_climb + self.Pbat_climb + self.Pcomp1 + self.Pcooling_system1_fc
        Pfuelcell_engine = self.Ptotal_climb * 0.05
        Pfuelcell_taxing = self.Ptotal_climb * 0.1
        # self.Mass_fuel_old = (Pfuelcell_engine * 0.083 + Pfuelcell_taxing * 0.75 + (self.Pfuelcell_climb + self.Pcomp1 + self.Pcooling_system1_fc) * 0.234 + (self.Pfuelcell_cruise_charger + self.Pcomp3 + Pcooling_system3_fc) * 0.783 + (self.Pfuelcell_cruise + self.Pcomp2 + self.Pcooling_system2_fc) * 0.817) / (self.LHVfuel * 277.78 * self.eta_fuelcell)
        # # self.Mass_fuel_old = n_stacks_parallel*pemfcsys_sizing_results['mdot_h2']*160*60          # (mdot_H2 [kg/s] X 160*60 [s])
        # self.mfuel = self.Mass_fuel_old/(1 - 0.1)
        self.mfuel = n_stacks_parallel*pemfcsys_sizing_results['mdot_h2']*self.Rtotal/pemfcsys_sizing_results['v_cr']          # (mdot_H2 [kg/s] X 160*60 [s])
        print(f"mfuel: {self.mfuel:,.0f}")
        
        #Hydrogen tank sizing

        # Calculate hydrogen tank volume
        VH2 = self.mfuel * self.Coversize / self.rho_H2
        self.Vtankex = VH2
        Ltank = self.Vtankex / (math.pi * ((self.Dfus_imp / (2 * 3.281))**2))

        self.mtank = (self.mfuel * self.Coversize / self.eta_storage) - (self.mfuel * self.Coversize)

        #Calculate Wing sizing
        # self.S_wing_imp = (MTOM * 9.81 / self.W_S_imp)*(3.281 ** 2)
        MTOM_imp = Conversions.kg_pound(self, MTOM, "pound")
        self.S_wing_imp = MTOM_imp / self.W_S_imp
        self.b_wing_imp = math.sqrt(self.AR_wing * self.S_wing_imp)
        
        # print('b_wing: %.2f m' %(self.b_wing_imp/3.281))  
        print(f"b_wing: {Conversions.meter_feet(self, self.b_wing_imp, 'meter'):,.2f} m")

        #Calculate fuselage sizing
        # print('Ltank : %.2f' %Ltank)
        print(f"Ltank: {Ltank:,.2f} m")
        self.Ltank_imp = Ltank * 3.281 # change unit (m -> ft)
        self.Lfus_imp = self.Lnose_imp + self.Ltail_imp + self.Lcabin_imp + self.Ltank_imp


        ##### SH:~$ Swet_fus_imp IMPROVED by Torenbeek's geometric method of elliptic-function
        self.Swet_fus_imp = self.circum_fus_imp *((self.Lcabin_imp + self.Ltank_imp) + self.k_w_nose*self.Lnose_imp + self.k_w_tail*self.Ltail_imp)

        # print('Lnose: %.2fm Lcabin: %.2fm Ltail: %.2fm' %(self.Lnose_imp/3.281, self.Lcabin_imp/3.281, self.Ltail_imp/3.281))
        # print('Lfus: %.2f m' %(self.Lfus_imp/3.281))
        # self.Lnose = Conversions.meter_feet(self, self.Lnose_imp, "meter")
        # self.Lcabin = Conversions.meter_feet(self, self.Lcabin_imp, "meter")
        # self.Ltail = Conversions.meter_feet(self, self.Ltail_imp, "meter")

        print(f"Lnose: {Conversions.meter_feet(self, self.Lnose_imp, "meter"):,.2f} m, Lcabin: {Conversions.meter_feet(self, self.Lcabin_imp, "meter"):,.2f} m, Ltail: {Conversions.meter_feet(self, self.Ltail_imp, "meter"):,.2f} m")
        print(f"Lfus: {Conversions.meter_feet(self, self.Lfus_imp, "meter"):,.2f} m")
        # self.Sfus_imp = math.pi * self.Dfus_imp * self.Lfus_imp + 2 * math.pi * ((self.Dfus_imp/2)**2)
        # self.W_fus_imp = ( 0.052 * ((self.Sfus_imp)**1.086) * ((self.nz * MTOM_imp * 2.205)**0.177) * (self.lHT_imp**(-0.051)) * (self.L_D1**(-0.072)) * (self.q_cruise_imp**0.241))
        
        
        # self.W_wing_imp = (0.036 * (self.S_wing_imp**0.758) *((self.AR_wing/(math.cos(math.radians(self.lc4)))**2))**0.6) * (self.q_cruise_imp**0.006) * (self.lam**0.04) * ((100 * self.t_c / (math.cos(math.radians(self.lc4))))**(-0.3))*((MTOM_imp * 2.205 * self.nz)**(0.49))
        #Wing sizing(lb -> kg) composite correction factor 0.78
        # self.W_wing = self.W_wing_imp * 0.4536 * 0.85 ##################
        # self.W_wing = Conversions.kg_pound(self, self.W_wing_imp, "kg")* 0.85

        #fuselage sizing(lb -> kg), composite correction factor 0.85
        # self.W_fus = self.W_fus_imp * 0.4536 * 0.7 ###############
        # self.W_fus = Conversions.kg_pound(self, self.W_fus_imp, "kg")* 0.7

        # Swet_fus_imp2 = math.pi * (self.Dfus_imp) * self.Lfus_imp + 2 * math.pi * ((self.Dfus_imp/2)**2)
        # print(f"Swet_fus_imp2: {Swet_fus_imp2} ft2")


        ##### SH:~$ FIXED // Wing mass equation receiving METRIC variable(MTOM).
        ##### SH:~$ WeightEst class __init__ magic method accepts SI unit ONLY! ( AUTO CONVERTS TO FT-LBS UNITS )
        ##### SH:~$ cessnamethod / usafmethod / raymermethod / .../ methods accepts IMPERIAL units(FT-LBS) ONLY!
        WEst = WeightEst(W=MTOM, W_fw=0., S=Conversions.m2_ft2(self, self.S_wing_imp, 'm2'), 
                         b=Conversions.meter_feet(self, self.b_wing_imp, 'meter'), rho_a_cruise=self.rho_cruise, 
                         v_cruise=self.Vcruise, t_r_HT=0.09, S_HT=4.5404, S_VT=4.4004, t_r_VT=0.09*1.361, 
                         L_HT_act=(Conversions.meter_feet(self, self.lHT_imp, "meter")), b_HT=4.765, b_VT=2.907, 
                         FL=(Conversions.meter_feet(self, self.Lfus_imp, "meter")), 
                         Wf_mm=(Conversions.meter_feet(self, self.Dfus_imp, "meter")*1000), 
                         hf_mm=(Conversions.meter_feet(self, self.hfus_imp, "meter")*1000), 
                         W_press=0, l_n_mm=(Conversions.meter_feet(self, self.Lnose_imp, "meter")*1000), 
                         Croot=2.430, tc_r=self.t_c, n_ult=self.nz, 
                         Swet_fus=(Conversions.m2_ft2(self, self.Swet_fus_imp, "m2")))
        [self.W_wing_imp, self.W_wing, self.W_HT_imp, self.W_HT, self.W_VT_imp, self.W_VT, self.W_fus_imp, self.W_fus, 
         self.W_lndgearmain_imp, self.W_lndgearmain, self.W_lndgearnose_imp, self.W_lndgearnose] =\
              WEst.raymermethod(A=self.AR_wing, theta_c4=(math.radians(self.lc4)), lamda=self.lam, tc=self.t_c, 
                                theta_c4_HT=(math.radians(2.671)), lamda_HT=0.4, theta_c4_VT=(math.radians(19.53)), 
                                lamda_VT=0.4, LD=self.L_D1, Kmp=1.0, W_l=Conversions.kg_pound(self, 8000, 'pound'), N_l=3.0*1.5, L_m=Conversions.meter_inch(self, 0.9, "inch"), N_mw=1, N_mss=1, V_stall=Conversions.km1h_kn(self, 58.32, 'kn'), Knp=1.0, L_n=Conversions.meter_inch(self, 0.90, "inch"), N_nw=1, K_uht=1., F_w=Conversions.meter_feet(self, 1.11, 'ft'), L_t=Conversions.meter_feet(self, 10.74, 'ft'), K_y=Conversions.meter_feet(self, 3.22, 'ft'), A_ht=5, S_e=Conversions.m2_ft2(self, 0.73, 'ft2'), H_t__H_v=1., K_z=Conversions.meter_feet(self, 10.7, 'ft'), t_c=0.09)
        [self.W_motor, self.W_flight_control, self.W_els, self.W_iae, self.W_fur, self.W_hydraulics] =\
        WEst.miscweightest(pow_max=380, N_f=6, N_m=2, S_cs=Conversions.m2_ft2(self, 8.36, "ft2"), Iyaw=4.49*10**6, R_kva=50, L_a=Conversions.meter_feet(self, 20, "ft"), N_gen=1, N_e=4, W_TO=Conversions.kg_pound(self, MTOM, "pound"), N_c=2, W_c=Conversions.kg_pound(self, 10*19, "pound"), S_f=self.Swet_fus_imp*23, N_pil=2)

        #Calculate aircraft sizing
        self.OEMmisc = self.OEMmisc_ + self.W_HT + self.W_VT + self.W_lndgearmain + self.W_lndgearnose + self.W_flight_control + self.W_els + self.W_iae  + self.W_hydraulics #+ self.W_fur + self.W_motor
        self.OEM = self.OEMmisc + self.mpt + self.mtank + self.W_wing + self.W_fus + self.mbatt
        MTOM = self.OEM + self.mfuel + self.mpayload# + self.OEMmisc
        # print('MTOM: %f' %MTOM)
        print(f"MTOM: {MTOM:,.0f} kg")

        return MTOM, Pfuelcell_engine, Pfuelcell_taxing, pemfcsys_sizing_results
    
class FuelCellSystem:
    def __init__(self):
        self.sigma_t = 0.03                                         # S/m, Proton conductivity in the catalyst (Data from Table 7)
        self.b_tfslope = 0.05                                       # V, Tafel slope (Data from Table 7)
        self.l_t_ccl = 0.001                                       # cm, CCL thickness
        self.R_omega = 0.126                            # Ohm cm^2, ohmic resistance - -Generally from 0.2 to 0.045
        self.c_h = 7.36*10**(-6)                                    # mol/cm^3, Oxygen concentration in the channel p=1bar
        self.c_ref = 8.58*10**(-6)                                  # mol/cm^3, Reference oxygen concentration in teh channel
        self.F_const = 96458                                        # C/mol, Faraday constant
        self.D_b = 0.015                                           # cm^2/s, Effective diffusion coefficient of GDL
        self.V_oc = 1.145                                           # V, Open circuit voltage
        self.istar = 0.01                                       # A/cm^3, Volumetric exchange current density, from "A Physically–Based Analytical Polarization Curve of a PEM Fuel Cell"
        self.beta = 0.

        self.jstar = 2*10**(-3) #self.sigma_t*self.b_tfslope/self.l_t_ccl                    # A/cm^3, Exchange current, [Variable parameter]
        self.l_b = 0.0312                               # cm, GDL thickness -Generally from 0.015 to 0.04, [Variable parameter]
        self.D = 1.37*10**(-4)                                         # cm^2/s, conductiviy in the CCL, [Variable parameter]

        # self.j_0 = self.b_tfslope / self.l_t_ccl
        self.j_sigma = math.sqrt(2*self.istar*self.sigma_t*self.b_tfslope)
        self.jstar_lim = 4*self.F_const * self.D_b * self.c_h / self.l_b

    
    def singlecell(self):
        self.j_0 = np.linspace(0.06, 1.5, 1000)
        self.beta = np.sqrt( 2* self.j_0/self.jstar ) / ( 1 + np.sqrt(1.12 * self.j_0/self.jstar) * np.exp(2*np.sqrt(2*self.j_0/self.jstar)) ) + (np.pi * self.j_0/self.jstar) / (2 + self.j_0/self.jstar)
        # print(self.beta)
        eta_0 = np.arcsinh( (self.j_0 / self.j_sigma)**2 / ( 2*(self.c_h / self.c_ref)*( 1 - np.exp(-1*self.j_0/(2*self.jstar)) ) ) ) + self.sigma_t * self.b_tfslope**2 / (4*self.F_const*self.D*self.c_h) * ( self.j_0/self.jstar - np.log( 1 + (self.j_0**2/(self.jstar**2 * self.beta**2)) ) ) * ( 1- self.j_0/(self.jstar_lim * self.c_h/self.c_ref) )**(-1) - self.b_tfslope * np.log( 1- self.j_0 / (self.jstar_lim * self.c_h/self.c_ref) )
        Vcell = self.V_oc - self.R_omega * self.j_0 - eta_0
        # Voc = min(Vcell)
        # plt.plot(self.j_0, Vcell)
        # plt.show()

    
    def size_system(power_fc_sys, volt_req, h_cr, mach_cr, oversizing, beta, comp_bool, n_stacks_series):


        """
        Size a fuel cell system.

        :param n_stacks_series: Number of stacks in series
        :param power_fc_sys: Effective power to be delivered by the fuel cell system (in W)
        :param volt_req: Voltage to be delivered by the fuel cell system (in V)
        :param h_cr: Cruise altitude in m
        :param mach_cr: Cruise Mach number
        :param oversizing: Oversizing factor, 1-oversizing_factor = j_operating_point / j_max_power
        :param beta: Compression factor for cathode, i.e. a factor of 2 means that the cathode inlet pressure is twice the
        ambient pressure at the given altitude
        :param comp_bool: Boolean for whether to include a compressor
        :return: results: List of sizing results
        """

        figs = []

        # Atmospheric conditions
        atm_cr = Atmosphere(h_cr)
        c_cr = atm_cr.speed_of_sound[0]  # speed of sound at cruise in m
        v_cr = mach_cr * c_cr  # cruise true airspeed in m/s
        p_cr = atm_cr.pressure[0]  # static pressure at cruise altitude in Pa
        p_cr_tot = p_cr * (1 + 0.4 / 2 * mach_cr ** 2) ** (1.4 / 0.4)  # total pressure at cruise in Pa
        t_cr = atm_cr.temperature[0]  # static temperature at cruise altitude in K
        t_cr_tot = t_cr * (1 + 0.4 / 2 * mach_cr ** 2)  # total temperature at cruise in K
        rho_cr = atm_cr.density[0]  # air density at cruise altitude in kg/m3
        mu_cr = atm_cr.dynamic_viscosity[0]  # dynamic viscosity at cruise altitude in Pa s

        print(f"Reynolds_number: {rho_cr*v_cr*1.8/mu_cr:,.0f}")

        # Other inputs
        cell_temp = 273.15 + 80  # operating temperature inside cell
        mu_f = 0.95  # fuel utilisation

        # Compressor outlet conditions
        if comp_bool:
            pres_cathode_in = beta * p_cr_tot  # assuming the flow slows down completely before the compressor, see
            #                                  compressor_performance.py
        else:
            pres_cathode_in = p_cr_tot

        # Cell model
        pres_h = Atmosphere(0).pressure[0]  # assume that the anode inlet pressure is equal to sea level air pressure
        volt_cell, power_dens_cell, eta_cell, fig = cell_model(pres_cathode_in, pres_h, cell_temp, oversizing)
        figs.append(fig)
        # print(f"volt_cell: {volt_cell:,.2f} V")

        # Compressor models
        if comp_bool:
            # Iterate until the fuel cell stacks produce enough power for propulsion AND to run their compressor
            power_req = 0  # initiated as 0 to get iteration going
            power_req_new = power_fc_sys  # initially, the stacks only need to produce the propulsive power
            while abs(power_req_new - power_req) > 1e-3:  # while not converged within tolerance
                power_req = power_req_new  # this is the power produced by stacks
                geom_comp, power_comp, rho_humid_in, m_dot_comp = compressor_performance_model(power_req, volt_cell, beta,
                                                                                            p_cr_tot, t_cr_tot, mu_cr)
                power_req_new = power_fc_sys + power_comp  # (new) compressor power has been determined, add this to
                # propulsive power
            m_comp = compressor_mass_model(geom_comp, power_comp)  # determine compressor mass
        else:
            # no compressor
            m_comp = 0
            power_comp = 0
            power_req_new = power_fc_sys
            m_dot_comp = mass_flow_stack(power_req_new, volt_cell)  # mass flow of air for cathode in kg/s
            rho_humid_in = rho_cr  # humidifier inlet air density in kg/m3

        # Remaining BOP models
        m_humid = humidifier_model(m_dot_comp, rho_humid_in)  # mass of humidifier
        q_all, m_hx, dim_hx = heat_exchanger_model(power_req_new, volt_cell, cell_temp, mu_f, v_cr, mach_cr, p_cr_tot, t_cr_tot, rho_cr,
                                    mu_cr)

        # Stack model
        m_stacks, dim_stack, res_stack = stack_model(n_stacks_series, volt_req, volt_cell, power_req_new, power_dens_cell)  # mass of stack(s)

        # Sum up to find mass of FC system (all masses in kg)
        m_sys = m_stacks + m_comp + m_humid + m_hx
        # print("Stack(s): {} kg, Compressor: {} kg, Humidifier: {} kg, Heat Exchanger: {} kg"
        #     .format(m_stacks, m_comp, m_humid, m_hx))
        # print(mdot_h2)
        print(f"Stack(s): {m_stacks:,.0f} kg, Compressor: {m_comp:,.0f} kg, Humidifier: {m_humid:,.0f} kg, HX: {m_hx:,.0f} kg")
        # print("Power density of system in kW/kg: ", round(power_fc_sys/1000/m_sys, 3))
        print(f"Stack prop output power: {power_fc_sys/1000:,.0f} kW, Pcomp: {power_comp/1000:,.1f} kW")
        # print(f"Power density of system in kW/kg: {power_fc_sys/1000/m_sys:,.3f}")
        # print("Stack prop output power: {} kW, Comp power: {} kW".format(power_fc_sys, power_comp))

        # Determine FC system efficiency
        eta_fcsys = eta_cell * power_fc_sys / (power_comp + power_fc_sys) * mu_f
        # print("Cell efficiency: {}, Output efficiency: {}".format(eta_cell, eta_fcsys))
        print(f"Cell efficiency: {eta_cell:,.3f}, Output efficiency: {eta_fcsys:,.3f}")

        # Hydrogen comsumption
        mdot_h2 = 1.05e-8 * (power_comp + power_fc_sys) / volt_cell

        print(f"mdot_h2: {mdot_h2*1000:,.2f} g/s")

        # Make list of values to return
        results = {
            'm_sys': m_sys, 
            'm_stacks': m_stacks, 
            'm_comp': m_comp, 
            'm_humid': m_humid, 
            'm_hx': m_hx, 
            'eta_fcsys': eta_fcsys, 
            'mdot_h2': mdot_h2, 
            'power_comp': power_comp, 
            'figs': figs, 
            'dim_stack': dim_stack, 
            'n_stacks_series': n_stacks_series, 
            'dim_hx': dim_hx, 
            'res_stack': res_stack,
            'q_all': q_all,
            'v_cr': v_cr
            }

        return results



    def othersys(self):
        pass


    def fcstack(self):
        pass






if __name__ == "__main__":
    PFC1 = 200000
    mtom_a = 0
    FCS = FuelCellSizing()
    FCsys = FuelCellSystem()
    FCsys.singlecell()

    iter2 = 1
    while True:
        print(f"======================================================")
        print(f"======================= ITER {iter2} =======================")
        ptotal_cruise = FCS.cal_Ptotal_cruise(PFC1)
        [ptotal_cruise_charger, Pcooling_system3_fc] = FCS.cal_Ptotal_cruise_charger(PFC1) 
        ptotal_takeoff = FCS.cal_Ptotal_takeoff(PFC1) 
        Pshaft1 = FCS.cal_Ptotal_climb(PFC1)
        [FCS.MTOM, Pfuelcell_engine, Pfuelcell_taxing, pemfcsys_sizing_results] = FCS.cal_MTOM()
        # print("ptotal_climb: %f mtom_a: %f" %(FCS.Ptotal_climb, MTOM))
        print(f"ptotal_climb: {FCS.Ptotal_climb/1000:,.0f} kW, mtom_a: {FCS.MTOM:,.0f} kg")
        print(f"------------------------------------------------------")
        pnet = Pshaft1 * FCS.eta_PDU * FCS.eta_em

        if abs(mtom_a-FCS.MTOM) <= 1:
            print(f"Ptotal_climb: {FCS.Ptotal_climb/1000:,.0f} kW")
            print(f"Ptotal_cruise: {ptotal_cruise/1000:,.0f} kW")
            print(f"Ptotal_takeoff: {ptotal_takeoff/1000:,.0f} kW")
            print(f"Pelectricnet: {Pshaft1/1000:,.0f} kW")
            print(f"Pcomp: {FCS.Pcomp1/1000:,.0f} kW")
            print(f"Pcoolingsystem: {FCS.Pcooling_system1_fc/1000:,.0f} kW")
            print(f"Pnet: {pnet/1000:,.0f} kW")

            print(f"\neta_pt: {FCS.eta_pt:,.4f}")
            print(f"Mfftotal: {FCS.Mfftotal:,.4f} ")                # Not Calculated

            print(f"\nmFC: {FCS.mFC:,.0f} kg")
            print(f"mcomp: {FCS.mcomp:,.0f} kg")
            print(f"mcoolingsystem: {FCS.mcoolingsystem:,.0f} kg")
            print(f"mPMAD: {FCS.mPMAD:,.0f} kg")
            print(f"mem: {FCS.mem:,.0f} kg")

            print(f"\n=======================")
            print(f"mpt: {FCS.mpt:,.0f} kg")
            print(f"mtank: {FCS.mtank:,.0f} kg")
            print(f"W_wing: {FCS.W_wing:,.0f} kg")
            print(f"W_HT: {FCS.W_HT:,.0f} kg")
            print(f"W_VT: {FCS.W_VT:,.0f} kg")
            print(f"W_fus: {FCS.W_fus:,.0f} kg")
            print(f"W_lndgearmain: {FCS.W_lndgearmain:,.0f} kg")
            print(f"W_lndgearnose: {FCS.W_lndgearnose:,.0f} kg")
            print(f"W_motor: {FCS.W_motor:,.0f} kg")
            print(f"W_flight_control: {FCS.W_flight_control:,.0f} kg")
            print(f"W_els: {FCS.W_els:,.0f} kg")
            print(f"W_iae: {FCS.W_iae:,.0f} kg")
            print(f"W_hydraulics: {FCS.W_hydraulics:,.0f} kg")
            print(f"W_fur: {FCS.W_fur:,.0f} kg")
            print(f"OEM: {FCS.OEM:,.0f} kg")
            print(f"OEMmisc: {FCS.OEMmisc:,.0f} kg")
            print(f"mfuel: {FCS.mfuel:.1f} kg")
            print(f"mbatt: {FCS.mbatt:.1f} kg")
            print(f"-----------------------")
            print(f"MTOM: {FCS.MTOM:,.0f} kg")
            

            print(f"\nVtankex: {FCS.Vtankex:,.1f} m^3")
            break
        else:
            mtom_a = FCS.MTOM
            iter2 += 1

    #Hybrid power calculation
    pfc_ready =  Pfuelcell_engine
    pfc_taxing = Pfuelcell_taxing
    #pfc_takeoff = Pfuelcell_takeoff + Pcomp4 + Pcooling_system4_fc
    pfc_climb = FCS.Pfuelcell_climb + FCS.Pcomp1 + FCS.Pcooling_system1_fc
    pbat_climb = FCS.Pbat_climb
    pfc_cruise1 = FCS.Pfuelcell_cruise_charger + FCS.Pcomp2 + FCS.Pcooling_system2_fc #Pcomp3 + Pcooling_system3_fc
    pfc_cruise2 = FCS.Pfuelcell_cruise + FCS.Pcomp2 + FCS.Pcooling_system2_fc
    #pbat_takeoff = Pbat_takeoff
    pbat_charge = FCS.Pbat_cruise_charger
    ####################

    figs = pemfcsys_sizing_results['figs']
    figs[-1].show()
    figs[-1].write_image(f"./figs/pemfc_fig.png")

    #Mission profile plot code
    # print('Pfuel_ready: %.2f kW Pfuel_taxing: %.2f kW Pfuel_climb: %.2f kW Pfuel_cruise: %.2f kW Pfuel_cruise_charger: %.2f kW Pbat_climb: %.2f kW Pbat_charge: %.2f kW' %(pfc_ready/1000, pfc_taxing/1000, pfc_climb/1000 , pfc_cruise2/1000, pfc_cruise1/1000 , pbat_climb/1000,  pbat_charge/1000))
    print(f"\nPfuel_ready: {pfc_ready/1000:,.0f} kW, Pfuel_taxing: {pfc_taxing/1000:,.0f} kW, Pfuel_climb: {pfc_climb/1000:,.0f} kW, Pfuel_cruise: {pfc_cruise2/1000:,.0f} kW, Pfuel_cruise_charger: {pfc_cruise1/1000:,.0f} kW, Pbat_climb: {pbat_climb/1000:,.0f} kW, Pbat_charge: {pbat_charge/1000:,.0f} kW")
    Mission_time = [5, 15, 16, 23, 25, 70, 71, 100, 110, 112, 116, 118, 135, 135, 135, 140, 150, 160, 160]
    Power_fc = [pfc_ready, pfc_taxing, pfc_climb, pfc_climb, pfc_cruise1, pfc_cruise1, pfc_cruise2, pfc_cruise2, pfc_taxing, pfc_climb, pfc_climb, pfc_cruise2, pfc_cruise2, pfc_cruise2, pfc_cruise2, pfc_taxing, pfc_taxing, pfc_taxing, pfc_ready]
    Power_bat = [0, 0, pbat_climb, pbat_climb, pbat_charge, pbat_charge, 0, 0, 0, pbat_climb, pbat_climb, 0, 0, 0, 0, 0, 0, 0, 0]


    x = Mission_time
    y1 = np.array(Power_bat)/1000
    y2 = np.array(Power_fc)/1000
    y3 = y1 + y2

    plt.plot(x,y1, linestyle='solid', color = 'gray', label = 'Battery')
    plt.plot(x,y2, linestyle='dashed', color = 'orange', label = 'Fuel Cell')
    plt.plot(x,y3, linestyle='solid', color = 'blue', label = 'Total')

    plt.xlabel('Time(min)')
    plt.ylabel('Power(kW)')
    plt.axis([0, 180, -200, 2500])
    plt.title('Power Mission Profile')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig("Power Mission Profile.png", dpi=400)
    # plt.show()
    plt.close()

    ReqPow = np.array([y3, y2, y1])
    ReqPow = ReqPow.T

    # print(ReqPow)
    df = pd.DataFrame(ReqPow, columns=["ReqPow_AC", "ReqPow_FC", "ReqPow_Batt"])
    df = df.T
    df.to_excel("ReqPowDATA.xlsx", index=True)
