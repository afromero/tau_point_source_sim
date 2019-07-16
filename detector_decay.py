
import numpy as np
from Tau_Exit_Simulator_2 import Tau_Exit_Simulator


class tau_event:

    # initialization
    def __init__(self, theta_src, e_dot, phi_e, t_e, rho, E_nu):
        self.N = len(e_dot)
        self.theta_src = theta_src
        self.e_dot = e_dot
        self.phi_e = phi_e
        self.t_e =t_e
        self.rho = rho
        self_e_nu = E_nu
        self.elev = np.degrees(self.theta_src) - 90
        
        TES1 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+15_eV.npz'
        TES2 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+15_eV.npz'
        TES3 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+16_eV.npz'
        TES4 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+16_eV.npz'
        TES5 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+17_eV.npz'
        TES6 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+17_eV.npz'
        TES7 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+18_eV.npz'
        TES8 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+18_eV.npz'
        TES9 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+19_eV.npz'
        TES10 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+19_eV.npz'
        TES11 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+20_eV.npz'
        TES12 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_3e+20_eV.npz'
        TES13 = './high_statistics/4.0km_ice_midCS_stdEL/LUT_1e+21_eV.npz'

        TESchoices = [TES1,TES2,TES3,TES4,TES5,TES6,TES7,TES8,TES9,TES10,TES11,TES12,TES13]       
        energychoices = [15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0,19.5,20.0,20.5,21.0]
        
        # selecting the simulation with the appropriate incident tau neutrino energy
        k = energychoices.index(E_nu)
        self.TES = Tau_Exit_Simulator(TESchoices[k])
    
    def Exit_P(self):
        exit_p_arr = self.TES.P_exit_th_exit(self.elev)
        return exit_p_arr
    
    def E_tau(self):
        vals =np.asarray([])
        for i in range(self.N):
            vals = np.append(vals,self.TES.sample_energies_th_exit(90+self.elev))
        return vals
    
    def event_retention(self):
        self.tau_energy = self.E_tau()
        
        self.ret_tau_energy = self.tau_energy *  (self.tau_energy > 14.0) 
        self.ret_tau_energy=self.ret_tau_energy[np.nonzero(self.ret_tau_energy)]
        
        self.ret_e_dot = self.e_dot*  (self.tau_energy > 14) 
        self.ret_e_dot=self.ret_e_dot[np.nonzero(self.ret_e_dot)]
        
        self.ret_phi_e = self.phi_e*  (self.tau_energy > 14) 
        self.ret_phi_e=self.ret_phi_e[np.nonzero(self.ret_phi_e )]
        
        self.ret_t_e = self.t_e*  (self.tau_energy > 14) 
        self.ret_t_e=self.ret_t_e[np.nonzero(self.ret_t_e )]
        
        return self.ret_e_dot, self.ret_phi_e, self.ret_t_e, self.ret_tau_energy



