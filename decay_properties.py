
import numpy as np
from Tau_Exit_Simulator_2 import Tau_Exit_Simulator
import Tau_Decay_Simulator as TauDecaySimulator
reload(TauDecaySimulator)
from scipy import stats

class tau_event:

    # initialization
    def __init__(self, theta_src, e_dot, phi_e, t_e, rho, E_nu,h,R,view_angle):
        self.N = len(e_dot)
        self.theta_src = theta_src
        self.e_dot = e_dot
        self.phi_e = phi_e
        self.t_e =t_e
        self.rho = rho
        self_e_nu = E_nu
        self.phi_src = 0
        self.h =h
        self.R = R
        self.th_v = view_angle
        self.cos_theta_e_hor = self.R/(self.R+self.h)
        self.theta_e_hor = np.arccos(self.cos_theta_e_hor)
        self.A0 = (2*np.pi*self.R**2)*(1-self.cos_theta_e_hor)
        
        
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
        self.TDS = TauDecaySimulator.Tau_Decay_Simulator()
    
    def E_tau(self):
        vals =np.asarray([])
        for i in range(self.N):
            vals = np.append(vals,self.TES.sample_energies_th_exit(np.degrees(self.theta_src)))
        return vals
    
    def decay_distance_det(self,tau_energies):
        decay_dist = self.TDS.sample_range(tau_energies, len(tau_energies))
        return decay_dist
    
    def event_energy_cut(self):
        tau_energy = self.E_tau()
        decay_dist= self.decay_distance_det(tau_energy)

        cut_tau_energy = tau_energy *  (tau_energy > 14.0) * (decay_dist < self.rho)
        cut_tau_energy=cut_tau_energy[np.nonzero(cut_tau_energy)]
        
        cut_decay_dist = decay_dist *  (tau_energy > 14.0) * (decay_dist < self.rho)
        cut_decay_dist=cut_decay_dist[np.nonzero(cut_decay_dist)]
        
        cut_e_dot = self.e_dot*  (tau_energy > 14) * (decay_dist < self.rho)
        cut_e_dot=cut_e_dot[np.nonzero(cut_e_dot)]
        
        cut_phi_e = self.phi_e*  (tau_energy > 14) * (decay_dist < self.rho)
        cut_phi_e=cut_phi_e[np.nonzero(cut_phi_e)]
        
        cut_t_e = self.t_e*  (tau_energy > 14) * (decay_dist < self.rho)
        cut_t_e= cut_t_e[np.nonzero(cut_t_e)]
        
        cut_rho = self.rho*(tau_energy > 14) * (decay_dist < self.rho)
        cut_rho=cut_rho[np.nonzero(cut_rho)]
        
        return cut_e_dot, cut_phi_e, cut_t_e, cut_tau_energy, cut_rho
    
    def P_exit(self):
        exit_p_arr = self.TES.P_exit_th_exit(np.degrees(self.theta_src))
        return exit_p_arr
    
    def P_decay(self,rho,decay_distances):
        lam = 4.9 * rho
        p_decay_arr = stats.poisson.sf(lam, rho)
        return p_decay_arr
        
    def E_shower(self,tau_energies):
        cc_counts=[]
        fractions_h=[]
        fractions_e=[]
        for i in range(len(tau_energies)):
            cc = 0
            f_h = []
            f_e = []
            for k in range(0,10000):
                stp = self.TDS.sample_shower_type() 
                cc+=stp
                if stp==0: f_h.append(self.TDS.sample_energy_fraction(stp))
                if stp==1: f_e.append(self.TDS.sample_energy_fraction(stp))
            cc_counts.append(cc)
            fractions_h.append(f_h)
            fractions_e.append(f_e)
        return cc_counts, fractions_h,fractions_e
    
    #P_shower
    
    def view_angle_det(self,earth_theta,earth_phi,d,h,R):
        obs_x = 0
        obs_y = 0
        h_decay = d * np.cos(self.theta_src)
        h_eff = h - h_decay
        obs_z = R+h_eff
        
        chord = (R*np.sin(earth_theta) - d*np.sin(self.theta_src)) 
        earth_theta_eff = np.arcsin(chord /R)
        e_x_eff = R*np.sin(earth_theta_eff) * np.cos(earth_phi)
        e_y_eff = R*np.sin(earth_theta_eff) * np.sin(earth_phi)
        e_z_eff = R*np.cos(earth_theta_eff) 
        
        point_to_obs_x = obs_x - e_x_eff
        point_to_obs_y = obs_y - e_y_eff
        point_to_obs_z = obs_z - e_z_eff
        
        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        
        point_to_obs_hat_x = point_to_obs_x / norm
        point_to_obs_hat_y = point_to_obs_y / norm
        point_to_obs_hat_z = point_to_obs_z / norm
        
        r_x = np.sin(self.theta_src) * np.cos(self.phi_src)
        r_y = np.sin(self.theta_src) * np.sin(self.phi_src)
        r_z = np.cos(self.theta_src) 
        
        obs_dot = point_to_obs_hat_x*r_x+point_to_obs_hat_y*r_y+point_to_obs_hat_z*r_z
        angle = np.arccos(obs_dot)
   
        return angle
    
    def event_retention(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_rho = self.event_energy_cut()
        ret_decay_dist = self.decay_distance_det(ret_tau_energy)
        ret_p_decay = self.P_decay(ret_rho, ret_decay_dist)
        ret_cc_counts, ret_fractions_h,ret_fractions_e = self.E_shower(ret_tau_energy)
        ret_view_angle = self.view_angle_det(ret_t_e,ret_phi_e,ret_decay_dist,self.h,self.R)
        return ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_decay_dist, ret_p_decay ,ret_view_angle
    
    
##########
    def degree_eff_area(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_decay_dist, ret_p_decay ,ret_view_angle = self.event_retention()
        N = len(ret_e_dot)
        self.A_deg = self.A0 * 1./float(N) * np.sum(ret_e_dot * (ret_view_angle < self.th_v) )
        return self.A_deg


