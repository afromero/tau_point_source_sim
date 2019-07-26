
import numpy as np
import Tau_Exit_Simulator_2 as TauExitSimulator
import Tau_Decay_Simulator as TauDecaySimulator
reload(TauDecaySimulator)
reload(TauExitSimulator)
from scipy import stats

class tau_event:

    # initialization
    def __init__(self, theta_src, ice, E_cut, e_dot, phi_e, t_e, rho, E_nu,h,R,view_cut,  exit_angles,view_angles, A_g):
        self.N = len(e_dot)
        self.theta_src = theta_src
        self.ice = str(int(ice))+'.0'
        self.e_cut = E_cut
        self.e_dot = e_dot
        self.phi_e = phi_e
        self.t_e =t_e
        self.norm = rho
        self_e_nu = E_nu
        self.phi_src = 0
        self.h =h
        self.R = R
        self.th_v = view_cut
        self.view = view_angles
        self.exit = exit_angles
        self.A_g = A_g
        if np.floor(E_nu)==np.rint(E_nu):
            self.TEScall = '1e+'+str(self_e_nu)
        else:
            self.TEScall = '3e+'+str(self_e_nu)
 
        self.TES = TauExitSimulator.Tau_Exit_Simulator('./high_statistics/'+self.ice+'km_ice_midCS_stdEL/LUT_'+self.TEScall+'_eV.npz')
        self.TDS = TauDecaySimulator.Tau_Decay_Simulator()
    
    def E_tau(self):
        vals =np.asarray([])
        for i in range(self.N):
            try:
                vals = np.append(vals,self.TES.sample_energies_th_exit(np.degrees(self.exit[i])))
            except:
                vals =np.append(vals,0)
        return vals
    
    def decay_distance_det(self,tau_energies):
        decay_dist = self.TDS.sample_range(10**(tau_energies), len(tau_energies))
        return decay_dist
    
    def event_energy_cut(self):
        tau_energy = self.E_tau()
        decay_dist= self.decay_distance_det(tau_energy)

        cut_tau_energy = tau_energy * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_tau_energy=cut_tau_energy[np.nonzero(cut_tau_energy)]
        
        cut_decay_dist = decay_dist * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_decay_dist=cut_decay_dist[np.nonzero(cut_decay_dist)]
        
        cut_e_dot = self.e_dot * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_e_dot=cut_e_dot[np.nonzero(cut_e_dot)]
        
        cut_phi_e = self.phi_e * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_phi_e=cut_phi_e[np.nonzero(cut_phi_e)]
        
        cut_t_e = self.t_e * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_t_e= cut_t_e[np.nonzero(cut_t_e)]
        
        cut_rho = self.norm * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_rho=cut_rho[np.nonzero(cut_rho)]
        
        cut_exit = self.exit * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_exit=cut_exit[np.nonzero(cut_exit)]
        
        cut_view = self.view * (tau_energy > self.e_cut) * (decay_dist < self.norm) 
        cut_view=cut_view[np.nonzero(cut_view)]
        
        return cut_e_dot, cut_phi_e, cut_t_e, cut_tau_energy, cut_rho, cut_decay_dist, cut_exit, cut_view
    
    
    def P_exit(self,exit_angle_arr):
        exit_p_arr =[]
        for i in range(len(exit_angle_arr)):
            try:
                val= self.TES.P_exit_th_exit(np.degrees(exit_angle_arr[i]))
                exit_p_arr.append(val)
            except:
                val = 0 
                exit_p_arr.append(val)
        return exit_p_arr
    
#     def P_decay(self,rho,decay_distances):
#         lam = 4.9 * rho  #ENERGY
#         p_decay_arr = stats.poisson.sf(lam, rho)
#         return p_decay_arr
        
    def E_shower(self,tau_energies):
        fractions=[]
        shower_types=[]
        for i in range(len(tau_energies)):
            stp = self.TDS.sample_shower_type() 
            if stp==0: 
                fractions.append(self.TDS.sample_energy_fraction(stp))
                shower_types.append("h")
            if stp==1: 
                fractions.append(self.TDS.sample_energy_fraction(stp))
                shower_types.append("e")
        return fractions, shower_types
    
#     def view_angle_det(self,earth_theta,earth_phi,d,h,R):
#         obs_x = 0
#         obs_y = 0
#         h_decay = d * np.cos(self.exit)
#         h_eff = h - h_decay
#         obs_z = R+h_eff
        
#         chord = (R*np.sin(earth_theta) - d*np.sin(self.exit)) 
#         earth_theta_eff = np.arcsin(chord /R)
#         e_x_eff = R*np.sin(earth_theta_eff) * np.cos(earth_phi)
#         e_y_eff = R*np.sin(earth_theta_eff) * np.sin(earth_phi)
#         e_z_eff = R*np.cos(earth_theta_eff) 
        
#         point_to_obs_x = obs_x - e_x_eff
#         point_to_obs_y = obs_y - e_y_eff
#         point_to_obs_z = obs_z - e_z_eff
        
#         norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        
#         point_to_obs_hat_x = point_to_obs_x / norm
#         point_to_obs_hat_y = point_to_obs_y / norm
#         point_to_obs_hat_z = point_to_obs_z / norm
        
#         r_x = np.sin(self.theta_src) * np.cos(self.phi_src)
#         r_y = np.sin(self.theta_src) * np.sin(self.phi_src)
#         r_z = np.cos(self.theta_src) 
        
#         obs_dot = point_to_obs_hat_x*r_x+point_to_obs_hat_y*r_y+point_to_obs_hat_z*r_z
#         angle = np.arccos(obs_dot)
   
#         return angle
    
    def event_retention(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_rho, ret_decay_dist, ret_exit, ret_view = self.event_energy_cut()
        ret_fractions,ret_types = self.E_shower(ret_tau_energy)
        ret_p_exit = self.P_exit(ret_exit)
        return ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_rho,ret_decay_dist, ret_exit, ret_view, ret_fractions,ret_types, ret_p_exit
    
    
##########
    def degree_eff_area(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_rho,ret_decay_dist, ret_exit, ret_view, ret_fractions,ret_types, ret_p_exit = self.event_retention()
        # print np.sum((ret_view_angle < self.th_v))
        try:
            self.A_deg = self.A_g  * np.sum(ret_p_exit * ret_e_dot )#* np.sum((ret_view_angle < self.th_v))
        except:
            self.A_deg = 0
        return self.A_deg


