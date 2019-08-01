
import numpy as np
import Tau_Exit_Simulator_2 as TauExitSimulator
import Tau_Decay_Simulator as TauDecaySimulator
reload(TauDecaySimulator)
reload(TauExitSimulator)
from scipy import stats

class tau_event:

    # initialization
    def __init__(self, theta_src, ice, E_cut, e_dot, phi_e, t_e, rho, E_nu,h,R,view_cut,  exit_angles, emg_angles, view_angles, A_g, 
                 A0, N0):
        self.N = len(e_dot)
        self.theta_src = theta_src
        self.ice = str(int(ice))+'.0'
        self.e_cut = E_cut
        self.e_dot = e_dot
        self.phi_e = phi_e
        self.t_e =t_e
        self.norm = rho
        self_e_nu = E_nu
        self.phi_src = np.radians(180)
        self.h =h
        self.R = R
        self.th_v = view_cut
        self.view = view_angles
        self.exit = exit_angles
        self.emg = emg_angles
        self.A_g = A_g
        self.A0 = A0
        self.N0 = N0
        if np.floor(E_nu)-E_nu == 0:
            self.TEScall = '1e+'+str(self_e_nu)
        else:
            self.TEScall = '3e+'+str(int(np.floor(E_nu)))
        self.energy_cut = 1
        self.decay_cut = 1
 
        self.TES = TauExitSimulator.Tau_Exit_Simulator('./high_statistics/'+self.ice+'km_ice_midCS_stdEL/LUT_'+self.TEScall+'_eV.npz')
        self.TDS = TauDecaySimulator.Tau_Decay_Simulator()
    
    def E_tau(self):
        vals =np.asarray([])
        for i in range(self.N):
            try:
                vals = np.append(vals,self.TES.sample_energies_th_exit(np.degrees(self.exit[i])))
            except:
                vals =np.append(vals,0)
        #print vals[:10]
        return vals
    
    def decay_distance_det(self,tau_energies):
        decay_dist = self.TDS.sample_range(10**(tau_energies), len(tau_energies))
        #print decay_dist
        return decay_dist
    
    def event_energy_cut(self):
        tau_energy = self.E_tau()
        decay_dist= self.decay_distance_det(tau_energy)

        if self.energy_cut + self.decay_cut==0:
            cut_factor_idx = 1
        elif self.energy_cut + self.decay_cut == 1:
            cut_factor_idx =  np.asarray((tau_energy > self.e_cut))
        elif self.energy_cut + self.decay_cut == 2: 
            cut_factor_idx =  np.asarray((tau_energy > self.e_cut) * (decay_dist < self.norm))
       
        cut_tau_energy = tau_energy * cut_factor_idx
        cut_tau_energy=cut_tau_energy[np.nonzero(cut_tau_energy)]
        
        cut_decay_dist = decay_dist * cut_factor_idx
        cut_decay_dist=cut_decay_dist[np.nonzero(cut_decay_dist)]
        
        cut_e_dot = self.e_dot *cut_factor_idx
        cut_e_dot=cut_e_dot[np.nonzero(cut_e_dot)]
        
        cut_phi_e = self.phi_e *cut_factor_idx
        cut_phi_e=cut_phi_e[np.nonzero(cut_phi_e)]
        
        cut_t_e = self.t_e * cut_factor_idx
        cut_t_e= cut_t_e[np.nonzero(cut_t_e)]
        
        cut_rho = self.norm * cut_factor_idx
        cut_rho=cut_rho[np.nonzero(cut_rho)]
        
        cut_exit = self.exit * cut_factor_idx
        cut_exit=cut_exit[np.nonzero(cut_exit)]
        
        cut_emg = self.emg * cut_factor_idx
        cut_emg=cut_emg[np.nonzero(cut_emg)]
        
        cut_view = self.view * cut_factor_idx
        cut_view=cut_view[np.nonzero(cut_view)]
        
        return cut_e_dot, cut_phi_e, cut_t_e, cut_tau_energy, cut_rho, cut_decay_dist, cut_exit, cut_emg, cut_view
    
    
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
    
    def decay_angle_alt(self,earth_theta,earth_phi,d,h,R, zenith):
        obs_x = 0
        obs_y = 0
        obs_z = R+h
        
        emg_angle = np.pi/2 - zenith - self.theta_src
        h_decay = d*np.sin(emg_angle)
        x_decay = d*np.cos(emg_angle)
        e_x_decay = R*np.sin(earth_theta)* np.cos(earth_phi) +x_decay
        e_y_decay = R*np.sin(earth_theta)* np.sin(earth_phi)
        e_z_decay = R*np.cos(earth_theta) +h_decay
        
        
        
        point_to_obs_x = obs_x - e_x_decay
        point_to_obs_y = obs_y - e_y_decay
        point_to_obs_z = obs_z - e_z_decay
        
        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        
        point_to_obs_hat_x = point_to_obs_x / norm
        point_to_obs_hat_y = point_to_obs_y / norm
        point_to_obs_hat_z = point_to_obs_z / norm
        
        r_x = np.sin(self.theta_src) * np.cos(self.phi_src+np.pi)
        r_y = np.sin(self.theta_src) * np.sin(self.phi_src+np.pi)
        r_z = np.cos(self.theta_src) 
        
        obs_dot = point_to_obs_hat_x*r_x+point_to_obs_hat_y*r_y+point_to_obs_hat_z*r_z
        angle = np.arccos(obs_dot)
   
        return angle, norm, h_decay
    
    def event_retention(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_exit_obs, ret_exit_decay, ret_exit, ret_emg, ret_view = self.event_energy_cut()
        ret_fractions,ret_types = self.E_shower(ret_tau_energy)
        ret_p_exit = self.P_exit(ret_exit)
        ret_zenith = ret_exit - self.theta_src
        ret_decay_angle, ret_decay_obs, ret_decay_alt = self.decay_angle_alt(ret_t_e,ret_phi_e,ret_exit_decay,self.h,self.R,ret_zenith )
        return ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_exit_obs, ret_exit_decay, ret_decay_obs,ret_exit, ret_emg, ret_view, ret_decay_alt, ret_decay_angle, ret_fractions,ret_types, ret_p_exit
    
    
##########
    def degree_eff_area(self):
        ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_exit_obs, ret_exit_decay, ret_decay_obs, ret_exit, ret_emg, ret_view, ret_decay_alt, ret_decay_angle, ret_fractions,ret_types, ret_p_exit = self.event_retention()

        try:
            A_deg = self.A0 *1./float(self.N0) *  np.sum(ret_p_exit * ret_e_dot * (ret_view < self.th_v) * (ret_e_dot>0.) ) 
        except:
            A_deg = 0
        return A_deg, ret_e_dot, ret_phi_e, ret_t_e, ret_tau_energy, ret_exit_obs, ret_exit_decay, ret_decay_obs, ret_exit, ret_emg, ret_view, ret_decay_alt, ret_decay_angle, ret_fractions,ret_types, ret_p_exit, self.A0,self.N0


