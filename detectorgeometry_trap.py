
import numpy as np
import math

class Area:

    # initialization
    def __init__(self, theta_src, h, R, theta_view, N):
        self.t_src = theta_src
        self.h = h
        self.R = R
        self.th_v = theta_view
        self.phi_src = np.radians(180)
        self.n = N
        
        self.cos_theta_e_hor = self.R/(self.R+self.h)
        self.theta_e_hor = np.arccos(self.cos_theta_e_hor)
        
        ###################################
    def A_theta_patch(self,e_theta,e_phi):
        if np.degrees(self.t_src)<2:
            indicator =0
            t_max = max(e_theta)
            # Area = np.pi * (np.sin(t_max)*self.R)**2 
            Area = (2*np.pi*self.R**2)*(1-np.cos(t_max)) #t_max
            return Area
        else:
            indicator =1
            min_phi = min(e_phi)
            max_phi = max(e_phi)
            Area = self.R**2 * (max_phi-min_phi)*( np.cos(min(e_theta)) - np.cos(max(e_theta)) )
        return Area
    
    def thetaE_nadir(self,nadir): # earth angle based on nadir angle
        arg = np.arcsin((self.R+self.h) * np.sin(nadir) / self.R)
        if math.isnan(arg)==True:
            theta_E = self.theta_e_hor
        else:
            psi  = np.pi - arg
            theta_E = np.pi - psi - nadir
        return theta_E   

    def earth_patch(self,theta_src, phi_src, theta_view):    
        earth_t = self.thetaE_nadir(theta_src)
        earth_t_min = max(0,self.thetaE_nadir(theta_src - theta_view))
        
        arg = (self.R+self.h) * np.sin(theta_src+ 1.5*theta_view) / self.R
        if abs(arg) <=1:
            earth_t_max = self.thetaE_nadir(theta_src + 1.5*theta_view)
        elif abs(arg) >1:
            earth_t_max = np.arccos(self.R/(self.R+self.h))+ theta_view

        phi_E = self.phi_src
        arg2 = (self.h * np.tan(theta_src)) /(4*np.pi*np.sin(earth_t)*self.R)
        
        if np.degrees(theta_src)<2:
            phi_E_min = 0
            phi_E_max = 2*np.pi
        else: 
            #d_phi_E = arg2 * (40/np.degrees(theta_src))
            d_phi_E = 1.5*theta_view
            phi_E_min = phi_E - d_phi_E
            phi_E_max = phi_E + d_phi_E
        return earth_t_min, earth_t_max, phi_E_min, phi_E_max

    def coords(self,t_src,phi_src):
        r_x = np.sin(t_src) * np.cos(phi_src)
        r_y = np.sin(t_src) * np.sin(phi_src)
        r_z = np.cos(t_src) 
        return r_x, r_y,r_z

    def dot_prod(self,ax,ay,az,bx,by,bz):
        dot = ax*bx+ay*by+az*bz
        return dot

    def earth_locs(self,earth_t_min, earth_t_max, phi_E_min, phi_E_max):
        cos_theta_e = np.random.uniform(min(np.cos(earth_t_max), np.cos(earth_t_min)),max(np.cos(earth_t_max), np.cos(earth_t_min)), self.n)
        phi_e = np.random.uniform(0,2*np.pi, self.n)
        t_e = np.arccos(cos_theta_e)
        return  t_e, phi_e

    def view_angle_dist_det(self, e_x,e_y,e_z,r_x,r_y,r_z):
        obs_x = 0
        obs_y = 0
        obs_z = self.R+self.h

        point_to_obs_x = obs_x - self.R*e_x
        point_to_obs_y = obs_y - self.R*e_y
        point_to_obs_z = obs_z - self.R*e_z

        flight_path = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)

        point_to_obs_hat_x = point_to_obs_x / flight_path
        point_to_obs_hat_y = point_to_obs_y / flight_path
        point_to_obs_hat_z = point_to_obs_z / flight_path

        obs_dot = point_to_obs_hat_x*r_x+point_to_obs_hat_y*r_y+point_to_obs_hat_z*r_z
        view_angle = np.arccos(obs_dot)
        

        return view_angle, flight_path
   
        ###################################
    def event_retention(self):
        earth_t = self.thetaE_nadir(self.t_src)
        earth_t_min, earth_t_max, phi_E_min, phi_E_max = self.earth_patch(self.t_src,self.phi_src, self.th_v)
        r_x, r_y, r_z = self.coords(self.t_src, self.phi_src + np.pi) 
        t_e,phi_e = self.earth_locs(earth_t_min, earth_t_max, phi_E_min, phi_E_max)
        e_x,e_y,e_z = self.coords(t_e,phi_e)
        view_angle, flight_path = self.view_angle_dist_det(e_x,e_y,e_z,r_x, r_y,r_z)
        dot = self.dot_prod(e_x,e_y,e_z,r_x, r_y,r_z)
        exit_angle = np.arccos(dot)
        emg_angle = np.pi/2 - exit_angle
      
        ret_view_angle = view_angle *  (view_angle < self.th_v) * (dot>0.)
        ret_view_angle=ret_view_angle[np.nonzero(ret_view_angle)]
        ret_exit_angle = exit_angle *  (view_angle < self.th_v) * (dot>0.)
        ret_exit_angle=ret_exit_angle[np.nonzero(ret_exit_angle)]
        ret_emg_angle = emg_angle *  (view_angle < self.th_v) * (dot>0.)
        ret_emg_angle=ret_emg_angle[np.nonzero(ret_emg_angle)]
        ret_norm = flight_path *  (view_angle < self.th_v) * (dot>0.)
        ret_norm=ret_norm[np.nonzero(ret_norm)]
        ret_dot = dot *  (view_angle < self.th_v) * (dot>0.)
        ret_dot=ret_dot[np.nonzero(ret_dot)]
        
        ret_phi_e = phi_e*  (view_angle < self.th_v) * (dot>0.)
        ret_phi_e=ret_phi_e[np.nonzero(ret_phi_e)]
        ret_t_e = t_e *  (view_angle < self.th_v) * (dot>0.)
        ret_t_e=ret_t_e[np.nonzero(ret_t_e)]
        
        A0= self.A_theta_patch(t_e,phi_e)
        A_deg = A0 *1./float(self.n) *  np.sum(dot * (view_angle < self.th_v) * (dot>0.) ) 
        
        return A_deg, ret_phi_e, ret_t_e, ret_view_angle, ret_exit_angle, ret_emg_angle, ret_norm, ret_dot, A0, self.n



