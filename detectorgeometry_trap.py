
import numpy as np

class Area:

    # initialization
    def __init__(self, theta_src, h, R, theta_view, N):
        self.t_src = theta_src
        self.h = h
        self.R = R
        self.th_v = theta_view
        self.phi_src = np.radians(45)
        self.n = N
        
        self.cos_theta_e_hor = self.R/(self.R+self.h)
        self.theta_e_hor = np.arccos(self.cos_theta_e_hor)
        self.A0 = (2*np.pi*self.R**2)*(1-self.cos_theta_e_hor)
        ###################################
    
    def rho_nadir(self,nadir): # distance from observatory to Earth based on nadir angle
        psi  = np.pi - np.arcsin(self.R * np.sin(nadir) / (self.R+self.h))
        theta_E = np.pi - psi - nadir
        rho_2 = (self.R+self.h)**2 + self.R**2 - 2 * (self.R+self.h)*self.R*np.cos(theta_E)
        rho = np.sqrt(rho_2)
        return rho 
    
    def thetaE_nadir(self,nadir): # earth angle based on nadir angle
        self.earth_t = np.arcsin(self.rho_nadir(nadir) * np.sin(nadir) / self.R)
        return self.earth_t
    
    def earth_patch(self):    
        earth_t = self.thetaE_nadir(self.t_src)
        earth_t_min = self.thetaE_nadir(self.t_src - self.th_v)
        arg = 4*((self.R+self.h)**2)*np.cos(self.t_src + self.th_v)**2 - 4*(2*self.R*self.h+self.h**2)
        if arg >=0:
            earth_t_max = self.thetaE_nadir(self.t_src + self.th_v)
        elif arg < 0:
            earth_t_max = np.arccos(self.R/(self.R+self.h))
        phi_E = np.radians(45)
        d_phi_E = np.arcsin(self.rho_nadir(self.t_src) * np.sin(self.t_src) / self.R)
        phi_E_min = phi_E - d_phi_E
        phi_E_max = phi_E + d_phi_E
        return earth_t_min, earth_t_max, phi_E_min, phi_E_max
        
    def source_coords(self):
        self.r_x = np.sin(self.t_src) * np.cos(self.phi_src)
        self.r_y = np.sin(self.t_src) * np.sin(self.phi_src)
        self.r_z = np.cos(self.t_src) 
        return self.r_x, self.r_y, self.r_z
        
    def dot_prod(self):
        self.e_x, self.e_y, self.e_z = self.earth_coords()
        self.r_x, self.r_y, self.r_z = self.source_coords()
        self.dot = self.e_x*self.r_x +self.r_y*self.e_y + self.e_z*self.r_z
        return self.dot
    
    def earth_locs(self):
        earth_t_min, earth_t_max, phi_E_min, phi_E_max = self.earth_patch()
        self.cos_phi_e = np.random.uniform(np.cos(phi_E_max), np.cos(phi_E_min), self.n)
        self.cos_theta_e = np.random.uniform(np.cos(earth_t_min), np.cos(earth_t_max), self.n)
        self.phi_e = np.arccos(self.cos_phi_e)
        self.t_e = np.arccos(self.cos_theta_e)
        return self.t_e, self.phi_e
    
    def earth_coords(self):
        self.phi_e, self.t_e = self.earth_locs()
        self.e_x = np.sin(self.t_e) * np.cos(self.phi_e)
        self.e_y = np.sin(self.t_e) * np.sin(self.phi_e)
        self.e_z = np.cos(self.t_e) 
        return self.e_x, self.e_y, self.e_z
    
    def view_angle_det(self):
        dot = self.dot_prod()
        obs_x = 0
        obs_y = 0
        obs_z = self.R+self.h
        
        point_to_obs_x = obs_x - self.R*self.e_x
        point_to_obs_y = obs_y - self.R*self.e_y
        point_to_obs_z = obs_z - self.R*self.e_z
        
        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        
        point_to_obs_hat_x = point_to_obs_x / norm
        point_to_obs_hat_y = point_to_obs_y / norm
        point_to_obs_hat_z = point_to_obs_z / norm
        
        obs_dot = point_to_obs_hat_x*self.r_x+point_to_obs_hat_y*self.r_y+point_to_obs_hat_z*self.r_z
        self.angle = np.arccos(obs_dot)
        return self.angle
    
    def degree_eff_area(self):
        dot = self.dot_prod()
        angle = self.view_angle_det()
        self.A_deg = self.A0 * 1./float(self.n) * np.sum(dot * (angle < self.th_v) * (dot>0.) )
        return self.A_deg
    
    def event_retention(self):
        dot = self.dot_prod()
        angle = self.view_angle_det()
        self.ret_e_x = self.e_x *  (angle < self.th_v) * (dot>0.)
        self.ret_e_y = self.e_y *  (angle < self.th_v) * (dot>0.)
        self.ret_e_z = self.e_z *  (angle < self.th_v) * (dot>0.)
        self.ret_e_dot = dot*  (angle < self.th_v) * (dot>0.)
        self.ret_phi_e = np.arctan(self.ret_e_y / self.ret_e_x)
        self.ret_t_e = np.arccos(self.ret_e_z)
        return self.t_src, self.ret_phi_e, self.ret_t_e



