
import numpy as np

class Area:

    # initialization
    def __init__(self, theta_src, h, R, theta_view, N):
        self.t_src = theta_src
        self.h = h
        self.r = R
        self.t_v = theta_view
        self.phi_src = 0
        self.n = N
        
        self.cos_theta_e_hor = self.r/(R+h)
        self.theta_e_hor = np.arccos(self.cos_theta_e_hor)
        self.A0 = (2*np.pi*self.r**2)*(1-self.cos_theta_e_hor)
    
        
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
        self.phi_e = np.random.uniform(0., 2.*np.pi, self.n)
        self.cos_theta_e = np.random.uniform(self.cos_theta_e_hor, 1, self.n)
        self.t_e = np.arccos(self.cos_theta_e)
        return self.phi_e, self.t_e
    
    def earth_coords(self):
        self.phi_e, self.t_e = self.earth_locs()
        self.e_x = np.sin(self.t_e) * np.cos(self.phi_e)
        self.e_y = np.sin(self.t_e) * np.sin(self.phi_e)
        self.e_z = np.cos(self.t_e) 
        return self.e_x, self.e_y, self.e_z
    
    def view_angle_det(self):
        obs_x = 0
        obs_y = 0
        obs_z = self.r+self.h
        
        point_to_obs_x = obs_x - self.r*self.e_x
        point_to_obs_y = obs_y - self.r*self.e_y
        point_to_obs_z = obs_z - self.r*self.e_z
        
        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        
        point_to_obs_hat_x = point_to_obs_x / norm
        point_to_obs_hat_y = point_to_obs_y / norm
        point_to_obs_hat_z = point_to_obs_z / norm
        
        obs_dot =point_to_obs_hat_x*self.r_x+point_to_obs_hat_y*self.r_y+point_to_obs_hat_z*self.r_z
        self.angle = np.arccos(obs_dot)
    
        return self.angle
    
    def degree_eff_area(self):
        dot = self.dot_prod()
        angle = self.view_angle_det()
        self.A_deg = self.A0 * 1./float(self.n) * np.sum(dot * (angle < self.t_v) * (dot>0.) )
        return self.A_deg


