
import numpy as np

class Area:

    # initialization
    def __init__(self, theta_src, h, R, theta_view, N):
        self.t_src = theta_src
        self.h = h
        self.R = R
        self.th_v = theta_view
        self.phi_src = 0
        self.n = N
        
        self.cos_theta_e_hor = self.R/(self.R+self.h)
        self.theta_e_hor = np.arccos(self.cos_theta_e_hor)
        self.A0 = (2*np.pi*self.R**2)*(1-self.cos_theta_e_hor)
        
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
    
    def rho_nadir(self): # distance from observatory to Earth based on nadir angle
        psi  = np.pi - np.arcsin(self.R * np.sin(self.t_src) / (self.R+self.h))
        theta_E = np.pi - psi - self.t_src
        rho_2 = (self.R+self.h)**2 + self.R**2 - 2 * (self.R+self.h)*self.R*np.cos(theta_E)
        rho = np.sqrt(rho_2)
        return rho 
    
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
        self.angle = self.view_angle_det()
        rho = self.rho_nadir()
        self.ret_e_x = self.e_x *  (self.angle < self.th_v) * (dot>0.)
        self.ret_e_x=self.ret_e_x[np.nonzero(self.ret_e_x)]
        
        self.ret_e_y = self.e_y *  (self.angle < self.th_v) * (dot>0.)
        self.ret_e_y=self.ret_e_y[np.nonzero(self.ret_e_y)]
        
        self.ret_e_z = self.e_z *  (self.angle < self.th_v) * (dot>0.)
        self.ret_e_z=self.ret_e_z[np.nonzero(self.ret_e_z)]
        
        self.ret_e_dot = dot*  (self.angle < self.th_v) * (dot>0.)
        self.ret_e_dot=self.ret_e_dot[np.nonzero(self.ret_e_dot)]
        
        self.ret_rho = rho*  (self.angle < self.th_v) * (dot>0.)
        self.ret_rho=self.ret_rho[np.nonzero(self.ret_rho)]
        
        self.ret_phi_e = np.arctan(self.ret_e_y / self.ret_e_x)
        self.ret_t_e = np.arccos(self.ret_e_z)
        
        return self.ret_e_dot, self.ret_phi_e, self.ret_t_e, self.ret_rho


