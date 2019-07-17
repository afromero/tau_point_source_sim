
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
        
        ###################################
    def A_theta_patch(self,t_min,t_max,phi_min,phi_max):
        Area = self.R**2 * (phi_max-phi_min)*(np.cos(t_min) - np.cos(t_max))
        return Area

    def thetaE_nadir(self,nadir): # earth angle based on nadir angle
        psi  = np.pi - np.arcsin((self.R+self.h) * np.sin(nadir) / self.R)
        theta_E = np.pi - psi - nadir
        return theta_E   

    def earth_patch(self,theta_src, phi_src, theta_view):    
        earth_t = self.thetaE_nadir(theta_src)
        earth_t_min = self.thetaE_nadir(theta_src - theta_view)
        arg = 4*((self.R+self.h)**2)*np.cos(theta_src + theta_view)**2 - 4*(2*self.R*self.h+self.h**2)
        if arg >=0:
            earth_t_max = self.thetaE_nadir(theta_src + theta_view)
        elif arg < 0:
            earth_t_max = np.arccos(self.R/(self.R+self.h))

        phi_E = self.phi_src
        d_phi_E =  100* (earth_t_max - earth_t_min)
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
        cos_phi_e = np.random.uniform(np.cos(phi_E_max), np.cos(phi_E_min), self.n)
        cos_theta_e = np.random.uniform(np.cos(earth_t_max), np.cos(earth_t_min), self.n)
        phi_e = np.arccos(cos_phi_e)
        t_e = np.arccos(cos_theta_e)
        return  t_e, phi_e

    def view_angle_det(self, e_x,e_y,e_z,r_x,r_y,r_z):
        obs_x = 0
        obs_y = 0
        obs_z = self.R+self.h

        point_to_obs_x = obs_x - self.R*e_x
        point_to_obs_y = obs_y - self.R*e_y
        point_to_obs_z = obs_z - self.R*e_z

        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)

        point_to_obs_hat_x = point_to_obs_x / norm
        point_to_obs_hat_y = point_to_obs_y / norm
        point_to_obs_hat_z = point_to_obs_z / norm

        obs_dot = point_to_obs_hat_x*r_x+point_to_obs_hat_y*r_y+point_to_obs_hat_z*r_z
        angle = np.arccos(obs_dot)

        return angle

        ###################################
    def event_retention(self):
        earth_t_min, earth_t_max, phi_E_min, phi_E_max = self.earth_patch(self.t_src,self.phi_src, self.th_v)
        r_x, r_y, r_z = self.coords(self.t_src, self.phi_src + np.pi)
        t_e,phi_e = self.earth_locs(earth_t_min, earth_t_max, phi_E_min, phi_E_max)
        e_x,e_y,e_z = self.coords(t_e,phi_e)
        angle = self.view_angle_det(e_x,e_y,e_z,r_x, r_y,r_z)
        dot = self.dot_prod(e_x,e_y,e_z,r_x, r_y,r_z)
        A0 = self.A_theta_patch(earth_t_min, earth_t_max, phi_E_min, phi_E_max)
        A_deg = A0 * 1./float(self.n) * np.sum(dot * (angle < self.th_v) * (dot>0.) )
        
        self.ret_e_x = e_x *  (angle < self.th_v) * (dot>0.)
        self.ret_e_y = e_y *  (angle < self.th_v) * (dot>0.)
        self.ret_e_z = e_z *  (angle < self.th_v) * (dot>0.)
        self.ret_e_dot = dot *  (angle < self.th_v) * (dot>0.)
        self.ret_phi_e = np.arctan(self.ret_e_y / self.ret_e_x)
        self.ret_t_e = np.arccos(self.ret_e_z)
        return A_deg, self.t_src, self.ret_phi_e, self.ret_t_e



