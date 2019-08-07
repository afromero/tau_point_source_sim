import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib as mpl
import sys
sys.path.append('nutau_sim/03_Detector')
sys.path.append('nutau_sim/applications/ANITA_upper_bound')
import tau_Shower_Efield_ANITA_Sim_lib  as RF_functions
reload(RF_functions)
import decay_properties as decay
reload(decay)

class event_detection:
    def __init__(self, A_g, f_lo, f_high, Gain_dB, Nphased,
                 exit_angles,emg_angles, view_angles, decay_angle, ret_decay_alt, view_cut,
                 ret_exit_obs, ret_exit_decay, ret_decay_obs, ret_p_exit, ret_dot,
                 E_t, R, ice, h, e_theta, e_phi, src_theta, A0, N0): 

        self.N = len(exit_angles)
        self.A_ret = A_g
        self.A0 = A0
        self.N0 = N0
        self.zenith_angle_deg = np.degrees(exit_angles)
        self.view_angle_deg = np.degrees(view_angles)
        self.decay_angles = np.degrees(decay_angle)
        self.emg_angles = np.degrees(emg_angles)
        self.view_cut = view_cut #3.16
        self.exit_obs = ret_exit_obs
        self.exit_decay = ret_exit_decay
        self.decay_obs = ret_decay_obs
        self.decay_view_angle = decay_angle
        self.decay_zenith_angle = self.decay_view_angle + src_theta
        self.decay_alt = ret_decay_alt
        self.p_exit = ret_p_exit
        self.dot = ret_dot
       
        
        self.EFIELD_LUT_file_name = "anita_generic_parameterization.npz"
        self.tau_energy = E_t
        self.f_Lo = f_lo
        self.f_High = f_high
        self.Gain_dB = Gain_dB
        self.Nphased = Nphased
        self.R = R
        self.ice = ice
        self.h = h
        self.e_theta = e_theta
        self.e_phi = e_phi 
        self.src_theta = src_theta
        self.src_phi = np.radians(180)
        
        self.noise = 'default'
        self.Vpk_to_Vpkpk_conversion = 1.4
        self.df = 10.# MHz
        self.Epk_to_pk_threshold =  284e-6
        self.T_sys  = 140. # System temperature at the lower part of the ANITA band from elog 119
        self.T_ice  = 270 # Kelvin (water temperature)
        self.kB_W_Hz_K = 1.38064852e-23 # Watts / Hz / K
        self.frac_sky = 0.5 #fraction of sky visisble to antenna
        self.speed_of_light = 299792.458 # km / second
        self.Z_L = 50. # Ohms 50-Ohm characteristic impedance
        self.Z_A = 50. # Ohms antenna impedance
        self.Z_0 = 377. # Ohms impedance of free space
        self.Ieg = 1.0610e-20
        self.Ig  = 2.48e-20
        self.kB = 1.38064852e-23 # Watts / Hz / K
        self.freq_MHz = 300. #MHz
        self.Max_Delta_Theta_View = 4.0
        self.zenith_list = np.array([50, 55, 60, 65, 70, 75, 80, 85, 87, 89])
        self.decay_altitude_list = np.array([0,1,2,3,4,5,6,7,8,9])
        self.zhaires_sim_icethick = 2.0
        self.zhaires_sim_detector_altitude = 37.
        self.e_zhaires_tau_shower = 1e17
        
        self.Peak_Voltage_SNR = np.zeros(self.N)
        self.P_trig = np.zeros(self.N)  

############################################################################       

    def RF_eff_area(self):
        #efield_interpolator_list = self.load_efield_interpolator(self.EFIELD_LUT_file_name) 
        
        x_exit, y_exit, z_exit =  self.coords(self.e_theta, self.e_phi)
        x_exit, y_exit, z_exit = self.R * x_exit, self.R *y_exit, self.R *z_exit
        k_x, k_y, k_z = self.coords(self.src_theta, self.src_phi)
        x_det, y_det, z_det = 0,0,self.R + self.h 
        # print self.exit_decay
#         x_decay, y_decay, z_decay, decay_view_angle, dist_decay_to_detector = self.decay_point_geom_loop(k_x, k_y, k_z, 
#                                                                                                     x_exit, y_exit, z_exit,
#                                                                                                     self.exit_decay, 
#                                                                                                     x_det, y_det, z_det)
        
#         zhs_decay_altitude = self.get_altitude(x_decay, y_decay, z_decay, self.R+self.zhaires_sim_icethick)         
        
#         zenith_angle_decay = self.get_zenith_angle(k_x, k_y, k_z, x_decay, y_decay, z_decay) 
        parm_2d =self.load_efield_parameterization(self.EFIELD_LUT_file_name)
       
 
#         Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
#                                                                                           zhs_decay_altitude,
#                                                                                           np.degrees(zenith_angle_decay),  
#                                                                                           dist_decay_to_detector, 
#                                                                                           np.degrees(decay_view_angle), 
#                                                                                           parm_2d)
      
        Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
                                                                                  self.decay_alt,
                                                                                  self.zenith_angle_deg,
                                                                                  np.degrees(self.decay_view_angle),      
                                                                                  parm_2d)
        
      
       
    #energy, decay_altitude, zenith_exit_deg,distance_shower_to_detector, theta_view, parm_2d
    
    
        Peak_Voltage =  self.E_to_V_signal(Peak_Efield, self.Gain_dB, self.Z_A, self.Z_L, self.Nphased)
        #print np.average(Peak_Voltage)
        
        Noise_Voltage = self.Det_Noise_Voltage()
        
        Peak_Voltage_Threshold = self.E_to_V_signal(self.Epk_to_pk_threshold, self.Gain_dB, 
                                               self.Z_A,self.Z_L,self.Nphased) / self.Vpk_to_Vpkpk_conversion
        
        decay_delta_view_angle = np.abs(Theta_Peak - self.decay_angles)
        
        RF_area = self.trigger(self.Peak_Voltage_SNR, self.Vpk_to_Vpkpk_conversion, Peak_Voltage, 
                               Noise_Voltage,Peak_Voltage_Threshold,self.P_trig, 
                               decay_delta_view_angle, self.Max_Delta_Theta_View)
        
        return RF_area

        
#         self.voltage_test(self.Peak_Voltage_SNR, self.Vpk_to_Vpkpk_conversion, Peak_Voltage, 
#                                Noise_Voltage,Peak_Voltage_Threshold,self.P_trig, 
#                                decay_delta_view_angle, self.Max_Delta_Theta_View)
       
    

####################################################################################################
    def RF_efield_PLOT(self):
        #efield_interpolator_list = self.load_efield_interpolator(self.EFIELD_LUT_file_name) 
        
        x_exit, y_exit, z_exit =  self.coords(self.e_theta, self.e_phi)
        x_exit, y_exit, z_exit = self.R * x_exit, self.R *y_exit, self.R *z_exit
        k_x, k_y, k_z = self.coords(self.src_theta, self.src_phi)
        x_det, y_det, z_det = 0,0,self.R + self.h 
        # print self.exit_decay
#         x_decay, y_decay, z_decay, decay_view_angle, dist_decay_to_detector = self.decay_point_geom_loop(k_x, k_y, k_z, 
#                                                                                                     x_exit, y_exit, z_exit,
#                                                                                                     self.exit_decay, 
#                                                                                                     x_det, y_det, z_det)
        
#         zhs_decay_altitude = self.get_altitude(x_decay, y_decay, z_decay, self.R+self.zhaires_sim_icethick)         
        
#         zenith_angle_decay = self.get_zenith_angle(k_x, k_y, k_z, x_decay, y_decay, z_decay) 
        parm_2d =self.load_efield_parameterization(self.EFIELD_LUT_file_name)
       
 
#         Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
#                                                                                           zhs_decay_altitude,
#                                                                                           np.degrees(zenith_angle_decay),  
#                                                                                           dist_decay_to_detector, 
#                                                                                           np.degrees(decay_view_angle), 
#                                                                                           parm_2d)
      
        #PLOT_decay_alts = np.asarray([np.random.choice(np.linspace(0,9,10)) for x in range(self.N)])
        PLOT_decay_alts = self.decay_alt
        
        Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
                                                                                  PLOT_decay_alts,
                                                                                  self.zenith_angle_deg,
                                                                                  np.degrees(self.decay_view_angle),      
                                                                                  parm_2d)
        
        
            
        return PLOT_decay_alts, Peak_Efield, np.degrees(self.decay_view_angle), self.emg_angles  
    
    def RF_voltage(self):
        #efield_interpolator_list = self.load_efield_interpolator(self.EFIELD_LUT_file_name) 
        
        x_exit, y_exit, z_exit =  self.coords(self.e_theta, self.e_phi)
        x_exit, y_exit, z_exit = self.R * x_exit, self.R *y_exit, self.R *z_exit
        k_x, k_y, k_z = self.coords(self.src_theta, self.src_phi)
        x_det, y_det, z_det = 0,0,self.R + self.h 
        # print self.exit_decay
#         x_decay, y_decay, z_decay, decay_view_angle, dist_decay_to_detector = self.decay_point_geom_loop(k_x, k_y, k_z, 
#                                                                                                     x_exit, y_exit, z_exit,
#                                                                                                     self.exit_decay, 
#                                                                                                     x_det, y_det, z_det)
        
#         zhs_decay_altitude = self.get_altitude(x_decay, y_decay, z_decay, self.R+self.zhaires_sim_icethick)         
        
#         zenith_angle_decay = self.get_zenith_angle(k_x, k_y, k_z, x_decay, y_decay, z_decay) 
        parm_2d =self.load_efield_parameterization(self.EFIELD_LUT_file_name)
       
 
#         Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
#                                                                                           zhs_decay_altitude,
#                                                                                           np.degrees(zenith_angle_decay),  
#                                                                                           dist_decay_to_detector, 
#                                                                                           np.degrees(decay_view_angle), 
#                                                                                           parm_2d)
      
        #PLOT_decay_alts = np.asarray([np.random.choice(np.linspace(0,9,10)) for x in range(self.N)])
        PLOT_decay_alts = self.decay_alt
        
        Peak_Efield, Theta_Peak = self.efield_anita_generic_parameterization_decay_zenith(pow(10, self.tau_energy),
                                                                                  PLOT_decay_alts,
                                                                                  self.zenith_angle_deg,
                                                                                  np.degrees(self.decay_view_angle),      
                                                                                  parm_2d)
        Peak_Voltage =  self.E_to_V_signal(Peak_Efield, self.Gain_dB, self.Z_A, self.Z_L, self.Nphased)
        #print np.average(Peak_Voltage)
        
        Noise_Voltage = self.Det_Noise_Voltage()
        
        Peak_Voltage_Threshold = self.E_to_V_signal(self.Epk_to_pk_threshold, self.Gain_dB, 
                                               self.Z_A,self.Z_L,self.Nphased) / self.Vpk_to_Vpkpk_conversion
        print Theta_Peak
        Peak_Voltage_SNR = self.Vpk_to_Vpkpk_conversion*Peak_Voltage / (2.0 * Noise_Voltage )
        
        decay_delta_view_angle = np.abs(Theta_Peak - np.degrees(self.decay_view_angle))
        
        P_trig = self.P_trig
        for k in range(0,self.N):
        # ANITA defines SNR as = Vpk-pk / (2. * Vrms)
        # Asymmetry of the pulse means that the Vpk-pk != 2 Vpk
        # ZHAireS sims are Vpk, so we have to convert
            Max_Delta_Theta_View = self.Max_Delta_Theta_View
            if( (Peak_Voltage_SNR[k] > Peak_Voltage_Threshold) and (decay_delta_view_angle[k] < Max_Delta_Theta_View)):
                P_trig[k] = 1.
        
        return PLOT_decay_alts, Peak_Voltage_SNR, decay_delta_view_angle, np.degrees(self.decay_view_angle), self.emg_angles, P_trig, Peak_Voltage_Threshold
    
    
    def get_decay_zenith_angle(self, ground_elevation, decay_altitude, X0):
        R_e=self.R
        A = X0
        B = R_e + ground_elevation + decay_altitude
        C = R_e + ground_elevation

        cosZenithDecay = (A**2 + B**2 - C**2) / (2*A*B)
        return 180./np.pi * np.arccos(cosZenithDecay)

    def get_distance_decay_to_detector(self,ground_elevation, 
                                       decay_altitude, detector_altitude,
                                       zenith_decay_deg):
        R_e = self.R
        a = 1
        b = 2*np.cos(zenith_decay_deg*np.pi/180.)*(R_e + ground_elevation + decay_altitude)
        c = (R_e + ground_elevation + decay_altitude)**2 - (R_e + detector_altitude)**2
        d =( -b + np.sqrt(b**2 - 4*a*c) )/(2*a)
        return d

    def get_X0(self, ground_elevation, decay_altitude, zenith_exit_deg):
        R_e = self.R
        a = 1
        b = 2*np.cos(zenith_exit_deg*np.pi/180.)*(R_e + ground_elevation)
        c = (R_e + ground_elevation)**2 - (R_e + ground_elevation + decay_altitude)**2
        X0 =( -b + np.sqrt(b**2 - 4*a*c) )/(2*a)
        return X0

    def get_distance_decay_to_detector_zenith_exit(self, ground_elevation, 
                                   decay_altitude, detector_altitude,
                                   zenith_exit_deg):
        # finds the distance between the two points defined by the
        # decay altitude, detector altitude, and the zenith angle at the point
        ground_elevation = self.R
        if( decay_altitude ==0 ):
            return self.get_X0( ground_elevation, detector_altitude, zenith_exit_deg)

        X0 = self.get_X0(ground_elevation, decay_altitude, zenith_exit_deg)
        zenith_decay_deg = self.get_decay_zenith_angle(ground_elevation, decay_altitude, X0)
        return self.get_distance_decay_to_detector(ground_elevation, decay_altitude, 
                                              detector_altitude, zenith_decay_deg)
    
    def lorentzian_gaussian_background_func(self, psi, E_0, frac_gauss, gauss_peak,
                                        gauss_width, E_1, width2):
        v = (psi - gauss_peak) / gauss_width
        # background gaussian centered at psi=0
        # lorentizian + Gaussian to model the beam (takes care of some asymmetries inside vs. outside cone)
        E_field = E_0*(frac_gauss*np.exp(-v**2/2.)  + (1-frac_gauss)/(1+v**2)) + E_1*np.exp(-psi**2/2./width2**2)

        return E_field

    def coords(self,t_src,phi_src):
        r_x = np.sin(t_src) * np.cos(phi_src)
        r_y = np.sin(t_src) * np.sin(phi_src)
        r_z = np.cos(t_src) 
        return r_x, r_y,r_z
    
    def get_view_angle(self,k_x, k_y, k_z, x_pos, y_pos, z_pos, x_det, y_det, z_det):
        #x_pd = x_pos - x_det
        #y_pd = y_pos - y_det
        #z_pd = z_pos - z_det
        #cos_view_angle = -( k_x * x_pd + k_y * y_pd + k_z * z_pd ) / np.sqrt(x_pd**2 + y_pd**2 + z_pd**2)
        #return np.arccos(cos_view_angle) # radians
        x_pd = x_det - x_pos
        y_pd = y_det - y_pos
        z_pd = z_det - z_pos
        cos_view_angle = ( k_x * x_pd + k_y * y_pd + k_z * z_pd ) / np.sqrt(x_pd**2 + y_pd**2 + z_pd**2)
        return np.arccos(cos_view_angle) # radians

    def get_distance_to_detector(self,x_pos, y_pos, z_pos, x_det, y_det, z_det):
        x_pd = x_pos - x_det
        y_pd = y_pos - y_det
        z_pd = z_pos - z_det

        return np.sqrt( x_pd**2 + y_pd**2 + z_pd**2 )

    def decay_point_geom(self, k_x, k_y, k_z, x_exit, y_exit, z_exit, x_det, y_det, z_det, distance_exit):
        # Using euler angles is much cleaner.
        x_decay = distance_exit * k_x + x_exit
        y_decay = distance_exit * k_y + y_exit
        z_decay = distance_exit * k_z + z_exit
        
        decay_view_angle = self.get_view_angle(k_x, k_y, k_z, x_decay, y_decay, z_decay, x_det, y_det, z_det)
        decay_dist_to_detector = self.get_distance_to_detector(x_decay, y_decay, z_decay, x_det, y_det, z_det)

        return x_decay, y_decay, z_decay, decay_view_angle, decay_dist_to_detector
        
    
    def decay_point_geom_loop(self, k_x, k_y, k_z, x_exit, y_exit, z_exit,rho, x_det, y_det,z_det):
        x_decay= np.zeros(self.N)
        y_decay= np.zeros(self.N)
        z_decay= np.zeros(self.N)
        decay_view_angle= np.zeros(self.N)
        dist_decay_to_detector= np.zeros(self.N)
        for k in range(self.N):
            x_decay[k], y_decay[k], z_decay[k], decay_view_angle[k], dist_decay_to_detector[k] = self.decay_point_geom(k_x, k_y, k_z, x_exit[k], y_exit[k], z_exit[k], x_det, y_det, z_det, self.exit_obs[k])
        return x_decay, y_decay, z_decay, decay_view_angle, dist_decay_to_detector
    
    def get_zenith_angle(self,k_x, k_y, k_z, x, y, z):
        # finds the zenith angle relative to the Earth normal for a vector
        # defined by a Cartesian point (x,y,z) and it's propagation direction (k-hat)
        r = np.sqrt(x*x + y*y + z*z)
        # the zenith angle is the angle between n-hat (the normal to the surface)
        # and the propagation direction of the shower k-hat
        cos_zenith = k_x * x / r + k_y * y / r + k_z * z / r
        return np.arccos(cos_zenith) # radians

    def get_altitude(self,x, y, z, ground_altitude):
       # finds the altitude of the decay point, given a ground altitude relative to the center of the
       # Earth and the decay position in Cartesian coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        return r - ground_altitude

    def load_efield_parameterization(self,EFIELD_file_name):
        npzfile = np.load(EFIELD_file_name)
        parm_2d = npzfile['parm_2d']
        return parm_2d

    def find_nearest(self,array, values, lower_bound=None, upper_bound=None):
        # finds the nearest values in the arrays
        # if the values are outside the desire range, sets the index to -1
        values = np.atleast_1d(values)
        indices = np.abs(np.int64(np.subtract.outer(array, values))).argmin(0)
        out = array[indices]
        if( lower_bound != None):
            bound_ind = np.where(values < lower_bound)
            indices[bound_ind] = -1
        if( upper_bound != None):
            bound_ind = np.where(values > upper_bound)
            indices[bound_ind] = -1
        return indices

#     def efield_anita_generic_parameterization_decay_zenith(self, energy, decay_altitude, zenith_exit_deg,
#                                                            distance_shower_to_detector, theta_view, parm_2d):
    def efield_anita_generic_parameterization_decay_zenith(self, energy, decay_altitude, zenith_exit_deg,
                                                           theta_view, parm_2d):
        
        #MAIN
        # Parameterization for a 10^17 eV tau shower at zenith angle of 60deg / emergence angle 30deg
        # ground elevation of 3 km, detector altitude of 37 km
        #
        # The first row in the array sis for decay altitude of 0 km; the last is 9 km.
        # these are the simulation arrays used to generate the parameterizations
        zenith_list = self.zenith_list
        decay_altitude_list = self.decay_altitude_list

        escaled = np.zeros(len(energy))
        theta_peak = np.zeros(len(energy))
        
        zhaires_decay_obs = self.decay_dist_det()
        for i in range(len(theta_view)):
            if decay_altitude[i] >= 0: # if the decay altitude < 0, leave the electric field at 0
                # find the nearest neighbor for both the zenith angle at the exit point and the decay altitude
                i_ze = self.find_nearest(zenith_list, zenith_exit_deg[i])[0]
                i_d  = self.find_nearest(decay_altitude_list, decay_altitude[i], lower_bound = 0)[0]
                
#                 self.zenith_list = np.array([50, 55, 60, 65, 70, 75, 80, 85, 87, 89])
#                 self.decay_altitude_list = np.array([0,1,2,3,4,5,6,7,8,9])
                
                # if the decay altitude is < 0, then throw this event out
                if i_d >= 0:
                    nearest_zenith_angle = zenith_list[i_ze]
                    nearest_decay_altitude = decay_altitude_list[i_d]
                    #print zenith_exit_deg[i], nearest_zenith_angle
                    
                    parms = parm_2d[i_ze, i_d]

                    epeak = self.lorentzian_gaussian_background_func(theta_view[i], *parms)
                    #epeak = parms[0]
                    # Distance from the shower to the detector for the parameterized LDFs 
                    # at different decay altitudes and decay zenith angles
                   
#                     r_zhaires_tau_shower = self.get_distance_decay_to_detector_zenith_exit(self.zhaires_sim_icethick,
#                                                                                       nearest_decay_altitude,
#                                                                                       self.zhaires_sim_detector_altitude,
#                                                                                       nearest_zenith_angle)
                    #dist_ratio = r_zhaires_tau_shower / distance_shower_to_detector[i]                   
                    
                    #print epeak
            
                    dist_ratio_calc = zhaires_decay_obs[i]/self.decay_obs[i] #self.decay_obs[i]/self.exit_obs[i]
                    #print  dist_ratio_calc 
                    escaled[i] = epeak * (energy[i] / self.e_zhaires_tau_shower) * (dist_ratio_calc)
                    theta_peak[i] = parms[2]
        return escaled, theta_peak
    
    def decay_dist_det(self):
        earth_theta = self.e_theta
        earth_phi = self.e_phi
        d = self.exit_decay
        h = self.h
        R = self.R +2
        exit = np.radians(self.zenith_angle_deg)
        
        obs_x = 0
        obs_y = 0
        h_decay = d * np.cos(exit)
        h_eff = h - h_decay
        obs_z = R+h_eff
        
        chord = (R*np.sin(earth_theta) - d*np.sin(exit)) 
        earth_theta_eff = np.arcsin(chord /R)
        e_x_eff = R*np.sin(earth_theta_eff) * np.cos(earth_phi)
        e_y_eff = R*np.sin(earth_theta_eff) * np.sin(earth_phi)
        e_z_eff = R*np.cos(earth_theta_eff) 
        
        point_to_obs_x = obs_x - e_x_eff
        point_to_obs_y = obs_y - e_y_eff
        point_to_obs_z = obs_z - e_z_eff
        
        norm = np.sqrt(point_to_obs_x**2 + point_to_obs_y**2 + point_to_obs_z**2)
        return norm

    def load_efield_interpolator(self,EFIELD_LUT_file_name):
        # the interpolator is for 10-MHz subbands and 
        # is called as interpolator(]zenith_angle, starting_frequency,psi_angle)
        # zenith angle is the shower zenith angle in deg.
        # staring_frequency is the lowest frequency in the band in MHz
        # psi_angle is the angel off the shower axis, 
        # equivalent to view angle in deg.
        interp_file = np.load(EFIELD_LUT_file_name)
        return interp_file['efield_interpolator_list'][()]

    def Voltage_interp(self,efield_interpolator_list, zenith_angle_deg, view_angle_deg,
                   log10_tau_energy, distance_exit_km, distance_decay_km):
        # Lorentzian beam pattern based on 10-MHz filtered subbands of Harm's results
        # Returns electric field peak in V/m

        # Since the efields are stored in 10-MHz subbands
        # integrate over the range from f_Lo to f_High in 10-MHz bands
        Voltage = np.zeros(self.N)
        #E_field = 0.
        # TODO: Right now forcing the parameters outside the interpolation range to the edges
        # shoudl replace with extrapolation
        z = zenith_angle_deg.copy()
        v = view_angle_deg.copy()
        z[z>89.] = 89.
        z[z<55.] = 55.
        v[v<0.04] = 0.04
        v[v>3.16] = 3.16

        for freq in np.arange(self.f_Lo, self.f_High, self.df):
            i_f_Lo = int(round(freq / self.df - 1))
        # using the average frequency in the bin to calculate the voltage
        Voltage += E_to_V_signal(efield_interpolator_list[i_f_Lo](z,v), self.Gain_dB, 
                                 (freq+self.df)/2., self.Z_A, self.Z_L, self.Nphased)
        # account for ZHAIReS sims only extending to 3.16 deg 
        Voltage[view_angle_deg>self.view_cut] = Voltage[view_angle_deg>self.view_cut]*np.exp( -(view_angle_deg[view_angle_deg>self.view_cut]-0.)**2 / (2*3.16)**2)

        Voltage *= self.distance_exit_km/self.distance_decay_km   # distance to tau decay point correction
        Voltage *= 10**(log10_tau_energy - 17.) # Energy scaling
        return Voltage


    def E_to_V_signal(self,E_pk, Gain_dB, freq_MHz, Z_A, Z_L, Nphased=1):
        # Derived assuming:
        # P_r   : the power received at the antenna P_r = A P_inc = |V_A|^2/(8 R_A) = V_A  Z_L / (Z_L + Z_A) 
        # Note: V_A = 2 V_L for a perfectly matched antenna -- which we assume here
        # P_inc : incident power at the antenna P_inc = |E|^2 / (2 Z_0) 
        # A : antenna aperture = \lambda^2/(4pi) G eff_load eff_pol 
        # eff_load: load mismatch factor eff_load = (1 - Gamma^2), where Gamma is the reflection coefficient 
        # = (Z_L - Z_A*)/(Z_L + A_Z); assuming that we have a perfect match
        # eff_pol : polarization mismatch factor; assuming that this is built into ZHAireS pulses
        # calculate the reflection coefficient
        Gamma = self.reflection_coefficient()
        eff_load = 1. - np.abs(Gamma)**2

        # Radiation resistance of the antennas
        R_A = np.real(Z_A)

        #print Gamma, R_A, Z_L / (Z_A + Z_L), eff_load, Z_A, Z_L 
        V_A = 2. * E_pk * (self.speed_of_light*1.e3)/(freq_MHz * 1.e6) * np.sqrt(R_A/self.Z_0 * pow(10., Gain_dB/10.)/4./np.pi * eff_load) * Nphased
        
        V_L = V_A * Z_L / (Z_A + Z_L) # V_L = 1/2 * V_A for a perfectly matched antenna
        return V_L

    def galactic_temperature(self,f_MHz):
        # Dulk 2001
        nu = f_MHz # Hz
        tau = 5.0 * pow(nu, -2.1)
        # Iv in  W/m^2/Hz/sr
        Iv = self.Ig * pow(nu, -0.52) * (1-np.exp(-tau))/tau + self.Ieg * pow(nu, -0.80) * np.exp(-tau)
        c = self.speed_of_light * 1e3 # m/s
        temp = Iv * c**2 / (2*(nu*1e6)**2)/self.kB
        ## IV is the intensity
        ## temp is the galactic noise temperature in Kelvin
        return Iv, temp # W/m^2/Hz/sr, K

    def reflection_coefficient(self):
        # Z_A is the antenna impedance
        # Z_L is the load impedance characteristic to the system
        return (self.Z_A - np.conj(self.Z_L))/(self.Z_A + self.Z_L)
 
    def noise_voltage(self):
        freq_min_MHz = self.f_Lo
        freq_max_MHz = self.f_High
        # Note that Gain_dB isn't used here because the noise is integrated over the full beam of the antenna
        # TODO: Update with more detailed antenna model

        # calculate the reflection coefficient
        Gamma = self.reflection_coefficient()
        eff_load = 1. - np.abs(Gamma)**2

        # Radiation resistance of the antennas
        R_A = np.real(self.Z_A)

        # this is in V^2/Hz
        T_gal = self.galactic_temperature(np.arange(freq_min_MHz, freq_max_MHz + self.df, self.df))[1] 
        gal_noise = np.sqrt(self.Nphased * np.trapz(T_gal) * self.frac_sky * self.df 
                            * 1e6 * self.kB_W_Hz_K * R_A * eff_load  )
        sys_noise = np.sqrt(self.Nphased * (self.T_ice*(1.-self.frac_sky) * eff_load + self.T_sys) * 
                            (freq_max_MHz - freq_min_MHz) * self.kB_W_Hz_K * self.Z_L  ) 

        # assuming we're phasing after the amplifier rather than before.
        # if before, then the system temperature would also decrease as 1/N 
        # note that temperature ~ power, so this is decreasing the noise voltage by 1/sqrt(N)
        # as you would expect from incoherent noise
        combined_temp = self.Nphased*(T_gal*self.frac_sky * eff_load + self.T_ice*(1.-self.frac_sky) * eff_load + self.T_sys)
        
        # this is in V (rms)
        combined_noise = np.sqrt( np.trapz(combined_temp) * self.df * 1e6 * self.kB_W_Hz_K * self.Z_L  )
        return combined_noise, gal_noise, sys_noise
    
    def Det_Noise_Voltage(self):
        noises = self.noise_voltage() 
        if(self.noise == 'sys'):
            Noise_Voltage = noises[2]
        elif(self.noise == 'gal'):
            Noise_Voltage = noises[1]
        else: # default is the combination
            Noise_Voltage = noises[0]
        return Noise_Voltage

    def trigger(self, Peak_Voltage_SNR, Vpk_to_Vpkpk_conversion, Peak_Voltage, Noise_Voltage,Peak_Voltage_Threshold, P_trig, decay_delta_view_angle, Max_Delta_Theta_View):
        sum_P_trig = 0
        for k in range(0,self.N):
        # ANITA defines SNR as = Vpk-pk / (2. * Vrms)
        # Asymmetry of the pulse means that the Vpk-pk != 2 Vpk
        # ZHAireS sims are Vpk, so we have to convert
            Peak_Voltage_SNR[k] = Vpk_to_Vpkpk_conversion*Peak_Voltage[k] / (2.0 * Noise_Voltage )
            #if (Peak_Voltage[k] > Peak_Voltage_Threshold):
            if( (Peak_Voltage[k] > Peak_Voltage_Threshold) and (decay_delta_view_angle[k] < Max_Delta_Theta_View)):
                P_trig[k] = 1.
            sum_P_trig  += P_trig[k]

        self.A_trig = self.A0 *1./float(self.N0) *  np.sum(P_trig * self.p_exit* self.dot * (np.radians(self.view_angle_deg) < self.view_cut) * (self.dot>0.) ) 
        return self.A_trig
    
    def voltage_test(self, Peak_Voltage_SNR, Vpk_to_Vpkpk_conversion, Peak_Voltage, Noise_Voltage,Peak_Voltage_Threshold, P_trig, decay_delta_view_angle, Max_Delta_Theta_View):
        for k in range(0,self.N):
            Peak_Voltage_SNR[k] = Vpk_to_Vpkpk_conversion*Peak_Voltage[k] / (2.0 * Noise_Voltage )

        cutoff = Peak_Voltage_Threshold
        nan_num = sum(np.isnan(Peak_Voltage_SNR))
        num_num = len(Peak_Voltage_SNR) - nan_num
        print num_num,nan_num
        Peak_Voltage_SNR_num = Peak_Voltage_SNR[~np.isnan(Peak_Voltage_SNR)]
        avg = np.average(Peak_Voltage_SNR_num)
       
    ##########
        fig, ax = plt.subplots(num=1,figsize=(8,8))
        
        plt.xlabel("Voltage", fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.axvline(cutoff)
        n, bins, patches = plt.hist(x=Peak_Voltage_SNR_num, bins='auto', color='#0504aa',
                           rwidth=0.85)
        plt.title("ANITA RF Electric Field Voltages "+str(round(np.degrees(self.src_theta)))+r'$^{\circ}$', fontsize=16)
        #ax.get_xaxis().set_major_formatter(mpl.ticker.StrMethodFormatter('{:.3g}'))
        #plt.figtext(.5,-0.1,"Voltage Cutoff "+'{:.3g}'.format(cutoff), fontsize=12, ha='center')
        plt.figtext(.5,-0.1,"Voltage Cutoff "+str(round(cutoff,6)), fontsize=12, ha='center')
        # plt.semilogx()

       
        