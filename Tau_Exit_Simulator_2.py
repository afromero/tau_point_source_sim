#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:36:45 2019

@author: romerowo
"""

import numpy as np

class Tau_Exit_Simulator:
    def __init__(self,LUT_file_name):
        self.load_tau_LUTs(LUT_file_name)


    def load_tau_LUTs(self, LUT_file_name):
        f = np.load(LUT_file_name)
        data_array         = f['data_array']
        th_exit_array      = f['th_exit_array']
        # pathlength_m_array = f['pathlength_m_array']
        num_sim            = f['num_sim']
        f.close()
        P_exit = []
        for k in range(0, len(th_exit_array)):
            P_exit.append(float(len(data_array[k]))/float(num_sim[k]))
        self.P_exit               = np.array(P_exit)
        self.th_exit              = np.array(90. + th_exit_array)
        #self.pathlength_m   = np.array(pathlength_m_array)
        self.data_array = np.array(data_array)

    def get_unit_interval(self, arr):
        return (np.cumsum(np.ones(len(arr)))-1.)/float(len(arr)-1)
    
    def lin_interp(self, x, x1, x2, y1, y2):
        m = (y2-y1)/(x2-x1)
        b = y2 - m*x2
        return m*x + b 
        
    def get_closest_indices(self, _val, _array):
        idx = np.argmin(np.abs(_array-_val))
        idx_lo = idx
        idx_hi = idx+1
        if _array[idx_hi] > _array[idx_lo]:
            if _val>_array[idx]:
                idx_lo = idx
                idx_hi = idx+1
            if _val<=_array[idx]:
                idx_lo = idx-1
                idx_hi = idx
        if _array[idx_hi] < _array[idx_lo]:
            if _val<_array[idx]:
                idx_lo = idx+1
                idx_hi = idx
            if _val>=_array[idx]:
                idx_lo = idx
                idx_hi = idx-1
                
        return idx_lo, idx_hi
        
    def sample_array_energies(self, u, data_index):
        if len(self.data_array[data_index]) == 0: return 0.
        N = float(len(self.data_array[data_index])-1.)
        while N<0:
            N+=1.
        #print 'N', N
        idx    = u*N
        idx_lo = np.rint(idx)
        while idx_lo>idx: 
            idx_lo-=1.
        while idx_lo>= N:
            idx_lo -= 1.
        if idx_lo<0: idx_lo = 0.
        idx_hi = idx_lo + 1.
        #print '\t', idx, idx_lo, idx_hi
        if len(self.data_array[data_index])>=2:
            try:
                val =self.lin_interp(idx, idx_lo, idx_hi, 
                                       self.data_array[data_index][int(idx_lo)], 
                                       self.data_array[data_index][int(idx_hi)])
            except:
                print '!!!! ERROR!!!!! in sample_array_energies(self, u, data_index)'
                print 'data_index', data_index
                print 'u', u
                print 'N', N
                print 'idx', idx
                print 'idx_lo', idx_lo
                print 'idx_hi', idx_hi
                print 'len(self.data_array[data_index])', len(self.data_array[data_index])
                print 'self.data_array[data_index]', self.data_array[data_index]
                print 'self.data_array[data_index][int(idx_lo)]', self.data_array[data_index][int(idx_lo)]
                print 'self.data_array[data_index][int(idx_hi)]', self.data_array[data_index][int(idx_hi)]
                print 'returning self.data_array[0] = ', self.data_array[0]
                #val = 0.
                val = self.data_array[0]
            #print idx, idx_lo, idx_hi, val, self.data_array[data_index][int(idx_lo)], self.data_array[data_index][int(idx_hi)]
        if len(self.data_array[data_index])==1:
            print '\n!!!! WARNING !!!!'
            print 'len(self.data_array[data_index])=',len(self.data_array[data_index])
            print 'self.data_array[data_index] = ', self.data_array[data_index]
            print 'returning self.data_array[data_index][0] = ', self.data_array[data_index][0]
            val = self.data_array[data_index][0]
        return val
    
#     def P_exit_pathlength(self, path_len_m_val):
#         # find closest indices
#         idx_lo, idx_hi = self.get_closest_indices(path_len_m_val, self.pathlength_m)
#         P_lo = self.P_exit[idx_lo]
#         P_hi = self.P_exit[idx_hi]
#         val =self.lin_interp(path_len_m_val, 
#                              self.pathlength_m[idx_lo], 
#                              self.pathlength_m[idx_hi],
#                              P_lo,
#                              P_hi)
#         return val

#     def sample_energies_pathlength(self, path_len_m_val):
#         # find closest indices
#         idx_lo, idx_hi = self.get_closest_indices(path_len_m_val, self.pathlength_m)
#         while idx_lo<0:
#             idx_lo+=1
#             idx_hi = idx_lo + 1

#         u = np.random.uniform(0., 1.)
#         E_lo = self.sample_array_energies(u, idx_lo)
#         E_hi = self.sample_array_energies(u, idx_hi)
#         val =self.lin_interp(path_len_m_val, 
#                              self.pathlength_m[idx_lo], 
#                              self.pathlength_m[idx_hi],
#                              E_lo,
#                              E_hi)
#         return val

    def P_exit_th_exit(self, th_exit_val):
        # find closest indices
        idx_lo, idx_hi = self.get_closest_indices(th_exit_val, self.th_exit)
        P_lo = self.P_exit[idx_lo]
        P_hi = self.P_exit[idx_hi]
        val =self.lin_interp(th_exit_val, 
                             self.th_exit[idx_lo], 
                             self.th_exit[idx_hi],
                             P_lo,
                             P_hi)
        return val
    def sample_energies_th_exit(self, th_exit_val):
        # find closest indices
        idx_lo, idx_hi = self.get_closest_indices(th_exit_val, self.th_exit)
        #print '*', th_exit_val, self.th_exit[idx_lo], self.th_exit[idx_hi]
        u = np.random.uniform(0., 1.)
        E_lo = self.sample_array_energies(u, idx_lo)
        E_hi = self.sample_array_energies(u, idx_hi)
        val =self.lin_interp(th_exit_val, 
                             self.th_exit[idx_lo], 
                             self.th_exit[idx_hi],
                             E_lo,
                             E_hi)
        return val
        
    #def get_P_exit(th_exit):
        # get closest index

        
    #def sample_tau_energy():