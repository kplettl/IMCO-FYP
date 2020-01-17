# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:06:23 2019

@author: plettlk
"""

from os.path import abspath
#from PARRECread import PARRECread
import imageio
from skimage.restoration import unwrap_phase
import pyqtgraph as pg
import os
import numpy as np
import cv2
from skimage.util import img_as_ubyte, img_as_uint
from skimage import exposure
import numpngw 

#sonalleveNativeFile = abspath(      
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-1_2019-10-11_11-14-17_Native.PAR",
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-2_2019-10-11_11-46-22_Native.PAR",
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-3_2019-10-11_12-20-01_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\preSonication-with-motion-artefacts_2019-11-12_13-45-31_Native.PAR"
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-baseline-scan_2019-10-14_14-17-13_Native.PAR"
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-baseline-scan-2_2019-10-14_15-15-00_Native.PAR"
#            )
#sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)
##
#phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
#MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)
#pg.image(MMaps, title='phase')
#
##phaseMaps = unwrap_phase(phaseMaps,seed=100)
##pg.image(MMaps, title='phase')
#
#phaseMaps=np.zeros_like(phaseMapsOriginal)

''' tofu phantom '''
#for i in range(0,203):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#    
#phaseMapsCropped1 = phaseMaps[0:203, 42:106, 39:103]
#MMapscropped = MMaps[0:203, 42:106, 37:101]
#pg.image(MMapscropped)
#for i in range(0,709):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#phaseMapsCropped2 = phaseMaps[0:709, 42:106, 39:103]
#MMapscropped = MMaps[0:709, 42:106, 37:101]
#pg.image(MMapscropped)
#for i in range(0,771):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#phaseMapsCropped3 = phaseMaps[0:771,  42:106, 39:103]
###MMapscropped = MMaps[0:709, 42:106, 37:101]
###pg.image(MMapscropped)
###
##
#phaseMapsCropped = np.vstack((phaseMapsCropped1, phaseMapsCropped2, phaseMapsCropped3))
#phaseMapsCropped_recentered_tofu = np.zeros_like(phaseMapsCropped)
#for i,pmc in enumerate(phaseMapsCropped):
#    phaseMapsCropped_recentered_tofu[i] = pmc-np.average(pmc[30:34,30:34])
#pg.image(phaseMapsCropped_recentered_tofu)


''' QA  motion '''
#for i in range(113,361):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#    
#phaseMapsCropped = phaseMaps[113:361, 49:113, 50:114]
#MMapsCropped = MMaps[113:361, 49:113, 50:114]
#
#pg.image(MMapsCropped, title='phase')
#phaseMapsCropped_recentered_QA_motion = np.zeros_like(phaseMapsCropped)
#for i,pmc in enumerate(phaseMapsCropped):
#    phaseMapsCropped_recentered_QA_motion[i] = pmc-np.average(pmc[18:22,30:34])
#pg.image(phaseMapsCropped_recentered_QA_motion)

'''QA stationary '''
#for i in range(0,693):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#
#phaseMapsCropped4 = phaseMaps[0:693,49:113, 50:114]

#for i,pmc in enumerate(phaseMapsCropped):
#    phaseMapsCropped_recentered[i] = pmc-np.average(pmc[30:34,30:34])
#for i in range(728,1429):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#
#phaseMapsCropped5 = phaseMaps[728:1429,49:113, 50:114]
##MMapsCropped = MMaps[0:693,49:113, 50:114]
#phaseMapsCropped  = np.vstack((phaseMapsCropped4, phaseMapsCropped5))
#
#pg.image(phaseMapsCropped)
#phaseMapsCropped_recentered_QA = np.zeros_like(phaseMapsCropped)
#
#for i,pmc in enumerate(phaseMapsCropped):
#    phaseMapsCropped_recentered_QA[i] = pmc-np.average(pmc[18:22,30:34])
#pg.image(phaseMapsCropped_recentered_QA)


#%% 

#scale imgs to [0,1] and then to pixel values 
rescaled = exposure.rescale_intensity(phaseMapsCropped_recentered_QA_motion, in_range=(np.min(phaseMapsCropped_recentered_QA_motion), np.max(phaseMapsCropped_recentered_QA_motion)), out_range = (0, 1))

rescaled_ = img_as_ubyte(rescaled) #8 bit
#rescaled_ = img_as_uint(rescaled) # 16 bit 

directory = r"C:\Users\plettlk\DCGAN_Image_Completion\120120\8bit_QA_baseline_motion_train\\"


def make_dir(name):
    # Works on python 2.7, where exist_ok arg to makedirs isn't available.
    p = os.path.join(name)
    if not os.path.exists(p):
        os.makedirs(p)
        
make_dir(directory)

#save imgs to directory 
for i in range(rescaled_.shape[0]):
    stack = np.stack((rescaled_[i,:,:],)*3, axis=-1)
    numpngw.write_png(directory + 'scaled_phase_image_' + str(i) +'.png', stack)

