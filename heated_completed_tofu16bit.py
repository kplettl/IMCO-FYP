# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:29:42 2019

@author: plettlk
"""
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath
from PARRECread import PARRECread
from skimage.restoration import unwrap_phase
import pyqtgraph as pg



sonalleveNativeFile = abspath(    
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_Native.PAR"
        )

sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)

phaseMaps=np.zeros_like(phaseMapsOriginal)
#
#
for i in range(9,26):
    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
  
phaseMapsCropped = phaseMaps[9:26,42:106, 39:103] 

phaseMapsCropped_recentered_tofu_heated = np.zeros_like(phaseMapsCropped)

for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_tofu_heated[i] = pmc-np.average(pmc[30:34,30:34])
pg.image(phaseMapsCropped_recentered_tofu_heated)


alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3

completed_images = []
original_images = []


phaseMapsCroppedTest = phaseMapsCropped_recentered_tofu_heated
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**16-1) - 0
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\111219\16bit_tofu_test_set_recentered_heated\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    
completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-16bit-tofu-100epochs-heated-recentered\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_100 = np.array(completed_images)
    
unscaled_completed_tofu_heated = (((completed_images_array_100 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_16bit = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

deltaT100_tofu = np.zeros_like(unscaled_completed_tofu_heated)

average_peak_temp = []
peak_temp_STD = []
for i in range(1,17):
    deltaT100_tofu[i] = ((unscaled_GT_16bit[i] - unscaled_completed_tofu_heated[i])/(B0*alpha*gamma*echoTime))
    average_peak_temp.append(np.average(deltaT100_tofu[i, 37:38, 39:41]))
    peak_temp_STD.append(np.std(deltaT100_tofu[i, 37:38, 39:41]))
    
average_peak_temp = np.array(average_peak_temp)
peak_temp_STD = np.array(peak_temp_STD)
#pg.image(deltaT100_tofu)

sonalleveTMapFile = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_TMap.PAR"
            )

sonalleveTMapData,Params,Dims = PARRECread(sonalleveTMapFile)
#
TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
TMapsCropped = TMaps[10:26,42:106, 39:103] - 37

tags = Params['tags']
times = tags[:,31][::4] - tags[:,31][0]

HIFUaverage_peak_temp = []
HIFUpeak_temp_STD = []
for i in range(0,16):
    
    HIFUaverage_peak_temp.append(np.average(TMapsCropped[i, 37:38, 39:41]))
    HIFUpeak_temp_STD.append(np.std(TMapsCropped[i, 37:38, 39:41]))
    
HIFUaverage_peak_temp = np.array(HIFUaverage_peak_temp)
HIFUpeak_temp_STD = np.array(HIFUpeak_temp_STD)

time = np.linspace(0, 15, 16)


plt.figure()
plt.errorbar(times, average_peak_temp, yerr=peak_temp_STD, capsize=5, label = 'Completed', marker = '.', markersize=6, color='black')
plt.errorbar(times, HIFUaverage_peak_temp, yerr=HIFUpeak_temp_STD, capsize=5, label ='HIFU Software', marker = '.', markersize=6, color='xkcd:dark orange', linestyle=':')
plt.xlabel('Time after Starting HIFU (s)', size=18)
plt.ylabel('∆T(˚C)', size=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Average Temperature of Sonicated Region over Time, 16-bit', size=18)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.axvline(x=36, linestyle='--', color='gray', linewidth=0.7)
plt.legend(fontsize=18)
plt.show()

