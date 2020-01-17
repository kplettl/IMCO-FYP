# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 01:08:29 2020

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:54:23 2020

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:17:12 2019

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:38:35 2019

@author: plettlk
"""

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


alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3

sonalleveNativeFile = abspath(    
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-w-motion-artefact_2019-11-12_14-07-50_Native.PAR",
        )
sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)

#pg.image(MMaps, title='mag')
##
phaseMaps=np.zeros_like(phaseMapsOriginal)
 
for i in range(0,17):
    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
    
phaseMapsCropped = phaseMaps[0:17, 49:113, 50:114]
MMapsCropped = MMaps[0:17, 49:113, 50:114]
#
#pg.image(MMapsCropped, title='phase')
#
phaseMapsCropped_recentered_QA_motion = np.zeros_like(phaseMapsCropped)

for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_QA_motion[i] = pmc-np.average(pmc[18:22,30:34])
pg.image(phaseMapsCropped_recentered_QA_motion)
#    
#%%
completed_images = []
original_images = []

phaseMapsCroppedTest = phaseMapsCropped_recentered_QA_motion
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**8-1) - 0
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\120120\8bit_QA_test_30W_heating_w_motion\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)

completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bitQA-motion-30W-592epochs-120120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_592 = np.array(completed_images)
    

unscaled_completed_motion = (((completed_images_array_592 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_motion = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

deltaT_motion = np.zeros_like(unscaled_completed_motion)

average_peak_temp_motion = []
peak_temp_STD_motion = []
for i in range(0,16):
    deltaT_motion[i] = ((unscaled_GT_motion[i] - unscaled_completed_motion[i])/(B0*alpha*gamma*echoTime))
    average_peak_temp_motion.append(np.average(deltaT_motion[i, 28:33, 26:30]))
    peak_temp_STD_motion.append(np.std(deltaT_motion[i, 28:33, 26:30]))
#    
average_peak_temp_motion = np.array(average_peak_temp_motion)
peak_temp_STD_motion = np.array(peak_temp_STD_motion)
pg.image(deltaT_motion)

sonalleveTMapFile = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-w-motion-artefact_2019-11-12_14-07-50_TMap.PAR"
            )

sonalleveTMapData,Params,Dims = PARRECread(sonalleveTMapFile)
#
TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
TMapsCropped = TMaps[1:17,49:113, 50:114] - 37
#pg.image(TMapsCropped)
tags = Params['tags']
times = tags[:,31][::4] - tags[:,31][0]

#HIFUaverage_peak_temp = []
#HIFUpeak_temp_STD = []
#for i in range(0,16):
#    
#    HIFUaverage_peak_temp.append(np.average(TMapsCropped[i,30:34, 28:32]))
#    HIFUpeak_temp_STD.append(np.std(TMapsCropped[i, 30:34, 28:32]))
#    
#HIFUaverage_peak_temp = np.array(HIFUaverage_peak_temp)
#HIFUpeak_temp_STD = np.array(HIFUpeak_temp_STD)

#time = np.linspace(0, 15, 16)
#
#plt.figure()
#plt.errorbar(times, average_peak_temp_motion, yerr=peak_temp_STD_motion, capsize=5, label = 'IMCO', marker = '.', markersize=6, color='black')
#plt.errorbar(times, HIFUaverage_peak_temp_motion, yerr=HIFUpeak_temp_STD_motion, capsize=5, label ='SBL', marker = '^', markersize=6, color='xkcd:dark orange', linestyle=':')
#plt.errorbar(times, average_peak_temp_rless_motion, yerr=peak_temp_STD_rless_motion, capsize=5, label ='RLESS', marker = 's', markersize=6, color='xkcd:steel blue', linestyle='--')
#
#plt.xlabel('Time after Starting HIFU (s)', size=18)
#plt.ylabel('∆T(˚C)', size=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.title('Average Temperature Difference of Sonicated Region of \n QA Phantom over Time with Motion Artefact,\n 30W Sonication', size=18)
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.axvline(x=36, linestyle='--', color='gray', linewidth=0.7)
#plt.legend(fontsize=18)
#plt.show()

Avg_temp_sonalleve_motion = np.average(HIFUaverage_peak_temp)
Sd_temp_sonalleve_motion = np.std(HIFUaverage_peak_temp)

Avg_temp_comp_motion = np.average(average_peak_temp_motion)
Sd_temp_comp_motion = np.std(average_peak_temp_motion)

#
#
#MMaps = np.fliplr(MMaps)
#
#
#TMaps_masked_rless = np.ma.masked_where(innerDeltaTMap < 1, innerDeltaTMap)
#
#TMaps_masked_completed = np.ma.masked_where(deltaT_motion < 1, deltaT_motion)
#MMaps_cropped = MMaps[0:17,49:113, 50:114]
#TMaps_cropped_rless = TMaps_masked_rless[:,49:113, 50:114]
#
#cmap = cm.get_cmap('plasma')
#vmax = 14
#fig=plt.figure(figsize=(8,6))
#ax3 = fig.add_subplot(334,adjustable='box', aspect='auto')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[1],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped_rless[1],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(335,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[7],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(TMaps_cropped_rless[7],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=vmax,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(336,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped_rless[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#plt.axis('off')
#plt.show()
#
#print('Images acquired at: {:.2f} s, {:.2f}s, {:.2f}s'.format(times[1], times[7], times[15]))
#
#sonalleveTMapFile = sonalleveNativeFile.replace('Native','TMap')
#    
#
#
## read temperature data from PARREC files
#sonalleveTMapData,tmapParams,tmapDims = PARRECread(sonalleveTMapFile)
#
#
## reshape data to be displayed by pyqtgraph
#GT_TMaps = sonalleveTMapData[:,:,0,0,0,0,1:].swapaxes(0,2)
#GT_TMaps = GT_TMaps[0:16,:,:] - 37
#GT_TMaps_masked = np.ma.masked_where(GT_TMaps < 1, GT_TMaps)
#
#GT_TMaps_cropped = GT_TMaps_masked[:,49:113, 50:114]
#
#GTtags = tmapParams['tags']
#GTtime = GTtags[:,31][::4]
#GTtimes = GTtime - GTtime[0]
#
#ax3 = fig.add_subplot(331,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[1],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[1],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#\ax3 = fig.add_subplot(332,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[7],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(GT_TMaps_cropped[7],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=vmax,cmap = cmap)
#\ax3 = fig.add_subplot(333,adjustable='box')
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#plt.axis('off')
#
#cb_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
#ax3.get_xaxis().set_ticks([])
#ax3.get_yaxis().set_ticks([])
#plt.autoscale(False)
#plt.show()
#cbar = fig.colorbar(t_ax1, cb_ax)
#cbar.ax.tick_params(labelsize=12)
#
#
#
#ax3 = fig.add_subplot(337,adjustable='box', aspect='auto')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[1],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_masked_completed[1],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#ax3 = fig.add_subplot(338,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[7],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(TMaps_masked_completed[7],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=vmax,cmap = cmap)
#ax3 = fig.add_subplot(339,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_masked_completed[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=vmax,cmap = cmap)
#plt.axis('off')
#plt.show()
