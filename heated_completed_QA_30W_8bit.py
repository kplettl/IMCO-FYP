# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:40:38 2019

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
#
sonalleveNativeFile = abspath(    
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-wo-motion-artefact_2019-11-12_13-55-06_Native.PAR",
            )

sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)

pg.image(MMaps, title='mag')
##
   
phaseMaps = np.zeros_like(phaseMapsOriginal)
for i in range(9,27):
    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
  
phaseMapsCropped = phaseMaps[9:27,49:113, 50:114]
pg.image(phaseMapsCropped)
#
phaseMapsCropped_recentered_QA_heated = np.zeros_like(phaseMapsCropped)
#
for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_QA_heated[i] = pmc-np.average(pmc[18:22,30:34])
pg.image(phaseMapsCropped_recentered_QA_heated)

#%%

completed_images = []
original_images = []

phaseMapsCroppedTest = phaseMapsCropped_recentered_QA_heated
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**8-1) - 0
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\201219\8bit_QA_test_30W_heating\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    
completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-QA-100epochs-30W-heating\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_100 = np.array(completed_images)
    
unscaled_completed_QA_heated = (((completed_images_array_100 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_16bit = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

deltaT100_QA = np.zeros_like(unscaled_completed_QA_heated)

average_peak_temp = []
peak_temp_STD = []
for i in range(1,17):
    deltaT100_QA[i] = ((unscaled_GT_16bit[i] - unscaled_completed_QA_heated[i])/(B0*alpha*gamma*echoTime))
    average_peak_temp.append(np.average(deltaT100_QA[i, 29:33, 28:31]))
    peak_temp_STD.append(np.std(deltaT100_QA[i,29:33, 28:31]))
    
average_peak_temp = np.array(average_peak_temp)
peak_temp_STD = np.array(peak_temp_STD)
#pg.image(deltaT100_QA)

sonalleveTMapFile = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-wo-motion-artefact_2019-11-12_13-55-06_TMap.PAR"
            )

sonalleveTMapData,Params,Dims = PARRECread(sonalleveTMapFile)
#
TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
TMapsCropped = TMaps[10:26,49:113, 50:114] - 37
#pg.image(TMapsCropped)
tags = Params['tags']
times = tags[:,31][::4] - tags[:,31][0]

HIFUaverage_peak_temp = []
HIFUpeak_temp_STD = []
for i in range(0,16):
    
    HIFUaverage_peak_temp.append(np.average(TMapsCropped[i, 29:33, 28:31]))
    HIFUpeak_temp_STD.append(np.std(TMapsCropped[i,29:33, 28:31]))
    
HIFUaverage_peak_temp = np.array(HIFUaverage_peak_temp)
HIFUpeak_temp_STD = np.array(HIFUpeak_temp_STD)

#time = np.linspace(0, 15, 16)

#plt.figure()
#plt.errorbar(times[0:16], average_peak_temp, yerr=peak_temp_STD, capsize=5, label = 'IMCO', marker = '.', markersize=6, color='black')
#plt.errorbar(times[0:16], HIFUaverage_peak_temp, yerr=HIFUpeak_temp_STD, capsize=5, label ='SBL', marker = '^', markersize=6, color='xkcd:dark orange', linestyle=':')
#plt.errorbar(times[0:16], average_peak_temp_rless_qa_30w, yerr=peak_temp_STD_rless_qa_30w, capsize=5, label ='RLESS', marker = 's', markersize=6, color='xkcd:steel blue', linestyle='--')
#
#plt.xlabel('Time after Starting HIFU (s)', size=18)
#plt.ylabel('∆T(˚C)', size=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.title('Average Temperature of Sonicated Region of \n Stationary QA Phantom over Time, 30W Sonication', size=18)
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.axvline(x=36, linestyle='--', color='gray', linewidth=0.7)
#plt.legend(fontsize=18)
#plt.show()



#
#TMaps_masked_completed = np.ma.masked_where(deltaT100_QA < 1, deltaT100_QA)
#MMaps_cropped = MMaps[9:27,49:113, 50:114]
#
#
#''' plotting t maps ''' 
#cmap = cm.get_cmap('plasma')
#xWindow = 50e-3
#yWindow = 50e-3  
#
#
#TMaps_masked = np.ma.masked_where(innerDeltaTMap < 1, innerDeltaTMap)
#TMaps_cropped = TMaps_masked[:,49:113, 50:114]
#
#cmap = cm.get_cmap('plasma')
##
#fig=plt.figure(figsize=(8,6))
#ax3 = fig.add_subplot(334,adjustable='box', aspect='auto')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(335,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(336,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#plt.axis('off')
#
#
#
#print('Images acquired at: {:.2f} s, {:.2f}s, {:.2f}s'.format(times[1], times[8], times[15]))
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
#GT_TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
#
#
#GT_TMaps = GT_TMaps[10:26,:,:] - 37
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
#m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ax3 = fig.add_subplot(332,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(GT_TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
#ax3 = fig.add_subplot(333,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#plt.axis('off')
#
#cb_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
#ax3.get_xaxis().set_ticks([])
#ax3.get_xaxis().set_ticklabels([])
#ax3.get_yaxis().set_ticks([])
#ax3.get_yaxis().set_ticklabels([])
#plt.autoscale(False)
#plt.show()
#cbar = fig.colorbar(t_ax1, cb_ax)
#cbar.ax.tick_params(labelsize=12) 
#
#ax3 = fig.add_subplot(337,adjustable='box', aspect='auto')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_masked_completed[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ax3 = fig.add_subplot(338,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(TMaps_masked_completed[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
#ax3 = fig.add_subplot(339,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_masked_completed[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#plt.axis('off')
#plt.show()
