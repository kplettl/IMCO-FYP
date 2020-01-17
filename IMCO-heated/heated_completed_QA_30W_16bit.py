# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:06:23 2020

@author: plettlk
"""

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
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_ssim
from os.path import abspath
#from PARRECread import PARRECread
import imageio
from skimage.restoration import unwrap_phase
import pyqtgraph as pg
import numpy as np
import os
from skimage.util import img_as_ubyte, img_as_uint
from skimage import exposure
import matplotlib.cm as cm


alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3
#
sonalleveNativeFile = abspath(    
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-1_2019-10-11_11-14-17_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_Native.PAR"
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-01\longTemperatureMapping_2019-08-28_16-02-23_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-w-motion-artefact_2019-11-12_14-07-50_Native.PAR",
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-wo-motion-artefact_2019-11-12_13-55-06_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-20W-sonication-2-36s_2019-10-14_15-16-09_Native.PAR"  
            )

sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)
##phaseMaps = unwrap_phase(phaseMaps,seed=100)
#pg.image(phaseMapsOriginal, title='phase')
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
#unscaled_completed=[]
#unscaled_GT = []
#deltaT=[]


phaseMapsCroppedTest = phaseMapsCropped_recentered_QA_heated
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**16-1) - 0
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\100120\16bit_QA_test_30W_heating_wo_motion\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    
completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-16bitQA-100epochs-30Wheating-recentered-100120\completed\*.png"), key=numericalSort): 
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

plt.figure()
plt.errorbar(times[0:16], average_peak_temp, yerr=peak_temp_STD, capsize=5, label = 'IMCO', marker = '.', markersize=6, color='black')
plt.errorbar(times[0:16], HIFUaverage_peak_temp, yerr=HIFUpeak_temp_STD, capsize=5, label ='SBL', marker = '^', markersize=6, color='xkcd:dark orange', linestyle=':')
plt.errorbar(times[0:16], average_peak_temp_rless_qa_30w, yerr=peak_temp_STD_rless_qa_30w, capsize=5, label ='RLESS', marker = 's', markersize=6, color='xkcd:steel blue', linestyle='--')

plt.xlabel('Time after Starting HIFU (s)', size=18)
plt.ylabel('∆T(˚C)', size=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Average Temperature Difference of Sonicated Region of \n Stationary QA Phantom over Time, 30W Sonication', size=18)
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
plt.axvline(x=36, linestyle='--', color='gray', linewidth=0.7)
plt.legend(fontsize=18)
plt.show()



#rmse_8bit = np.sqrt(mse(HIFUaverage_peak_temp, average_peak_temp))


#deltaT100_tofu = np.fliplr(deltaT100_tofu)
#MMaps = np.fliplr(MMaps)

TMaps_masked_completed = np.ma.masked_where(deltaT100_QA < 1, deltaT100_QA)
MMaps_cropped = MMaps[9:27,49:113, 50:114]
#TMaps_cropped = TMaps_masked[:,42:106, 39:103]



cmap = cm.get_cmap('plasma')
xWindow = 50e-3
yWindow = 50e-3  

#innerDeltaTMap = np.fliplr(innerDeltaTMap)
#MMaps = np.fliplr(MMaps)

TMaps_masked = np.ma.masked_where(innerDeltaTMap < 1, innerDeltaTMap)
#MMaps_cropped = MMaps[9:26:,42:106, 39:103]
TMaps_cropped = TMaps_masked[:,49:113, 50:114]
#TMaps_cropped = TMaps_masked[:,42:106, 39:103]

#time = tags[:,31][::8]
#times = time - time[0]

cmap = cm.get_cmap('plasma')
#
#bounds = np.linspace(1, 21, 6)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig=plt.figure(figsize=(8,6))
ax3 = fig.add_subplot(334,adjustable='box', aspect='auto')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(335,adjustable='box')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax1 = ax3.imshow(TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(336,adjustable='box')

m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
ROI_ax = ax3.imshow(frameMask[0,49:113, 50:114],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
plt.axis('off')

#cb_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])



#ax3.text(.013,.021,"1 cm",fontsize=20)
#ax3.plot([0.013,0.023],[0.022,0.022],'k',linewidth=5)
#ax3.get_xaxis().set_ticks([])
#ax3.get_xaxis().set_ticklabels([])
#ax3.get_yaxis().set_ticks([])
#ax3.get_yaxis().set_ticklabels([])
#plt.autoscale(False)
#plt.show()
#cbar = fig.colorbar(t_ax1, cb_ax)

print('Images acquired at: {:.2f} s, {:.2f}s, {:.2f}s'.format(times[1], times[8], times[15]))

sonalleveTMapFile = sonalleveNativeFile.replace('Native','TMap')
    


# read temperature data from PARREC files
sonalleveTMapData,tmapParams,tmapDims = PARRECread(sonalleveTMapFile)


# reshape data to be displayed by pyqtgraph
GT_TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
#reshapedData = sonalleveTMapData[:,:,3,0,0,0,-1]
#pg.image(GT_TMaps)
# display image in pg.image

#pg.image(reshapedData1,title='test')
#pg.image(reshapedData2)

#imv = pg.ImageView()
#imv.show()
#imv.setImage(GT_TMaps)

GT_TMaps = GT_TMaps[10:26,:,:] - 37
GT_TMaps_masked = np.ma.masked_where(GT_TMaps < 1, GT_TMaps)

GT_TMaps_cropped = GT_TMaps_masked[:,49:113, 50:114]

GTtags = tmapParams['tags']
GTtime = GTtags[:,31][::4]
GTtimes = GTtime - GTtime[0]

#fig=plt.figure(figsize=(12,4))
ax3 = fig.add_subplot(331,adjustable='box')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(GT_TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(332,adjustable='box')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax1 = ax3.imshow(GT_TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(333,adjustable='box')

m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(GT_TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
plt.axis('off')

cb_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
ax3.get_xaxis().set_ticks([])
ax3.get_xaxis().set_ticklabels([])
ax3.get_yaxis().set_ticks([])
ax3.get_yaxis().set_ticklabels([])
plt.autoscale(False)
plt.show()
cbar = fig.colorbar(t_ax1, cb_ax)
cbar.ax.tick_params(labelsize=12) 
#cbar.ax.set_ylabel('∆T (˚C)', rotation=270, labelpad=16, fontsize=18)

#bounds = np.linspace(1, 21, 6)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

ax3 = fig.add_subplot(337,adjustable='box', aspect='auto')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(TMaps_masked_completed[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[1,41:121, 40:120],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(338,adjustable='box')
plt.axis('off')
m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax1 = ax3.imshow(TMaps_masked_completed[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[1,41:121, 40:120],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
ax3 = fig.add_subplot(339,adjustable='box')

m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
t_ax = ax3.imshow(TMaps_masked_completed[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=9.3,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[1,41:121, 40:120],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
plt.axis('off')
plt.show()
