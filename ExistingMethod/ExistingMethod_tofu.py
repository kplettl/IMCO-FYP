#Original code from Lukas Sebeke
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:52:44 2019

@author: plettlk
"""


from os.path import abspath

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyqtgraph as pg
from PARRECread import PARRECread
import numpy as np
import scipy.ndimage as ndimage 
from scipy.ndimage.filters import convolve
from skimage.restoration import unwrap_phase
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.model_selection import train_test_split
#import matplotlib.style as style

#style.use('seaborn-white') #sets the size of the charts
#
#plt.ioff()

alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3

''' Importing Data '''

#Tofu phantom data
sonalleveNativeFile = abspath(
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_Native.PAR"
                )

# read temperature data from PARREC files
sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMaps = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)

phaseMaps = phaseMaps[9,:,:]
phaseMaps = phaseMaps[None,:,:]

MMaps = MMaps[9,:,:]
MMaps = MMaps[None,:,:]

#pg.image(phaseMaps,title='phase')
#pg.image(MMaps,title='magnitude')

#%%

masks = np.ones_like(MMaps)
    
masks[np.where(MMaps<np.max(MMaps)*0.2)]=0
xInBaseline,yInBaseline = np.where(np.logical_not(masks[0]))
masks[:,xInBaseline,yInBaseline] = 0

k = np.zeros((7,7,7))
k[[2,3,4],:,:] = 1
k[:,[0,0,-1,-1],[0,-1,0,-1]] = 0
k[:,[0,0,-1,-1],[1,-2,1,-2]] = 0
k[:,[1,1,-2,-2],[0,-1,0,-1]] = 0
masks_conv = (convolve(masks,k,mode='constant',cval=0)/np.sum(k))*3
masks_conv[np.where(masks_conv<0.9)]=0
 
tags = Params['tags']
vSize_xy = tags[1,[28,29]]
vSize_z = tags[1,22]
vSize = (np.append(vSize_xy, vSize_z))*10e-4
shape = Params['scan_resolution']

echoTime = tags[1,30]*10e-4

xLocs = vSize[0]*np.arange(shape[0])-vSize[0]*(shape[0]-1)/2
yLocs = vSize[1]*np.arange(shape[1])-vSize[1]*(shape[1]-1)/2
xMap,yMap = np.meshgrid(xLocs,yLocs)

#1.8 cm circle in target map, all else zeros
targetSize = 1.8e-2
xTarget,yTarget = np.where(xMap**2+yMap**2<targetSize**2)

targetMap = np.zeros_like(MMaps)
targetMap[:,xTarget,yTarget]=5

if len(targetMap.shape)>2:
    targetMap = targetMap[0,:,:]
else:
    targetMap = targetMap.squeeze()
   
# mask for coherent areas of high signal
coherentMask = np.ones_like(MMaps)
for i,mask in enumerate(masks_conv):
    labeledMask, IDs = ndimage.label(mask)
    labelsInTarget = labeledMask[np.where(targetMap)[0],np.where(targetMap)[1]]
    dominantLabel=np.bincount(labelsInTarget).argmax()
    labelMap = np.ones_like(targetMap)*dominantLabel
    coherentMask[i][np.where(labeledMask==labelMap)]=0
   
#defining size of frame roi 
innerROIRad = 16e-3
outerROIRad = 30e-3

xInsideRing,yInsideRing = np.where((xMap+0.2e-2)**2+(yMap-0.15e-2)**2<innerROIRad**2)
xOutsideRing,yOutsideRing = np.where((xMap+0.2e-2)**2+(yMap-0.15e-2)**2>outerROIRad**2)

#creating masks for reference area - frame ROI used for learning
referenceMask = np.zeros_like(MMaps)
referenceMask[:,xInsideRing,yInsideRing]=1
referenceMask[:,xOutsideRing,yOutsideRing]=1

#pg.image(referenceMask, title='Ref mask')
#pg.image(coherentMask, title='Coherent mask')

maskedPhase = np.ma.masked_where(coherentMask[:phaseMaps.shape[0]],phaseMaps[:coherentMask.shape[0]],copy=True)

unwrappedPhase_init = unwrap_phase(maskedPhase,seed=100)

#pg.image(unwrappedPhase_init,title ='unwrappedPhase')


inclusionMask = np.logical_and(np.logical_not(referenceMask),np.logical_not(coherentMask))  
exclusionMask = np.logical_not(inclusionMask)

#pg.image(inclusionMask,title='inclusion mask')


referencePhase = unwrappedPhase_init[np.where(inclusionMask[:unwrappedPhase_init.shape[0]])]

referencePhaseMap = np.zeros_like(unwrappedPhase_init)
referencePhaseMap[np.where(inclusionMask[:referencePhaseMap.shape[0]])] = referencePhase
#pg.image(referencePhaseMap,title ='reference Phase')

#mask used for inner temperature region prediction
predictionMask = np.zeros_like(MMaps)
predictionMask[:,xInsideRing,yInsideRing]=1

#pg.image(predictionMask, title='pred mask')

ROI = MMaps[np.where(predictionMask)]

#frame mask for plotting 
frameMask = np.ones_like(MMaps)
frameMask[:,xInsideRing,yInsideRing]=0
frameMask[:,xOutsideRing,yOutsideRing]=0
#pg.image(frameMask)


#%% 
regs =[]
X_trains =[]
X_tests =[]
Y_trains =[]
Y_tests =[]
w_trains = []
w_tests = []
errors = []
innerStdDevTs = []
outerStdDevTs = []
deltaTs= []
meanOffsets =[]
degrees = np.arange(1,8)

print('Start Learning')

baselineMaps = np.zeros_like(unwrappedPhase_init)
innerDeltaTMap = np.zeros_like(predictionMask)
outerDeltaTMap = np.zeros_like(frameMask)
''' 

Initialisation Stage: polynomial orders between first and seventh are tested
to see which best approximates the phase in the background.

'''

for deg in degrees:
    
    for i,iMask in enumerate(inclusionMask[:unwrappedPhase_init.shape[0]]):


        X = np.where(iMask)
        X = np.transpose(np.array(X))
    
    
        Y = unwrappedPhase_init[i][np.where(iMask)]

        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        weights = (MMaps[i][np.where(iMask)])**2

        X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(X_poly, Y, weights, test_size=0.33, random_state=815)

        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(X_train, Y_train, sample_weight=w_train)
        
        y_test_pred = reg.predict(X_test)
        error = mean_absolute_error(Y_test,y_test_pred)
        
        regs.append(reg)
        X_trains.append(X_train)
        X_tests.append(X_test)
        Y_trains.append(Y_train)
        Y_tests.append(Y_test)
        w_trains.append(w_train)
        w_tests.append(w_test)
        errors.append(error)
        
        x = np.where(np.ones_like(iMask))
        x = np.transpose(np.array(x))

        x_poly = poly_features.fit_transform(x)
        phasePred = reg.predict(x_poly)
        
        baselineMaps[i][np.where(np.ones_like(iMask))]=phasePred
               
        
    phase_diff = unwrappedPhase_init-baselineMaps

    deltaT = phase_diff/(B0*alpha*gamma*echoTime) # full temperature map

    innerDeltaT = deltaT[np.where(predictionMask[:deltaT.shape[0]])] #inner ROI map for SD testing
    innerDeltaTMap[np.where(predictionMask[:innerDeltaTMap.shape[0]])] = innerDeltaT

    outerDeltaT = deltaT[np.where(frameMask[:deltaT.shape[0]])] #frame ROI map for SD testing 
    outerDeltaTMap[np.where(frameMask[:outerDeltaTMap.shape[0]])] = outerDeltaT


#    pg.image(deltaT,title='Temperature Difference, Degree: {}'.format(deg))
#    pg.image(innerDeltaTMap,title='Inner Temperature Difference, Degree: {}'.format(deg))
#    pg.image(outerDeltaTMap, title='Frame Temperature Difference, Degree: {}'.format(deg))

    innerStdDevT = np.std(innerDeltaTMap)
    innerStdDevTs.append(innerStdDevT)
    
    outerStdDevT = np.std(outerDeltaTMap)
    outerStdDevTs.append(outerStdDevT)
    
    meanOffset = np.mean(innerDeltaTMap)
    meanOffsets.append(meanOffset)
    
#    
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(degrees, innerStdDevTs, 'k', label='Inner Region', linewidth=1.5, marker='o', markersize = 6)
#ax1.set_title('Standard Deviation of RLESS Temperature \n Estimation of the Tofu Phantom',size=18)
#ax1.plot(degrees, outerStdDevTs, color='xkcd:dark orange', label='Outer Region', linestyle='--', marker='^', markersize=6, linewidth=1.5)
#ax1.legend(loc='best', fontsize=18)
#ax1.set_xlabel('Degree of polynomial', size=18)
#ax1.set_ylabel('Standard deviation (˚C)',size=18)
#ax1.tick_params(axis='both', labelsize=18)
#ax1.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.show()

min_deg = np.argmin(innerStdDevTs)+1
if meanOffsets[min_deg] < 3: 
    print('Best fit is degree {}'.format(min_deg))
else:
    print('Mean offset greater than 3')
  #%%  
'''
Estimation stage: best fit polynomial order used to estimate the temperature
of the phantom during heating
''' 

sonalleveNativeFile = abspath(             
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_Native.PAR"
           )

# read temperature data from PARREC files
sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMaps = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)

phaseMaps = phaseMaps[10:26,:,:]
MMaps = MMaps[10:26,:,:]

#pg.image(phaseMaps,title='phase')
#pg.image(MMaps,title='magnitude')

masks = np.ones_like(MMaps)
masks[np.where(MMaps<np.max(MMaps)*0.2)]=0
xInBaseline,yInBaseline = np.where(np.logical_not(masks[1]))
masks[:,xInBaseline,yInBaseline] = 0

k = np.zeros((7,7,7))
k[[2,3,4],:,:] = 1
k[:,[0,0,-1,-1],[0,-1,0,-1]] = 0
k[:,[0,0,-1,-1],[1,-2,1,-2]] = 0
k[:,[1,1,-2,-2],[0,-1,0,-1]] = 0
masks_conv = convolve(masks,k,mode='constant',cval=0)/np.sum(k)
masks_conv[np.where(masks_conv<0.9)]=0
 
tags = Params['tags']
vSize_xy = tags[1,[28,29]]
vSize_z = tags[1,22]
vSize = (np.append(vSize_xy, vSize_z))*10e-4
shape = Params['scan_resolution']

echoTime = tags[1,30]*10e-4

xLocs = vSize[0]*np.arange(shape[0])-vSize[0]*(shape[0]-1)/2
yLocs = vSize[1]*np.arange(shape[1])-vSize[1]*(shape[1]-1)/2
xMap,yMap = np.meshgrid(xLocs,yLocs)

#1.8 cm circle in target map, all else zeros
targetSize = 1.8e-2
xTarget,yTarget = np.where(xMap**2+yMap**2<targetSize**2)

targetMap = np.zeros_like(MMaps)
targetMap[:,xTarget,yTarget]=5

if len(targetMap.shape)>2:
    targetMap = targetMap[0,:,:]
else:
    targetMap = targetMap.squeeze()
   
# mask for coherent areas of high signal
coherentMask = np.ones_like(MMaps)
for i,mask in enumerate(masks_conv):
    labeledMask, IDs = ndimage.label(mask)
    labelsInTarget = labeledMask[np.where(targetMap)[0],np.where(targetMap)[1]]
    dominantLabel=np.bincount(labelsInTarget).argmax()
    labelMap = np.ones_like(targetMap)*dominantLabel
    coherentMask[i][np.where(labeledMask==labelMap)]=0
   
#creating masks for reference area - frame ROI used for learning
referenceMask = np.zeros_like(MMaps)
referenceMask[:,xInsideRing,yInsideRing]=1
referenceMask[:,xOutsideRing,yOutsideRing]=1

#pg.image(referenceMask, title='Ref mask')
#pg.image(coherentMask, title='Coherent mask')

maskedPhase = np.ma.masked_where(coherentMask[:phaseMaps.shape[0]],phaseMaps[:coherentMask.shape[0]],copy=True)

unwrappedPhase = unwrap_phase(maskedPhase,seed=100)
#pg.image(unwrappedPhase,title ='unwrappedPhase')

inclusionMask = np.logical_and(np.logical_not(referenceMask),np.logical_not(coherentMask))
exclusionMask = np.logical_not(inclusionMask)

#pg.image(exclusionMask,title='Exclusion mask')

referencePhase = unwrappedPhase[np.where(inclusionMask[:unwrappedPhase.shape[0]])]

referencePhaseMap = np.zeros_like(unwrappedPhase)
referencePhaseMap[np.where(inclusionMask[:referencePhaseMap.shape[0]])] = referencePhase
#pg.image(referencePhaseMap,title ='reference Phase')

predictionMask = np.zeros_like(MMaps)
predictionMask[:,xInsideRing,yInsideRing]=1
#pg.image(predictionMask, title='pred mask')

regs =[]
X_trains =[]
X_tests =[]
Y_trains =[]
Y_tests =[]
w_trains = []
w_tests = []
errors = []



print('Start Prediction of Temperature Map')

baselineMaps = np.zeros_like(unwrappedPhase)
innerDeltaTMap = np.zeros_like(predictionMask)

 
for i,iMask in enumerate(inclusionMask[:unwrappedPhase.shape[0]]):


    X = np.where(iMask)
    X = np.transpose(np.array(X))


    Y = unwrappedPhase[i][np.where(iMask)]

    poly_features = PolynomialFeatures(degree=min_deg, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    weights = (MMaps[i][np.where(iMask)])**2
  
    X_train, X_test, Y_train, Y_test, w_train, w_test = train_test_split(X_poly, Y, weights, test_size=0.33, random_state=815)

    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(X_train, Y_train, sample_weight=w_train)
    
    y_test_pred = reg.predict(X_test)
    error = mean_absolute_error(Y_test,y_test_pred)
    
    regs.append(reg)
    X_trains.append(X_train)
    X_tests.append(X_test)
    Y_trains.append(Y_train)
    Y_tests.append(Y_test)
    w_trains.append(w_train)
    w_tests.append(w_test)
    errors.append(error)
    
    x = np.where(np.ones_like(iMask))
    x = np.transpose(np.array(x))

    x_poly = poly_features.fit_transform(x)
    phasePred = reg.predict(x_poly)
    
    baselineMaps[i][np.where(np.ones_like(iMask))]=phasePred
           
    
phase_diff = unwrappedPhase-baselineMaps

deltaT = phase_diff/(B0*alpha*gamma*echoTime) # full temperature map

#
innerDeltaT = deltaT[np.where(predictionMask[:deltaT.shape[0]])] #inner ROI map for SD testing
innerDeltaTMap[np.where(predictionMask[:innerDeltaTMap.shape[0]])] = innerDeltaT

#
#pg.image(deltaT,title='Temperature Difference, Degree: {}'.format(min_deg))
#pg.image(innerDeltaTMap,title='Inner Temperature Difference, Degree: {}'.format(min_deg))

''' Plotting TMaps ''' 

#xWindow = 50e-3
#yWindow = 50e-3  
#
#TMaps_masked = np.ma.masked_where(innerDeltaTMap < 1, innerDeltaTMap)
#MMaps_cropped = MMaps[:,42:106, 39:103]
#TMaps_cropped = TMaps_masked[:,42:106, 39:103]
##TMaps_cropped = TMaps_masked[:,42:106, 39:103]
#
#time = tags[:,31][::8]
#times = time - time[0]
#
#cmap = cm.get_cmap('plasma')
#
#fig=plt.figure(figsize=(8,6))
#ax3 = fig.add_subplot(234,adjustable='box', aspect='auto')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=8.0,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(235,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=8.0,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#ax3 = fig.add_subplot(236,adjustable='box')
#
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=8.0,cmap = cmap)
#ROI_ax = ax3.imshow(frameMask[0,42:106, 39:103],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2], alpha=0.1,cmap=cm.gray)
#plt.axis('off')
#
#print('Images acquired at: {:.2f} s, {:.2f}s, {:.2f}s'.format(times[1], times[8], times[15]))
#
#''' GT temperature map data ''' 
#
#sonalleveTMapFile = sonalleveNativeFile.replace('Native','TMap')
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
#GT_TMaps_cropped = GT_TMaps_masked[:,42:106, 39:103]
#
#GTtags = tmapParams['tags']
#GTtime = GTtags[:,31][::4]
#GTtimes = GTtime - GTtime[0]
#
#ax3 = fig.add_subplot(231,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[2],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[2],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=8.0,cmap = cmap)
#ax3 = fig.add_subplot(232,adjustable='box')
#plt.axis('off')
#m_ax = ax3.imshow(MMaps_cropped[8],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax1 = ax3.imshow(GT_TMaps_cropped[8],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1, vmax=8.0,cmap = cmap)
#ax3 = fig.add_subplot(233,adjustable='box')
#m_ax = ax3.imshow(MMaps_cropped[15],interpolation='none',extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 200,cmap=cm.gray)
#t_ax = ax3.imshow(GT_TMaps_cropped[15],interpolation='none', extent=[-xWindow/2,xWindow/2,xWindow/2,-yWindow/2],vmin = 1,vmax=8.0,cmap = cmap)
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
#average_peak_temp_rless_tofu = []
#peak_temp_STD_rless_tofu = []
#for i in range(0,16):
#    average_peak_temp_rless_tofu = np.append(average_peak_temp_rless_tofu, np.average(innerDeltaTMap[i, 78:82, 78:81]))
#    peak_temp_STD_rless_tofu = np.append(peak_temp_STD_rless_tofu,np.std(innerDeltaTMap[i, 78:82, 78:81]))
#    
#
#HIFUaverage_peak_temp = []
#HIFUpeak_temp_STD = []
#for i in range(0,16):
#    
#    HIFUaverage_peak_temp = np.append(HIFUaverage_peak_temp,np.average(GT_TMaps[i, 78:82, 78:81]))
#    HIFUpeak_temp_STD= np.append(HIFUpeak_temp_STD,np.std(GT_TMaps[i, 78:82, 78:81]))
#
#time = np.linspace(0, 15, 16)
#
#plt.figure()
#plt.errorbar(times[0:16], average_peak_temp_rless_tofu, yerr=peak_temp_STD_rless_tofu, capsize=5, label = 'Referenceless', marker = '.', markersize=6, color='black')
#plt.errorbar(times[0:16], HIFUaverage_peak_temp, yerr=HIFUpeak_temp_STD, capsize=5, label ='HIFU Software', marker = '.', markersize=6, color='xkcd:dark orange', linestyle=':')
#plt.xlabel('Time after Starting HIFU (s)', size=18)
#plt.ylabel('∆T(˚C)', size=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.title('Average Temperature of Sonicated Region of Tofu Phantom over Time', size=18)
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.axvline(x=36, linestyle='--', color='gray', linewidth=0.7)
#plt.legend(fontsize=18)
#plt.show()
