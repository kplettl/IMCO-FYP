# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:56:32 2019

@author: plettlk
"""

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

from os.path import abspath
from PARRECread import PARRECread
import pyqtgraph as pg
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_ssim
from skimage.restoration import unwrap_phase


alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3

''' Importing data ''' 

sonalleveNativeFile1 = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-baseline-scan_2019-10-14_14-17-13_Native.PAR"
            )

sonalleveNativeFile2 = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-baseline-scan-2_2019-10-14_15-15-00_Native.PAR"
            )
# read temperature data from PARREC files
sonalleveNativeData1,Params,Dims = PARRECread(sonalleveNativeFile1)

phaseMaps1 = sonalleveNativeData1[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps1 = sonalleveNativeData1[:,:,0,0,0,0,:].swapaxes(0,2)



phaseMaps=np.zeros_like(phaseMaps1)
for i in range(0,693):
    phaseMaps[i,:,:] = unwrap_phase(phaseMaps1[i,:,:], seed=100)

phaseMapsCropped4 = phaseMaps[0:693,49:113, 50:114]

sonalleveNativeData2,Params,Dims = PARRECread(sonalleveNativeFile2)

phaseMaps2= sonalleveNativeData2[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps2 = sonalleveNativeData2[:,:,0,0,0,0,:].swapaxes(0,2)


phaseMaps=np.zeros_like(phaseMaps2)

for i in range(728,1429):
    phaseMaps[i,:,:] = unwrap_phase(phaseMaps2[i,:,:], seed=100)

phaseMapsCropped5 = phaseMaps[728:1429,49:113, 50:114]

phaseMapsCropped  = np.vstack((phaseMapsCropped4, phaseMapsCropped5))

phaseMapsCropped_recentered_QA = np.zeros_like(phaseMapsCropped)

for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_QA[i] = pmc-np.average(pmc[18:22,30:34])
pg.image(phaseMapsCropped_recentered_QA)

#%% 

phaseMapsCroppedTest = phaseMapsCropped_recentered_QA[100::100, :,:] #every 100th image for new test set 
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**16-1) - 0

completed_images = []
original_images = []
 ''' importing images ''' 
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-16bitQA-baseline-50epochs-100120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_50 = np.array(completed_images)
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\100120\16bit_QA_baseline_test_recentered_notincentre\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    
completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-16bitQA-baseline-100epochs-100120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_100 = np.array(completed_images)
    

completed_images = []
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-16bitQA-baseline-150epochs-100120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_150 = np.array(completed_images)
    


unscaled_completed_50epochs_16bit = (((completed_images_array_50 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_completed_100epochs_16bit = (((completed_images_array_100 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_completed_150epochs_16bit = (((completed_images_array_150 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_16bit = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

deltaT50 = np.zeros_like(unscaled_completed_50epochs_16bit)
deltaT100 = np.zeros_like(unscaled_completed_100epochs_16bit)
deltaT150 = np.zeros_like(unscaled_completed_150epochs_16bit)

for i in range(0,13):
    deltaT50[i] = ((unscaled_GT_16bit[i] - unscaled_completed_50epochs_16bit[i])/(B0*alpha*gamma*echoTime))

for i in range(0,13):
    deltaT100[i] = ((unscaled_GT_16bit[i] - unscaled_completed_100epochs_16bit[i])/(B0*alpha*gamma*echoTime))

average_baseline_temp_qa_16bit_100 = []
baseline_temp_STD_qa_16bit_100 = []
for i in range(0,13):
    average_baseline_temp_qa_16bit_100 = np.append(average_baseline_temp_qa_16bit_100, np.average(deltaT100[i, 25:39, 27:37]))
    baseline_temp_STD_qa_16bit_100 = np.append(baseline_temp_STD_qa_16bit_100,np.std(deltaT100[i,  25:39, 27:37]))
baseline_avg_qa_16bit_100 = np.average(average_baseline_temp_qa_16bit_100)   
baseline_std_qa_16bit_100 = np.std(baseline_temp_STD_qa_16bit_100)

for i in range(0,13):
    deltaT150[i] = ((unscaled_GT_16bit[i] - unscaled_completed_150epochs_16bit[i])/(B0*alpha*gamma*echoTime))

#pg.image(deltaT50)
#pg.image(deltaT100)
#pg.image(deltaT150)

ssims50=[]
for i in range(0,13):
    ssims50.append(compare_ssim(unscaled_completed_50epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgssim50 = np.average(ssims50)
ssimSTD50 = np.std(ssims50)

ssims100=[]
for i in range(0,13):
    ssims100.append(compare_ssim(unscaled_completed_100epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgssim100 = np.average(ssims100)
ssimSTD100 = np.std(ssims100)

ssims150=[]
for i in range(0,13):
    ssims150.append(compare_ssim(unscaled_completed_150epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgssim150 = np.average(ssims150)
ssimSTD150 = np.std(ssims150)

mses50=[]
for i in range(0,13):
    mses50.append(mse(unscaled_completed_50epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgmse50 = np.average(mses50)
mseSTD50 = np.std(mses50)

mses100=[]
for i in range(0,13):
    mses100.append(mse(unscaled_completed_100epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgmse100 = np.average(mses100)
mseSTD100 = np.std(mses100)
mses150=[]
for i in range(0,13):
    mses150.append(mse(unscaled_completed_150epochs_16bit[i,16:48,16:48], unscaled_GT_16bit[i,16:48,16:48] ))
avgmse150 = np.average(mses150)
mseSTD150 = np.std(mses150)


epochs = ['50 Epochs', '100 Epochs', '150 Epochs']
plt.rc('axes', axisbelow=True)
mses = [avgmse50, avgmse100, avgmse150]
ssims = [avgssim50, avgssim100, avgssim150]
mseerror = [mseSTD50, mseSTD100, mseSTD150] 
ssimerror=  [ssimSTD50, ssimSTD100, ssimSTD150]

#https://www.weirdgeek.com/2018/11/plotting-multiple-bar-graph/
#fig = plt.figure()
#x = np.arange(len(mses))
#ax1 = plt.subplot(1,1,1)
#w = 0.1
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#
##plt.xticks(), will label the bars on x axis with the respective country names.
#plt.xticks(0.3*x + w /2, labels=epochs, fontsize=18)
#bar1 =ax1.bar(0.3*x, mses, yerr=mseerror, width=w, color='xkcd:light gray', align='center', capsize=5,linewidth=1.2, edgecolor='black' )
#plt.yticks(fontsize=18)
#plt.ylabel('MSE', size=18)
##The trick is to use two different axes that share the same x axis, we have used ax1.twinx() method.
#ax2 = ax1.twinx()
##We have calculated GDP by dividing gdpPerCapita to population.
#bar2 =ax2.bar(0.3*x + w, ssims, yerr=ssimerror,width=w,color='xkcd:orange brown',align='center', capsize=5,linewidth=1.2, edgecolor='black')
#plt.yticks(fontsize=18)
##Set the Y axis label as GDP.
#plt.ylabel('SSIM Index', size=18)
##To set the legend on the plot we have used plt.legend()
#plt.legend([bar1,bar2],['MSE', 'SSIM Index'], loc='lower left', fontsize=18)
#plt.title('Similarity Measures of Original and Completed \n 16-bit Stationary QA Phantom Baseline Phase Images', size=18)
##To show the plot finally we have used plt.show().
#plt.show()

deltaTavgs = []

for i in range (0,13):
    deltaTavgs.append(np.average(deltaT100[i,16:48,16:48], axis=0))
    
deltaTavgs_ = np.array(deltaTavgs)    
deltaTavgs__ = np.average(deltaTavgs_, axis=0)
deltaTSTDs = np.std(deltaTavgs_, axis=0)


#distance = np.linspace(16,47, 32)
#plt.figure()
#plt.axhline(y=0, linestyle='--', color='gray', linewidth=1.5)
#plt.errorbar(distance, deltaTavgs__, yerr=deltaTSTDs, capsize=5, marker='.', markersize=8, ecolor='black', color='xkcd:orange brown')
#plt.title('Average Temperature Difference Of Completed Region \n of Baseline 16-bit Stationary QA Phantom Image Set', size=18)
#plt.ylim([-0.9, 1.5])
#
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.yticks(fontsize=18)
#plt.xticks(fontsize=18)
#
#plt.xlabel('Pixel Position', size=18)
#plt.ylabel('∆T(˚C)',size=18)
#plt.show()


avg_baseline_temp_16bit_qa = np.average(deltaTavgs__)
sd_baseline_temp_16bit_qa = np.std(deltaTavgs__)
