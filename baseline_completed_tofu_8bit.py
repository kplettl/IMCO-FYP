# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:35:46 2019

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:44:13 2019

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:51:50 2019

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
from PARRECread import PARRECread
from skimage.restoration import unwrap_phase
import pyqtgraph as pg


alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3

 ''' importing MR data '''

sonalleveNativeFile1 = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-1_2019-10-11_11-14-17_Native.PAR",
        )

sonalleveNativeFile2 = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-2_2019-10-11_11-46-22_Native.PAR",
           )

sonalleveNativeFile3 = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-3_2019-10-11_12-20-01_Native.PAR",
           )
sonalleveNativeData1,Params,Dims = PARRECread(sonalleveNativeFile1)

sonalleveNativeData2,Params,Dims = PARRECread(sonalleveNativeFile2)

sonalleveNativeData3,Params,Dims = PARRECread(sonalleveNativeFile3)
#
phaseMaps1 = sonalleveNativeData1[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps1 = sonalleveNativeData1[:,:,0,0,0,0,:].swapaxes(0,2)
#pg.image(MMaps, title='phase')



phaseMaps=np.zeros_like(phaseMaps1)
##
for i in range(0,203):
    phaseMaps[i,:,:] = unwrap_phase(phaseMaps1[i,:,:], seed=100)
    
phaseMapsCropped1 = phaseMaps[0:203, 42:106, 39:103]

phaseMaps2 = sonalleveNativeData2[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps2 = sonalleveNativeData2[:,:,0,0,0,0,:].swapaxes(0,2)

phaseMaps=np.zeros_like(phaseMaps2)
for i in range(0,709):
    phaseMaps[i,:,:] = unwrap_phase(phaseMaps2[i,:,:], seed=100)
phaseMapsCropped2 = phaseMaps[0:709, 42:106, 39:103]


phaseMaps3 = sonalleveNativeData3[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps3 = sonalleveNativeData3[:,:,0,0,0,0,:].swapaxes(0,2)
phaseMaps=np.zeros_like(phaseMaps3)
for i in range(0,771):
    phaseMaps[i,:,:] = unwrap_phase(phaseMaps3[i,:,:], seed=100)
phaseMapsCropped3 = phaseMaps[0:771,  42:106, 39:103]

phaseMapsCropped = np.vstack((phaseMapsCropped1, phaseMapsCropped2, phaseMapsCropped3))
phaseMapsCropped_recentered_tofu = np.zeros_like(phaseMapsCropped)
for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_tofu[i] = pmc-np.average(pmc[30:34,30:34])
pg.image(phaseMapsCropped_recentered_tofu)
#%% 

completed_images = []
original_images = []

phaseMapsCroppedTest = phaseMapsCropped_recentered_tofu[100::100, :,:] #every 100th image for new test set 
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**8-1) - 0

''' importing images ''' 
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-tofu-50epochs-baseline\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_50 = np.array(completed_images)
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\171219\8bit_tofu_test_recentered_baseline\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    

completed_images = []

for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-tofu-100epochs-baseline\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_100 = np.array(completed_images)

completed_images = []

for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-tofu-150epochs-baseline\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_150 = np.array(completed_images)

unscaled_completed_50epochs_8bit = (((completed_images_array_50 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_8bit = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_completed_100epochs_8bit = (((completed_images_array_100 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_completed_150epochs_8bit = (((completed_images_array_150 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))


mean_temp_50_8bit=[]
for i in range(0,16):
    mean_temp_50_8bit.append(np.mean(unscaled_completed_50epochs_8bit[i, 16:48, 16:48]))
 
mean_temp_100_8bit=[]
for i in range(0,16):
    mean_temp_100_8bit.append(np.mean(unscaled_completed_100epochs_8bit[i, 16:48, 16:48]))
    
mean_temp_150_8bit=[]
for i in range(0,16):
    mean_temp_150_8bit.append(np.mean(unscaled_completed_150epochs_8bit[i, 16:48, 16:48]))
    

deltaT50 = np.zeros_like(unscaled_completed_50epochs_8bit)
deltaT100 = np.zeros_like(unscaled_completed_100epochs_8bit)
deltaT150 = np.zeros_like(unscaled_completed_150epochs_8bit)

for i in range(0,16):
    deltaT50[i] = ((unscaled_GT_8bit[i] - unscaled_completed_50epochs_8bit[i])/(B0*alpha*gamma*echoTime))

for i in range(0,16):
    deltaT100[i] = ((unscaled_GT_8bit[i] - unscaled_completed_100epochs_8bit[i])/(B0*alpha*gamma*echoTime))
#pg.image(deltaT100)

average_baseline_temp_tofu_8bit_100 = []
baseline_temp_STD_tofu_8bit_100 = []
for i in range(0,16):
    average_baseline_temp_tofu_8bit_100 = np.append(average_baseline_temp_tofu_8bit_100, np.average(deltaT100[i, 31:44, 35:47]))
    baseline_temp_STD_tofu_8bit_100 = np.append(baseline_temp_STD_tofu_8bit_100,np.std(deltaT100[i,  31:44, 35:47]))
baseline_avg_tofu_8bit_100 = np.average(average_baseline_temp_tofu_8bit_100)   
baseline_std_tofu_8bit_100 = np.std(baseline_temp_STD_tofu_8bit_100)


for i in range(0,16):
    deltaT150[i] = ((unscaled_GT_8bit[i] - unscaled_completed_150epochs_8bit[i])/(B0*alpha*gamma*echoTime))



ssims50=[]
for i in range(0,16):
    ssims50.append(compare_ssim(unscaled_completed_50epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim50 = np.average(ssims50)
ssimSTD50 = np.std(ssims50)

ssims100=[]
for i in range(0,16):
    ssims100.append(compare_ssim(unscaled_completed_100epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim100 = np.average(ssims100)
ssimSTD100 = np.std(ssims100)

ssims150=[]
for i in range(0,16):
    ssims150.append(compare_ssim(unscaled_completed_150epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim150 = np.average(ssims150)
ssimSTD150 = np.std(ssims150)

mses50=[]
for i in range(0,16):
    mses50.append(mse(unscaled_completed_50epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgmse50 = np.average(mses50)
mseSTD50 = np.std(mses50)

mses100=[]
for i in range(0,16):
    mses100.append(mse(unscaled_completed_100epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgmse100 = np.average(mses100)
mseSTD100 = np.std(mses100)
mses150=[]
for i in range(0,16):
    mses150.append(mse(unscaled_completed_150epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
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
#plt.legend([bar1,bar2],['MSE', 'SSIM Index'], loc='lower right', fontsize=18)
#plt.title('Similarity Measures of Original and Completed \n 8-bit Tofu Baseline Phase Images', size=18)
##To show the plot finally we have used plt.show().
#plt.show()
#

deltaTavgs = []

for i in range (0,16):
    deltaTavgs.append(np.average(deltaT100[i,16:48,16:48], axis=0))
    
deltaTavgs_ = np.array(deltaTavgs)    
deltaTavgs__ = np.average(deltaTavgs_, axis=0)
deltaTSTDs = np.std(deltaTavgs_, axis=0)

#distance = np.linspace(16,47, 32)
#plt.figure()
#plt.axhline(y=0, linestyle='--', color='gray', linewidth=1.5)
#plt.errorbar(distance, deltaTavgs__, yerr=deltaTSTDs, capsize=5, marker='.', markersize=8, ecolor='black', color='xkcd:orange brown')
#plt.title('Average Temperature Difference Of Completed Region \n of Baseline 8-bit Tofu Phantom Image Set', size=18)
#plt.ylim([-0.4, 0.6])
#
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.yticks(fontsize=18)
#plt.xticks(fontsize=18)
#
#plt.xlabel('Pixel Position', size=18)
#plt.ylabel('∆T(˚C)',size=18)
#plt.show()


avg_baseline_temp_8bit_tofu = np.average(deltaTavgs__)
sd_baseline_temp_8bit_tofu = np.std(deltaTavgs__)
