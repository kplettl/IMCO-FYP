# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:48:06 2020

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:06:22 2020

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

alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3
sonalleveNativeFile = abspath(      
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\preSonication-with-motion-artefacts_2019-11-12_13-45-31_Native.PAR"
            )
sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)
#
phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)
#pg.image(MMaps, title='phase')


phaseMaps=np.zeros_like(phaseMapsOriginal)

for i in range(113,361):
    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
    
phaseMapsCropped = phaseMaps[113:361,  49:113, 50:114]
MMapsCropped = MMaps[113:361,  49:113, 50:114]

pg.image(MMapsCropped, title='phase')
phaseMapsCropped_recentered_QA_motion = np.zeros_like(phaseMapsCropped)
for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_QA_motion[i] = pmc-np.average(pmc[18:22,30:34])
pg.image(phaseMapsCropped_recentered_QA_motion)

#%%
completed_images = []
original_images = []
generated_images = []


phaseMapsCroppedTest = phaseMapsCropped_recentered_QA_motion[20::20, :,:] #every 20th image for new test set 
OldRange = (np.max(phaseMapsCroppedTest) - np.min(phaseMapsCroppedTest))  
NewRange = (2**8-1) - 0

for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-qa-motion-baseline-296epochs-120120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_296 = np.array(completed_images)
    
for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\120120\8bit_QA_baseline-motion-test\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    original_images.append(im_)
    original_images_array = np.array(original_images)
    
completed_images = []

for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-qa-motion-baseline-592epochs-120120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_592 = np.array(completed_images)
#
completed_images = []

for filename in sorted(glob.glob(r"C:\Users\plettlk\DCGAN_Image_Completion\outputImages\completed\completed-8bit-qa-motion-baseline-888epochs-120120\completed\*.png"), key=numericalSort): 
    im=cv2.imread(filename, -1)
    im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    completed_images.append(im_)
    completed_images_array_888 = np.array(completed_images)


unscaled_completed_296epochs_8bit = (((completed_images_array_296 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_GT_8bit = (((original_images_array /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))

unscaled_completed_592epochs_8bit = (((completed_images_array_592 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))
#
unscaled_completed_888epochs_8bit = (((completed_images_array_888 /NewRange ) * OldRange ) + np.min(phaseMapsCroppedTest))


mean_temp_296_8bit=[]
for i in range(0,12):
    mean_temp_296_8bit.append(np.mean(unscaled_completed_296epochs_8bit[i, 16:48, 16:48]))
 
mean_temp_592_8bit=[]
for i in range(0,12):
    mean_temp_592_8bit.append(np.mean(unscaled_completed_592epochs_8bit[i, 16:48, 16:48]))
#    
mean_temp_888_8bit=[]
for i in range(0,12):
    mean_temp_888_8bit.append(np.mean(unscaled_completed_888epochs_8bit[i, 16:48, 16:48]))
    




deltaT50 = np.zeros_like(unscaled_completed_296epochs_8bit)
deltaT100 = np.zeros_like(unscaled_completed_592epochs_8bit)
deltaT150 = np.zeros_like(unscaled_completed_888epochs_8bit)

for i in range(0,12):
    deltaT50[i] = ((unscaled_GT_8bit[i] - unscaled_completed_296epochs_8bit[i])/(B0*alpha*gamma*echoTime))
pg.image(deltaT50)
for i in range(0,12):
    deltaT100[i] = ((unscaled_GT_8bit[i] - unscaled_completed_592epochs_8bit[i])/(B0*alpha*gamma*echoTime))
pg.image(deltaT100)

average_baseline_temp_qa_motion_592 = []
baseline_temp_STD__qa_motion_592 = []
for i in range(0,12):
    average_baseline_temp_qa_motion_592 = np.append(average_baseline_temp_qa_motion_592, np.average(deltaT100[i, 25:39, 27:37]))
    baseline_temp_STD__qa_motion_592 = np.append(baseline_temp_STD__qa_motion_592,np.std(deltaT100[i,  25:39, 27:37]))
baseline_avg_qa_motion_592 = np.average(average_baseline_temp_qa_motion_592)   
baseline_std_qa_motion_592 = np.std(baseline_temp_STD__qa_motion_592)


for i in range(0,12):
    deltaT150[i] = ((unscaled_GT_8bit[i] - unscaled_completed_888epochs_8bit[i])/(B0*alpha*gamma*echoTime))

#pg.image(deltaT150)

ssims50=[]
for i in range(0,12):
    ssims50.append(compare_ssim(unscaled_completed_296epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim50 = np.average(ssims50)
ssimSTD50 = np.std(ssims50)

ssims100=[]
for i in range(0,12):
    ssims100.append(compare_ssim(unscaled_completed_592epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim100 = np.average(ssims100)
ssimSTD100 = np.std(ssims100)

ssims150=[]
for i in range(0,12):
    ssims150.append(compare_ssim(unscaled_completed_888epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgssim150 = np.average(ssims150)
ssimSTD150 = np.std(ssims150)

mses50=[]
for i in range(0,12):
    mses50.append(mse(unscaled_completed_296epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgmse50 = np.average(mses50)
mseSTD50 = np.std(mses50)

mses100=[]
for i in range(0,12):
    mses100.append(mse(unscaled_completed_592epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgmse100 = np.average(mses100)
mseSTD100 = np.std(mses100)

mses150=[]
for i in range(0,12):
    mses150.append(mse(unscaled_completed_888epochs_8bit[i,16:48,16:48], unscaled_GT_8bit[i,16:48,16:48] ))
avgmse150 = np.average(mses150)
mseSTD150 = np.std(mses150)
#

epochs = ['296 Epochs', '592 Epochs', '888 Epochs']
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
#plt.title('Similarity Measures of Original and Completed Baseline \n 8-bit QA Phantom Phase Images with Motion Artefacts', size=18)
##To show the plot finally we have used plt.show().
#plt.show()


deltaTavgs = []

for i in range (0,12):
    deltaTavgs.append(np.average(deltaT100[i,16:48,16:48], axis=0))
    
deltaTavgs_ = np.array(deltaTavgs)    
deltaTavgs__ = np.average(deltaTavgs_, axis=0)
deltaTSTDs = np.std(deltaTavgs_, axis=0)


#distance = np.linspace(16,47, 32)
#plt.figure()
#plt.axhline(y=0, linestyle='--', color='gray', linewidth=1.5)
#plt.errorbar(distance, deltaTavgs__, yerr=deltaTSTDs, capsize=5, marker='.', markersize=8, ecolor='black', color='xkcd:orange brown')
#plt.title('Average Temperature Difference Of Completed Region of Baseline 8-bit \n QA Phantom Image Set with Motion Artefacts', size=18)
#plt.ylim([-1.6, 0.8])
#
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
#plt.yticks(fontsize=18)
#plt.xticks(fontsize=18)
#
#plt.xlabel('Pixel Position', size=18)
#plt.ylabel('∆T(˚C)',size=18)
#plt.show()


avg_baseline_temp_8bit_motion = np.average(deltaTavgs__)
sd_baseline_temp_8bit_motion = np.std(deltaTavgs__)

