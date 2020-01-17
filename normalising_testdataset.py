# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:47:06 2019

@author: plettlk
"""
from os.path import abspath
from PARRECread import PARRECread
import imageio
from skimage.restoration import unwrap_phase
import pyqtgraph as pg
import numpy as np
import os
from skimage.util import img_as_ubyte, img_as_uint
from skimage import exposure

alpha = -10.3e-9
gamma = 0.267513e9
B0 = 3
echoTime = 19.5e-3

sonalleveNativeFile = abspath(    
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\baseline-scan-1_2019-10-11_11-14-17_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_Native.PAR"
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-01\longTemperatureMapping_2019-08-28_16-02-23_Native.PAR",
        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-w-motion-artefact_2019-11-12_14-07-50_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-wo-motion-artefact_2019-11-12_13-55-06_Native.PAR",
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-20W-sonication-2-36s_2019-10-14_15-16-09_Native.PAR"  
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-20W-sonication-3-36s_2019-10-14_15-32-29_Native.PAR"
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-04\qa-phantom-20W-sonication-4-36s_2019-10-14_15-49-11_Native.PAR"
        )
#sonalleveTMapFile = abspath(      
##        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-05\30W-sonication-wo-motion-artefact_2019-11-12_13-55-06_TMap.PAR"
#        r"C:\Users\plettlk\DCGAN_Image_Completion\19-PLEK-RLES-03\20W-sonication-1_2019-10-11_12-24-33_TMap.PAR"
#
#            )

sonalleveNativeData,Params,Dims = PARRECread(sonalleveNativeFile)

phaseMapsOriginal = sonalleveNativeData[:,:,0,0,0,1,:].swapaxes(0,2)
MMaps = sonalleveNativeData[:,:,0,0,0,0,:].swapaxes(0,2)
##phaseMaps = unwrap_phase(phaseMaps,seed=100)
#pg.image(phaseMapsOriginal, title='phase')
pg.image(MMaps, title='mag')
##
phaseMaps=np.zeros_like(phaseMapsOriginal)
##
##
#for i in range(9,26):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#  
#phaseMapsCropped = phaseMaps[9:26,42:106, 39:103] 

##for i in range(0,600):
##    phaseMaps[i,:,:] = unwrap_phase(phaseMaps[i,:,:], seed=100)
#  
for i in range(0,17):
    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
    
phaseMapsCropped = phaseMaps[0:17, 49:113, 50:114]
MMapsCropped = MMaps[0:17, 49:113, 50:114]
#
pg.image(phaseMapsCropped, title='phase')

phaseMapsCropped_recentered_QA_motion = np.zeros_like(phaseMapsCropped)

for i,pmc in enumerate(phaseMapsCropped):
    phaseMapsCropped_recentered_QA_motion[i] = pmc-np.average(pmc[18:22,30:34])
pg.image(phaseMapsCropped_recentered_QA_motion)
##    
#phaseMaps = np.zeros_like(phaseMapsOriginal)
#for i in range(9,27):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#  
#phaseMapsCropped = phaseMaps[9:27,49:113, 50:114]
#pg.image(phaseMapsCropped)
#MMapsCropped = MMaps[9:27, 49:113, 50:114]
#pg.image(MMapsCropped)

#for i in range(1446,1463):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#  
#phaseMapsCropped = phaseMaps[1446:1463,49:113, 50:114]
#pg.image(phaseMapsCropped)

#for i in range(1463,1480):
#    phaseMaps[i,:,:] = unwrap_phase(phaseMapsOriginal[i,:,:], seed=100)
#  
#phaseMapsCropped = phaseMaps[1463:1480,49:113, 50:114]
#MMapsCropped = MMaps[1463:1480,49:113, 50:114]
#pg.image(MMapsCropped)
#pg.image(phaseMapsCropped)
#  
#phaseMap
#
#phaseMapsCropped_recentered_QA_heated = np.zeros_like(phaseMapsCropped)
##
#for i,pmc in enumerate(phaseMapsCropped):
#    phaseMapsCropped_recentered_QA_heated[i] = pmc-np.average(pmc[18:22,30:34])
#pg.image(phaseMapsCropped_recentered_QA_heated)

#MMapsCropped = MMaps[9:27,49:113, 50:114]
#phaseMapsCroppedCenter = phaseMaps[9:26,60:90, 55:87] 
#pg.image(phaseMapsCropped)
#pg.image(MMapsCropped)
#
#sonalleveTMapData,Params,Dims = PARRECread(sonalleveTMapFile)
##
#TMaps = sonalleveTMapData[:,:,0,0,0,0,:].swapaxes(0,2)
#TMapsCropped = TMaps[9:26,42:106, 39:103] 
#pg.image(TMapsCropped, title='temperature maps from sonalleve')
##
##distance = np.linspace(0,63,64)
#horizontalProfile_heated_sonalleve = TMapsCropped[15,37,:] - 37
#plt.figure()
#plt.plot(distance,horizontalProfile_heated_sonalleve, 'k', label='unheated',linewidth=1.5)
##plt.plot(distance,horizontalProfile_sonalleve,'k', label='HIFU software', linestyle=':',linewidth=1.5, alpha=0.75)
#plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
##plt.legend(loc='upper right')
#plt.xlabel('Pixel position')
#plt.xticks()
#
#plt.title('Horizontal slice through Sonalleve temperature map with 30W sonication')
#plt.ylabel('∆T (˚C)')
#plt.show()
#phaseMaps = unwrap_phase(phaseMaps,seed=100)
#pg.image(TMapsCropped, title='temperature maps from sonalleve')
#%%
#phaseMapsCropped_Norm = phaseMapsCropped / np.max(phaseMapsCropped)
#OldRange = (np.max(phaseMapsCropped) - np.min(phaseMapsCropped))
#NewRange = (2**16-1) - 0
#Scaled = ((((phaseMapsCropped - np.min(phaseMapsCropped)) * NewRange) / OldRange) + 0).astype(np.uint8)
#Scaled = ((((phaseMapsCropped - np.min(phaseMapsCropped)) / OldRange) * NewRange) + 0).astype(np.uint8)
#Scaled_ = img_as_ubyte(Scaled)
rescaled = exposure.rescale_intensity(phaseMapsCropped_recentered_QA_motion, in_range=(np.min(phaseMapsCropped_recentered_QA_motion), np.max(phaseMapsCropped_recentered_QA_motion)), out_range = (0, 1))
rescaled_ = img_as_ubyte(rescaled)
#rescaled_ = img_as_uint(rescaled)
#

#delta = phaseMapsCropped[:,:,:] - phaseMapsCropped[0,:,:]
#deltaT = (phaseMapsCropped[:,:,:] - phaseMapsCropped[0,:,:])/(B0*alpha*gamma*echoTime) 
#pg.image(deltaT, title='temperature difference of phase images, unwrapped one by one')

#unscaled = (( rescaled_ /NewRange ) * OldRange ) + np.min(phaseMapsCropped)

#delta_unscaled = unscaled[16,:,:] - unscaled[0,:,:]
#pg.image(delta_unscaled)
#directory = r"C:\Users\plettlk\DCGAN_Image_Completion\tofu_phase-first-tofu-imaging\\"
new_dir = r"C:\Users\plettlk\DCGAN_Image_Completion\120120\8bit_QA_test_30W_heating_w_motion\\"
#img = np.zeros((1774, 64, 64))
#
#for i in range(0, 1774):
#    img[i,:,:] = imageio.imread(directory + 'cropped_phase_image_' + str(i) + '.jpeg')
#
##img /= np.max(img) 
#OldRange = (np.max(img) - np.min(img))  
#NewRange = 255 - 0
#NewValue__ = ((((img - np.min(img)) * NewRange) / OldRange) + 0).astype(np.uint8)
#
def make_dir(name):
    # Works on python 2.7, where exist_ok arg to makedirs isn't available.
    p = os.path.join(name)
    if not os.path.exists(p):
        os.makedirs(p)
        
make_dir(new_dir)

for i in range(rescaled_.shape[0]):
    stack = np.stack((rescaled_[i,:,:],)*3, axis=-1)
    numpngw.write_png(new_dir + 'scaled_phase_image_' + str(i) +'.png', stack)
#    imageio.imwrite(new_dir + 'scaled_phase_image_' + str(i) +'.png', stack)
    
#img = imageio.imread(r"C:\Users\plettlk\DCGAN_Image_Completion\131119\16bit_tofu_wo_motion_test_set\scaled_phase_image_0.png")  
#reader = png.Reader(r"C:\Users\plettlk\DCGAN_Image_Completion\131119\16bit_tofu_wo_motion_test_set\scaled_phase_image_0.png")
#data = reader.asDirect()
#pixels = data[2]
#image = []
#for row in pixels:
#  row = np.asarray(row)
#  row = np.reshape(row, [-1, 3])
#  image.append(row)
#image = np.stack(image, 1)