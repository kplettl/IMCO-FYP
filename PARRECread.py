# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:34:54 2019

@author: plettlk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:53:36 2015

@author: Lukas
"""


#==============================================================================
# 
# Created by      Benoit Desjardins, MD, PhD
#                 Department of Radiology, University of Michigan
#==============================================================================



import tkinter as tk #macht GUI
import sys
from tkinter.filedialog import * #zum speichern
import numpy as np
import math

#import time

def PARRECread(parfile = '', out_fmt = 'FP', verboseflag='', imagenumber = ''):
#==============================================================================
#   reads Philips PAR/REC files
#   
#   Syntax: 
#   (flt_data, params, dims)= PARRECread(filename,output format,verbose flag)
#   
#   Inputs:
#       * If no inputs, select parfile via UI. 
#       * .REC file is assumed to be in same folder as .PAR file
#       * default value for output format then is 'FP'
#       With inputs:
#           filename = filename with path of PAR file
#           out_fmt  = data output type (FP, DV, SV)
#           verboseflag = if not empty, the program prints some information 
#                         about the .PAR file and the process to the command 
#                         window
#
#    Output: 
#        flt_data = matrix of images (default: as FP values)
#        params   = structure containing PAR file info
#        dims: [stride nline nslice necho nphase ntype ndyn]
#        
#    Can process both V3 and V4 types of PAR files automatically
#    
#    
#==============================================================================

    
    

    if parfile == '':
        root=tk.Tk()
        root.withdraw()
        parfile = askopenfilename()
        
    recfile=parfile.replace('.PAR','.REC')
    (params,ver)=read_parfile(parfile)
    (flt_data,dims)=read_recfile(recfile,params,ver,out_fmt,imagenumber)

    return (flt_data, params, dims)


def read_parfile(filename):
    #reads all the lines of the .PAR-file
    fid= open(filename)
    lines = fid.readlines()
    fid.close()
    NG=0    
    geninfo=[]
    # identify the header lines
    for line in lines:
        if(line[0]=='.'):        
            geninfo.append(line)
            NG+=1
    
    if(NG < 1):
        sys.exit('.PAR file has invalid format!')
    
    # test fpr V3 or V4 PAR files
    
    testkey= '*Image pixel size*' # only present in v3 files
    if testkey in lines:
        ver = 'V3'
        template = get_template_v3()
    else:
        ver = 'V4'
        template = get_template_v4()
    
    
    
    # parse the header information
#    value_keys  = list(template[:,0])
    value_types = list(template[:,1])
    field_names = list(template[:,2])
           
    
    
    values = [value.split(':',1)[-1] for value in geninfo[:]]  #split the values from the rest of the line
    values = [value.lstrip() for value in values]           #strip the values of the leading spaces
    values = [value.replace('\n','') for value in values]   #remove the newline string
#    values = ['  VALUE NOT FOUND  ' for value in values if value == '']
    
    
    # identify missing information in the .par-file 
    # and fill corresponding list places with NaN 
    if len(value_types) != len(values):
        paranames = [value.split(':',1)[0]+':' for value in geninfo[:]]
        i=0
        for name in template[:,0]:
            i += 1
            if name not in paranames:
                values.insert(i, np.nan)
    
    
    
    # compile a dictionary that pairs the parameter names and the corresponding values in the proper datatype
    i=0
    params = {} 
    for value_type in value_types:
        if value_type in ['float scalar','int   scalar']: # scalar numbers are converted to doubles
            params[field_names[i]]=np.double(values[i])
        elif value_type in ['float vector','int   vector']: # vector numbers are split, then converted to doubles
            params[field_names[i]]=np.double(values[i].split())
#            print "error on line %d" % i
        elif value_type in ['char  scalar','char  vector']: # scalars are stripped from their leading and trailing blanks
            params[field_names[i]]=values[i].strip()
        else: # if no datatype applies, the corresponding value is not recorded
            params[field_names[i]]=''
        i+=1
        
        
    # parse the tags for each line of data
    tags=[] # empty data list
#    taglabels = [line for line in lines if line.endswith(")")]
#    tags.append(taglabels)
    for line in lines:
        if (line[0] != '#' and line[0] != '.' and len(line)>1): # datalines are identified
            tags.append(np.double(line.split())) # data is appended to the prepared data list
    tags=np.array(tags) # data list is converted to ndarray

    params['tags']=tags
    params['tagnames']=get_tagnames()
    if tags.shape[1] == 0:
        sys.exit('Missing scan information in .PAR file')
    tags = dict(zip(get_tagnames(),tags))

    return (params,ver) 






def read_recfile(recfile,params,ver,out_fmt = 'FP',imagenumber = ''):
    
    types_list      = list(params['tags'][:,4])
    types_list      = erase_duplicates(types_list)
    ntype           = len(types_list)
    scan_tag_size   = params['tags'].shape
    nimg            = int(scan_tag_size[0])
    nslice          = int(params['max_slices'])
    nphase          = int(params['max_card_phases'])
    necho           = int(params['max_echoes'])
    ndyn            = int(params['max_dynamics'])


    if 'recon_resolution' in params.keys():
        nline = int(params['recon_resolution'][1])
        stride = int(params['recon_resolution'][2])
    else:
        nline = int(params['tags'][0,9])
        stride = int(params['tags'][0,10])



    if ver =='V3':
        pixel_bits = params['pixel_bits']
    elif ver == 'V4':
        pixel_bits = params['tags'][0,7]
    else:
        sys.exit('invalid version string')
        

    if pixel_bits == 8:
        read_type = np.int8
        byts = 1
    elif pixel_bits == 16:
        read_type = np.int16
        byts =2
    else:
        read_type = np.char
        byts =1
        
        

#HIFU T mapping sequence
#==============================================================================
 
    if ndyn == 9999:    
        ndyn = int(math.ceil(nimg/nslice/necho/nphase/ntype)) # if ndyn not an integer: add 1 to it
        dims = np.array([stride,nline,nslice,necho,nphase,ntype,ndyn]) # generate the final matrix of images

        if imagenumber == '': # Dataset is read as a whole
            image_data = np.zeros(dims) # Allocate disk space
            fid = open(recfile,'rb')

            for I in np.arange(0,nimg):
                imslice     = int(params['tags'][I,0]-1)
                phase       = int(params['tags'][I,3]-1)
                imtype      = params['tags'][I,4]
                type_idx    = int(types_list.index(imtype))
                echo        = int(params['tags'][I,1]-1)
                dyn         = int(params['tags'][I,2]-params['tags'][0,2])
                seq         = params['tags'][I,5]
                rec         = params['tags'][I,6]
                block_size  = int(stride*nline)
                binary_1D   = np.fromfile(fid, dtype = read_type, count = block_size)
                read_size   = len(binary_1D)
                img         = binary_1D.reshape(stride,nline)
                img         = np.rot90(img)
                # rescale data to produce FP information (not SV, not DV)
                img = rescale_rec(img,params['tags'][I,:], ver, out_fmt)
                image_data[:,:,imslice,echo,phase,type_idx,dyn] = img
            fid.close()
            print('.REC file read sucessfully')
        else: #single images are read from the dataset
            fid = open(recfile,'rb')
            if imagenumber<=nimg:
                I           = imagenumber
                imslice     = params['tags'][I,0]-1
                phase       = params['tags'][I,3]-1
                imtype      = params['tags'][I,4]
                type_idx    = types_list.index(imtype)
                echo        = params['tags'][I,1]-1
                dyn         = params['tags'][I,2]-params['tags'][0,2]
                seq         = params['tags'][I,5]
                rec         = params['tags'][I,6]
                block_size  = stride*nline
                start_image = imagenumber*nline*stride*byts
                fid.seek(start_image, 0)
                binary_1D   = np.fromfile(fid, dtype = read_type, count = block_size)
                img         = binary_1D.reshape(stride,nline)
                img         = np.rot90(img)                               
                # rescale data to produce FP information (not SV, not DV)
                img = rescale_rec(img,params['tags'][I,:], ver, out_fmt)
                #type_idx; # for debugging
                image_data = img
                print('Single image read sucessfully')
            else:
                print('Imagenumber is bigger then number of dynamics scans')
                image_data= []
            fid.close()
  
#read the .REC file         
#==============================================================================
    else:
        fid = open(recfile,'rb')
        binary_1D = np.fromfile(fid,dtype = read_type, count = -1)
        read_size = len(binary_1D)
        fid.close()
    
        if (read_size != nimg*nline*stride):
            print('Expected %d int.  Found %d int', (nimg*nline*stride,read_size))
            if read_size > nimg*nline*stride:
                sys.exit('.REC file has more data than expected from .PAR file')
            else:
                sys.exit('.REC file has less data than expected from .PAR file')
            
        else:
            print('.REC file read sucessfully')
        
    
        # generate the final matrix of images
        dims = (stride,nline,nslice,necho,nphase,ntype,ndyn)
        image_data = np.zeros(dims)

        for I  in np.arange(nimg):
            imslice = int(params['tags'][I,0]-1)
            phase = int(params['tags'][I,3]-1)
            imtype = params['tags'][I,4]
            type_idx = int(types_list.index(imtype))
            echo = int(params['tags'][I,1]-1)
            dyn = int(params['tags'][I,2]-1)
            seq = params['tags'][I,5]
            rec = params['tags'][I,6]
            start_image = int(rec*nline*stride)
            end_image = int(start_image + stride*nline)
#            print end_image-start_image
#            print stride*nline
            img = binary_1D[start_image:end_image].reshape(stride,nline)
            img = np.rot90(img)               
            
            # rescale data to produce FP information (not SV, not DV)
            
            
            img = rescale_rec(img,params['tags'][I,:], ver, out_fmt)
            image_data[:,:,imslice,echo,phase,type_idx,dyn] = img
    
        

    return (image_data, dims)


def get_tags_():
    
    tagslist = ['slice number', 'echo number', 'dynamic scan number',\
    'cardiac phase number', 'image type number', 'index in REC file (in images)',\
    'image pixel size (bits)', 'scan percentage', 'recon resolution (x)',\
    'recon resolution (y)', 'rescale intercept', 'rescale slope', 'scale slope',\
    'window center', 'window width', 'image angulation (ap)', ] 
    
    return tagslist

def rescale_rec(img,tag,ver,out_fmt):

# transforms SV data in REC files to SV, DV or FP data for output
    if ver == 'V3':
        ri_i = 7
        rs_i = 8
        ss_i = 9
    elif ver == 'V4':
        ri_i = 11
        rs_i = 12
        ss_i = 13
    else:
        sys.exit('can\'t rescale unsuported file version')
        
    RI = tag[ri_i]  # 'rescale inter' --> 'RI'; offsets the pixel values
    RS = tag[rs_i]  # 'rescale slope' --> 'RS'; scales the offset
    SS = tag[ss_i]  # 'scale slope'   --> 'SS'; scales the pixel values and the offset
    
    if out_fmt == 'FP': # floating-point image (?)
        img = (RS*img + RI)/(RS*SS)
    elif out_fmt == 'DV': # difference variance (?)
        img = (RS*img + RI)
    elif out_fmt == 'SV': # sum variance (?)
        img = img
    else:
        sys.exit('invalid output format')
    
    return img



def get_template_v3(): # header information for V3 PAR files

    template = np.array([
        '.    Patient name                       :','char  scalar','patient',
        '.    Examination name                   :','char  scalar','exam_name',
        '.    Protocol name                      :','char  vector','protocol',
        '.    Examination date/time              :','char  vector','exam_date',
        '.    Acquisition nr                     :','int   scalar','acq_nr',
        '.    Reconstruction nr                  :','int   scalar','recon_nr',
        '.    Scan Duration [sec]                :','float scalar','scan_dur',
        '.    Max. number of cardiac phases      :','int   scalar','max_card_phases',
        '.    Max. number of echoes              :','int   scalar','max_echoes',
        '.    Max. number of slices/locations    :','int   scalar','max_slices',
        '.    Max. number of dynamics            :','int   scalar','max_dynamics',
        '.    Max. number of mixes               :','int   scalar','max_mixes',
        '.    Image pixel size [8 or 16 bits]    :','int   scalar','pixel_bits',
        '.    Technique                          :','char  scalar','technique',
        '.    Scan mode                          :','char  scalar','scan_mode',
        '.    Scan resolution  (x, y)            :','int   vector','scan_resolution',
        '.    Scan percentage                    :','int   scalar','scan_percentage',
        '.    Recon resolution (x, y)            :','int   vector','recon_resolution',
        '.    Number of averages                 :','int   scalar','num_averages',
        '.    Repetition time [msec]             :','float scalar','repetition_time',
        '.    FOV (ap,fh,rl) [mm]                :','float vector','fov',
        '.    Slice thickness [mm]               :','float scalar','slice_thickness',
        '.    Slice gap [mm]                     :','float scalar','slice_gap',
        '.    Water Fat shift [pixels]           :','float scalar','water_fat_shift',
        '.    Angulation midslice(ap,fh,rl)[degr]:','float vector','angulation',
        '.    Off Centre midslice(ap,fh,rl) [mm] :','float vector','offcenter',
        '.    Flow compensation <0=no 1=yes> ?   :','int   scalar','flowcomp',
        '.    Presaturation     <0=no 1=yes> ?   :','int   scalar','presaturation',
        '.    Cardiac frequency                  :','int   scalar','card_frequency',
        '.    Min. RR interval                   :','int   scalar','min_rr_interval',
        '.    Max. RR interval                   :','int   scalar','max_rr_interval',
        '.    Phase encoding velocity [cm/sec]   :','float vector','venc',
        '.    MTC               <0=no 1=yes> ?   :','int   scalar','mtc',
        '.    SPIR              <0=no 1=yes> ?   :','int   scalar','spir',
        '.    EPI factor        <0,1=no EPI>     :','int   scalar','epi_factor',
        '.    TURBO factor      <0=no turbo>     :','int   scalar','turbo_factor',
        '.    Dynamic scan      <0=no 1=yes> ?   :','int   scalar','dynamic_scan',
        '.    Diffusion         <0=no 1=yes> ?   :','int   scalar','diffusion',
        '.    Diffusion echo time [msec]         :','float scalar','diffusion_echo_time',
        '.    Inversion delay [msec]             :','float scalar','inversion_delay',
        ])
        
    template.resize(40,3)
    return template

#==========================================================================

def get_template_v4():    # header information for V4 PAR files

    template = np.array([
        '.    Patient name                       :','char  scalar','patient',
        '.    Examination name                   :','char  vector','exam_name',
        '.    Protocol name                      :','char  vector','protocol',
        '.    Examination date/time              :','char  vector','exam_date',
        '.    Series Type                        :','char  vector','series_type',
        '.    Acquisition nr                     :','int   scalar','acq_nr',
        '.    Reconstruction nr                  :','int   scalar','recon_nr',
        '.    Scan Duration [sec]                :','float scalar','scan_dur',
        '.    Max. number of cardiac phases      :','int   scalar','max_card_phases',
        '.    Max. number of echoes              :','int   scalar','max_echoes',
        '.    Max. number of slices/locations    :','int   scalar','max_slices',
        '.    Max. number of dynamics            :','int   scalar','max_dynamics',
        '.    Max. number of mixes               :','int   scalar','max_mixes',
        '.    Patient position                   :','char  vector','patient_position',
        '.    Preparation direction              :','char  vector','preparation_dir',
        '.    Technique                          :','char  scalar','technique',
        '.    Scan resolution  (x, y)            :','int   vector','scan_resolution',
        '.    Scan mode                          :','char  scalar','scan_mode',
        '.    Repetition time [ms]               :','float scalar','repetition_time',
        '.    FOV (ap,fh,rl) [mm]                :','float vector','fov',
        '.    Water Fat shift [pixels]           :','float scalar','water_fat_shift',
        '.    Angulation midslice(ap,fh,rl)[degr]:','float vector','angulation',
        '.    Off Centre midslice(ap,fh,rl) [mm] :','float vector','offcenter',
        '.    Flow compensation <0=no 1=yes> ?   :','int   scalar','flowcomp',
        '.    Presaturation     <0=no 1=yes> ?   :','int   scalar','presaturation',
        '.    Phase encoding velocity [cm/sec]   :','float vector','venc',
        '.    MTC               <0=no 1=yes> ?   :','int   scalar','mtc',
        '.    SPIR              <0=no 1=yes> ?   :','int   scalar','spir',
        '.    EPI factor        <0,1=no EPI>     :','int   scalar','epi_factor',
        '.    Dynamic scan      <0=no 1=yes> ?   :','int   scalar','dynamic_scan',
        '.    Diffusion         <0=no 1=yes> ?   :','int   scalar','diffusion',
        '.    Diffusion echo time [ms]           :','float scalar','diffusion_echo_time',
        '.    Max. number of diffusion values    :','int   scalar','max_diffusion_values',
        '.    Max. number of gradient orients    :','int   scalar','max_gradient_orients',
        '.    Number of label types <0=no ASL>   :','int   scalar','N_label_types'
        ])
        
    template.resize(35,3)    
    return template

def get_tagnames():
    tagnames = ['slice number',
               'echo number',
               'dynamic scan number',
               'cardiac phase number',
               'image_type_mr',
               'scanning sequence',
               'index in REC file (in images)',
               'image pixel size (in bits)',
               'scan percentage',
               'recon resolution (x)',
               'recon resolution (y)',
               'rescale intercept',
               'rescale slope',
               'scale slope',
               'window center',
               'window width',
               'image angulation (ap [deg])',
               'image angulation (fh [deg])',
               'image angulation (rl [deg])',
               'image offcentre (ap,fh,rl in mm )',
               'image offcentre (ap,fh,rl in mm )',
               'image offcentre (ap,fh,rl in mm )',
               'slice thickness [mm]',
               'slice gap [mm]',
               'image_display_orientation',
               'slice orientation (TRA/SAG/CO)',
               'fmri_status_indication',
               'image_type_ed_es (end diast/end syst)',
               'pixel spacing (x) [mm]',
               'pixel spacing (y) [mm]',
               'echo_time',
               'dyn_scan_begin_time',
               'trigger_time',
               'diffusion_b_factor',
               'number of averages',
               'image_flip_angle [deg]',
               'cardiac frequency [bpm]',
               'minimum RR-interval [ms]',
               'maximum RR-interval [ms]',
               'TURBO factor  <0=no turbo>',
               'Inversion delay [ms]']
    
    return tagnames


def erase_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]









