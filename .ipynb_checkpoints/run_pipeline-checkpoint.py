import sys
sys.path.insert(0, "/home/pwahle/proliferating_RGC/")

import yaml
import importlib
import modules
importlib.reload(modules)
import os
from pathlib import Path
import subprocess

# load global variables and parameters
with open("params.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

globals().update(cfg)

import re
import cv2
import copy
import numpy as np
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma

#MIP
experiment = 'exp5/cycle2/'
data_path = data_path + experiment
rawpath = data_path + 'raw/'
metadf = modules.get_metadata(rawpath)
os.makedirs(data_path + 'MIPs/' , exist_ok=True)
os.makedirs(data_path + 'stacks/' , exist_ok=True)
os.makedirs(data_path + 'denoised/' , exist_ok=True)


for n in np.unique(metadf['well_id']):
    for channel in np.unique(metadf['channel']):
        filenames = metadf[(metadf['well_id'] == n)& (metadf['channel'] == channel)]['file'].values
        if not os.path.isfile(data_path + 'MIPs/' + filenames[0]):
            print('MIPing ' + filenames[0])
            imgs = []
            for filename in filenames:
                imgs.append(io.imread(rawpath +filename))
                
            stack = np.dstack(imgs)
            save3d = copy.copy(stack)
            save3d = stack.swapaxes(0,2)
            save3d = save3d.swapaxes(1,2)
            io.imsave(data_path + 'stacks/' + filename, save3d,photometric='minisblack')
    
            MIP = np.max(stack, axis=2)
            io.imsave(data_path + 'MIPs/' + filenames[0], MIP)
        else:
            print(filenames[0] + ' already MIPed. moving on!')
                
                
#denoise
metadf = modules.get_metadata(data_path + 'MIPs/')

for file in metadf['file'].values:
    if not os.path.isfile(Path(data_path, 'denoised', file)):
        print('denoising ' + file)
        image = io.imread(data_path + 'MIPs/' + file ).astype('int16')
        sigma_est = np.mean(estimate_sigma(image, multichannel=False))
        patch_kw = dict(patch_size=5,      # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=False)
        denoised = denoise_nl_means(image, h=0.8 * sigma_est, sigma=sigma_est,
                                     fast_mode=True, **patch_kw, preserve_range=True)
        
        io.imsave(Path(data_path, 'denoised', file), denoised.astype('int16'))
    else:
        print(file + ' already denoised. moving on')