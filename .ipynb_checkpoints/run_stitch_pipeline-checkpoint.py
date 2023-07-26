import sys
sys.path.insert(0, "/home/pwahle/microglia/")

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
r_path = str(Path('/usr/local/R-4.0.3/bin/Rscript'))  # path to r interpreter

from scipy import ndimage
import re
import cv2
from scipy import ndimage as ndi
import copy
import numpy as np
import matplotlib as mpl
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import nd2
from nd2reader import ND2Reader
import imageio

def get_metadata_nis(dir_images):
    import os
    import pandas as pd
    images = os.listdir(dir_images)
    images = [image for image in images if '.nd2' in image]
    images.sort()
    regex = r'Point(?P<well>\d+)_Point(?P<tile>\d+)_ChannelPW2_SCF_SD (?P<channel>\d+)'
    df = pd.DataFrame({'file': images})
    df = df.join(df['file'].astype(str).str.extractall(regex).groupby(level=0).last())
    df['well'] = df['well'].apply(lambda x: int(x))
    df['tile'] = df['tile'].apply(lambda x: int(x))
    df['channel'] = df['channel'].apply(lambda x: int(x))
    return df

def get_metadata_tif(dir_images):
    import os
    import pandas as pd
    images = os.listdir(dir_images)
    images = [image for image in images if '.tif' in image]
    images.sort()
    regex = r'(?P<iteration>\d{1})Point(?P<well>\d+)_Point(?P<tile>\d+)'
    df = pd.DataFrame({'file': images})
    df = df.join(df['file'].astype(str).str.extractall(regex).groupby(level=0).last())
    df['well'] = df['well'].apply(lambda x: int(x))
    df['tile'] = df['tile'].apply(lambda x: int(x))
    df['iteration'] = df['iteration'].apply(lambda x: int(x))
    return df

def zproject(image, channel = 0):
    ims = []
    for i in np.arange(len(image)):
       ims.append(np.array(image.get_frame_2D(c=channel, t=0, z=i, x=0, y=0, v=0)))
    redstack = np.dstack(ims)
    return(np.max(redstack, axis = 2))

path = '/links/groups/treutlein/DATA/imaging/PW/microglia/livecell/test6/raw8_4i_C01/'
dir_raws = ['20230327_215214_894/',
           '20230328_093242_414/',
           '20230328_140306_373/',
           '20230328_112112_457/',
           '20230328_150439_899/']

dir_ins = []
for i in dir_raws:
    dir_ins.append(path + i)
    
dir_MIPs = '/links/groups/treutlein/DATA/imaging/PW/microglia/livecell/test6/raw8_4i_C01/multi_MIP/'
dir_stitched = data_path6 + 'raw8_4i_C01/stitched/'
os.makedirs(dir_MIPs , exist_ok=True)
os.makedirs(dir_stitched , exist_ok=True)


it = 1
for dir_in in dir_ins:
    metadata = get_metadata_nis(dir_in)
    
    for well in np.unique(metadata['well']):
        print('MPIng well ' +str(well))
        for tile in np.unique(metadata['tile']):
            pre, ext = os.path.splitext(metadata[(metadata['well'] == well) & (metadata['tile'] == tile) & (metadata['channel'] == 405)]['file'].values[0])
            if not os.path.isfile(dir_MIPs + pre + '.tif'):
                channels = []
                for channel in np.unique(metadata['channel']):
                    tmp = ND2Reader(dir_in + metadata[(metadata['well'] == well) & (metadata['tile'] == tile) & (metadata['channel'] == channel)]['file'].values[0])
                    MIP = zproject(image = tmp, channel =  0)
                    channels.append(MIP)
                stack = np.dstack(channels)
                imageio.imwrite(dir_MIPs + str(it) + pre + '.tif', stack)
                print('well ' +str(well) + ' tile ' + str(tile) + ' MIPed')
    it += 1    
    
metadata = get_metadata_tif(dir_MIPs)
img_avg = io.imread('/links/groups/treutlein/DATA/imaging/PW/microglia/livecell/test6/raw6_smfish_C02/avg_MIP.tif')
cutoff = 115
for iteration in np.unique(metadata['iteration']):
    for well in np.unique(metadata[(metadata['iteration'] == iteration)]['well']):
        if not os.path.isfile(dir_stitched + str(well) + '.tif'):
            print('stitching well ' +str(well))
            files = metadata[(metadata['well'] == well) & (metadata['iteration'] == iteration)]['file'].values
            
            img0 = io.imread(dir_MIPs + files[0])
            length = img0.shape[0]
            
            row1 = []
            row2 = []
            iterator = np.arange(49)[::7]
            for x in iterator:
                if x in [0,14,28,42]:
                    imgs = []
                    for i in np.arange(49)[x:x+7]:
                        img = io.imread(dir_MIPs + files[i]) - img_avg
                        img = img[cutoff:length-cutoff,cutoff:length-cutoff,:]
                        imgs.append(img)
                    
                    row1.append(np.hstack(imgs[::-1]))
                else:
                    imgs = []
                    for i in np.arange(49)[x:x+7]:
                        img = io.imread(dir_MIPs + files[i]) - img_avg
                        img = img[cutoff:length-cutoff,cutoff:length-cutoff,:]
                        imgs.append(img)
                    
                    row2.append(np.hstack(imgs))
        
            grid = np.vstack([row1[3],row2[2],row1[2],row2[1],row1[1],row2[0],row1[0]])
            
            io.imsave(dir_stitched + str(iteration) +'_'+ str(well) + '.tif',grid.astype('int16'))
