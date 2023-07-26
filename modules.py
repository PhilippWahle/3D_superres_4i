import sys
sys.path.insert(0, "/home/pwahle/proliferating_RGC/")
import yaml
from skimage import img_as_uint
from scipy import ndimage
from skimage.filters.rank import median
from skimage.morphology import disk
from time import gmtime, strftime
import re
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import gaussian, threshold_otsu, threshold_multiotsu
from skimage import measure
from scipy import ndimage as ndi
from fnnls import fnnls
from tqdm import tqdm
from cellpose import models
import matplotlib.patches as mpatches
from scipy.stats import zscore
import seaborn as sns
import random
import os.path
import copy
import numpy as np
import phenograph
import matplotlib as mpl
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops_table
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

# load global variables
with open("/home/pwahle/proliferating_RGC/params.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
globals().update(cfg)

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# generic functions
def sorted_nicely(l):
    # sort list of strings alphanumerically as intuited by humans
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

#get white on transparent background outline of label image
def get_outline(mask):
    B = mask * (np.abs(laplace(mask)) > 0)
    binary = (B > 0).astype(int)
    dilated_outline = ndimage.binary_dilation(binary, iterations=1).astype(binary.dtype)
    
    frame = np.zeros(dilated_outline.shape)
    outline = np.dstack([dilated_outline*250, dilated_outline*250, dilated_outline*250, dilated_outline*250])
    return outline



#dir_images = Path(data_path, 'stitched')
def get_metadata(dir_images):
    import os
    import pandas as pd
    images = os.listdir(dir_images)
    images = [image for image in images if '.tif' in image]
    images.sort()
    regex = r'xy(?P<well_id>\d+)z(?P<plane>\d+)c(?P<channel>\d{1})'
    df = pd.DataFrame({'file': images})
    df = df.join(df['file'].astype(str).str.extractall(regex).groupby(level=0).last())
    df['well_id'] = df['well_id'].apply(lambda x: int(x))
    df['channel'] = df['channel'].apply(lambda x: int(x))
    df['plane'] = df['plane'].apply(lambda x: int(x)) 
    return df


def scale_image(image, percentile=0):
    if percentile == 0:
        image = np.interp(image, (image.min(), image.max()), (0, +65535))
    else:
        image = np.interp(image, (np.percentile(image, percentile), np.percentile(image, 100 - percentile)), (0, +65535))
    return image


