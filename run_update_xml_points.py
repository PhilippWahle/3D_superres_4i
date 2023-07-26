import sys
sys.path.insert(0, "/home/pwahle/proliferating_RGC/")

import yaml
import modules
import utils
import numpy as np
from bs4 import BeautifulSoup

# load global variables and parameters
with open("../params.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

globals().update(cfg)

path = data_path + 'July_2023_optimization/20_vs_4_degree/cycle1/'

outname = 'multipoints_C1_updated.xml'
origina_file = 'multipoints.xml'
new_file = 'multipoints_C1.xml'
npositions = 5

with open(path + origina_file, 'rb') as f:
    data = f.read()
Bs_orig = BeautifulSoup(data, "xml")

with open(path + new_file, 'rb') as f:
    data = f.read()
Bs_new = BeautifulSoup(data, "xml")

a1 = [float(Bs_orig.find_all('dXPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
a2 = [float(Bs_orig.find_all('dYPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
a3 = [float(Bs_orig.find_all('dZPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
v0 = np.array([a1,a2,a3])

b1 = [float(Bs_new.find_all('dXPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
b2 = [float(Bs_new.find_all('dYPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
b3 = [float(Bs_new.find_all('dZPosition', {'runtype':'double'})[i]['value']) for i in np.arange(npositions)]
v1 = np.array([b1,b2,b3])

T = utils.affine_matrix_from_points(v0, v1,shear=False, scale=False, usesvd=True)

bX = [float(i['value']) for i in Bs_orig.find_all('dXPosition', {'runtype':'double'})[npositions:]] 
bY = [float(i['value']) for i in Bs_orig.find_all('dYPosition', {'runtype':'double'})[npositions:]] 
bZ = [float(i['value']) for i in Bs_orig.find_all('dZPosition', {'runtype':'double'})[npositions:]] 
v3 = np.stack([bX,bY,bZ,np.ones(len(bZ))])

projected_points = np.matmul(T,v3)

for tag in np.arange(len(Bs_orig.find_all('dXPosition', {'runtype':'double'})))[npositions:]:
    Bs_orig.find_all('dXPosition', {'runtype':'double'})[tag]['value'] = str(projected_points[0,tag-npositions])
    
for tag in np.arange(len(Bs_orig.find_all('dXPosition', {'runtype':'double'})))[:npositions]:
    Bs_orig.find_all('dXPosition', {'runtype':'double'})[tag]['value'] = str(v1[0,tag])
    
for tag in np.arange(len(Bs_orig.find_all('dYPosition', {'runtype':'double'})))[npositions:]:
    Bs_orig.find_all('dYPosition', {'runtype':'double'})[tag]['value'] = str(projected_points[1,tag-npositions])
    
for tag in np.arange(len(Bs_orig.find_all('dYPosition', {'runtype':'double'})))[:npositions]:
    Bs_orig.find_all('dYPosition', {'runtype':'double'})[tag]['value'] = str(v1[1,tag])
    
for tag in np.arange(len(Bs_orig.find_all('dZPosition', {'runtype':'double'})))[npositions:]:
    Bs_orig.find_all('dZPosition', {'runtype':'double'})[tag]['value'] = str(projected_points[2,tag-npositions])
    
for tag in np.arange(len(Bs_orig.find_all('dZPosition', {'runtype':'double'})))[:npositions]:
    Bs_orig.find_all('dZPosition', {'runtype':'double'})[tag]['value'] = str(v1[2,tag])
    
with open(path + outname, "w") as f:
    f.write(Bs_orig.prettify())
    
print('updated points in file ' + outname)