"""
@author: Zhe XU
resort the original UAMT_LA data
1. rename and copy the files to new LA_data_h5 folder
2. Normalization
3. Convert to h5 file
"""


import glob
import os
import re
import h5py
import numpy as np
import shutil

def resave_LA_data(folderDir, NewDir):
    img_foldername_list = os.listdir(folderDir)
    for case in img_foldername_list:
#        print(case)
        new_name_h5_img = case
        ori_path = os.path.join(folderDir, case, 'mri_norm2.h5')
        new_path = os.path.join(NewDir, case+'.h5')
        print(ori_path)
        print(new_path)
        shutil.copyfile(ori_path,new_path)
 
    print("copy LA files to new folder")


if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    folderDir = '../../data/2018LA_Seg_Training Set/'
    NewDir = '../../data/LA_data_h5/data/'

    resave_LA_data(folderDir, NewDir)