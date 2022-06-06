"""
@author: JackXu
1. corrupt and save the LA data
"""


import glob
import os
import re
import h5py
import numpy as np
import shutil
import scipy.ndimage
import random
import SimpleITK as sitk

def dilation_mask(ori_mask, structure):
    cor_mask = scipy.ndimage.binary_dilation(ori_mask, structure)
    return cor_mask


def erosion_mask(ori_mask, structure):
    cor_mask = scipy.ndimage.binary_erosion(ori_mask, structure)
    return cor_mask


def add_noise(ori_mask, min_radius=3, max_radius=15):
    random_num = random.random()
    radius = random.randint(min_radius, max_radius)
    structure=np.ones((radius,radius,radius)).astype(ori_mask.dtype)
    
    if random_num < 0.5:
        dst_label_np = dilation_mask(ori_mask, structure)
        noise_type = 'dilate'
    elif random_num >= 0.5:
        dst_label_np = erosion_mask(ori_mask, structure)
        noise_type = 'erode'
    return dst_label_np, noise_type


def Noisy_3D_LA_generation(clean_num, train_list_path, ori_Dir, corrupt_Dir, test_list_path):
    with open(train_list_path, 'r') as f:
        train_image_list = f.readlines()
    
    with open(test_list_path, 'r') as f:
        test_image_list = f.readlines()

    num = 0
    #print(train_image_list)
    # we will corrupt the cases except for clean_num (labeled_num)
    for case_name in train_image_list[clean_num:]:
        case_name = case_name.replace('\n', '')
        print(case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        new_path = os.path.join(corrupt_Dir, case_name+'.h5')
        # read the labels, add_noise, and save h5 to new_path
        print(ori_path)
        h5f_ori = h5py.File(ori_path, 'r')
        label_ori = h5f_ori['label']
        print('ori:', label_ori.shape)
        dst_label, noise_type = add_noise(label_ori, min_radius=3, max_radius=15)
        print('corruptshape, type:', dst_label.shape, noise_type)

        f = h5py.File(new_path, 'w')
        f.create_dataset('image', data=h5f_ori['image'], compression="gzip")
        f.create_dataset('label', data=dst_label, compression="gzip")
        f.close()
        num += 1
    print("create noisy LA files to new folder")
        
    for case_name in train_image_list[:clean_num]:
        case_name = case_name.replace('\n', '')
        print('clean:', case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        new_path = os.path.join(corrupt_Dir, case_name+'.h5')        
        shutil.copyfile(ori_path, new_path)
        num += 1
    print("copy clean LA files to new folder")

    for case_name in test_image_list:
        case_name = case_name.replace('\n', '')
        print('test:', case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        new_path = os.path.join(corrupt_Dir, case_name+'.h5')        
        shutil.copyfile(ori_path, new_path)
        num += 1
    print("copy test LA files to new folder")
    print('total number:', num)
    
    
def VIS_noisy_LA_generation(test_list_path, ori_Dir, corruptVIS_Dir):
    with open(test_list_path, 'r') as f:
        test_image_list = f.readlines()
    num = 0
    for case_name in test_image_list:
        case_name = case_name.replace('\n', '')
        print(case_name)
        ori_path = os.path.join(ori_Dir, case_name+'.h5')
        
        # read the labels, add_noise, and save h5 to new_path
        h5f_ori = h5py.File(ori_path, 'r')
        label_ori = h5f_ori['label']
        dst_label, noise_type = add_noise(label_ori, min_radius=3, max_radius=15)
        print('corruptshape, type:', dst_label.shape, noise_type)
        new_path = os.path.join(corruptVIS_Dir, case_name+'_'+noise_type+'.nii.gz')
        dst_label_itk = sitk.GetImageFromArray(dst_label.astype(np.uint8))
        dst_label_itk.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(dst_label_itk, new_path)
        num += 1
    print('total number:', num)

 
 
if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    clean_num = 8 # clean percentage: 10%, 20%, 30% (80 cases in total)
    train_list_path = '../../data/train.txt'
    test_list_path = '../../data/test.txt'
    ori_Dir = '../../data/LA_data_h5/data/'
    corrupt_Dir = '../../data/LA_corrupt_data_{}_h5/data/'.format(clean_num)
    if not os.path.exists(corrupt_Dir):
        os.makedirs(corrupt_Dir)
    corrupt_VIS_Dir = '../../data/LA_VIS_corrupt_v2/'
    if not os.path.exists(corrupt_VIS_Dir):
        os.makedirs(corrupt_VIS_Dir)
    
    Noisy_3D_LA_generation(clean_num, train_list_path, ori_Dir, corrupt_Dir, test_list_path)
    #VIS_noisy_LA_generation(test_list_path, ori_Dir, corrupt_VIS_Dir)
    

    