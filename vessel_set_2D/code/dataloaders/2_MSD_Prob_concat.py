"""
@author: Zhe XU
Data Preprocessing for IRCAD Probability Map Dataset
1. slice extraction for training set
2. Normalization
3. Convert to h5 file
"""


import glob
import os
import re
import h5py
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def findidx(file_name):
    # find the idx
    cop = re.compile("[^0-9]")
    idx = cop.sub('', file_name)
    return idx


def training_concat_2D_slice_process(train_img_Dir, train_prob_Dir, msk_baseDir, organ):
    # training slice generate
    slice_num = 0
    train_img_path = sorted(glob.glob(train_img_Dir))

    for case in train_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case)

        prob_path = os.path.join(train_prob_Dir, 'image_' + str(idx) + '.nii.gz')
        print(prob_path)
        prob_itk = sitk.ReadImage(prob_path)
        prob = sitk.GetArrayFromImage(prob_itk)

        label_file_name = 'image_' + str(idx) + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)

        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)

            print('image shape:', image.shape)
            image = image.astype(np.float32)
            prob = prob.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            for slice_ind in range(image.shape[0]):
                # tutorial of h5py write file: https://blog.csdn.net/yudf2010/article/details/50353292

                # add z to make the MSD image to the bottom of train_slice.txt
                f = h5py.File(
                    '../../data/IRCAD_c/training_slice_{}_concat_SSL_h5/z{}_slice_{}.h5'.format(organ, item, slice_ind), 'w')
                img = np.expand_dims(image[slice_ind], axis=0)
                prob_ = np.expand_dims(prob[slice_ind], axis=0)
                concat_img = np.concatenate((img, prob_), axis=0)
                # print(concat_img.shape)
                f.create_dataset('image', data=concat_img, compression="gzip")
                f.create_dataset('label_{}'.format(organ), data=mask[slice_ind], compression="gzip")
                f.close()
                slice_num += 1
    print("Converted MSD volumes to concatenated 2D slices")
    print("Total {} slices".format(slice_num))


# Noise VIS volume h5 generate
def Noise_VIS_volume_process(test_img_Dir, test_prob_Dir, msk_baseDir, organ, vis_num=20):
    test_img_path = sorted(glob.glob(test_img_Dir))
    for case in test_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case)

        prob_path = os.path.join(test_prob_Dir, 'image_' + str(idx) + '.nii.gz')
        print(prob_path)
        prob_itk = sitk.ReadImage(prob_path)
        prob = sitk.GetArrayFromImage(prob_itk)

        label_file_name = 'image_' + str(idx) + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            print('image shape:', image.shape)
            image = image.astype(np.float32)
            prob = prob.astype(np.float32)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                '../../data/MSD_c/image_{}_noise_concat_VIS_h5/image_{}.h5'.format(organ, str(idx)), 'w')
            concat_img = np.concatenate((image, prob), axis=0)
            print(concat_img.shape)
            f.create_dataset('image', data=concat_img, compression="gzip")
            f.create_dataset('label_{}'.format(organ), data=mask, compression="gzip")
            f.close()
    print("Converted test concatenated IRCAD volumes to h5 files")



if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_img_Dir =  '../../data/MSD_c/image_ROI_ori/*.nii.gz'
    train_prob_Dir = '../../data/MSD_NEW/image_ROI/'

    # New ROI vessel
    msk_baseDir = '../../data/MSD_NEW/label_vessel_ROI/'
    organ = 'ROI'
    # training_concat_2D_slice_process(train_img_Dir, train_prob_Dir, msk_baseDir, organ)

    # Noise CL Visualization preparation
    Noise_VIS_volume_process(train_img_Dir, train_prob_Dir, msk_baseDir, organ)




