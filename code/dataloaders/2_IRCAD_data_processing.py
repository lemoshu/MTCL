"""
@author: Zhe XU
Data Preprocessing for IRCAD Dataset
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

def training_2D_slice_process(train_img_Dir, msk_baseDir, organ):
    # training slice generate
    slice_num = 0
    train_img_path = sorted(glob.glob(train_img_Dir))
    for case in train_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case) # ERROR

        label_file_name = 'image_' + str(idx)[:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            # convert the label from 255 to 1
            # mask = mask / 255
            # print('mask shape:', mask.shape)
            #
            # # Normalize the image
            # if organ == 'liver':
            #     image = CT_liver_normalize(image)
            # else:
            #     image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            for slice_ind in range(image.shape[0]):
                # tutorial of h5py write file: https://blog.csdn.net/yudf2010/article/details/50353292
                f = h5py.File(
                    '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_NEW/training_slice_{}_h5/{}_slice_{}.h5'.format(organ, item, slice_ind), 'w')
                f.create_dataset('image', data=image[slice_ind], compression="gzip")
                f.create_dataset('label_{}'.format(organ), data=mask[slice_ind], compression="gzip")
                f.close()
                slice_num += 1
    print("Converted IRCAD volumes to 2D slices")
    print("Total {} slices".format(slice_num))


# val volume h5 generate
def val_volume_process(val_img_Dir, msk_baseDir, organ):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[1:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            # convert the label from 255 to 1
            # mask = mask / 255
            # print('mask shape:', mask.shape)
            #
            # # Normalize the image
            # if organ == 'liver':
            #     image = CT_liver_normalize(image)
            # else:
            #     image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_NEW/val_{}_h5/image_{}.h5'.format(organ, str(idx)[1:]), 'w')
            f.create_dataset('image', data=image, compression="gzip")
            f.create_dataset('label_{}'.format(organ), data=mask, compression="gzip")
            f.close()
    print("Converted val IRCAD volumes to h5 slices")


# test volume h5 generate
def test_volume_process(test_img_Dir, msk_baseDir, organ):
    test_img_path = sorted(glob.glob(test_img_Dir))
    for case in test_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[1:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            # convert the label from 255 to 1
            # mask = mask / 255
            # print('mask shape:', mask.shape)
            #
            # Normalize the image
            # if organ == 'liver':
            #     image = CT_liver_normalize(image)
            # else:
            #     image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_NEW/test_{}_h5/image_{}.h5'.format(organ, str(idx)[1:]), 'w')
            f.create_dataset('image', data=image, compression="gzip")
            f.create_dataset('label_{}'.format(organ), data=mask, compression="gzip")
            f.close()
    print("Converted test IRCAD volumes to h5 slices")

# option: train volume h5 generate
def train_volume_process(test_img_Dir, msk_baseDir, organ):
    test_img_path = sorted(glob.glob(test_img_Dir))
    for case in test_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path, better than the re, sub code
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[1:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            # convert the label from 255 to 1
            # mask = mask / 255
            # print('mask shape:', mask.shape)
            #
            # # Normalize the image
            # if organ == 'liver':
            #     image = CT_liver_normalize(image)
            # else:
            #     image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/train_{}_h5/image_{}.h5'.format(organ, str(idx)[1:]), 'w')
            f.create_dataset('image', data=image, compression="gzip")
            f.create_dataset('label_{}'.format(organ), data=mask, compression="gzip")
            f.close()
    print("Option: Converted train IRCAD volumes to h5 slices")

# Preprocessing Library
def CT_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    MIN_BOUND = -40.0
    MAX_BOUND = 180.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0.43] = 0.
    return nii_data


if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_img_Dir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/image_ROI_ori_train/*.nii.gz'
    val_img_Dir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/image_ROI_ori_val/*.nii.gz'
    test_img_Dir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/image_ROI_ori_test/*.nii.gz'

    # # portalvein
    # msk_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/label_portalvein/'
    # organ = 'portalvein'
    #
    # training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    # val_volume_process(val_img_Dir, msk_baseDir, organ)
    # test_volume_process(test_img_Dir, msk_baseDir, organ)
    #
    # # venacava
    # msk_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/label_venacava/'
    # organ = 'venacava'
    #
    # training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    # val_volume_process(val_img_Dir, msk_baseDir, organ)
    # test_volume_process(test_img_Dir, msk_baseDir, organ)

    # liver
    # msk_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/label_liver/'
    # organ = 'liver'
    # training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    # val_volume_process(val_img_Dir, msk_baseDir, organ)
    # test_volume_process(test_img_Dir, msk_baseDir, organ)

    # New ROI vessel
    msk_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/label_vessel_ROI/'
    organ = 'ROIori'
    training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    val_volume_process(val_img_Dir, msk_baseDir, organ)
    test_volume_process(test_img_Dir, msk_baseDir, organ)

    # Option: train set to volume to see if overfit?
    # train_volume_process(train_img_Dir, msk_baseDir, organ)

    ## test
    # file_path = '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/image/image_01.nii.gz'
    # nii_initial = nib.load(file_path)
    # nii_data = nii_initial.get_data()
    # nii_data_p = CT_normalize(nii_data)
    #
    # img = nib.Nifti1Image(nii_data_p, nii_initial.affine)
    # nib.save(img, '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/preprocess_test/image_01_n.nii.gz')




