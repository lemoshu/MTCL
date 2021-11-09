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

# ROI bounding box extract lib
from skimage.measure import label
from skimage.measure import regionprops


def findidx(file_name):
    # find the idx
    cop = re.compile("[^0-9]")
    idx = cop.sub('', file_name)
    return idx

def combine_vessel_mask(mask1, mask2):
    mask = mask1 + mask2
    mask[mask >= 1] = 1
    return mask

def liver_ROI(mask_npy):
    # regionprops tutorial: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
    labeled_img, num = label(mask_npy, return_num=True)
    print(labeled_img.shape)
    print('There are {} regions'.format(num))
    # print(np.max(labeled_img))
    if num > 0:
        regions = regionprops(labeled_img, cache=True)
        for prop in regions:
            box = prop.bbox #Bounding box (min_row, min_col, max_row, max_col)
            area = prop.area #Number of pixels of the region
            ratio = prop.extent #Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
            print(box)
            print(area)
            print(ratio)
            # print(centroid)
            if area >= 500:
                return box

def crop_ROI(npy_data, box):
    xmin, xmax = box[1], box[4]
    ymin, ymax = box[2], box[5]
    zmin, zmax = box[0], box[3]
    # crop to z x 320 x 320
    npy_data_aftercrop = npy_data[zmin:zmax, xmin-5:xmin+315, ymin-5:ymin+315]
    print('crop size:', npy_data_aftercrop.shape)
    return npy_data_aftercrop

# val volume h5 generate
def ROI_preprocess(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        # direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        
        # change to mask path
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name)
        vessel_msk2_path = os.path.join(vessel_msk2_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk) / 255

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk) / 255

            vesselmsk2_itk = sitk.ReadImage(vessel_msk2_path)
            vessel_mask2 = sitk.GetArrayFromImage(vesselmsk2_itk) / 255
            # combine two vessel labels
            vessel_mask = combine_vessel_mask(vessel_mask1, vessel_mask2)

            print('mask shape:', mask.shape)

            # Normalize the image
            if organ == 'liver':
                image = CT_liver_normalize(image)
            else:
                image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = mask * image
            vessel_mask = mask * vessel_mask

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            img_itk.SetSpacing(spacing)
            label_itk = sitk.GetImageFromArray(mask.astype(np.float32))
            label_itk.SetSpacing(spacing)
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            sitk.WriteImage(img_itk, '../../data/IRCAD/image_ROI_{}/image_{}.nii.gz'.format(mode, str(idx)[1:]))
            sitk.WriteImage(vessel_label_itk,
                            '../../data/IRCAD/label_vessel_ROI/image_{}_gt.nii.gz'.format(str(idx)[1:]))
    print("Converted val IRCAD volumes to ROI")

# val volume h5 generate
def ROI_crop_preprocess(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        # direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[1:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name)
        vessel_msk2_path = os.path.join(vessel_msk2_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask_ = sitk.GetArrayFromImage(msk_itk)
            if np.max(mask_) == 255: # fix the bug that some labels are valued 1 not 255
                mask = mask_ / 255 #fix the bug that did not find the region using normalized masks
            else:
                mask = mask_

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            if np.max(vessel_mask1) == 255:
                vessel_mask1 = vessel_mask1 / 255

            vesselmsk2_itk = sitk.ReadImage(vessel_msk2_path)
            vessel_mask2 = sitk.GetArrayFromImage(vesselmsk2_itk)
            if np.max(vessel_mask2) == 255:
                vessel_mask2 = vessel_mask2 / 255
            # combine two vessel labels
            vessel_mask = combine_vessel_mask(vessel_mask1, vessel_mask2)

            print('mask shape:', mask.shape)

            # Normalize the image
            if organ == 'liver':
                image = CT_liver_normalize(image)
            else:
                image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = mask * image
            vessel_mask = mask * vessel_mask

            # crop the liver area
            # Get the liver area
            box = liver_ROI(mask_)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            mask = crop_ROI(mask, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            img_itk.SetSpacing(spacing)
            label_itk = sitk.GetImageFromArray(mask.astype(np.float32))
            label_itk.SetSpacing(spacing)
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            sitk.WriteImage(img_itk, '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/image_ROI_{}_ori/image_{}.nii.gz'.format(mode, str(idx)[1:]))
            sitk.WriteImage(vessel_label_itk,
                            '/home/xuzhe/Segment/SSL4MIS/data/IRCAD_c/label_vessel_ROI/image_{}_gt.nii.gz'.format(str(idx)[1:]))
    print("Converted val IRCAD volumes to ROI")


def ROI_crop_preprocess_nosplit(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        # direction = img_itk.GetDirection()

        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[1:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name)
        vessel_msk2_path = os.path.join(vessel_msk2_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask_ = sitk.GetArrayFromImage(msk_itk)
            if np.max(mask_) == 255: # fix the bug that some labels are valued 1 not 255
                mask = mask_ / 255 #fix the bug that did not find the region using normalized masks
            else:
                mask = mask_

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            if np.max(vessel_mask1) == 255:
                vessel_mask1 = vessel_mask1 / 255

            vesselmsk2_itk = sitk.ReadImage(vessel_msk2_path)
            vessel_mask2 = sitk.GetArrayFromImage(vesselmsk2_itk)
            if np.max(vessel_mask2) == 255:
                vessel_mask2 = vessel_mask2 / 255
            # combine two vessel labels
            vessel_mask = combine_vessel_mask(vessel_mask1, vessel_mask2)

            print('mask shape:', mask.shape)

            # Normalize the image
            if organ == 'liver':
                image = CT_liver_normalize(image)
            else:
                image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = mask * image
            vessel_mask = mask * vessel_mask

            # crop the liver area
            # Get the liver area
            box = liver_ROI(mask_)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            mask = crop_ROI(mask, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('---------------')
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            img_itk.SetSpacing(spacing)
            label_itk = sitk.GetImageFromArray(mask.astype(np.float32))
            label_itk.SetSpacing(spacing)
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            sitk.WriteImage(img_itk, '../../data/IRCAD_c/image_ROI_ori/image_{}.nii.gz'.format(str(idx)[1:]))
            sitk.WriteImage(vessel_label_itk,
                            '../../data/IRCAD_c/label_vessel_ROI/image_{}_gt.nii.gz'.format(str(idx)[1:]))
    print("Converted val IRCAD volumes to ROI")


# Preprocessing Library
def CT_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = -75.0
    MAX_BOUND = 250.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data


# Preprocessing Library
def CT_liver_normalize(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    MIN_BOUND = -100.0
    MAX_BOUND = 400.0

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data


if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_img_Dir = '../../data/IRCAD/train_image/*.nii.gz'
    val_img_Dir = '../../data/IRCAD/val_image/*.nii.gz'
    test_img_Dir = '../../data/IRCAD/test_image/*.nii.gz'

    #
    # training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    # val_volume_process(val_img_Dir, msk_baseDir, organ)
    # test_volume_process(test_img_Dir, msk_baseDir, organ)
    #
    # training_2D_slice_process(train_img_Dir, msk_baseDir, organ)
    # val_volume_process(val_img_Dir, msk_baseDir, organ)
    # test_volume_process(test_img_Dir, msk_baseDir, organ)

    # liver
    ROI_baseDir = '../../data/IRCAD/label_liver/'
    vessel_msk1_baseDir = '../../data/IRCAD/label_venacava/'
    vessel_msk2_baseDir = '../../data/IRCAD/label_portalvein/'
    organ = 'vessel'

    # # train process
    # ROI_preprocess(train_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='train')
    #
    # # val process
    # ROI_preprocess(val_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='val')
    #
    # # test process
    # ROI_preprocess(test_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='test')

    ## crop ROI region
    # # train process
    # ROI_crop_preprocess(train_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='train')
    #
    # # val process
    # ROI_crop_preprocess(val_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='val')
    #
    # # test process
    # ROI_crop_preprocess(test_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode='test')


    # preprocess all cases
    all_img_Dir = '../../data/IRCAD/image/*.nii.gz'
    ROI_crop_preprocess_nosplit(all_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ)


