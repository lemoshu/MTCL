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
# import nibabel as nib

# ROI bounding box extract lib
from skimage.measure import label
from skimage.measure import regionprops

# Preprocess Vessel
from skimage.filters import frangi, hessian, sato

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


# Preprocessing Library
def normalize_after_prob(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = np.min(nii_data)
    MAX_BOUND = np.max(nii_data)

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data


def VesselEnhance(img, type):
    if type == 'sato':
        filter_img = sato(img, sigmas=range(1, 4, 1), black_ridges=False, mode='constant')
    elif type == 'frangi':
        filter_img = frangi(img, sigmas=range(1, 4, 1), scale_range=None,
                            scale_step=None, alpha=0.5, beta=0.5, gamma=5, black_ridges=False, mode='constant', cval=1)
    return filter_img


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

# all in one
# Input: Abdominal CT
# Output: Concatenated h5 file
def ROI_crop_sato_concat_preprocess(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ, mode):
    val_img_path = sorted(glob.glob(val_img_Dir))
    # print(val_img_path)
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        # direction = img_itk.GetDirection()
        
        # img_itk, new_spacing = Resampling(img_itk, label=False)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'image_' + str(idx)[:] + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name)
        vessel_msk2_path = os.path.join(vessel_msk2_baseDir, label_file_name)
        print(msk_path)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            # resampling label
            # msk_itk, new_spacing = Resampling(msk_itk, label=True)
            mask_ = sitk.GetArrayFromImage(msk_itk)
            if np.max(mask_) == 255: # fix the bug that some labels are valued 1 not 255
                mask = mask_ / 255 #fix the bug that did not find the region using normalized masks
            else:
                mask = mask_

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            #resampling label
            # vesselmsk1_itk, new_spacing = Resampling(vesselmsk1_itk, label=True)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            if np.max(vessel_mask1) == 255:
                vessel_mask1 = vessel_mask1 / 255

            vesselmsk2_itk = sitk.ReadImage(vessel_msk2_path)
            # resampling label
            # vesselmsk2_itk, new_spacing = Resampling(vesselmsk2_itk, label=True)
            vessel_mask2 = sitk.GetArrayFromImage(vesselmsk2_itk)
            if np.max(vessel_mask2) == 255:
                vessel_mask2 = vessel_mask2 / 255
            # combine two vessel labels
            vessel_mask = combine_vessel_mask(vessel_mask1, vessel_mask2)

            print('mask shape:', mask.shape)

            # Normalize the image
            image = CT_normalize(image)
            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = mask * image
            vessel_mask = mask * vessel_mask

            # Vessel Enhancement
            prob = VesselEnhance(image, type='sato')
            # Normalize
            prob = normalize_after_prob(prob)
            prob = prob.astype(np.float32)
            print('vessel prob completed')

            # crop the liver area
            # Get the liver area
            box = liver_ROI(mask_)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            prob = crop_ROI(prob, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            item = case.split("/")[-1].split(".")[0]
            # if image.shape != mask.shape:
            #     print("Error")
            print(item)
            print('---------------')
            f = h5py.File(
                    '../../data/IRCAD_c/test_ROI_allinone_h5/image_{}.h5'.format(str(idx)), 'w')
            concat_img = np.concatenate((image, prob), axis=0)
            print(concat_img.shape)
            f.create_dataset('image', data=concat_img, compression="gzip")
            f.create_dataset('label_{}'.format(organ), data=vessel_mask, compression="gzip")
            f.close()
    print("Converted test concatenated IRCAD volumes to h5 files")


if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # train_img_Dir = '../../data/IRCAD/train_image/*.nii.gz'
    # val_img_Dir = '../../data/IRCAD/val_image/*.nii.gz'
    test_img_Dir = '../../data/IRCAD_c/image/*.nii.gz'


    # liver
    ROI_baseDir = '../../data/IRCAD_c/label_liver/'
    vessel_msk1_baseDir = '../../data/IRCAD_c/label_venacava/'
    vessel_msk2_baseDir = '../../data/IRCAD_c/label_portalvein/'
    organ = 'ROI'

    ## crop ROI region
    # val process
    ROI_crop_sato_concat_preprocess(test_img_Dir, ROI_baseDir, vessel_msk1_baseDir, vessel_msk2_baseDir, organ,  mode='test')



