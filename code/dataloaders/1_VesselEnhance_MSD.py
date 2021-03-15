"""
Sato: https://www.kite.com/python/docs/skimage.filters.sato

We should use the 'vessel' environment as the skimage should be 0.18.1


"""

import glob
import os
import re
import numpy as np
import SimpleITK as sitk

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


def choose_mask(mask_npy):
    target = [1]
    ix = np.isin(mask_npy, target) # bool array
    # print(ix)
    idx = np.where(ix)
    idx_inv = np.where(~ix) # inverse the bool array
    # print(idx)
    mask_npy[idx] = 1
    mask_npy[idx_inv] = 0
    return mask_npy


def liver_ROI(mask_npy):
    # regionprops tutorial: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
    labeled_img, num = label(mask_npy, return_num=True)
    print(labeled_img.shape)
    print('There are {} regions'.format(num))
    # print(np.max(labeled_img))
    if num > 0:
        regions = regionprops(labeled_img, cache=True)
        for prop in regions:
            box = prop.bbox  # Bounding box (min_row, min_col, max_row, max_col)
            area = prop.area  # Number of pixels of the region
            ratio = prop.extent  # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
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
    npy_data_aftercrop = npy_data[zmin:zmax, xmin - 5:xmin + 315, ymin - 5:ymin + 315]
    print('crop size:', npy_data_aftercrop.shape)
    return npy_data_aftercrop


def ROI_crop_preprocess(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, organ='ROI'):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()

        # Resampling the img to 1x1x1
        # img_itk, new_spacing = Resampling(img_itk, label=False)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'hepaticvessel_' + str(idx)[:]  + '.nii.gz'
        liver_msk_path = os.path.join(msk_baseDir, label_file_name) # liver
        print(liver_msk_path)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name) # vessel(2) and tumor(2)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(liver_msk_path):
            print(liver_msk_path)
            liver_msk_itk = sitk.ReadImage(liver_msk_path)
            liver_mask = sitk.GetArrayFromImage(liver_msk_itk)
            print('mask shape:', liver_mask.shape)

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            vessel_mask1 = choose_mask(vessel_mask1)

            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = liver_mask * image
            vessel_mask = liver_mask * vessel_mask1

            # Vessel Enhancement
            image = VesselEnhance(image, type='sato')
            # Normalize
            image = normalize_after_prob(image)

            # crop the liver area
            box = liver_ROI(liver_mask)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            mask = crop_ROI(liver_mask, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('-'*20)
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            img_itk.SetSpacing(spacing)
            img_itk.SetDirection(direction)
            label_itk = sitk.GetImageFromArray(mask.astype(np.float32))
            label_itk.SetSpacing(spacing)
            label_itk.SetDirection(direction)
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            vessel_label_itk.SetDirection(direction)
            sitk.WriteImage(img_itk, '/home/xuzhe/Segment/SSL4MIS/data/MSD_NEW/image_ROI/image_{}.nii.gz'.format(str(idx)[1:]))
            # sitk.WriteImage(label_itk,
            #                 '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/label_ROI_{}/image_{}_gt.nii.gz'.format(mode, str(idx)[1:]))
            sitk.WriteImage(vessel_label_itk,
                            '/home/xuzhe/Segment/SSL4MIS/data/MSD_NEW/label_vessel_ROI/image_{}_gt.nii.gz'.format(str(idx)[1:]))
    print("Converted MSD volumes to ROI")


def ROI_crop_preprocess_F(val_img_Dir, msk_baseDir, vessel_msk1_baseDir, organ='ROI'):
    val_img_path = sorted(glob.glob(val_img_Dir))
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()

        # Resampling the img to 1x1x1
        # img_itk, new_spacing = Resampling(img_itk, label=False)
        image = sitk.GetArrayFromImage(img_itk)
        # change to mask path
        idx = findidx(case)
        label_file_name = 'hepaticvessel_' + str(idx)[1:]  + '.nii.gz'
        liver_msk_path = os.path.join(msk_baseDir, label_file_name) # liver
        print(liver_msk_path)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name) # vessel(2) and tumor(2)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(liver_msk_path):
            print(liver_msk_path)
            liver_msk_itk = sitk.ReadImage(liver_msk_path)
            liver_mask = sitk.GetArrayFromImage(liver_msk_itk)
            print('mask shape:', liver_mask.shape)

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            vessel_mask1 = choose_mask(vessel_mask1)

            print('image shape:', image.shape)
            image = image.astype(np.float32)

            # mask the liver?
            image = liver_mask * image
            vessel_mask = liver_mask * vessel_mask1

            # Vessel Enhancement
            image = VesselEnhance(image, type='frangi')
            # Normalize
            image = normalize_after_prob(image)

            # crop the liver area
            box = liver_ROI(liver_mask)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            mask = crop_ROI(liver_mask, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print(item)
            print('-'*20)
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            img_itk.SetSpacing(spacing)
            img_itk.SetDirection(direction)
            label_itk = sitk.GetImageFromArray(mask.astype(np.float32))
            label_itk.SetSpacing(spacing)
            label_itk.SetDirection(direction)
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            vessel_label_itk.SetDirection(direction)
            sitk.WriteImage(img_itk, '/home/xuzhe/Segment/SSL4MIS/data/MSD_NEW/image_ROI_F/image_{}.nii.gz'.format(str(idx)[1:]))
            # sitk.WriteImage(label_itk,
            #                 '/home/xuzhe/Segment/SSL4MIS/data/IRCAD/label_ROI_{}/image_{}_gt.nii.gz'.format(mode, str(idx)[1:]))
            # sitk.WriteImage(vessel_label_itk,
            #                 '/home/xuzhe/Segment/SSL4MIS/data/MSD_NEW/label_vessel_ROI/image_{}_gt.nii.gz'.format(str(idx)[1:]))
    print("Converted MSD volumes to F_ROI")

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
    if type == 'sato': # https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sato
        filter_img = sato(img, sigmas=range(1, 3, 1), black_ridges=False, mode='constant') #mode{‘constant’, ‘reflect’, ‘wrap’, ‘nearest’, ‘mirror’}
    elif type == 'frangi':
        filter_img = frangi(img, sigmas=range(1, 4, 1), scale_range=None,
                            scale_step=None, alpha=0.5, beta=0.5, gamma=5, black_ridges=False, mode='constant', cval=1)
    return filter_img


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    train_img_Dir = '/home/xuzhe/Segment/SSL4MIS/data/MSD/imagesTr/*.nii.gz'

    # liver
    ROI_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/MSD/imagesTr_Seg/'
    vessel_msk1_baseDir = '/home/xuzhe/Segment/SSL4MIS/data/MSD/labelsTr/'

    # ROI_crop_preprocess(train_img_Dir, ROI_baseDir, vessel_msk1_baseDir)

    ROI_crop_preprocess_F(train_img_Dir, ROI_baseDir, vessel_msk1_baseDir)

"""
Filter an image with the Sato tubeness filter.

This filter can be used to detect continuous ridges, e.g. tubes,
wrinkles, rivers. It can be used to calculate the fraction of the
whole image containing such objects.

Defined only for 2-D and 3-D images. Calculates the eigenvectors of the
Hessian to compute the similarity of an image region to tubes, according to
the method described in [1]_.

Parameters
----------
image : (N, M[, P]) ndarray
    Array with input image data.
sigmas : iterable of floats, optional
    Sigmas used as scales of filter.
black_ridges : boolean, optional
    When True (the default), the filter detects black ridges; when
    False, it detects white ridges.

Returns
-------
out : (N, M[, P]) ndarray
    Filtered image (maximum of pixels across all scales).

References
----------
.. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
    Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
    filter for segmentation and visualization of curvilinear structures in
    medical images. Medical image analysis, 2(2), 143-168.
    :DOI:`10.1016/S1361-8415(98)80009-1`




"""