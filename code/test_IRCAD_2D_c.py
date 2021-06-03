import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# post-process lib
from skimage import measure
import scipy.ndimage as nd

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/IRCAD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MTCL_c', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='model_name')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    print('Dice, HD95 for this case:', dice, hd95)
    return dice, hd


# If use the all volume metric, you should concat all the test volumes
def calculate_metric_all_volume(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    print('Dice, HD95 for this case:', dice, hd95)
    return dice, hd95


def post_processing(prediction):
    label_cc, num_cc = measure.label(prediction, return_num=True)
    total_cc = np.sum(prediction)
    for cc in range(1, num_cc+1):
        single_cc = (label_cc == cc)
        single_vol = np.sum(single_cc)
        # remove small regions
        if single_vol/total_cc < 0.001:
            prediction[single_cc] = 0
    return prediction


def test_single_concat_volume(case, net, test_save_path, post_process=False):
    # get image spacing
    img_itk = sitk.ReadImage(FLAGS.root_path + "/image_ROI_ori_test/{}.nii.gz".format(case))
    spacing = img_itk.GetSpacing()

    h5f = h5py.File(FLAGS.root_path + "/test_ROI_concat_h5/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label_ROI'][:]
    prediction = np.zeros_like(label)
    for ind in range(int(image.shape[0]/2)):
        slice = image[ind, :, :] # assume that total slices for one img volume is 115, img is from from [0,115)
        prob_slice = image[int(image.shape[0]/2)+ind, :, :] # prob map is from [115, 230)
        img = np.expand_dims(slice, axis=0)
        prob_ = np.expand_dims(prob_slice, axis=0)
        concat_input = np.concatenate((img, prob_), axis=0) # (2, H, W)
        input = torch.from_numpy(concat_input).unsqueeze(0).float().cuda()
        net.eval()
        # start predict
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out

    if post_process:
        prediction = post_processing(prediction)

    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/IRCAD_c/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/IRCAD_c/{}/{}_Prediction/".format(FLAGS.exp,
                                                         FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=2,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    for case in tqdm(image_list):
        print(case)
        first_metric = test_single_concat_volume(case, net, test_save_path, post_process=True)
        first_total += np.asarray(first_metric)

    avg_metric = first_total / len(image_list)
    return avg_metric


# If you want to test in a whole test spaces, you should use the following function, not the case-by-case function above
def test_whole_volumes(case, net, test_save_path, post_process=False):
    # get image spacing
    img_itk = sitk.ReadImage(FLAGS.root_path + "/image_ROI_ori_test/{}.nii.gz".format(case))
    spacing = img_itk.GetSpacing()

    h5f = h5py.File(FLAGS.root_path + "/test_ROI_concat_h5/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label_ROI'][:]
    prediction = np.zeros_like(label)
    for ind in range(int(image.shape[0]/2)):
        slice = image[ind, :, :] # from [0,115), assume that total slices for one img volume is 115
        prob_slice = image[int(image.shape[0]/2)+ind, :, :] # from [115, 230)
        img = np.expand_dims(slice, axis=0)
        prob_ = np.expand_dims(prob_slice, axis=0)
        concat_input = np.concatenate((img, prob_), axis=0) # (2, H, W)
        input = torch.from_numpy(concat_input).unsqueeze(0).float().cuda()
        net.eval()
        # start predict
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out

    if post_process:
        prediction = post_processing(prediction)

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing(spacing)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return label, prediction


def Inference_whole(FLAGS):
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/IRCAD_c/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/IRCAD_c/{}/{}_Prediction/".format(FLAGS.exp,
                                                         FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=2,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    prediction_all = np.zeros((1, 320, 320))
    label_all = np.zeros((1, 320, 320))
    for case in tqdm(image_list):
        print(case)
        label, prediction = test_whole_volumes(case, net, test_save_path, post_process=True)
        prediction_all = np.concatenate((prediction_all, prediction), axis=0)
        label_all = np.concatenate((label_all, label), axis=0)
        print(label_all.shape, prediction_all.shape)  # D x H x W

    avg_metric = calculate_metric_all_volume(prediction_all == 1, label_all == 1)
    return avg_metric


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    FLAGS = parser.parse_args()

    mode = 'whole'
    if mode == 'per_case':
        metric = Inference(FLAGS)
        print('Dice, HD95:', metric)

    elif mode == 'whole':
        metric = Inference_whole(FLAGS)
