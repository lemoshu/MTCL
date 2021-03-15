"""
A Exploratory Experiment on how to use CleanLab API to characterize the noise in labels

Implemented by Zhe Xu (Jan.2021)

Reference:
Simple Tutorial: https://l7.curtisnorthcutt.com/cleanlab-python-package
Prune Config: https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/pruning.py


"""
import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
import cleanlab

# post-process lib
from skimage import measure

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/MSD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MT_concat_fixCL3_14_labeled', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='model_name')
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')


def test_single_concat_volume(case, net, test_save_path):
    # get image spacing
    img_itk = sitk.ReadImage(FLAGS.root_path + "/image_ROI_ori/{}.nii.gz".format(case))
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()

    h5f = h5py.File(FLAGS.root_path + "/image_ROI_noise_concat_VIS_h5/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label_ROI'][:]

    preds_softmax_np_accumulated = 0
    preds_np_accumulated = 0
    prediction = np.zeros_like(label)

    for ind in range(int(image.shape[0]/2)):
        slice = image[ind, :, :] # from [0,115), assume that total slices for one img volume is 115
        prob_slice = image[int(image.shape[0]/2)+ind, :, :] # from [115, 230)
        img = np.expand_dims(slice, axis=0)
        prob_ = np.expand_dims(prob_slice, axis=0)
        concat_input = np.concatenate((img, prob_), axis=0) # (2, H, W)
        input = torch.from_numpy(concat_input).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            pred_mask = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            pred_mask = pred_mask.cpu().detach().numpy()
            prediction[ind] = pred_mask
            pred_np = out_main.cpu().detach().numpy() #(1, 2, 320, 320)
            pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy() #(1, 2, 320, 320)
            print('Shape of pred_np:', pred_np.shape)
            print('Shape of pred_soft_np:', pred_soft_np.shape)
        if ind == 0:
            preds_np_accumulated = pred_np
            preds_softmax_np_accumulated = pred_soft_np
        else:
            preds_np_accumulated = np.concatenate((preds_np_accumulated, pred_np), axis=0)
            preds_softmax_np_accumulated = np.concatenate((preds_softmax_np_accumulated, pred_soft_np), axis=0)

    print('Shape of preds_np_accumulated:', preds_np_accumulated.shape)  # (115, 2, 320, 320)
    print('Shape of preds_softmax_np_accumulated:', preds_softmax_np_accumulated.shape) # (115, 2, 320, 320)

    # permulate the dimension
    preds_np_accumulated = np.swapaxes(preds_np_accumulated, 1, 2)
    preds_np_accumulated = np.swapaxes(preds_np_accumulated, 2, 3)
    preds_np_accumulated = preds_np_accumulated.reshape(-1, FLAGS.num_classes)
    preds_np_accumulated = np.ascontiguousarray(preds_np_accumulated) #(11776000, 2)
    print('(After_4)Shape of preds_np_accumulated:', preds_np_accumulated.shape)

    preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 1, 2)
    preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
    preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, FLAGS.num_classes)
    preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated) #(11776000, 2)
    print('(After_4)Shape of preds_softmax_np_accumulated:', preds_softmax_np_accumulated.shape)

    masks_np_accumulated = label.reshape(-1).astype(np.uint8) #(11776000,)
    print('(After_Reshape)Shape of masks_np_accumulated:', masks_np_accumulated.shape)

    assert preds_np_accumulated.shape[0] == masks_np_accumulated.shape[0] == preds_softmax_np_accumulated.shape[0]

    CL_type = FLAGS.CL_type

    if CL_type in ['both', 'Qij']:
        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                   prune_method='both', n_jobs=1)
    elif CL_type == 'Cij':
        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_np_accumulated, prune_method='both',
                                                   n_jobs=1)
    elif CL_type == 'intersection':
        noise_qij = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                       prune_method='both', n_jobs=1)
        noise_cij = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_np_accumulated, prune_method='both',
                                                       n_jobs=1)
        noise = noise_qij & noise_cij
    elif CL_type == 'union':
        noise_qij = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                       prune_method='both', n_jobs=1)
        noise_cij = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_np_accumulated, prune_method='both',
                                                       n_jobs=1)
        noise = noise_qij | noise_cij
    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated, prune_method=CL_type,
                                                   n_jobs=1)

    print('Number of estimated errors in training set:', sum(noise))
    print('Shape of Noise:', noise.shape)
    confident_maps_np = noise.reshape(-1, image.shape[1], image.shape[2]).astype(np.uint8) # (115, 320, 320)
    print('Shape of confident_maps_np:', confident_maps_np.shape)

    # Correct the label
    correct_type = 'hard'
    if correct_type == 'smooth':
        smooth_arg = 0.8
        label_corrected = label + confident_maps_np * np.power(-1, label) * smooth_arg
    else:
        label_corrected = label + confident_maps_np * np.power(-1, label)

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing(spacing)
    prd_itk.SetDirection(direction)
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")

    confident_map_itk = sitk.GetImageFromArray(confident_maps_np.astype(np.float32))
    confident_map_itk.SetSpacing(spacing)
    confident_map_itk.SetDirection(direction)
    sitk.WriteImage(confident_map_itk, test_save_path + case + "_noise.nii.gz")

    label_corrected_itk = sitk.GetImageFromArray(label_corrected.astype(np.float32))
    label_corrected_itk.SetSpacing(spacing)
    label_corrected_itk.SetDirection(direction)
    sitk.WriteImage(label_corrected_itk, test_save_path + case + "_corrected_label.nii.gz")


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test_noise.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/IRCAD_c/{}/{}".format(FLAGS.exp, FLAGS.model)
    test_save_path = "../model/IRCAD_c/{}/NoiseCorrect/".format(FLAGS.exp)
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

    for case in tqdm(image_list):
        print(case)
        test_single_concat_volume(case, net, test_save_path)
        print('Finish')


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    FLAGS = parser.parse_args()
    errors = Inference(FLAGS)
    print(errors)


    """Returns the indices of most likely (confident) label errors in s. The
    number of indices returned is specified by frac_of_noise. When
    frac_of_noise = 1.0, all "confident" estimated noise indices are returned.
    * If you encounter the error 'psx is not defined', try setting n_jobs = 1.
    Parameters
    ----------
    s : np.array
      A binary vector of labels, s, which may contain mislabeling. "s" denotes
      the noisy label instead of \tilde(y), for ASCII encoding reasons.
    psx : np.array (shape (N, K))
      P(s=k|x) is a matrix with K (noisy) probabilities for each of the N
      examples x.
      This is the probability distribution over all K classes, for each
      example, regarding whether the example has label s==k P(s=k|x).
      psx should have been computed using 3+ fold cross-validation.
    inverse_noise_matrix : np.array of shape (K, K), K = number of classes
      A conditional probability matrix of the form P(y=k_y|s=k_s) representing
      the estimated fraction observed examples in each class k_s, that are
      mislabeled examples from every other class k_y. If None, the
      inverse_noise_matrix will be computed from psx and s.
      Assumes columns of inverse_noise_matrix sum to 1.
    confident_joint : np.array (shape (K, K), type int) (default: None)
      A K,K integer matrix of count(s=k, y=k). Estimates a a confident
      subset of the joint distribution of the noisy and true labels P_{s,y}.
      Each entry in the matrix contains the number of examples confidently
      counted into every pair (s=j, y=k) classes.
    frac_noise : float
      When frac_of_noise = 1.0, return all "confident" estimated noise indices.
      Value in range (0, 1] that determines the fraction of noisy example
      indices to return based on the following formula for example class k.
      frac_of_noise * number_of_mislabeled_examples_in_class_k, or equivalently
      frac_of_noise * inverse_noise_rate_class_k * num_examples_with_s_equal_k
    num_to_remove_per_class : list of int of length K (# of classes)
      e.g. if K = 3, num_to_remove_per_class = [5, 0, 1] would return
      the indices of the 5 most likely mislabeled examples in class s = 0,
      and the most likely mislabeled example in class s = 1.
      ***Only set this parameter if prune_method == 'prune_by_class'
      You may use with prune_method == 'prune_by_noise_rate', but
      if num_to_remove_per_class == k, then either k-1, k, or k+1
      examples may be removed for any class. This is because noise rates
      are floats, and rounding may cause a one-off. If you need exactly
      'k' examples removed from every class, you should use 'prune_by_class'.
    prune_method : str (default: 'prune_by_noise_rate')
      Possible Values: 'prune_by_class', 'prune_by_noise_rate', or 'both'.
      Method used for pruning.
      1. 'prune_by_noise_rate': works by removing examples with
      *high probability* of being mislabeled for every non-diagonal
      in the prune_counts_matrix (see pruning.py).
      2. 'prune_by_class': works by removing the examples with *smallest
      probability* of belonging to their given class label for every class.
      3. 'both': Finds the examples satisfying (1) AND (2) and
      removes their set conjunction.
    sorted_index_method : str [None, 'prob_given_label', 'normalized_margin']
      If None, returns a boolean mask (true if example at index is label error)
      If not None, returns an array of the label error indices
      (instead of a bool mask) where error indices are ordered by the either:
        'normalized_margin' := normalized margin (p(s = k) - max(p(s != k)))
        'prob_given_label' := [psx[i][labels[i]] for i in label_errors_idx]
    multi_label : bool
      If true, s should be an iterable (e.g. list) of iterables, containing a
      list of labels for each example, instead of just a single label.
    n_jobs : int (Windows users may see a speed-up with n_jobs = 1)
      Number of processing threads used by multiprocessing. Default None
      sets to the number of processing threads on your CPU.
      Set this to 1 to REMOVE parallel processing (if its causing issues).
    verbose : int
      If 0, no print statements. If 1, prints when multiprocessing happens."""

