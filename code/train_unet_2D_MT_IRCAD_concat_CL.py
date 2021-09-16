"""
Training code for MTCL(c)
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets_IRCAD_SSL_concat, RandomGenerator_IRCAD_concat,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_unet_2D import test_single_concat_volume

# HD loss and boundary loss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

# Confident Learning module
import cleanlab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/IRCAD_c', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='IRCAD_c/MTCL_c', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[2, 320, 320],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1000, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--gpu', type=str, default='2',
                    help='gpu id')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# pretrain
parser.add_argument('--pretrain_model', type=str, default=None, help='pretrained model')

# CL
parser.add_argument('--CL_type', type=str,
                    default='both', help='CL implement type')
args = parser.parse_args()


# BD and HD loss
def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis / np.max(posdis)

    return normalized_dtm


def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def compute_sdf1_1(img_gt, out_shape):
    """
    compute the normalized signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1, 1]
    """

    img_gt = img_gt.astype(np.uint8)

    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                            np.max(posdis) - np.min(posdis))
                sdf[boundary == 1] = 0
                normalized_sdf[b][c] = sdf

    return normalized_sdf


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:, 1, ...]
    dc = gt_sdf[:, 1, ...]
    multipled = torch.mul(pc, dc)
    bd_loss = multipled.mean()

    return bd_loss


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:, 1, ...] - gt.float()) ** 2  # [4, 320, 320]
    # print('delta shape:', delta_s.shape)
    s_dtm = seg_dtm[:, 1, ...] ** 2
    g_dtm = gt_dtm[:, 1, ...] ** 2
    dtm = s_dtm + g_dtm  # [4, 320, 320]
    # print('dtm shape:', dtm.shape)
    multipled = torch.mul(delta_s, dtm)
    # print('dtm shape:', multipled.shape)
    hd_loss = multipled.mean()

    return hd_loss


def labeled_slices(dataset, patiens_num):
    ref_dict = None
    if "IRCAD" in dataset:  # 1-1298 are IRCAD slices, others are MSD slices
        ref_dict = {"10": 1298}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    pretrain_model = args.pretrain_model

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=2,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # using pretrain?
    if pretrain_model:
        model.load_state_dict(torch.load(pretrain_model))
        print("Loaded Pretrained Model")
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets_IRCAD_SSL_concat(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator_IRCAD_concat(args.patch_size)]))

    db_val = BaseDataSets_IRCAD_SSL_concat(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = labeled_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    focal_loss = losses.FocalLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label_ROI']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            # focal loss
            loss_focal = focal_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())

            # boundary loss
            with torch.no_grad():
                # defalut using compute_sdf; however, compute_sdf1_1 is also worth to try;
                gt_sdf_npy = compute_sdf1_1(label_batch.cpu().numpy(), outputs_soft.shape)
                gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs_soft.device.index)
            loss_bd = boundary_loss(outputs_soft[:args.labeled_bs], gt_sdf[:args.labeled_bs])

            supervised_loss = 0.5 * (loss_ce + loss_dice) + loss_focal + 0.5 * loss_bd


            # Confident Learning - weakly supervised Loss
            noisy_label_batch = label_batch[args.labeled_bs:]
            CL_inputs = unlabeled_volume_batch
            if iter_num < 4000:
                loss_ce_weak = 0.0
            elif iter_num >= 4000:
                with torch.no_grad():
                    out_main = ema_model(CL_inputs)
                    pred_soft_np = torch.softmax(out_main, dim=1).cpu().detach().numpy() # (bs,2,H,W)
                    # print('Shape of pred_soft_np:', pred_soft_np.shape)

                masks_np = noisy_label_batch.cpu().detach().numpy() # (bs,2,H,W)
                # print('Shape of masks_np:', masks_np.shape)

                preds_softmax_np_accumulated = np.swapaxes(pred_soft_np, 1, 2)
                preds_softmax_np_accumulated = np.swapaxes(preds_softmax_np_accumulated, 2, 3)
                preds_softmax_np_accumulated = preds_softmax_np_accumulated.reshape(-1, num_classes)
                preds_softmax_np_accumulated = np.ascontiguousarray(preds_softmax_np_accumulated)

                masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)

                assert masks_np_accumulated.shape[0] == preds_softmax_np_accumulated.shape[0]

                CL_type = args.CL_type

                try:
                    if CL_type in ['both', 'Qij']:
                        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                   prune_method='both', n_jobs=1)
                    elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                        noise = cleanlab.pruning.get_noise_indices(masks_np_accumulated, preds_softmax_np_accumulated,
                                                                   prune_method=CL_type,
                                                                   n_jobs=1)

                    confident_maps_np = noise.reshape(-1, 320, 320).astype(np.uint8)  # (bs, 320, 320)

                    # Correct the label
                    correct_type = 'smooth'
                    if correct_type == 'smooth':
                        smooth_arg = 0.8
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                        print('Smoothly correct the noisy label')
                    else:
                        corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)

                    noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(outputs_soft.device.index)
                    # print('Shape of noisy_label_batch:', noisy_label_batch.shape)
                    loss_ce_weak = ce_loss(outputs[args.labeled_bs:], noisy_label_batch.long())
                    loss_focal_weak = focal_loss(outputs[args.labeled_bs:], noisy_label_batch.long())

                    supervised_loss = supervised_loss + 0.5 * (loss_ce_weak + loss_focal_weak)

                    # Iterative Update the Noisy GT
                    loop_type = 'hardloop'
                    if loop_type == 'smoothloop':
                        label_batch[args.labeled_bs:] = noisy_label_batch
                    else:
                        hard_correct_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                        hard_correct_masks = torch.from_numpy(hard_correct_masks_np).cuda(outputs_soft.device.index)
                        label_batch[args.labeled_bs:] = hard_correct_masks

                except Exception as e:
                    loss_ce_weak = loss_ce_weak


            # Unsupervised Consistency Loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)

            # Total Loss = Supervised + Consistency
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_consistency: %f, loss_weak: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), consistency_loss, loss_ce_weak))

            print('-'*50)

            # Validation
            if iter_num > 0 and iter_num % 100 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_concat_volume(
                        sampled_batch["image"], sampled_batch["label_ROI"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
