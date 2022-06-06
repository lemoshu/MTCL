import argparse
import os
import shutil
from glob import glob

import torch
from networks.net_factory_3d import net_factory_3d
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LA_corrupt_data_8_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_MTCL_hard_8', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/{}_Prediction".format(FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory_3d(net_type=FLAGS.model, in_chns=1, class_num=num_classes).cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric, std = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4, test_save_path=test_save_path)
    return avg_metric, std


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    avg_metric, std = Inference(FLAGS)
    print('dice, jc, hd, asd:', avg_metric)
    print('std:', std)
