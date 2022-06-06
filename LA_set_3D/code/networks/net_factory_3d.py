from networks.unet_3D import unet_3D
from networks.vnet import VNet, VNet_MTPD, VNet_MTPD_voting, MTPD_dual_VNet, Dualstream_VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "attention_unet":
        net = Attention_UNet(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "voxresnet":
        net = VoxResNet(in_chns=in_chns, feature_chns=64,
                        class_num=class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_MTPD":
        net = VNet_MTPD(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_MTPD_voting":
        net = VNet_MTPD_voting(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "MTPD_dual_VNet":
        net = MTPD_dual_VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "dualstream_vnet":
        net = Dualstream_VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    else:
        net = None
    return net
