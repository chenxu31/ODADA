import random
import numpy as np
import numpy
import torch
import platform
import skimage.io
import os
import sys
import pdb
from datetime import datetime

if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_pelvic_pt as common_pelvic


def set_rand_seed(seed=1):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 保证每次返回得的卷积算法是确定的


set_rand_seed(seed=1)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import math
# import visdom
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable

from distutils.version import LooseVersion

from Models.model_mia1201 import UNet_DA
# import lib

from metrics import LogNLLLoss

from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC
from torch.optim import lr_scheduler
from data import PrepareDataset, Rescale, ToTensor, Normalize
from time import *
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch import nn

from itertools import cycle

from network import deeplabv3plus_resnet50, deeplabv3_resnet50


def main(device, args):
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.num_input = 3
    if args.task == "pelvic":
        common_file = common_pelvic

        _, test_data_t, _, test_label_t = common_pelvic.load_test_data(args.root_path)
    else:
        assert 0

    num_classes = common_file.NUM_CLASSES
    print('Loading is done\n')

#     model = UNet_DA()
    model = deeplabv3_resnet50(num_classes=num_classes)
    ckpt = torch.load(os.path.join(args.ckpt), map_location='cpu')
    model.load_state_dict(ckpt.get("state_dict"))
    model.eval()
    model.to(device)

    patch_shape = (args.num_input, test_data_t.shape[2], test_data_t.shape[3])
    test_t_dsc = numpy.zeros((test_data_t.shape[0], num_classes - 1), numpy.float32)
    test_t_assd = numpy.zeros((test_data_t.shape[0], num_classes - 1), numpy.float32)
    with torch.no_grad():
        for i in range(test_data_t.shape[0]):
            pred = common_net.produce_results(device, lambda x: model(x)[0].softmax(1), [patch_shape, ],
                                              [test_data_t[i], ], data_shape=test_data_t[i].shape,
                                              patch_shape=patch_shape, is_seg=True, num_classes=num_classes)
            pred = pred.argmax(0).astype(numpy.float32)
            test_t_dsc[i] = common_metrics.calc_multi_dice(pred, test_label_t[i], num_cls=num_classes)
            test_t_assd[i] = common_metrics.calc_multi_assd(pred, test_label_t[i], num_cls=num_classes)

            if args.output_dir:
                common_file.save_nii(pred, os.path.join(args.output_dir, "syn_%d.nii.gz" % i))

    msg = "test_t_dsc:%f/%f  test_t_assd:%f/%f" % \
          (test_t_dsc.mean(), test_t_dsc.std(), test_t_assd.mean(), test_t_assd.std())
    print(msg)

    if args.output_dir:
        with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
            f.write(msg)

        numpy.save(os.path.join(args.output_dir, "test_t_dsc.npy"), test_t_dsc)
        numpy.save(os.path.join(args.output_dir, "test_t_assd.npy"), test_t_assd)


if __name__ == '__main__':


    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
    parser = argparse.ArgumentParser(description='ODADA')

    parser.add_argument('--id', default="UNet_DA",
                        help='Unet...')  # 模型名字
    parser.add_argument('--id_load', default="UNet_individual",
                        help='Unet...')
    # Path related arguments
    parser.add_argument('--root_path', default='/home/chenxu/datasets/pelvic/h5_data',
                        help='root directory of data')
    parser.add_argument('--ckpt',
                        default='/home/chenxu/training/checkpoints/odada/pelvic',
                        help='folder to output checkpoints')  # 模型保存的文件夹
    parser.add_argument('--output_dir', default='', help='output_dir')

    parser.add_argument('--task', default='pelvic', choices=["pelvic"], help='task')
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str,
                        help='which cross validation folder')  # 五折训练


    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')  # 训练数据

    args = parser.parse_args()

    print('Models are saved at %s' % (args.ckpt))
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
