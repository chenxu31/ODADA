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
from Models.model_mia1201_resnet import UNet_DA_resnet
# import lib

from utils.dice_loss import get_soft_label, val_dice, SoftDiceLoss
from utils.dice_loss import Intersection_over_Union
from utils_new.dice_loss_github import SoftDiceLoss_git, CrossentropyND
from metrics import jaccard_index, f1_score, LogNLLLoss, classwise_f1
from utils_tf import JointTransform2D, ImageToImage2D, Image2D

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
import surface_distance as surfdist
from medpy import metric

from network import deeplabv3plus_resnet50, deeplabv3_resnet50



criterion = "loss_MedT"  # loss_A-->SoftDiceLoss;  loss_B-->softdice_git;  loss_C-->CE_softdice_git


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


criterion_tf = LogNLLLoss()
bce = torch.nn.BCELoss()
device = torch.device("cuda")
"""adding label smoothing"""
real_label=1.
fake_label=0.

def train(trainloader_a, trainloader_b, model, criterion, scheduler, optimizer1, optimizer2, args, epoch):
    losses = AverageMeter()

    model.train()
    for step, data in enumerate(zip(trainloader_a, cycle(trainloader_b))):
    
        
        for name, param in model.named_parameters():
            if "feature_extractor" in name:
                param.requires_grad = True

        (x_a, y_a) = data[0]
        (x_b, y_b) = data[1]
        if not (x_a.shape[0] == args.batch_size):
            # print(step)
            continue
        if not (x_b.shape[0] == args.batch_size):
            # print(step)
            continue
        image_a = Variable(x_a.cuda())
        target_a = Variable(y_a.long().squeeze(dim=1).cuda())
        image_b = Variable(x_b.cuda())
        target_b = Variable(y_b.long().squeeze(dim=1).cuda())

        final_a, _, _, _ = model(x_a)

        loss_seg = criterion_tf(final_a, target_a)


        loss = loss_seg
        losses.update(loss.data, image_a.size(0))

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()


        for name, param in model.named_parameters():
            if "feature_extractor" in name:
                param.requires_grad = False

        (x_a, y_a) = data[0]
        (x_b, y_b) = data[1]
        if not (x_a.shape[0] == args.batch_size):
            continue
        if not (x_b.shape[0] == args.batch_size):
            # print(step)
            continue
        image_a = Variable(x_a.cuda())
        target_a = Variable(y_a.long().squeeze(dim=1).cuda())
        image_b = Variable(x_b.cuda())
        target_b = Variable(y_b.long().squeeze(dim=1).cuda())


        final_a, loss_orthogonal_a, prob_di_a, prob_ds_a = model(x_a, 2)
        final_b, loss_orthogonal_b, prob_di_b, prob_ds_b = model(x_b, 2)

        prob_di_source = torch.full((args.batch_size,), real_label).cuda()
        prob_di_target = torch.full((args.batch_size,), fake_label).cuda()
        prob_ds_source = torch.full((args.batch_size,), real_label).cuda()         
        prob_ds_target = torch.full((args.batch_size,), fake_label).cuda()

        loss_seg = criterion_tf(final_a, target_a)
        loss_class = bce(prob_di_a, prob_di_source) + bce(prob_di_b, prob_di_target) + bce(prob_ds_a, prob_ds_source) + bce(prob_ds_b, prob_ds_target)

        loss = loss_seg + loss_orthogonal_a.mean() + loss_orthogonal_b.mean() + loss_class * 0.1
        losses.update(loss.data, image_a.size(0))

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        

        if step % (math.ceil(float(len(trainloader_a.dataset)) / args.batch_size)) == 0:
            print('current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                optimizer.state_dict()['param_groups'][0]['lr'],
                epoch, step * len(image_a), len(trainloader_a.dataset),
                       100. * step / len(trainloader_a), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg


def main(device, args):
    best_score = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'test', 'test'))

    args.num_input = 3
    if args.task == "pelvic":
        args.num_classes = common_pelvic.NUM_CLASSES
        dataset_s = common_pelvic.Dataset(args.data_dir, "ct", n_slices=args.num_input, debug=args.debug)
        dataset_t = common_pelvic.Dataset(args.data_dir, "cbct", n_slices=args.num_input, debug=args.debug)

        trainloader_a = torch.utils.data.DataLoader(dataset_s,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS,
                                                    pin_memory=True,
                                                    drop_last=True)
        trainloader_b = torch.utils.data.DataLoader(dataset_t,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=NUM_WORKERS,
                                                    pin_memory=True,
                                                    drop_last=True)

        _, val_data_t, _, val_label_t = common_pelvic.load_val_data(args.data_dir)
    else:
        assert 0

    print('Loading is done\n')

#     model = UNet_DA()
    model = deeplabv3_resnet50(num_classes=2)
    model = nn.DataParallel(model)
    if args.gpu >= 0:
        model = model.cuda()

    print("------------------------------------------")
    print("Network Architecture of Model {}:".format(args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    print("------------------------------------------")

    # Define optimizers and loss function
    optimizer1 = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr_rate, 'weight_decay': args.weight_decay},
    ])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=10, T_mult=2, eta_min=0.000001)  # lr_3
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
    optimizer2 = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr_rate, 'weight_decay': args.weight_decay},
    ])

    print("Start training ...")
    best_dsc = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader_a, trainloader_b, model, criterion, scheduler, optimizer1, optimizer2, args, epoch)

        model.eval()
        patch_shape = (args.num_input, val_data_t[0].shape[1], val_data_t[0].shape[2])
        dsc_list = np.zeros((len(val_data_t), args.num_classes - 1), np.float32)
        with torch.no_grad():
            for i in range(len(val_data_t)):
                pred = common_net.produce_results(device, model, [patch_shape, ], [val_data_t[i], ],
                                                  data_shape=val_data_t[i].shape, patch_shape=patch_shape,
                                                  is_seg=True, num_classes=args.num_classes)
                dsc_list[i] = common_metrics.calc_multi_dice(pred, val_label_t[i], num_cls=args.num_classes)

        model.train()

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        if dsc_list.mean() > best_dsc:
            best_dsc = dsc_list.mean()
            torch.save(state, os.path.join(args.ckpt, 'best.pth'))

        print("%s  Epoch:%d  dsc:%f/%f  best_dsc:%f" %
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, dsc_list.mean(), dsc_list.std(), best_dsc))

        torch.save(state, os.path.join(args.ckpt, 'last.pth'))


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

    parser.add_argument('--ckpt_a',
                        default='./0406_prostate_BIDMC_lossmedt_img224_400epoch_16bc_lr-4_UNet_individual_cuda7',
                        help='folder to output checkpoints')
    parser.add_argument('--ckpt_b',
                        default='./0406_prostate_HK_lossmedt_img224_400epoch_16bc_lr-4_UNet_individual_cuda6',
                        help='folder to output checkpoints')
    parser.add_argument('--data', default='2018', help='choose the dataset')  # 训练数据

    parser.add_argument('--dataset_a', default='BIDMC', help='choose the dataset')
    parser.add_argument('--dataset_b', default='HK', help='choose the dataset')
    parser.add_argument('--task', default='pelvic', choices=["pelvic"], help='task')
    parser.add_argument('--out_size', default=(256, 256), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str,
                        help='which cross validation folder')  # 五折训练

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')  # 初始学习率
    parser.add_argument('--num_classes', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=200, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    parser.add_argument('--gpu', default=0, type=int, help='GPU ID')  # 训练数据
    parser.add_argument('--debug', default=0, type=int, help='debug flag')  # 训练数据

    args = parser.parse_args()

    """
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id + "_{}".format(criterion))  # 模型保存地址
    args.ckpt_a = os.path.join(args.ckpt_a, args.data, args.val_folder,
                               args.id_load + "_{}".format(criterion))  # 模型保存地址
    args.ckpt_b = os.path.join(args.ckpt_b, args.data, args.val_folder,
                               args.id_load + "_{}".format(criterion))  # 模型保存地址
    """
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    logfile = os.path.join(args.ckpt, 'logs.txt')  # 训练日志保存地址
    sys.stdout = Logger(logfile)

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

    # if args.start_epoch > 1:
    #args.resume_a = args.ckpt_a + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    #args.resume_b = args.ckpt_b + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    main(device, args)
