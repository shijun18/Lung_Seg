from __future__ import print_function
import numpy as np
from trainer import AverageMeter, accuracy, compute_dice
from run import get_cross_validation,get_cross_validation_by_specificed,VAL_SAMPLE

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms

import os
import logging

import nni

import data_utils.transformer as tr
from data_utils.data_loader import DataGenerator,Trunc_and_Normalize,To_Tensor,CropResize
from utils import get_path_with_annotation



TR_OP = ['RE','RZ','RD','RR','RF','RA','RN']

TR_COMPOSE = {
    'RE':tr.RandomErase2D(scale_flag=False),
    'RZ':tr.RandomZoom2D(),
    'RD':tr.RandomDistort2D(),
    'RR':tr.RandomRotate2D(),
    'RF':tr.RandomFlip2D(mode='hv'),
    'RA':tr.RandomAdjust2D(),
    'RN':tr.RandomNoise2D()
}

_logger = logging.getLogger("Lung_Seg_automl")

mode = 'seg'
train_loader = None
val_loader = None
net = None
criterion = None
optimizer = None
lr_scheduler = None
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

FOLD_NUM = 5

loss_fun = 'TopkCEPlusDice'
net_name = 'm_unet'
channels = 1
num_classes = 2
roi_number = 1
input_shape = (512,512)
crop = 0
batch_size = 8
scale = [-1100,500]

train_path = []
val_path = []


def val_on_epoch(epoch):

    net.eval()

    val_loss = AverageMeter()
    val_dice = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for _, sample in enumerate(val_loader):

            data = sample['image']
            target = sample['mask']
            label = sample['label']

            data = data.cuda()
            target = target.cuda()
            label = label.cuda()

            output = net(data)
            if mode == 'cls':
                loss = criterion(output[1], label)
            elif mode == 'seg':
                loss = criterion(output[0], target)
            else:
                loss = criterion(output,[target,label])


            cls_output = output[1]
            cls_output = F.sigmoid(cls_output).float()

            seg_output = output[0].float()
            seg_output = F.softmax(seg_output, dim=1)

            loss = loss.float()

            # measure acc
            acc = accuracy(cls_output.detach(), label)
            val_acc.update(acc.item(),data.size(0))

            # measure dice and record loss
            dice = compute_dice(seg_output.detach(), target)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(dice.item(), data.size(0))

            torch.cuda.empty_cache()

    return val_loss.avg, val_dice.avg, val_acc.avg


def train_on_epoch(epoch):

    net.train()

    train_loss = AverageMeter()
    train_dice = AverageMeter()
    train_acc = AverageMeter()

    for step, sample in enumerate(train_loader):
        
        data = sample['image']
        target = sample['mask']
        label = sample['label']

        data = data.cuda()
        target = target.cuda()
        label = label.cuda()

        output = net(data)
        if mode == 'cls':
            loss = criterion(output[1], label)
        elif mode == 'seg':
            loss = criterion(output[0], target)
        else:
            loss = criterion(output,[target,label])


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_output = output[1] #N*C
        cls_output = F.sigmoid(cls_output).float()

        seg_output = output[0].float() #N*C*H*W
        seg_output = F.softmax(seg_output, dim=1)

        loss = loss.float()

        # measure acc
        acc = accuracy(cls_output.detach(), label)
        train_acc.update(acc.item(), data.size(0))

        # measure dice and record loss
        dice = compute_dice(seg_output.detach(), target)
        train_loss.update(loss.item(), data.size(0))
        train_dice.update(dice.item(), data.size(0))

        torch.cuda.empty_cache()

        print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_acc:{:.5f},lr:{}'
                .format(epoch, step, loss.item(), dice.item(),acc.item(), optimizer.param_groups[0]['lr']))


    return train_loss.avg, train_dice.avg, train_acc.avg


def get_net(net_name):
    if '_unet' in net_name:
        from model import unet
        net = unet.__dict__[net_name](n_channels=channels,n_classes=num_classes)
    return net


def get_loss(loss_fun):
    if loss_fun == 'TopkCEPlusDice':
        from loss.combine_loss import TopkCEPlusDice
        loss = TopkCEPlusDice(weight=None, ignore_index=0, k=20)
    return loss


def prepare(args, train_path, val_path):
    global train_loader
    global val_loader
    global net
    global criterion
    global optimizer
    global lr_scheduler

    tr_list = [Trunc_and_Normalize(scale),
               CropResize(dim=input_shape,num_class=num_classes,crop=crop)
    ]
    for index in args["tr_index"]:
        tr_list.append(TR_COMPOSE[TR_OP[index]])
    
    tr_list.append(To_Tensor(num_class=num_classes))
    # Data
    print('==> Preparing data..')
    train_transformer = transforms.Compose(tr_list)

    val_transformer = transforms.Compose([
        Trunc_and_Normalize(scale),
        CropResize(dim=input_shape,num_class=num_classes,crop=crop),
        To_Tensor(num_class=num_classes)
    ])

    train_dataset = DataGenerator(train_path, 
                                  roi_number=roi_number,
                                  num_class=num_classes,
                                  transform=train_transformer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_dataset = DataGenerator(val_path,
                                roi_number=roi_number,
                                num_class=num_classes,
                                transform=val_transformer)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    net = get_net(net_name)
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    criterion = get_loss(loss_fun)

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            net.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    if args['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    if args['lr_scheduler'] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args['milestones'], gamma=args['gamma'])
    if args['lr_scheduler'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                       optimizer, T_max=args['T_max'])        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cur_fold", type=int, default=1)

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = nni.get_next_parameter()
        _logger.debug(RCV_CONFIG)

        csv_path = '/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/lung_tumor.csv'
        path_list = get_path_with_annotation(csv_path, 'path', 'GTV')
        fold_metric = []

        for cur_fold in range(1, FOLD_NUM+1):
            train_path, val_path = get_cross_validation_by_specificed(path_list, VAL_SAMPLE)
            prepare(RCV_CONFIG, train_path, val_path)

            fold_best_val_metric = 0.
            for epoch in range(start_epoch, start_epoch+args.epochs):
                epoch_train_loss, epoch_train_dice, epoch_train_acc = train_on_epoch(epoch)
                epoch_val_loss, epoch_val_dice, epoch_val_acc = val_on_epoch(epoch)

                if lr_scheduler is not None:
                    lr_scheduler.step()
                if mode == 'cls':
                    print('Fold %d | Epoch %d | Val Loss %.5f | Acc %.5f'
                        % (cur_fold, epoch, epoch_val_loss, epoch_val_acc))
                    nni.report_intermediate_result(epoch_val_acc)
                    fold_best_val_metric = max(fold_best_val_metric, epoch_val_acc)
                else:
                    print('Fold %d | Epoch %d | Val Loss %.5f | Dice %.5f'
                        % (cur_fold, epoch, epoch_val_loss, epoch_val_dice))
                    nni.report_intermediate_result(epoch_val_dice)
                    fold_best_val_metric = max(fold_best_val_metric, epoch_val_dice)

            fold_metric.append(fold_best_val_metric)
            break
        nni.report_final_result(np.mean(fold_metric))
    except Exception as exception:
        _logger.exception(exception)
        raise