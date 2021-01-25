import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil

from torch.nn import functional as F

from rcnn.data_utils.transformer import RandomAdjustHalf,RandomDistortHalf,RandomEraseHalf,RandomFlipHalf,RandomNoiseHalf,RandomTranslationRotationZoomHalf
from rcnn.data_utils.data_loader import DataGenerator, To_Tensor, CropResizeHalf, Trunc_and_Normalize

import torch.distributed as dist
# GPU version.

class SemanticSeg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    - mode: string __all__ = ['cls','seg','cls_and_seg','cls_or_seg']
    '''
    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 channels=1,
                 num_classes=2,
                 roi_number=1,
                 scale=None,
                 seq_len=3,
                 input_shape=None,
                 crop=0,
                 batch_size=6,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 ckpt_point=True,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 mode='cls',
                 topk=10,
                 freeze=None):
        super(SemanticSeg, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.roi_number = roi_number
        self.scale = scale
        self.seq_len = seq_len
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.net = self._get_net(self.net_name)
        self.pre_trained = pre_trained
        self.ckpt_point = ckpt_point
        self.weight_path = weight_path

        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 2.0

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max

        self.mode = mode
        self.topk = topk
        self.freeze = freeze

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        if self.pre_trained:
            self._get_pre_trained(self.weight_path,ckpt_point)


        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

    def trainer(self,
                train_path,
                val_path,
                cur_fold,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None):

        torch.manual_seed(1000)
        np.random.seed(1000)
        torch.cuda.manual_seed_all(1000)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)
        self.step_pre_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path[0]) / self.batch_size)

        net = self.net

        # only for deeplab
        if self.freeze is not None and 'deeplab' in self.net_name:
            if self.freeze == 'backbone':
                net.freeze_backbone()
            elif self.freeze == 'classifier':
                net.freeze_classifier()

        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        if self.mode == 'cls':
            train_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                RandomEraseHalf(scale_flag=False),
                RandomDistortHalf(),
                RandomTranslationRotationZoomHalf(num_class=self.num_classes),
                RandomFlipHalf(mode='hv'),
                RandomAdjustHalf(),
                To_Tensor(num_class=self.num_classes)
            ])
        else:
            train_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                RandomEraseHalf(scale_flag=False),
                RandomDistortHalf(),
                RandomTranslationRotationZoomHalf(num_class=self.num_classes),
                # RandomFlipHalf(mode='hv'),
                # RandomAdjustHalf(),
                RandomNoiseHalf(),
                To_Tensor(num_class=self.num_classes)
            ])
        train_dataset = DataGenerator(train_path,
                                      roi_number=self.roi_number,
                                      num_class=self.num_classes,
                                      transform=train_transformer,
                                      seq_len=self.seq_len)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)
        if self.pre_trained and self.ckpt_point:
            checkpoint = torch.load(self.weight_path)
            optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        # loss_threshold = 1.0
        for epoch in range(self.start_epoch, self.n_epoch):
            train_loss, train_dice, train_acc = self._train_on_epoch(epoch, net, loss, optimizer, train_loader)

            val_loss, val_dice, val_acc = self._val_on_epoch(epoch, net, loss, val_path)

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)

            torch.cuda.empty_cache()
            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))

            print('epoch:{},train_dice:{:.5f},val_dice:{:.5f}'.format(epoch, train_dice, val_dice))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],epoch)

            if val_loss <= self.loss_threshold:
                self.loss_threshold = val_loss

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_acc:{:.5f}-val_loss:{:.5f}-val_dice:{:.5f}-val_acc:{:.5f}.pth'.format(
                    epoch, train_loss, train_dice, train_acc, val_loss,
                    val_dice, val_acc)
                
                save_path = os.path.join(output_dir, file_name)
                print("save as %s" % file_name)

                torch.save(saver, save_path)

        self.writer.close()

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader):

        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()
        train_acc = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=[0,1],ignore_label=-1)
        for step, sample in enumerate(train_loader):

            data = sample['image']  # img:(N,cin,seq_len, H, W),
            target = sample['mask'] # mask:(N,num_class,seq_len, H, W),
            label = sample['label'] # label (N,seq_len,num_class-1)
            # print(data.size(),target.size(),label.size())

            data = data.cuda() 
            target = target.cuda()
            label = label.cuda()

            output = net(data) #(N,seq_len,num_class,H,W) (N,seq_len,num_class-1)
            # print(output[0].size(),output[1].size())
            loss = 0.
            if self.mode == 'cls':
                for i in range(data.size(2)):
                    loss += criterion(output[1][:,i], label[:,i])
            elif self.mode == 'seg':
                for i in range(data.size(2)):
                    loss += criterion(output[0][:,i], target[:,:,i])
            else:
                for i in range(data.size(2)):
                    loss += criterion([output[0][:,i],output[1][:,i]],[target[:,:,i],label[:,i]])
            
            loss /= data.size(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.float()
            train_loss.update(loss.item(), data.size(0))

            total_dice = 0.
            total_acc = 0.
            for i in range(data.size(2)):
                cls_output = output[1][:,i] #(N,num_class-1)
                cls_output = F.sigmoid(cls_output).float()

                seg_output = output[0][:,i].float() #(N,num_class,H,W)
                seg_output = F.softmax(seg_output, dim=1)

                # measure acc
                acc = accuracy(cls_output.detach(), label[:,i])
                train_acc.update(acc.item(), data.size(0))
                total_acc += acc.item()

                # measure dice and record loss
                dice = compute_dice(seg_output.detach(), target[:,:,i])
                train_dice.update(dice.item(), data.size(0))
                total_dice += dice.item()

                # measure run dice  
                seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()  #N*H*W 
                tmp_target = torch.argmax(target[:,:,i],1).detach().cpu().numpy()
                run_dice.update_matrix(tmp_target,seg_output)

            torch.cuda.empty_cache()

            if self.global_step % 10 == 0:
                if self.mode == 'cls':
                    print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'.format(epoch, step, loss.item(), total_acc/data.size(2), optimizer.param_groups[0]['lr']))
                elif self.mode == 'seg':
                    rundice, dice_list = run_dice.compute_dice() 
                    print("Category Dice: ", dice_list)
                    print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},run_dice:{:.5f},lr:{}'.format(epoch, step, loss.item(), total_dice/data.size(2), rundice, optimizer.param_groups[0]['lr']))
                    run_dice.init_op()
                else:
                    print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_acc:{:.5f},lr:{}'.format(epoch, step, loss.item(), total_dice/data.size(2),total_acc/data.size(2), optimizer.param_groups[0]['lr']))

                
                self.writer.add_scalars('data/train_loss_dice', {
                    'train_loss': loss.item(),
                    'train_dice': total_dice/data.size(2),
                    'train_acc': total_acc/data.size(2)
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_dice.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, val_transformer=None):

        net.eval()

        if self.mode == 'cls':
            val_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                To_Tensor(num_class=self.num_classes)
            ])
        else:
            val_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                To_Tensor(num_class=self.num_classes)
            ])

        val_dataset = DataGenerator(val_path,
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=val_transformer,
                                    seq_len=-1)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_dice = AverageMeter()
        val_acc = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=[0,1],ignore_label=-1)
        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['mask']
                label = sample['label']

                data = data.cuda()
                target = target.cuda()
                label = label.cuda()

                output = net(data)
                loss = 0.
                if self.mode == 'cls':
                    for i in range(data.size(2)):
                        loss += criterion(output[1][:,i], label[:,i])
                elif self.mode == 'seg':
                    for i in range(data.size(2)):
                        loss += criterion(output[0][:,i], target[:,:,i])
                else:
                    for i in range(data.size(2)):
                        loss += criterion([output[0][:,i],output[1][:,i]],[target[:,:,i],label[:,i]])

                loss /= data.size(2)

                loss = loss.float()
                val_loss.update(loss.item(), data.size(0))
                
                total_dice = 0.
                total_acc = 0.
                for i in range(data.size(2)):
                    cls_output = output[1][:,i] #(N,num_class-1)
                    cls_output = F.sigmoid(cls_output).float()

                    seg_output = output[0][:,i].float() #(N,num_class,H,W)
                    seg_output = F.softmax(seg_output, dim=1)

                    # measure acc
                    acc = accuracy(cls_output.detach(), label[:,i])
                    val_acc.update(acc.item(), data.size(0))
                    total_acc += acc.item()

                    # measure dice and record loss
                    dice = compute_dice(seg_output.detach(), target[:,:,i])
                    val_dice.update(dice.item(), data.size(0))
                    total_dice += dice.item()

                    # measure run dice  
                    seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()  #N*H*W 
                    tmp_target = torch.argmax(target[:,:,i],1).detach().cpu().numpy()
                    run_dice.update_matrix(tmp_target,seg_output)

                torch.cuda.empty_cache()

                if step % 10 == 0:
                    if self.mode == 'cls':
                        print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(epoch, step, loss.item(), total_acc/data.size(2)))
                    elif self.mode == 'seg':
                        rundice, dice_list = run_dice.compute_dice() 
                        print("Category Dice: ", dice_list)
                        print('epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},rundice:{:.5f}'.format(epoch, step, loss.item(), total_dice/data.size(2),rundice))
                        run_dice.init_op()
                    else:
                        print('epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},val_acc:{:.5f}'.format(epoch, step, loss.item(), total_dice/data.size(2), total_acc/data.size(2)))

        return val_loss.avg, val_dice.avg, val_acc.avg
    
    
    def test(self, test_path, save_path, net=None, mode='seg', save_flag=False):
        if net is None:
            net = self.net
        
        net = net.cuda()
        net.eval()
        
        if self.mode == 'cls':
            test_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                RandomEraseHalf(scale_flag=False),
                RandomTranslationRotationZoomHalf(num_class=self.num_classes),
                RandomFlipHalf(mode='hv'),
                RandomAdjustHalf(),
                To_Tensor(num_class=self.num_classes)
            ])
        else:
            test_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale),
                CropResizeHalf(dim=self.input_shape,num_class=self.num_classes,crop=self.crop),
                To_Tensor(num_class=self.num_classes)
            ])

        test_dataset = DataGenerator(test_path,
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=test_transformer,
                                    seq_len=-1)

        test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        test_dice = AverageMeter()
        test_acc = AverageMeter()
        from PIL import Image
        from metrics import RunningDice
        run_dice = RunningDice(labels=[0,1],ignore_label=-1)

        cls_result = {
            'true': [],
            'pred': [],
            'prob': []
        }

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['mask']
                label = sample['label'] 
                # print(label)

                data = data.cuda()
                target = target.cuda()
                label = label.cuda()

                output = net(data)
                total_dice = 0.
                total_acc = 0.
                for i in range(data.size(2)):
                    cls_output = output[1][:,i]
                    cls_output = F.sigmoid(cls_output).float()

                    seg_output = output[0][:,i].float()
                    seg_output = F.softmax(seg_output, dim=1)

                    # measure acc
                    acc = accuracy(cls_output.detach(), label[:,i])
                    test_acc.update(acc.item(),data.size(0))
                    total_acc += acc.item()

                    # measure dice and iou for evaluation (float)
                    dice = compute_dice(seg_output.detach(), target[:,:,i], ignore_index=0)
                    test_dice.update(dice.item(), data.size(0))
                    total_dice += dice.item()

                    cls_result['prob'].extend(cls_output.detach().squeeze().cpu().numpy().tolist())
                    cls_output = (cls_output > 0.5).float() # N*C
                    cls_result['pred'].extend(cls_output.detach().squeeze().cpu().numpy().tolist())
                    cls_result['true'].extend(label.detach().squeeze().cpu().numpy().tolist())
                    # print(cls_output.detach())
                    if mode == 'mtl':
                        b, c, _, _ = seg_output.size()
                        seg_output[:,1:,...] = seg_output[:,1:,...] * cls_output.view(b,c-1,1,1).expand_as(seg_output[:,1:,...])

                    seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()  #N*H*W N=1
                    tmp_target = torch.argmax(target[:,:,i],1).detach().cpu().numpy()
                    run_dice.update_matrix(tmp_target,seg_output)
                    # print(np.unique(seg_output),np.unique(target))

                    # save
                    if mode != 'cls' and save_flag:
                        seg_output = np.squeeze(seg_output).astype(np.uint8) 
                        seg_output = Image.fromarray(seg_output, mode='L')
                        seg_output.save(os.path.join(save_path,test_path[step].split('.')[0] + '_' + str(i) +'.png'))

                torch.cuda.empty_cache()
                
                print('step:{},test_dice:{:.5f},test_acc:{:.5f}'.format(step,total_dice/data.size(2),total_acc/data.size(2)))
            
        rundice, dice_list = run_dice.compute_dice() 
        print("Category Dice: ", dice_list)
        print('avg_dice:{:.5f},avg_acc:{:.5f}，rundice:{:.5f}'.format(test_dice.avg, test_acc.avg, rundice))

        return cls_result
    

    def _get_net(self, net_name):
        if net_name == 'rcnn_unet' :
            from rcnn.model.unet import rcnn_unet
            net = rcnn_unet(n_channels=self.channels,n_classes=self.num_classes,seq_len=self.seq_len)    
   
        return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from rcnn.loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight)
        if loss_fun == 'DynamicTopKLoss':
            from rcnn.loss.cross_entropy import DynamicTopKLoss
            loss = DynamicTopKLoss(weight=class_weight,step_threshold=self.step_pre_epoch)
        
        elif loss_fun == 'DynamicTopkCEPlusDice':
            from rcnn.loss.combine_loss import DynamicTopkCEPlusDice
            loss = DynamicTopkCEPlusDice(weight=class_weight, ignore_index=0, step_threshold=self.step_pre_epoch)
        
        elif loss_fun == 'TopKLoss':
            from rcnn.loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight, k=self.topk)
        
        elif loss_fun == 'DiceLoss':
            from rcnn.loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=1)
        elif loss_fun == 'ShiftDiceLoss':
            from rcnn.loss.dice_loss import ShiftDiceLoss
            loss = ShiftDiceLoss(weight=class_weight,ignore_index=0, reduction='topk',shift=0.5, p=1, k=self.topk)
        elif loss_fun == 'TopkDiceLoss':
            from rcnn.loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0,reduction='topk', k=self.topk)

        elif loss_fun == 'PowDiceLoss':
            from rcnn.loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=2)
        
        elif loss_fun == 'TverskyLoss':
            from rcnn.loss.tversky_loss import TverskyLoss
            loss = TverskyLoss(weight=class_weight, ignore_index=0, alpha=0.7)
        
        elif loss_fun == 'FocalTverskyLoss':
            from rcnn.loss.tversky_loss import TverskyLoss
            loss = TverskyLoss(weight=class_weight, ignore_index=0, alpha=0.7, gamma=0.75)

        elif loss_fun == 'BCEWithLogitsLoss':
            loss = nn.BCEWithLogitsLoss(class_weight)
        
        elif loss_fun == 'BCEPlusDice':
            from rcnn.loss.combine_loss import BCEPlusDice
            loss = BCEPlusDice(weight=class_weight,ignore_index=0,p=1)
        
        elif loss_fun == 'CEPlusDice':
            from rcnn.loss.combine_loss import CEPlusDice
            loss = CEPlusDice(weight=class_weight, ignore_index=0)
        
        elif loss_fun == 'CEPlusTopkDice':
            from rcnn.loss.combine_loss import CEPlusTopkDice
            loss = CEPlusTopkDice(weight=class_weight, ignore_index=0, reduction='topk', k=self.topk)
        
        elif loss_fun == 'TopkCEPlusTopkDice':
            from rcnn.loss.combine_loss import TopkCEPlusTopkDice
            loss = TopkCEPlusTopkDice(weight=class_weight, ignore_index=0, reduction='topk', k=self.topk)
        
        elif loss_fun == 'TopkCEPlusDice':
            from rcnn.loss.combine_loss import TopkCEPlusDice
            loss = TopkCEPlusDice(weight=class_weight, ignore_index=0, k=self.topk)
        
        elif loss_fun == 'TopkCEPlusShiftDice':
            from rcnn.loss.combine_loss import TopkCEPlusShiftDice
            loss = TopkCEPlusShiftDice(weight=class_weight,ignore_index=0, shift=0.5,k=self.topk)
        
        elif loss_fun == 'TopkCEPlusTopkShiftDice':
            from rcnn.loss.combine_loss import TopkCEPlusTopkShiftDice
            loss = TopkCEPlusTopkShiftDice(weight=class_weight,ignore_index=0, reduction='topk',shift=0.5,k=self.topk)
        
        return loss

    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=lr,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        return lr_scheduler

    def _get_pre_trained(self, weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1
            # self.loss_threshold = eval(os.path.splitext(self.weight_path.split(':')[-1])[0])


# computing tools


class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def binary_dice(predict, target, smooth=1e-5):
    """Dice of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        dice according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    inter = torch.sum(torch.mul(predict, target), dim=1)
    union = torch.sum(predict + target, dim=1)

    dice = (2 * inter + smooth) / (union + smooth)

    return dice.mean()

'''
def compute_dice(predict, target, ignore_index=0, shift=0.5):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'

    predict_shift = F.relu(predict-shift)
    alpha = predict / (predict-shift)
    alpha_relu = F.relu(alpha)
    predict = predict_shift * alpha_relu

    total_dice = 0.
    dice_list = []
    for i in range(target.shape[1]):
        if i != ignore_index:
            dice = binary_dice(predict[:, i], target[:, i])
            # print(dice)
            total_dice += dice
            dice_list.append(round(dice.item(),4))
    # print(dice_list)

    if ignore_index is not None:
        return total_dice / (target.shape[1] - 1)
    else:
        return total_dice / target.shape[1]
'''
def compute_dice(predict,target,ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    total_dice = 0.
    # predict = F.softmax(predict, dim=1)

    onehot_predict = torch.argmax(predict,dim=1)#N*H*W
    onehot_target = torch.argmax(target,dim=1) #N*H*W

    dice_list = []
    for i in range(target.shape[1]):
        if i != ignore_index:
            # dice = binary_dice(predict[:, i], target[:, i])
            dice = binary_dice((onehot_predict==i).float(), (onehot_target==i).float())
            total_dice += dice
            dice_list.append(round(dice.item(),4))
    # print(dice_list)

    if ignore_index is not None:
        return total_dice/(target.shape[1] - 1)
    else:
        return total_dice/target.shape[1]

def accuracy(output, target):
    '''
    Computes the precision acc
    - output shape: N*C
    - target shape: N*C
    '''
    batch_size, class_num = target.size()
    pred = (output > 0.5).float()
    correct = pred.eq(target)
    acc = correct.float().sum() / (batch_size*class_num) 

    return acc
