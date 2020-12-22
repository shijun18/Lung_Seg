import os
import argparse
from trainer import SemanticSeg
import pandas as pd
import random
from PIL import Image

from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, PATH_LIST, FOLD_NUM

import time


def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])
    random.shuffle(train_id)
    random.shuffle(validation_id)
    print(len(train_id), len(validation_id))
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train_cross_val',
                        choices=["train", 'train_cross_val', "inf","test"],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train_cross_val':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
    path_list = PATH_LIST
    path_list.sort()
    # Training
    ###############################################
    if args.mode == 'train_cross_val':
        path_list = path_list[:int(len(path_list) * 0.8)]
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path, val_path = get_cross_validation(path_list, FOLD_NUM, current_fold)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        path_list = path_list[:int(len(path_list) * 0.8)]
        train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    if args.mode == 'test':
        start_time = time.time()
        test_path = path_list[int(len(path_list) * 0.9):]
        print("test set len:",len(test_path))
        save_path = './result/test'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        segnetwork.test(test_path,save_path,mode='seg')
        
        print('run time:%.4f' % (time.time() - start_time))
