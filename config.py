import os
import json
import glob

from utils import get_path_with_annotation,get_path_with_annotation_ratio
from utils import get_weight_path

__disease__ = ['Covid-Seg','Lung_Tumor']
__net__ = ['m_unet','mr_unet','e_unet','er_unet','ResUNet18','ResUNet34','ResUNet50','deeplabv3plus_resnet18','deeplabv3plus_resnet34','deeplabv3plus_resnet50','deeplabv3plus_resnet101']
__mode__ = ['cls','seg','mtl']


json_path = {
    'Cervical':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Oar.json',
    'Nasopharynx':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Oar.json',
    'Structseg_HaN':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_HaN.json',
    'Structseg_THOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_THOR.json',
    'SegTHOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/SegTHOR.json',
    'Covid-Seg':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Covid-Seg.json', # competition
    'Lung':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Oar.json',
    'Lung_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Tumor.json',
    'EGFR':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/EGFR.json',
}
    
DISEASE = 'Lung_Tumor' 
MODE = 'seg'
NET_NAME = 'deeplabv3plus_resnet18'
VERSION = 'v8.3-zero'

with open(json_path[DISEASE], 'r') as fp:
    info = json.load(fp)

DEVICE = '4'
# Must be True when pre-training and inference
PRE_TRAINED = False 
CKPT_POINT = False
# 1,2,...,8
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = 1# or 0,1,2,3,4,5,6 
NUM_CLASSES = info['annotation_num'] + 1  # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    NUM_CLASSES = 2
    ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
else:
    ROI_NAME = 'All'
SCALE = info['scale'][ROI_NAME]
#---------------------------------

#--------------------------------- mode and data path setting
#all
# PATH_LIST = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))
#zero
PATH_LIST = get_path_with_annotation(info['2d_data']['csv_path'],'path',ROI_NAME)
#half
# PATH_LIST = get_path_with_annotation_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=0.5)
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (256,256)
BATCH_SIZE = 24

# CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE, 'cls', 'v1.0', ROI_NAME, str(1))
CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))

WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-3, 
  'n_epoch':200,
  'channels':1,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER,
  'scale':SCALE,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.001,
  'momentum': 0.99,
  'gamma': 0.1,
  'milestones': [40,80],
  'T_max':5,
  'mode':MODE,
  'topk':70
 }
#---------------------------------

__seg_loss__ = ['DiceLoss','TverskyLoss','FocalTverskyLoss','TopkCEPlusDice','TopkCEPlusTopkShiftDice','TopkCEPlusShiftDice','PowDiceLoss','Cross_Entropy','TopkDiceLoss','DynamicTopKLoss','DynamicTopkCEPlusDice','TopKLoss','CEPlusDice','TopkCEPlusDice','CEPlusTopkDice','TopkCEPlusTopkDice']
__cls_loss__ = ['BCEWithLogitsLoss']
__mtl_loss__ = ['BCEPlusDice']
# Arguments when perform the trainer 

if MODE == 'cls':
    LOSS_FUN = 'BCEWithLogitsLoss'
elif MODE == 'seg' :
    LOSS_FUN = 'TopkCEPlusDice'
else:
    LOSS_FUN = 'BCEPlusDice'

SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME), 
  'optimizer':'Adam',
  'loss_fun':LOSS_FUN,
  'class_weight':None, #[1,4]
  'lr_scheduler':'CosineAnnealingLR', #'CosineAnnealingLR'
  }
#---------------------------------
