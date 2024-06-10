import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.optim as optim
from pathlib import Path
import os
import numpy as np

from models.semseg import SemsegModel
from models.resnet.resnet_pyramid import *
from models.loss import BoundaryAwareFocalLoss
from data.transform import *
# from data.cityscapes import Cityscapes
from data.DeepGlobeLLC.DeepGlobeLLC import DeepGlobeLLC
from evaluation import StorePreds

from models.util import get_n_params

torch.cuda.empty_cache()

root = Path('datasets/deepglobelcc/455_207_141')      # add symbolic link to datasets folder for different datasets
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

evaluating = True
random_crop_size = 1440

scale = 1
mean =  [104.09488634 , 96.66853759 , 71.80576166] 
std =  [37.4961336 , 29.24727474, 26.74629693]  
mean_rgb = tuple(np.uint8(scale * np.array(mean)))

num_classes = (
    DeepGlobeLLC.num_classes
    )


ignore_id = DeepGlobeLLC.num_classes
class_info = DeepGlobeLLC.class_info
color_info = DeepGlobeLLC.color_info
mapping = DeepGlobeLLC.map_to_id

print(f"Training with :{DeepGlobeLLC.num_classes} classes")
num_levels = 3
ostride = 4
target_size_crops = (random_crop_size, random_crop_size)
target_size_crops_feats = (random_crop_size // ostride, random_crop_size // ostride)

eval_each = 4
dist_trans_bins = (16, 64, 128)
dist_trans_alphas = (8., 4., 2., 1.)
target_size = (2448,2448)
target_size_feats = (2448 // ostride, 2448 // ostride)

trans_val = Compose(
    [Open(),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Tensor(),
     ]
)

if evaluating:
    trans_train = trans_train_val = trans_val
else:
    trans_train = Compose(
        [Open(),
         RandomFlip(),
         RandomSquareCropAndScale(random_crop_size, ignore_id=ignore_id, mean=mean_rgb),
         SetTargetSize(target_size=target_size_crops, target_size_feats=target_size_crops_feats),
         LabelDistanceTransform(num_classes=num_classes, reduce=True, bins=dist_trans_bins, alphas=dist_trans_alphas),
         Tensor(),
         ])

dataset_train = DeepGlobeLLC(root, transforms=trans_train, subset='train')
dataset_val = DeepGlobeLLC(root, transforms=trans_val, subset='val')
if evaluating:
    dataset_test = DeepGlobeLLC(root, transforms=trans_val, subset='test')


backbone = resnet18(pretrained=True,
                    pyramid_levels=num_levels,
                    k_upsample=3,
                    scale=scale,
                    mean=mean,
                    std=std,
                    k_bneck=1,
                    output_stride=ostride,
                    efficient=True)
model = SemsegModel(backbone, num_classes, k=1, bias=True)
if evaluating:
    model.load_state_dict(torch.load('/mnt/sdb/mharalovic/swiftnet/weights/rn18_pyramid/71-52_rn18_pyramid/stored/model_best.pt'), strict=False)
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)
else:
    model.criterion = BoundaryAwareFocalLoss(gamma=.5, num_classes=num_classes, ignore_id=ignore_id)

bn_count = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        bn_count += 1
print(f'Num BN layers: {bn_count}')

if not evaluating:
    lr = 4e-4
    lr_min = 1e-6
    fine_tune_factor = 4
    weight_decay = 1e-4
    epochs = 250

    optim_params = [
        {'params': model.random_init_params(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model.fine_tune_params(), 'lr': lr / fine_tune_factor,
         'weight_decay': weight_decay / fine_tune_factor},
    ]

    optimizer = optim.Adam(optim_params, betas=(0.9, 0.99))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr_min)

batch_size = bs = 6
print(f'Batch size: {bs}')
nw = 4

loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=nw)

if evaluating:
    loader_train = DataLoader(dataset_train, batch_size=1, collate_fn=custom_collate, num_workers=nw)
    loader_test = DataLoader(dataset_test,batch_size=1,collate_fn=custom_collate)
else:
    loader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=nw, pin_memory=True,
                              drop_last=True, collate_fn=custom_collate, shuffle=True)

total_params = get_n_params(model.parameters())
ft_params = get_n_params(model.fine_tune_params())
ran_params = get_n_params(model.random_init_params())
assert total_params == (ft_params + ran_params)
print(f'Num params: {total_params:,} = {ran_params:,}(random init) + {ft_params:,}(fine tune)')

if evaluating:
    # eval_loaders = [(loader_val, 'val'), (loader_train, 'train')]
    eval_loaders = [(loader_test, 'test')]
    
    dir_path = '/mnt/sdb/mharalovic/swiftnet/weights/rn18_pyramid/71-52_rn18_pyramid'
    store_dir = f'{dir_path}/out/'
    
    # for d in ['', 'val', 'train']:
    for d in ['test']:
        os.makedirs(store_dir + d, exist_ok=True)
    to_color = ColorizeLabels(color_info)
    to_image = Compose([Numpy(), to_color])
    eval_observers = [StorePreds(store_dir, to_image, to_color)]
