# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
import os
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_integrated import pascal_voc_integrated
from datasets.voc_clipart import voc_clipart
from datasets.voc_watercolor import voc_watercolor
from datasets.voc_comic import voc_comic

from datasets.clipart import clipart
from datasets.comic import comic
from datasets.watercolor import watercolor
from datasets.cityscapes import cityscapes


# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, devkit_path='../faster-rcnn.pytorch/data/VOCdevkit2007'))

# VOC clipart + amds
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02', '_50_0', '_50_1', '_25_0', '_25_1', '_25_2', '_25_3', '_10_0', '_10_1', '_10_2']:
        # clipart
        name = 'clipart{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_clipart('clipart{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'clipart{}_{}'.format(data_percentage_split, shift))))
        # amds
        name = 'amds{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_clipart('amds{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'amds{}_{}'.format(data_percentage_split, shift))))

        # bicycle
        name = 'bicycle{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_clipart('bicycle{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'bicycle{}_{}'.format(data_percentage_split, shift))))

# AMD  test splits
for split in ['test']:
  for data_percentage_split in ['', '_10_0', '_10_1']:
    name = 'clipart{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('clipart{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'clipart{}'.format(data_percentage_split))))

    name = 'comic{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: comic('comic{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'comic{}'.format(data_percentage_split))))

    name = 'watercolor{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: watercolor('watercolor{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'watercolor{}'.format(data_percentage_split))))

    name = 'amds{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('amds{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'amds{}'.format(data_percentage_split))))

    name = 'bicycle{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('bicycle{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'bicycle{}'.format(data_percentage_split))))

# AMD train splits
  
for split in ['train']:
  for data_percentage_split in ['', '_1_00', '_1_01', '_1_02', '_50_0', '_50_1', '_25_0', '_25_1', '_25_2', '_25_3', '_10_0', '_10_1', '_10_2']:
    name = 'clipart{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('clipart{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'clipart{}'.format(data_percentage_split))))

    name = 'comic{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: comic('comic{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'comic{}'.format(data_percentage_split))))

    name = 'watercolor{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: watercolor('watercolor{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'watercolor{}'.format(data_percentage_split))))

    name = 'bicycle{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('bicycle{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'bicycle{}'.format(data_percentage_split))))

    name = 'amds{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('amds{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'amds{}'.format(data_percentage_split))))

# VOC watercolor
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02', '_50_0', '_50_1', '_25_0', '_25_1', '_25_2', '_25_3', '_10_0', '_10_1', '_10_2']:
        name = 'watercolor{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_watercolor('watercolor{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'watercolor{}_{}'.format(data_percentage_split, shift))))

# cityscapes
for split in ['cityscapes_train', 'foggy_test']:
    name = split
    __sets[name] = (lambda split=split: cityscapes(name=split, image_set=split, devkit_path=os.path.join('datasets', 'voc_cityscapes')))

for split in ['cityscapes2foggy', 'cityscapes2kitti', 'foggy', 'kitti']:
    for data_percentage_split in ['', '_1_00', '_1_01', '_1_02', '_50_0', '_50_1', '_25_0', '_25_1', '_25_2', '_25_3', '_10_0', '_10_1', '_10_2', '_10_samples', '_10_samples_2']:
        if split == "cityscapes2foggy":
            for shift in ['CP', 'CPR', 'R']:
                name = "cityscapes2foggy{}_{}".format(data_percentage_split, shift)
                split = "cityscapes_train"
                __sets[name] = (lambda name=name, split=split: cityscapes(name=name, image_set=split, devkit_path=os.path.join('datasets', 'voc_{}'.format(name))))
        elif split == "cityscapes2kitti":
            for shift in ['CP', 'CPR', 'R']:
                name = "cityscapes2kitti{}_{}".format(data_percentage_split, shift)
                split = "cityscapes_train"
                __sets[name] = (lambda name=name, split=split: cityscapes(name=name, image_set=split, devkit_path=os.path.join('datasets', 'voc_{}'.format(name))))
        elif split == "kitti":
            name = "kitti{}_train".format(data_percentage_split) # e.g. kitti_10_samples_train
            split_file = "trainval"
            dataset_name = "kitti{}".format(data_percentage_split)
            __sets[name] = (lambda name=name, split=split_file, dataset_name=dataset_name: kitti(name=name, image_set=split, devkit_path=os.path.join('datasets', dataset_name)))
        else: # foggy train: we need foggy_train, foggy_1_00_train, foggy_1_01_train and foggy_1_02_train
            name = "foggy{}_train".format(data_percentage_split)
            split_file = "foggy_train"
            dataset_name = "voc_cityscapes2foggy{}".format(data_percentage_split)
            __sets[name] = (lambda name=name, split=split_file, dataset_name=dataset_name: cityscapes(name=name, image_set=split, devkit_path=os.path.join('datasets', dataset_name)))

# VOC comic
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02', '_50_0', '_50_1', '_25_0', '_25_1', '_25_2', '_25_3', '_10_0', '_10_1', '_10_2']:
        name = 'comic{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_comic('comic{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'comic{}_{}'.format(data_percentage_split, shift))))

# Set up voc_integrated
for split in ['trainval']:
  name = 'voc_integrated_{}'.format(split)
  __sets[name] = (lambda split=split: pascal_voc_integrated(split, devkit_path='datasets/Pascal/VOCdevkit/VOC_Integrated'))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
