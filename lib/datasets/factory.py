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

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, devkit_path='../faster-rcnn.pytorch/data/VOCdevkit2007'))

# VOC clipart
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02']:
        name = 'clipart{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_clipart('clipart{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'clipart{}_{}'.format(data_percentage_split, shift))))


# AMD  test splits
for split in ['test']:
    data_percentage_split = ''
    name = 'clipart{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('clipart{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'clipart{}'.format(data_percentage_split))))

    name = 'comic{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: comic('comic{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'comic{}'.format(data_percentage_split))))

    name = 'watercolor{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: watercolor('watercolor{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'watercolor{}'.format(data_percentage_split))))

# AMD train splits
  
for split in ['train']:
  for data_percentage_split in ['', '_1_00', '_1_01', '_1_02']:
    name = 'clipart{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: clipart('clipart{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'clipart{}'.format(data_percentage_split))))

    name = 'comic{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: comic('comic{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'comic{}'.format(data_percentage_split))))

    name = 'watercolor{}_{}'.format(data_percentage_split, split)
    __sets[name] = (lambda split=split, data_percentage_split=data_percentage_split: watercolor('watercolor{}'.format(data_percentage_split), split, devkit_path=os.path.join('datasets/', 'watercolor{}'.format(data_percentage_split))))

# VOC watercolor
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02']:
        name = 'watercolor{}_{}_{}'.format(data_percentage_split, shift, split)
        __sets[name] = (lambda shift=shift, split=split, data_percentage_split=data_percentage_split: voc_watercolor('watercolor{}'.format(data_percentage_split), shift, split, devkit_path=os.path.join('datasets/', 'watercolor{}_{}'.format(data_percentage_split, shift))))

# VOC comic
for split in ['trainval']:
    for shift in ['CP', 'R', 'CPR']:
      for data_percentage_split in ['', '_1_00', '_1_01', '_1_02']:
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
