# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg


class wider(imdb):
  def __init__(self, image_set):
    name = 'WIDER_' + image_set
    imdb.__init__(self, name)
    self._image_set = image_set
    
    # /data/wider
    self._data_path = self._get_default_path()
    # /data/wider/wider_face_split
    self._annotation_path = os.path.join(self._data_path, 'wider_face_split')
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)
    assert os.path.exists(self._annotation_path), \
      'Path does not exist: {}'.format(self._devkit_path)

    # target class
    self._classes = ('__background__',  # always index 0
                     'face')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

    # abcsup storing mat file for tons of purpose
    mat_file = os.path.join(self._annotation_path, 
                                  'wider_face_' + self._image_set + '.mat')
    assert os.path.exists(mat_file), \
      'Path does not exist: {}'.format(image_set_file)
    self._mat = sio.loadmat(mat_file)

    # image_index
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()

    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'matlab_eval': False,
                   'rpn_file': None}


  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence. done
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier. done
    """
    image_path = os.path.join(self._data_path, 'WIDER_' + self._image_set, 
                              'images', index + self._image_ext)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file. done
    """
    image_path_dict = ['0--Parade', '1--Handshaking', '10--People_Marching', '11--Meeting', '12--Group', '13--Interview', '14--Traffic', '15--Stock_Market', '16--Award_Ceremony', '17--Ceremony', '18--Concerts', '19--Couple', '2--Demonstration', '20--Family_Group', '21--Festival', '22--Picnic', '23--Shoppers', '24--Soldier_Firing', '25--Soldier_Patrol', '26--Soldier_Drilling', '27--Spa', '28--Sports_Fan', '29--Students_Schoolkids', '3--Riot', '30--Surgeons', '31--Waiter_Waitress', '32--Worker_Laborer', '33--Running', '34--Baseball', '35--Basketball', '36--Football', '37--Soccer', '38--Tennis', '39--Ice_Skating', '4--Dancing', '40--Gymnastics', '41--Swimming', '42--Car_Racing', '43--Row_Boat', '44--Aerobics', '45--Balloonist', '46--Jockey', '47--Matador_Bullfighter', '48--Parachutist_Paratrooper', '49--Greeting', '5--Car_Accident', '50--Celebration_Or_Party', '51--Dresses', '52--Photographers', '53--Raid', '54--Rescue', '55--Sports_Coach_Trainer', '56--Voter', '57--Angler', '58--Hockey', '59--people--driving--car', '6--Funeral', '61--Street_Battle', '7--Cheering', '8--Election_Campain', '9--Press_Conference']

    file_list = self._mat['file_list']
    image_index = []
    for idx, section in enumerate(file_list):
        for photo in section[0]:
            photo_path = os.path.join(image_path_dict[idx] , photo[0][0])
            image_index.append(photo_path)

    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed. done
    """
    return os.path.join(cfg.DATA_DIR, 'wider')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest. done

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self._load_wider_annotation(index)
                for index, _ in enumerate(self.image_index)]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    """
    done
    """
    #if int(self._year) == 2007 or self._image_set != 'test':
    #  gt_roidb = self.gt_roidb()
    #  rpn_roidb = self._load_rpn_roidb(gt_roidb)
    #  roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    #else:
    roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    """
    done
    """
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_wider_annotation(self, index):
    """
    Load image path and bounding boxes info from mat file in WIDER done
    """
    face_bbx_list = self._mat["face_bbx_list"]
    invalid_label_list = self._mat["invalid_label_list"]

    photo_idx = index
    # (N, [x1,y1,w,h])
    bboxes = np.array([], dtype=np.uint16)
    for section in face_bbx_list:
        section_size = section[0].shape[0]

        if photo_idx - section_size < 0:
            photo = section[0][photo_idx]
            for bbox in photo[0]:
                bbox = bbox.astype('uint16')
                bboxes = np.concatenate((bboxes, bbox))
            break
        else:
            photo_idx -= section_size
    bboxes = np.reshape(bboxes, (-1,4))

    # convert bboxes to x1,y1,x2,y2
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    # reset photo_idx
    photo_idx = index
    gt_classes = np.array([], dtype=np.int32)
    for section in invalid_label_list:
        section_size = section[0].shape[0]

        if photo_idx - section_size < 0:
            photo = section[0][photo_idx]
            for bbox in photo[0]:
                bbox = bbox.astype('uint16')
                gt_classes = np.concatenate((gt_classes, bbox))
            break
        else:
            photo_idx -= section_size
    gt_classes = np.reshape(gt_classes, (-1))
    # flip 1 and 0
    gt_classes = 1 - gt_classes

    if self._image_set == 'train':
        if index == 10969:
            bboxes = np.delete(bboxes, (27), axis=0)
            gt_classes = np.delete(gt_classes, 27)
        if index == 12381:
            bboxes = np.delete(bboxes, (358), axis=0)
            gt_classes = np.delete(gt_classes, 358)
    
    if self._image_set == 'val':
        if index == 1885:
            bboxes = np.delete(bboxes, (7), axis=0)
            gt_classes = np.delete(gt_classes, 7)

    assert(bboxes[:, 2] >= bboxes[:, 0]).all()
    assert(bboxes.shape[0] == gt_classes.shape[0])

    num_objs = bboxes.shape[0]

    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    overlaps[gt_classes == 1, 1] = 1

    seg_areas = np.zeros((num_objs), dtype=np.float32)
    seg_areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)

    # TODO: READ ABOUT THIS
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': bboxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  #def _get_comp_id(self):
  #  comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
  #             else self._comp_id)
  #  return comp_id

  #def _get_voc_results_file_template(self):
  #  # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
  #  filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
  #  path = os.path.join(
  #    self._devkit_path,
  #    'results',
  #    'VOC' + self._year,
  #    'Main',
  #    filename)
  #  return path

  #def _write_voc_results_file(self, all_boxes):
  #  for cls_ind, cls in enumerate(self.classes):
  #    if cls == '__background__':
  #      continue
  #    print('Writing {} VOC results file'.format(cls))
  #    filename = self._get_voc_results_file_template().format(cls)
  #    with open(filename, 'wt') as f:
  #      for im_ind, index in enumerate(self.image_index):
  #        dets = all_boxes[cls_ind][im_ind]
  #        if dets == []:
  #          continue
  #        # the VOCdevkit expects 1-based indices
  #        for k in range(dets.shape[0]):
  #          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
  #                  format(index, dets[k, -1],
  #                         dets[k, 0] + 1, dets[k, 1] + 1,
  #                         dets[k, 2] + 1, dets[k, 3] + 1))

  #def _do_python_eval(self, output_dir='output'):
  #  annopath = os.path.join(
  #    self._devkit_path,
  #    'VOC' + self._year,
  #    'Annotations',
  #    '{:s}.xml')
  #  imagesetfile = os.path.join(
  #    self._devkit_path,
  #    'VOC' + self._year,
  #    'ImageSets',
  #    'Main',
  #    self._image_set + '.txt')
  #  cachedir = os.path.join(self._devkit_path, 'annotations_cache')
  #  aps = []
  #  # The PASCAL VOC metric changed in 2010
  #  use_07_metric = True if int(self._year) < 2010 else False
  #  print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
  #  if not os.path.isdir(output_dir):
  #    os.mkdir(output_dir)
  #  for i, cls in enumerate(self._classes):
  #    if cls == '__background__':
  #      continue
  #    filename = self._get_voc_results_file_template().format(cls)
  #    rec, prec, ap = voc_eval(
  #      filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
  #      use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
  #    aps += [ap]
  #    print(('AP for {} = {:.4f}'.format(cls, ap)))
  #    with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
  #      pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
  #  print(('Mean AP = {:.4f}'.format(np.mean(aps))))
  #  print('~~~~~~~~')
  #  print('Results:')
  #  for ap in aps:
  #    print(('{:.3f}'.format(ap)))
  #  print(('{:.3f}'.format(np.mean(aps))))
  #  print('~~~~~~~~')
  #  print('')
  #  print('--------------------------------------------------------------')
  #  print('Results computed with the **unofficial** Python eval code.')
  #  print('Results should be very close to the official MATLAB eval code.')
  #  print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
  #  print('-- Thanks, The Management')
  #  print('--------------------------------------------------------------')

  #def _do_matlab_eval(self, output_dir='output'):
  #  print('-----------------------------------------------------')
  #  print('Computing results with the official MATLAB eval code.')
  #  print('-----------------------------------------------------')
  #  path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
  #                      'VOCdevkit-matlab-wrapper')
  #  cmd = 'cd {} && '.format(path)
  #  cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
  #  cmd += '-r "dbstop if error; '
  #  cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
  #    .format(self._devkit_path, self._get_comp_id(),
  #            self._image_set, output_dir)
  #  print(('Running:\n{}'.format(cmd)))
  #  status = subprocess.call(cmd, shell=True)

  #def evaluate_detections(self, all_boxes, output_dir):
  #  self._write_voc_results_file(all_boxes)
  #  self._do_python_eval(output_dir)
  #  if self.config['matlab_eval']:
  #    self._do_matlab_eval(output_dir)
  #  if self.config['cleanup']:
  #    for cls in self._classes:
  #      if cls == '__background__':
  #        continue
  #      filename = self._get_voc_results_file_template().format(cls)
  #      os.remove(filename)

  #def competition_mode(self, on):
  #  if on:
  #    self.config['use_salt'] = False
  #    self.config['cleanup'] = False
  #  else:
  #    self.config['use_salt'] = True
  #    self.config['cleanup'] = True


#if __name__ == '__main__':
#  from datasets.pascal_voc import pascal_voc
#
#  d = pascal_voc('trainval', '2007')
#  res = d.roidb
#  from IPython import embed;
#
#  embed()
