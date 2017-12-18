from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__', 'face')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'wider': ('WIDER_train',)}

def extract_faces(img, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []

    img = img.copy()

    x1 = dets[:, 0].astype(int)
    y1 = dets[:, 1].astype(int)
    x2 = dets[:, 2].astype(int)
    y2 = dets[:, 3].astype(int)
    scores = dets[:, -1]
    
    crop_imgs = []
    for i in inds:
        crop = img[y1[i]:y2[i], x1[i]:x2[i], :]
        crop_imgs.append(crop)

    return crop_imgs

def process_img(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'face', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3

    cls_ind = 1 # because we skipped background
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]

    # 300 x 5 (x1,y1,x2,y2,scores)
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)

    # NMS
    keep = nms(torch.from_numpy(dets), NMS_THRESH)
    dets = dets[keep.numpy(), :]

    crop_imgs = extract_faces(im, dets, thresh=CONF_THRESH)
    
    return crop_imgs

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('name', help='Save name')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0] % (5000))
    output_dir = os.path.join(cfg.DATA_DIR, args.name)


    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(2, tag='default', anchor_scales=[8, 16, 32])
    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    c = 0
    img_dir = os.path.join(cfg.DATA_DIR, 'face')
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        print(img_name)
        crop_imgs = process_img(net, img_name)
        
        for crop in crop_imgs:
            path = os.path.join(output_dir, '%04d.jpg' % c)
            print("Writing to {:s}".format(path))
            cv2.imwrite(path, crop)
            c += 1

