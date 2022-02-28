"""
 DeepFashion - pytorch dataset
 Category prediction
"""

from __future__ import division
import os
from typing import List

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class DeepFashion(Dataset):
    CLASSES = None

    def __init__(self,
                 img_path,
                 img_cate_file,
                 img_bbox_file,
                 img_size):
        # general img path
        self.img_path = img_path

        # Pytorch transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # from img-cate txt to list
        fp = open(img_cate_file, 'r')
        list_img_cate_file = [x.strip() for x in fp]
        #
        self.img_list: List[str] = [] # img list
        self.targets: List[int] = [] # cate label list
        for i, line in enumerate(list_img_cate_file):
            if i > 1:
                tmp_line = line.split()
                self.img_lis.append(tmp_line[0])
                self.targets.append(int(tmp_line[1]))

        self.img_size = img_size

        # load bbox
        if img_bbox_file:
            self.with_bbox = True
            # from img-bbox txt to list
            fp = open(img_bbox_file, 'r')
            list_img_bbox_file = [x.strip() for x in fp]
            #
            self.bboxes: List[list] = []
            for i, line in enumerate(list_img_bbox_file):
                if i > 1:
                    tmp_line = line.split()
                    self.bboxes.append([int(tmp_line[1]), int(tmp_line[2]), int(tmp_line[3]), int(tmp_line[4])])
        else:
            self.with_bbox = False
            self.bboxes = None

    def get_basic_item(self, idx):
        img = Image.open(os.path.join(self.img_path,
                                      self.img_list[idx])).convert('RGB')

        width, height = img.size
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]) - 10)
            y1 = max(0, int(bbox_cor[1]) - 10)
            x2 = int(bbox_cor[2]) + 10
            y2 = int(bbox_cor[3]) + 10
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            img = img.crop(box=(x1, y1, x2, y2))
        else:
            bbox_w, bbox_h = self.img_size[0], self.img_size[1]

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)

        # Target for Cate task
        cate = torch.LongTensor([int(self.targets[idx]) - 1])  #
        cate = torch.LongTensor(cate[0])

        data = {'img': img, 'cate': cate}
        return data

    def __getitem__(self, idx):
        data = self.get_basic_item(idx)
        return data['img'], data['cate'] #

    def __len__(self):
        return len(self.img_list)
