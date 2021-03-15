#encoding:utf-8
#author: an_chao1994@163.com

import os
import cv2
import json
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from configs.config import Configs as cfg

class yoloDataset(data.Dataset):
    def __init__(self, train=True, debug=False):
        print('Data Init')
        self.train = train
        self.debug = debug
        self.boxes = []
        self.labels = []
        self.trainf = cfg.Data["train_data_path"]
        self.valf = cfg.Data["val_data_path"]
        self.image_size = cfg.Inputs["image_size"]
        self.num_classes = cfg.Inputs["num_class"]
        self.grid_num = cfg.Inputs["grid_num"]
        self.transform = transforms.ToTensor()

        if self.train:
            with open(self.trainf) as f:
                lines  = f.readlines()
                self.fnames = [path.strip() for path in lines]
        else:
            with open(self.valf) as f:
                lines  = f.readlines()
                self.fnames = [path.strip() for path in lines]

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        label_path = fname.replace("ori_images", "labels").replace(
            ".png", ".json").replace(".jpg", ".json")
        boxes, labels, box_level = self.get_boxes(json_file=label_path, num_classes=self.num_classes)
        boxes = torch.from_numpy(boxes)
        h,w,_ = img.shape
        assert box_level == "cxcywh", "box type id not cxcywh"
        if self.train:
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)

        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
        if self.debug:
            cvimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            boxes = self.cxcywh2xyxy(boxes=boxes, absize=(self.image_size[0], self.image_size[1]))
            for i, box in enumerate(boxes):
                cv2.rectangle(cvimg, pt1=(box[0],box[1]), pt2=(box[2],box[3]), color=(0,255,0), thickness=1)
                cv2.putText(cvimg, "%d" % (labels[i]),
                            (box[0] + 6, box[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),
                            1, cv2.LINE_AA)  
            cv2.imshow("show", cvimg) 
            if cv2.waitKey(0)==ord("q"):
                cv2.destroyAllWindows()
        target = self.encoder(boxes, labels)# 7x7x14
        img = self.transform(img)
        img = img.float().div(255.)
        return img, target

    def __len__(self):
        return len(self.fnames)

    def get_boxes(self, json_file, num_classes=4):
        box_valid = []
        labels_valid = []
        with open(json_file, 'r') as fr:
            json_data = json.load(fr)
            boxes = json_data['boxes']['boxes']
            labels = json_data['labels']
            box_level = json_data['boxes']['box_level']
            if num_classes is not None and isinstance(num_classes, int):
                for i, gt_label in enumerate(labels):
                    if gt_label < num_classes:
                        box_valid.append(boxes[i])
                        labels_valid.append(labels[i])
            else:
                print("please give right num_classes")
            boxes = np.array(box_valid).astype(np.float32).reshape(-1, 4)
        return boxes, labels_valid, box_level

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[cx,cy,w,h],[cx,cy,w,h],...]
        labels (tensor) [...]
        return 7x7x(10+classname)
        '''
        target = torch.zeros((self.grid_num, self.grid_num, 10+self.num_classes))
        cell_size = 1./self.grid_num
        wh = boxes[:, 2:]
        cxcy = boxes[:, :2]
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #这里的ij表示GT_BOX的中心落到了哪一个网格里面
            target[int(ij[1]),int(ij[0]),4] = 1 #对应网格的第一个尺度的BOX预测的值匹配到前景
            target[int(ij[1]),int(ij[0]),9] = 1 #对应网格的第二个尺度的BOX预测的值匹配到前景
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1 #对应类别
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size #中心点在网格里相对网格左上角坐标的偏移距离
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

    def cxcywh2xyxy(self, boxes, absize):
        """
        boxes (tensor) [[cx,cy,w,h],[cx,cy,w,h],...]
        absize (h, w)
        """
        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        xmin = xmin - w / 2
        ymin = ymin - h / 2
        xmax = xmin + w
        ymax = ymin + h
        xmin *= absize[0]
        ymin *= absize[1]
        xmax *= absize[0]
        ymax *= absize[1]
        xmin = xmin.unsqueeze(-1)
        ymin = ymin.unsqueeze(-1)
        xmax = xmax.unsqueeze(-1)
        ymax = ymax.unsqueeze(-1)
        bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
        return bbox

    # transforms
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr


if __name__ == '__main__':
    train_dataset = yoloDataset()
    train_loader = DataLoader(train_dataset, 
                              batch_size=2, 
                              shuffle=False, 
                              num_workers=0)
    for img, target in train_loader:
        print(img.shape)
        

