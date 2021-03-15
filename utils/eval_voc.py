#encoding:utf-8
#author: an_chao1994@163.com

import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from collections import defaultdict
from tqdm import tqdm
from dataset.datasetv2 import yoloDataset
from configs.config import Configs as cfg
from backbone.resnet import resnet34
from demo import decoder
from torch.utils.data import DataLoader

CLASSES = ("person","non-motor","car","tricycle")

def voc_ap(rec,prec):
    # correct ap caculation
    mrec = np.concatenate(([0.],rec,[1.]))
    mpre = np.concatenate(([0.],prec,[0.]))

    for i in range(mpre.size -1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def voc_eval(preds,target,CLASSES=CLASSES,threshold=0.5):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i,class_ in enumerate(CLASSES):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            break
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

def predict_gpu(model,image,image_id):
    result = []
    with torch.no_grad():
        img = image.cuda()
        pred = model(img) #1x7x7x(10+class_num)
        pred = pred.cpu()
        boxes,cls_indexs,probs =  decoder(pred)
        b,c,h,w = image.shape

        for i,box in enumerate(boxes):
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1,y1),(x2,y2),CLASSES[cls_index],image_id,prob])
    return result

def test_eval():
    preds = {'cat':[['image01',0.9,20,20,40,40],['image01',0.8,20,20,50,50],['image02',0.8,30,30,50,50]],'dog':[['image01',0.78,60,60,90,90]]}
    target = {('image01','cat'):[[20,20,41,41]],('image01','dog'):[[60,60,91,91]],('image02','cat'):[[30,30,51,51]]}
    voc_eval(preds,target, CLASSES=['cat','dog'])

if __name__ == '__main__':


    test_dataset = yoloDataset(train=False, debug=False)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=0)

    print('---start test---')
    model = resnet34()
    model.load_state_dict(torch.load('./save_model/best.pth'))
    model.eval()
    model.cuda()
    count = 0
    targets =  defaultdict(list)
    preds = defaultdict(list)
    # (448,448)尺寸下评估
    for i,(imageid, image, boxes, labels) in enumerate(test_loader):
        print("eval schedule:%d/%d"%(i, len(test_loader)))
        for i in range(len(labels)):
            label = labels[i]
            classname = CLASSES[label]
            tx1, ty1, tx2, ty2 = boxes[0][i]
            targets[(imageid, classname)].append([int(tx1), int(ty1), int(tx2), int(ty2)])
        result = predict_gpu(model, image, imageid) #result[[left_up,right_bottom,class_name,image_path],]
        for (x1,y1),(x2,y2),class_name,image_id,prob in result: #image_id is actually image_path
            preds[class_name].append([image_id,prob,x1,y1,x2,y2])
    
    print('---start evaluate---')
    voc_eval(preds,targets, CLASSES = CLASSES)