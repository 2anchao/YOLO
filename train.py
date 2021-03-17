#encoding:utf-8
#author: an_chao1994@163.com

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.autograd import Variable
import numpy as np

from backbone.vovnet import Vovnet27_slim
from backbone.resnet import resnet34
from configs.config import Configs as cfg
from loss.yoloLoss import yoloLoss
from dataset.dataset import yoloDataset
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser(description="use distributed to train yolov1")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--visible_gpu",type=str, default="0,1")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
dist.init_process_group(backend='nccl', init_method='env://')

# model = Vovnet27_slim(cfg)
model = resnet34()
torch.cuda.set_device(args.local_rank)
model = model.cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

train_dataset = yoloDataset(train=True, debug=False)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset,batch_size=cfg.Solver["batch_size"], shuffle=False, num_workers=4, sampler=train_sampler)

test_dataset = yoloDataset(train=False, debug=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_loader = DataLoader(test_dataset,batch_size=cfg.Solver["batch_size"], shuffle=False, num_workers=4, sampler=test_sampler)

criterion = yoloLoss()
def train():
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.Solver["learning_rate"], momentum=0.9, weight_decay=5e-4)
    learning_rate = cfg.Solver["learning_rate"]
    for epoch in range(cfg.Solver["num_epochs"]):
        model.train()
        if (epoch+1)%(cfg.Solver["lr_step"]) == 0:
            learning_rate /= 10

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        total_loss = 0.
        for i,(images,target) in enumerate(train_loader):
            images = Variable(images)
            target = Variable(target)
            images,target = images.cuda(),target.cuda()
            
            pred = model(images)
            loss = criterion(pred, target)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], learning_rate:%.6f, Iter [%d/%d]' 
                %(epoch+1, cfg.Solver["num_epochs"], learning_rate, i+1, len(train_loader)))
        val()

best_test_loss = np.inf
def val():
    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i,(images,target) in enumerate(test_loader):
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = model(images)
            loss = criterion(pred,target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        global best_test_loss
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print("save model!")
            torch.save(model.state_dict(),'./save_model/best.pth')      
        torch.save(model.state_dict(),'./save_model/yolo.pth')

if __name__ == "__main__":
    train()
  
