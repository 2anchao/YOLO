#encoding:utf-8
#author: an_chao1994@163.com

class Configs:
    Data = {"train_data_path":"/workspace/mnt/storage/dingrui/traffic-detection-data/mini_traffic_data/train.json",
           "val_data_path": "/workspace/mnt/storage/dingrui/traffic-detection-data/Traffic_Fu_Xu/ImageSets/Main/val_json.txt"
           }
    Inputs = {"num_class": 4,
             "image_size": (448, 448),
             "grid_num": 7,
             }
    Pretrain_model="checkpoint/resnet34-333f7ec4.pth"
    Solver = {"learning_rate":0.001,
              "num_epochs":50,
              "batch_size":8,
              "lr_step":20,
              "loss_weight":{"loc_loss_weight":2,
                             "contain_loss_weight":1.5,
                             "not_contain_loss_weight":1,
                             "nooobj_loss_weight":1,
                             "class_loss_weight":1
                             }
              }
    
    
    