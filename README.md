# YOLOv1
you only look once
## 数据格式
一个图片对应一个json，json中是标注信息。    
{"width": "1920",    
"height": "1080",     
"task_type": "DET",     
"ori_image": "image.jpg",         
"boxes": {"box_type": "BBOX",     
         "box_level": "cxcywh",     
         "boxes": [[0.14, 0.24、, 0.011, 0.05],...],    
         "hard": [1],    
         "crowd": [1],    
         "extra": [[]]},    
"labels": [0,...],    
"ids": null,    
"segs": null   
}
## 训练
python train.py.  
参数在config.py 文件中配置。  
分布式支持在mutitrain分支。   
