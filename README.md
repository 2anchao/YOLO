# YOLOv1
you only look once
# 数据格式
一个图片对应一个json，json中是标注信息。    
{"width": "1920",    
"height": "1080",     
"task_type": "DET",     
"ori_image": "image.jpg",         
"boxes": {"box_type": "BBOX",     
         "box_level": "cxcywh",     
         "boxes": [[0.14010416666666667, 0.2412037037037037, 0.011458333333333348, 0.05092592592592593],...],    
         "hard": [1],    
         "crowd": [1],    
         "extra": [[]]},    
"labels": [0,...],    
"ids": null,    
"segs": null   
}
# 训练
python train.py.  
参数在config.py 文件中配置。  
分布式支持在mutitrain分支。   
