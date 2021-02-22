import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList


########### 读取数据 ############
#im = cv2.imread('./input.jpg')
# cv2_imshow(im)

########## 指定配置文件 #############
cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.merge_from_file('../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.WEIGHTS = 'model_final_a54504.pkl'   # COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml



############# 处理输入图像：PIL转成tensor ##########
image = cv2.imread('./input.jpg')
height, width = image.shape[:2]
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs = [{"image": image, "height": height, "width": width}]


############ 使用Models执行网络的一部分 ###############
model = build_model(cfg)   # returns a torch.nn.Module with random parameters
DetectionCheckpointer(model).load('model_final_a54504.pkl')

model.eval()
with torch.no_grad():
    images = model.preprocess_image(inputs)
    features = model.backbone(images.tensor)
    # outputs = model(image)
    # features = model.backbone(image)

# features是一个dict：
print(features.keys())
with open('./print_features.txt_segmentation', 'w+') as f:

    print("features是一个字典，key包括['p2', 'p3', 'p4', 'p5', 'p6']", features['p2'], file=f)
#print(type(model.named_children()))
#print(model.named_children())

'''
for name, child in model.named_children():
    for i in child:
        print(i)
    #print(type(child))
    #print(type(name))
'''
'''
child和name的type：
<class 'detectron2.modeling.backbone.fpn.FPN'>
<class 'str'>
<class 'detectron2.modeling.proposal_generator.rpn.RPN'>
<class 'str'>
<class 'detectron2.modeling.roi_heads.roi_heads.StandardROIHeads'>
<class 'str'>
'''
