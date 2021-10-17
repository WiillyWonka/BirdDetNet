import torch
import torch.nn as nn
import numpy as np
gt_list=[[0,0,1],[1,2,1],[0,2,2],[2,2,2]]
# set initial size for gt_mask for batch_size=1
gt_cls_mask=np.zeros((3,4,3))
# add labels to mask
for label in gt_list:
    gt_cls_mask[label[2]][label[0]][label[1]]=1
# create predictions with initial shape
predictions=np.random.random((3,4,3))
loss=nn.CrossEntropyLoss()
# unsqueeze batch dimention of predictions (only in example code) and reshape gt_mask after argmax from shape [h,w] to shape [1,h,w] where 1 - batch_size
print(loss(torch.Tensor(predictions).unsqueeze(0),torch.Tensor(np.argmax(gt_cls_mask,axis=0).reshape((1,4,3))).long()))