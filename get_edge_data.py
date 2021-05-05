import cv2
import numpy as np
import os

dataset = 'EGNet_ResNet50'
root = 'E:/sal/EGNet-master/Result/resnet/HKU-IS/{}/'.format(dataset)
save_root = 'E:/sal/EGNet-master/Result/resnet/HKU-IS/'
if not os.path.exists(save_root):
    os.makedirs(save_root)
# if dataset == 'DAVIS':
#     split_root = os.path.join(root, 'Annotations', '480p')
#     splits = os.listdir(split_root)
# else:
#     split_root = os.path.join(root, 'Annotations')
#     splits = os.listdir(split_root)

# for split in splits:
save_r = save_root
if not os.path.exists(save_r):
    os.makedirs(save_r)
gts_root = root
gts = os.listdir(gts_root)
for gt in gts:
    gt_root = os.path.join(gts_root, gt)
    gt1 = cv2.imread(gt_root, 0)
    gt1[np.where(gt1>128)]=255
    gt1[np.where(gt1 < 128)] = 0
    x, y = np.gradient(gt1)
    edge = np.sqrt(pow(x, 2) + pow(y, 2))
    edge[np.where(edge != 0)] = 1
    edge = edge * 255
    cv2.imwrite(os.path.join(save_r, gt), edge)


