import os
import pprint
import numpy as np
import pandas as pd
import cv2
from PIL import Image 
path='D:\\all'
# mask_list = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in v]

# for arr in mask_list:
#     summed_mask = np.add(summed_mask, arr)
# _, summed_mask_bw = cv2.threshold(
#     src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
# )
imgsum=cv2.imread('C:\python\Calc-Test_P_00038_LEFT_CC_1.png',cv2.IMREAD_GRAYSCALE)
# for arr in imgsum:
#     for k in arr:
#         if k>0:
#             print(k)
   
for i in os.listdir(path):
    summed_mask = np.zeros((224,224))
    for j in os.listdir(path+'\\{}'.format(i)):
        img=cv2.imread(path+'\\{}\{}'.format(i,j),cv2.IMREAD_GRAYSCALE)
        # img=cv2.resize(img,(224,224))
        summed_mask = np.add(summed_mask,img)
    _, summed_mask_bw = cv2.threshold(
            src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
        )
    cv2.imwrite('D:\\{}.png'.format(i), summed_mask_bw)


    