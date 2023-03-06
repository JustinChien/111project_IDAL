import SimpleITK as sitK
import numpy as np
import cv2
import os
def convert(img,low_window,high_window,save_path):
    breast=np.array([low_window*1,high_window*1])
    newimg=(img-breast[0])/(breast[1]-breast[0])
    newimg=(newimg*255).astype('uint8')
    cv2.imwrite(save_path,newimg)
thefromDIR = "D:\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"    
for i in os.listdir(thefromDIR):
    if i[-1]=='C' or i[-1]=='O':
        path=thefromDIR+'\\{}'.format(i)
        for j in os.listdir(path):
            path=path+'\\{}'.format(j)
            for k in os.listdir(path):
                path=path+'\\{}'.format(k)
                for p in os.listdir(path):
                    dcpath=path+'\\{}'.format(p) 
                    outputpath='D:\\fullimg\\{}.png'.format(i)
                    ds_array=sitK.ReadImage(dcpath)
                    img_array=sitK.GetArrayFromImage(ds_array)
                    shape=img_array.shape
                    img_array=np.reshape(img_array,(shape[1],shape[2]))
                    high=np.max(img_array)
                    low=np.min(img_array)
                    convert(img_array,low,high,outputpath)