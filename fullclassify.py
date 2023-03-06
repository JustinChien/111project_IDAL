import os
import shutil
path='C:\\archive\\k_CBIS-DDSM\\jpg_img'
destpath='C:\\full'

for i in os.listdir(path):
    path2=path+'\\{}'.format(i)
    arr=i.split('-',1)
    for j in os.listdir(path2):
        if j[0:4]=='full':
            shutil.copy(path2+'\\{}'.format(j),destpath+'\\{}.jpg'.format(arr[0]))