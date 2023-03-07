from IPython.core.display import Image
import cv2
import numpy as np
import os
import pprint
from PIL import Image
from pathlib import Path
l=0.01
r=0.01
d=0.04
u=0.04
thresh=0.1
maxval=1.0
ksize=23
operation='open'
reverse=True
top_x=1
clip=2.0
tile=8
def cropBorders(img, l=0.01, r=0.01, u=0.04, d=0.04):

    nrows,ncols=img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img
def checkLRFlip(mask):
  # Get number of rows and columns in the image.
  nrows, ncols = mask.shape
  x_center = ncols // 2
  y_center = nrows // 2

  # Sum down each column.
  col_sum = mask.sum(axis=0)
  # Sum across each row.
  row_sum = mask.sum(axis=1)

  left_sum = sum(col_sum[0:x_center])
  right_sum = sum(col_sum[x_center:-1])

  if left_sum < right_sum:
      LR_flip = True
  else:
      LR_flip = False

  return LR_flip
def makeLRFlip(img):
  flipped_img = np.fliplr(img)
  return flipped_img
def pad(img):
  nrows, ncols= img.shape

  # If padding is required...
  if nrows != ncols:

      # Take the longer side as the target shape.
      if ncols < nrows:
          target_shape = (nrows, nrows)
      elif nrows < ncols:
          target_shape = (ncols, ncols)

      # pad.
      padded_img = np.zeros(shape=target_shape)
      padded_img[:nrows, :ncols] = img

  # If padding is not required...
  elif nrows == ncols:

      # Return original image.
      padded_img = img

  return padded_img
def maskPreprocess(mask, lr_flip):

    # Step 1: Initial crop.
    mask = cropBorders(img=mask)

    # Step 2: Horizontal flip.
    if lr_flip:
        mask = makeLRFlip(img=mask)

    # Step 3: Pad.
    mask_pre = pad(img=mask)

    return mask_pre
thefromDIR='D:\\ROI'
for i in os.listdir(thefromDIR):
    for j in os.listdir(thefromDIR+'\{}'.format(i)):
        img = cv2.imread(thefromDIR+'\{}\{}'.format(i,j),cv2.CV_8UC1)
        lr_flip = checkLRFlip(mask=img)
        premask=maskPreprocess(mask=img, lr_flip=lr_flip)
        cv2.imwrite('D:\\imgpreROI\{}.png'.format(i),premask)