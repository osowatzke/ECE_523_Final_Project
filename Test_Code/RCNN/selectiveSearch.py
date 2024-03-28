import cv2
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat,savemat

selectiveSearch = True

path_this_file = os.path.split(os.path.abspath(__file__))[0]
# sub_path = 'images_rgb_train/data/video-2BARff2EP7ZWkiF7n-frame-000432-pEpYGJT3PDodWjHHN.jpg''
sub_path = 'images_thermal_train/data/video-2Af3dwvs6YPfwSSf6-frame-000000-imW24bapJsHpTahce.jpg'
image_path = os.path.join(path_this_file,'..','..','FLIR_ADAS_v2',sub_path)
image = Image.open(image_path)
image = np.array(image)
if len(image.shape) == 2:
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
if selectiveSearch:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    savemat('rects.mat',{'rects':rects})

matData = loadmat('rects.mat')
rects = matData['rects']

for rect in rects:
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255))

cv2.imshow('image', image)
while cv2.waitKey(0) != 27:
    pass
cv2.destroyAllWindows()