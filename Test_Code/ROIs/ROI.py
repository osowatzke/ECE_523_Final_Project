################################################################################################
# Authors: Nick Blanchard and Owen Sowatzke
# Code to generate ROIs for a given image.
################################################################################################


## Dependencies ################################################################################
import cv2
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat,savemat


## Global Variables ############################################################################
im_path = 'images_thermal_train/data/video-2Af3dwvs6YPfwSSf6-frame-000000-imW24bapJsHpTahce.jpg'


## Load Image ##################################################################################
def load_image(sub_path):

    path_this_file = os.path.split(os.path.abspath(__file__))[0]
    image_path = os.path.join(path_this_file,'..','..','FLIR_ADAS_v2',sub_path)
    image = Image.open(image_path)
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    return image


## Find ROIs ###################################################################################
def selective_search(image):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    savemat('rects.mat',{'rects':rects})

    matData = loadmat('rects.mat')
    rects = matData['rects']
    
    return rects


## Display Results #############################################################################
def display_results(rects, image):

    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255))

    cv2.imshow('image', image)
    while cv2.waitKey(0) != 27:
        pass
    cv2.destroyAllWindows()


## Save ROIs ###################################################################################
def save_rois(rects, image):
    
    i = 0
    for rect in rects:
        if i > 10:
            while cv2.waitKey(0) != 27:
                pass
            cv2.destroyAllWindows()
            return
        else:
            sub_im = image[rect[0]:rect[2], rect[1]: rect[3]]
            if rect[2] - rect[0] > 0 and rect[3] - rect[1] > 0:
                cv2.imshow(str(i), sub_im)
                i = i + 1



## Debug #######################################################################################
if __name__ == "__main__":

    # Load image
    im = load_image(im_path)

    # Find ROIs
    rects = selective_search(im)

    # Show results
    # display_results(rects, im)

    # Save ROIs
    save_rois(rects, im)