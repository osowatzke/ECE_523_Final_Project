import os

class PathConstants:
    SRC_DIR  = os.path.dirname(__file__)
    DATA_DIR = os.path.join(SRC_DIR,'..','FLIR_ADAS_v2')
    TRAIN_DIR = os.path.join(DATA_DIR,'images_thermal_train')
    VAL_DIR = os.path.join(TRAIN_DIR,'images_thermal_val')
    TEST_DIR = os.path.join(DATA_DIR,'images_thermal_test')