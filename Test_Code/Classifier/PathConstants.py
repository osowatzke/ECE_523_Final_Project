import os

class PathConstants():
    _instance = None
    _use_local_copy = None
    SRC_DIR = None
    DATA_DIR = None
    TRAIN_DIR = None
    VAL_DIR = None
    TEST_DIR = None
    
    def __new__(cls, use_local_copy=False):
        if not cls._instance:
            super().__new__(cls)
            cls.SRC_DIR = os.path.dirname(__file__)
            cls._use_local_copy = use_local_copy
            cls.__set_all_dir()
            cls._instance = cls
        return cls._instance
    
    @classmethod
    def source_from_repo(cls):
        cls._use_local_copy = False
        cls.__set_all_dir()
        
    @classmethod
    def source_from_local_copy(cls):
        cls._use_local_copy = True
        cls.__set_all_dir()

    @classmethod
    def __set_all_dir(cls):
        cls.__set_data_dir()
        cls.__set_train_dir()
        cls.__set_val_dir()
        cls.__set_test_dir

    @classmethod
    def __set_data_dir(cls):
        if cls._use_local_copy:
            cls.DATA_DIR = '../FLIR_ADAS_v2'
        else:
            cls.DATA_DIR = os.path.join(cls.SRC_DIR,'..','..','..','FLIR_ADAS_v2')
            cls.DATA_DIR = os.path.abspath(cls.DATA_DIR)

    @classmethod
    def __set_train_dir(cls):
        cls.TRAIN_DIR = os.path.join(cls.DATA_DIR,'images_thermal_train')

    @classmethod
    def __set_val_dir(cls):
        cls.VAL_DIR = os.path.join(cls.DATA_DIR,'images_thermal_val')

    @classmethod
    def __set_test_dir(cls):
        cls.TEST_DIR = os.path.join(cls.DATA_DIR,'images_thermal_test')

    '''
    SRC_DIR  = os.path.dirname(__file__)
    DATA_DIR = os.path.join(SRC_DIR,'..','..','..','FLIR_ADAS_v2')
    #DATA_DIR = '/tmp/FLIR_ADAS_v2'
    TRAIN_DIR = os.path.join(DATA_DIR,'images_thermal_train')
    VAL_DIR = os.path.join(DATA_DIR,'images_thermal_val')
    TEST_DIR = os.path.join(DATA_DIR,'images_thermal_test')
    '''

if __name__ == "__main__":
    PATH_CONSTANTS = PathConstants(use_local_copy=True)
    print(PATH_CONSTANTS.DATA_DIR)
    PATH_CONSTANTS.source_from_repo()
    print(PATH_CONSTANTS.DATA_DIR)