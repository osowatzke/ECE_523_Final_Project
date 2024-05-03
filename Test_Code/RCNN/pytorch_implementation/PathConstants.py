import os

class PathConstants():
    _instance = None
    _root_dir = None
    TRAIN_DIR = None
    VAL_DIR = None
    TEST_DIR = None
    
    def __new__(cls, root_dir):
        if not cls._instance:
            super().__new__(cls)
            cls._root_dir = root_dir
            cls.__set_all_dir()
        return cls._instance

    @classmethod
    def __set_all_dir(cls):
        cls.__set_train_dir()
        cls.__set_val_dir()
        cls.__set_test_dir()

    @classmethod
    def __set_train_dir(cls):
        cls.TRAIN_DIR = os.path.join(cls._root_dir,'images_thermal_train')

    @classmethod
    def __set_val_dir(cls):
        cls.VAL_DIR = os.path.join(cls._root_dir,'images_thermal_val')

    @classmethod
    def __set_test_dir(cls):
        cls.TEST_DIR = os.path.join(cls._root_dir,'images_thermal_test')