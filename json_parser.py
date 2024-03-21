import json
import shutil
import os

class json_parser:

    __LABELS = {
        'person'        : 1,
        'bike'          : 2,
        'car'           : 3,
        'motor'         : 4,
        'bus'           : 6,
        'train'         : 7,
        'truck'         : 8,
        'light'         : 10,
        'hydrant'       : 11,
        'sign'          : 12,
        'dog'           : 17,
        'skateboard'    : 37,
        'scooter'       : 77,
        'other vehicle' : 79
    }

    def __init__(self, json_file, out_dir):
        self.json_file = json_file    
        self.out_dir = out_dir
        self.json_data = None

    def run(self):
        self.create_out_dir()
        self.read_json_file(json_file)
        self.convert_to_yolo_format()

    def create_out_dir(self):
        if os.path.isdir(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=False)

    def read_json_file(self, json_file):
        with open(json_file) as fid:
            self.json_data = json.load(fid)

    def convert_to_yolo_format(self):
        frames = self.json_data['frames']
        for frame in frames:
            image_height = frame['height']
            image_width = frame['width']
            annotations = frame['annotations']
            frame_index = frame['videoMetadata']['frameIndex']
            video_id = frame['videoMetadata']['videoId']
            frame_id = frame['videoMetadata']['frameId']
            file_name = f'video-{video_id}-frame-{frame_index:>0{5}d}-{frame_id}.txt'
            file_path = os.path.join(self.out_dir, file_name)
            with open(file_path,'w') as file:
                for annotation in annotations:
                    h = annotation['boundingBox']['h']
                    w = annotation['boundingBox']['w']
                    x = annotation['boundingBox']['x']
                    y = annotation['boundingBox']['y']
                    x = x + w/2.0
                    y = y + h/2.0
                    h = h/image_height
                    w = w/image_width
                    x = x/image_width
                    y = y/image_height
                    for label in annotation['labels']:
                        if label in self.__LABELS.keys():
                            id = self.__LABELS[label]
                            file.write(f'{id} {x:>.6f} {y:>.6f} {w:>.6f} {h:>.6f}\n')

if __name__ == "__main__":
    json_file = './FLIR_ADAS_v2/images_thermal_train/index.json'
    out_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0],'run','labels')
    parser = json_parser(json_file, out_dir)
    parser.run()

