import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from utils import *


class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir,  transform=None, scale_up=1.5,
                 scale_down=1.0, img_size=256, map_size=32, UUID=-1):
        self.landmarks_frame = pd.read_csv(info_list, delimiter=",",
                                           header=None)
        self.root_dir = root_dir
        self.map_root_dir = root_dir
        self.transform = transform
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.img_size = img_size
        self.map_size = map_size
        self.UUID = UUID

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        face_img_filename = str(self.landmarks_frame.iloc[idx, 1])
        face_img_filepath = os.path.join(self.root_dir, face_img_filename)
        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1
        else:
            spoofing_label = 0
        image_x, map_x = self.get_single_image_x(face_img_filepath,
                                                 face_img_filename,
                                                 spoofing_label)
        sample = {'image_x': image_x, 'map_x': map_x, 'label': spoofing_label,
                  "UUID": self.UUID}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, img_path, img_name, spoofing_label):
        map_path = img_path.replace('.', '_depth.')
        bbx_path = img_path.rpartition('.')[0] + '.dat'
        img_ok = os.path.exists(img_path)
        map_ok = spoofing_label == 0 or os.path.exists(map_path)
        bbx_ok = os.path.exists(bbx_path)
        if not all([img_ok, map_ok, bbx_ok]):
            raise FileNotFoundError(f"File not found for {img_name}",
                                    f"{img_ok=}, {img_path=}",
                                    f"{map_ok=}, {map_path=}",
                                    f"{bbx_ok=}, {bbx_path=}")
        img_x_tmp = cv2.imread(img_path)
        if img_x_tmp is None:
            raise ValueError("img_x_tmp is None", img_path)
        img_x = cv2.resize(img_x_tmp, (self.img_size, self.img_size))
        if spoofing_label == 1:
            map_x_tmp = cv2.imread(map_path, 0)
            if map_x_tmp is None:
                raise ValueError("map_x_tmp is None", map_path)
            map_x = cv2.resize(map_x_tmp, (self.map_size, self.map_size))
        else:
            map_x = np.zeros((self.map_size, self.map_size))
        # face_scale = np.random.randint(int(self.scale_down*10),
        #                                int(self.scale_up*10))
        # face_scale = face_scale/10.0
        return img_x, map_x
