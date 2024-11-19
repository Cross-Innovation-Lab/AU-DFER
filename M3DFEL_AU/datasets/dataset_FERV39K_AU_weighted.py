import os.path
from torch.utils import data
import torch
import glob
import os
import numpy as np
import csv
import PIL.Image as Image
import torchvision

from .video_transform import *
from datasets import video_transform

class FERV39KDataset(data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else self.args.test_dataset
        self.image_size = self.args.crop_size
        self.num_frames = self.args.num_frames
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()
            
        pass

    def get_label_id(self, label):
        map = {"Angry": 3, "Disgust": 5, "Fear": 6, "Happy": 0, "Neutral": 2, "Sad": 1, "Surprise": 4}    
        return map[label]
    
    def get_data(self):
        full_data = []
        AU_path = './datasets/AU_FERV39K.txt'
        AU_labels = {}
        with open(AU_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.strip('\n').split(' ')
                for i in range (2,len(line)):
                    line[i] = int(line[i])
                frame_name = line[1]
                AU_labels[frame_name] = line[2:]
        npy_path = self.path.replace('csv', 'npy')
        if False:
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    scene = row[0].split('\/')[0]
                    emotion = row[0].split(' ')[1]
                    path = row[0].split(' ')[0]
                    path = os.path.join(self.args.root, "2_ClipsforFaceCrop", path)
                    full_num_frames = len(os.listdir(path))

                    full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg'))
                    full_video_frames_paths.sort()
                    full_data.append({"path": full_video_frames_paths, "emotion": self.get_label_id(emotion), "scene": scene, "num_frames": full_num_frames, "AU_labels":AU_labels})
                np.save(npy_path, full_data)
            
        return full_data

    def get_transform(self):

        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupRandomSizedCrop(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                            Stack(),
                                                            ToTorchFormatTensor()])
        
        return transform

    def __getitem__(self, index):
        data = self.data[index]

        AU_label = []
        full_video_frames_paths = data['path']
        #print(full_video_frames_paths)
        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)
        for i in range(0,self.num_frames):            
            frame = int(full_num_frames * i / self.num_frames)
            if self.args.random_sample:
                frame += int(random.random() * self.num_frames)
                frame = min(full_num_frames - 1, frame)
            video_frames_paths.append(full_video_frames_paths[frame])
            frame_name = full_video_frames_paths[frame][:-4]
            split_list = frame_name.split('/')
            frame_name = split_list[-4]+'/'+split_list[-3]+'/'+split_list[-2]+'/'+split_list[-1]
            if frame_name not in data['AU_labels']:
                continue
            AU_label.append(data['AU_labels'][frame_name])
        AU_avg = []
        for i in range(0,18):
            count = 0
            for j in range(0,len(AU_label)):
                if AU_label[j][i]>=0.01:
                    count+=1
            if count>=len(AU_label)/2:
                AU_avg.append(1)
            else:
                AU_avg.append(0)
        #print(AU_avg)
        images = []
        for video_frames_path in video_frames_paths:
            images.append(Image.open(video_frames_path).convert('RGB'))
        #print(len(images),len(AU_label))
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        AU_avg = torch.Tensor(AU_avg)
        return images, data["emotion"], AU_avg

    def __len__(self):
        return len(self.data)
