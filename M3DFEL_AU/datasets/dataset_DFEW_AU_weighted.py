import os
import torch
import glob
import os
import numpy as np
import csv
import PIL.Image as Image
import torchvision
from torch.utils import data

from .video_transform import *


class DFEWDataset(data.Dataset):
    def __init__(self, args, mode):
        """Dataset for DFEW

        Args:
            args
            mode: String("train" or "test")

            num_frames: the number of sampled frames from every video, default: 16
            image_size: crop images to 112*112

        """
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else self.args.test_dataset
        self.num_frames = self.args.num_frames
        self.image_size = self.args.crop_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()

        pass

    def get_data(self):
        """get data path, label from the csv file

        Returns:
            data_dict:{"path", "emotion", "num_frames"}
        """
        full_data = []

        npy_path = self.path.replace('csv', 'npy')
        print("loading data")

        AU_labels = {}
        with open('./datasets/AU_DFEW.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.strip('\n').split(' ')
                for i in range (2,len(line)):
                    line[i] = int(line[i])
                frame_name = line[1].replace('.csv/','\\')
                if len(frame_name) == 7:
                    frame_name = line[1].replace('.csv/','\\0')
                #print(frame_name)
                AU_labels[frame_name] = line[2:]
            #print(AU_labels.keys()[0])
            #print(data)
        # save/load the data to/from npy file
        #if os.path.exists(npy_path):
            #full_data = np.load(npy_path, allow_pickle=True)
        if False:
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    path = row[0]
                    emotion = int(row[1]) - 1

                    # modify the path
                    while len(path) < 5:
                        path = "0" + path

                    # combine the path
                    path = os.path.join(
                        self.args.root, "Clip/clip_224x224_16f/", path)
                    full_num_frames = len(os.listdir(path))

                    # get the paths of the frames of a video and sort
                    full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg'))
                    for i in range(0,len(full_video_frames_paths)):
                        if len(full_video_frames_paths[i].split('/')[-1])==5:
                            full_video_frames_paths[i] = full_video_frames_paths[i].replace(full_video_frames_paths[i].split('/')[-1],'0'+full_video_frames_paths[i].split('/')[-1])
                    full_video_frames_paths.sort()

                    full_data.append({"path": full_video_frames_paths,
                                      "emotion": emotion,
                                      "num_frames": full_num_frames,
                                      'AU_labels':AU_labels})
                #print(full_video_frames_paths)

                np.save(npy_path, full_data)
            #print(full_video_frames_paths)

        print("data loaded")
        return full_data

    def get_transform(self):
        """get trasform accorging to train/test mode and args including: crop, flip, color jitter

        Returns:
            transform
        """
        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupRandomSizedCrop(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(
                                                            self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        return transform

    def __getitem__(self, index):
        # get the data according to index
        data = self.data[index]
        full_video_frames_paths = data['path']
        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)

        AU_labels = []

        # when getting the frames, randomly choose the neighbour to augment
        for i in range(self.num_frames):
            frame = int(full_num_frames * i / self.num_frames)

            if self.args.random_sample:
                frame += int(random.random() * self.num_frames)
                frame = min(full_num_frames - 1, frame)
            frame_name =full_video_frames_paths[frame].split('/')[-2]+'/'+ full_video_frames_paths[frame].split('/')[-1][:-4]
            if frame_name[-2]=='0':
                frame_name = frame_name[:-2]+frame_name[-1]
            #print(frame_name)
            #print(frame_name)
            #print(data['AU_labels'][frame_name])
            #print(data['AU_labels'])
            #print(data['AU_labels'].keys())
            '''
            if frame_name not in data['AU_labels']:
                i=i-1
                continue
            '''
            if full_video_frames_paths[frame][-6]=='0':
                video_frames_paths.append(full_video_frames_paths[frame][:-6]+full_video_frames_paths[frame][-5:])
            else:
                video_frames_paths.append(full_video_frames_paths[frame])
            #frame_name.replace(r'\\','/')
            AU_labels.append(data['AU_labels'][frame_name])
        #print(len(video_frames_paths),len(AU_labels))
            #print(full_video_frames_paths[frame].split('/')[-1][:-4])

        AU_avg = []
        for i in range(0,18):
            count = 0
            for j in range(0,len(AU_labels)):
                if AU_labels[j][i]>=0.01:
                    count+=1
            if count>=len(AU_labels)/2:
                AU_avg.append(1)
            else:
                AU_avg.append(0)
        # get the images and transform
        images = []
        for video_frames_path in video_frames_paths:
            images.append(Image.open(video_frames_path).convert('RGB'))
        #print('imgs length',len(images))
        images = self.transform(images)
        images = torch.reshape(
            images, (-1, 3, self.image_size, self.image_size))

        #print('len check:', data["emotion"],data['AU_labels'][frame_name])
        AU_avg = torch.Tensor(AU_avg)
        return images, data["emotion"], AU_avg

    def __len__(self):
        return len(self.data)
