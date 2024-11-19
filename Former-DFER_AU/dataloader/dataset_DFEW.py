import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from dataloader.video_transform import *
import numpy as np


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size):
        self.list_file = list_file
        self.AU_file = list_file.replace('.txt','_AU.txt')
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1]) >= 16]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        #print(('video number:%d' % (len(self.video_list))))
        AU_list = [x.strip() for x in open(self.AU_file)]
        #print(AU_list)
        self.AU_dict = {}
        for line in AU_list:
            #print(line)
            labels = line[-35:].split(' ')
            #print(labels)
            l = []
            for x in labels:
                l.append(int(x))
            #print(l)
            self.AU_dict[line[:-36]] = l


    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_name = record.path.split('/')[-1]

        video_frames_path = []
        for i in range(1,17):
            video_frames_path.append(os.path.join(record.path, str(i)+'.jpg'))
        video_frames_path = glob.glob(os.path.join(record.path, '*.jpg'))

        images = list()
        AU_labels = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for frame in video_frames_path:
            AU_key = frame.split('/')[-2]+' '+frame.split('/')[-1][:-4]
            seg_imgs = [Image.open(frame).convert('RGB')]
            images.extend(seg_imgs)
            for i in range(0,18):
                AU_labels[i]+=self.AU_dict[AU_key][i]/16
        for i in range(0,18):
            if AU_labels[i]>=0.5:
                AU_labels[i] = 1.0
            else:
                AU_labels[i] = 0.0
        #print(AU_labels)
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))
        #print(len(images))
        #print(len(AU_labels))
        AU_labels = torch.Tensor(AU_labels)
        return images, record.label, AU_labels

    def __len__(self):
        return len(self.video_list)


def train_data_loader(data_set):
    image_size = 112
    train_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])
    train_data = VideoDataset(list_file="./annotation/DFEW_set_"+str(data_set)+"_train.txt",
                              num_segments=8,
                              duration=2,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(data_set):
    image_size = 112
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    test_data = VideoDataset(list_file="./annotation/DFEW_set_"+str(data_set)+"_test.txt",
                             num_segments=8,
                             duration=2,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data



