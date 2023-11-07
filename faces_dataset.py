#import libraries ---------------------------------------------

import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from custom_transform import RandomCrop,Rescale,ToTensor,RandomHorizontalFlip

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# show a single image -----------------------------------------
landmarks_frame = pd.read_csv('/faces_dataset/faces/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

# function to shwo an image  ----------------------------------
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('/faces_dataset/faces/faces', img_name)),
               landmarks)
plt.show()

# Costum dataset for the faces dataset -------------------------

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample




rawdata_transform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])

transformdata_transform = transforms.Compose([Rescale(256),RandomCrop(224),RandomHorizontalFlip(),ToTensor()])

# dataloader for the dataset 
raw_dataset = FaceLandmarksDataset(csv_file='faces_dataset/faces/faces/face_landmarks.csv', root_dir='C:/Users/abir/Desktop/faces_dataset/faces/faces/',
                                         transform=rawdata_transform)

                                       
transform_dataset = FaceLandmarksDataset(csv_file='/faces_dataset/faces/faces/face_landmarks.csv', root_dir='C:/Users/abir/Desktop/faces_dataset/faces/faces/',
                                         transform=transformdata_transform)

train_dataset  = torch.utils.data.ConcatDataset([transform_dataset,raw_dataset]) 

for i, sample in enumerate(train_dataset):
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 8:
        break
dataloader = DataLoader(train_dataset, batch_size=8,
                        shuffle=True, num_workers=0)


# function to show a batch ----------------------------------------------
    
def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 4

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
   
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch ==8:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break