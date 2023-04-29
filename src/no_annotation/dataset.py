import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from torchvision.io import read_image

class PCBDataset(Dataset):
    def __init__(self, root, fnames, transform=None, target_transform=None):
        self.root = root
        self.fnames = fnames
        self.transform = transform
        self.target_transform = target_transform
        
        f = open(os.path.join(root, self.fnames), 'r')
        fdata = f.read()
        list1 = fdata.replace(' ', '\n').split('\n')
        del list1[1::2]
        list2 = []
        for i in list1:
            list2.append(i.split('.')[0] + '_temp.jpg')
            list2.append(i.split('.')[0] + '_test.jpg')
        f.close()
        self.fnames_list = list2
    def __len__(self):
        return len(self.fnames_list)
    
    def __getitem__(self, idx):
        img_name = self.fnames_list[idx]
        img_path = os.path.join(self.root, img_name)
        image = read_image(img_path)
        if image.shape[0] != 1:
            image = image[2::]

        if 'temp' in img_name:
            label = 0 # 0 for non-defective
        else:
            label = 1 # 1 for defective
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label