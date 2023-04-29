import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import os
from torchvision.io import read_image
from PIL import Image
from torch.nn.functional import normalize

class PCBDataset(Dataset):
    def __init__(self, root, fnames, transforms=None):
        self.root = root
        self.fnames = fnames
        self.transforms = transforms

        f = open(os.path.join(root, self.fnames), 'r')
        fdata = f.read()
        list1 = fdata.replace(' ', '\n').split('\n')
        self.img_names = []
        self.note_names = []
        for i in range(len(list1)):
            if (i % 2 == 0):
                self.img_names.append(list1[i].split('.')[0] + '_test.jpg')
            else:
                self.note_names.append(list1[i])
        f.close()
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        image_id = int(img_name.split('/')[2].split("_")[0])
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert('L')

        note_name = self.note_names[idx]
        note_path = os.path.join(self.root, note_name)

        f = open(note_path, 'r')
        fdata = list(filter(None, f.read().split('\n')))
        f.close()

        boxes = []
        labels = []
        area = []
        for box in fdata:
            box_ = [int(i) for i in box.split(' ')[:-1]]
            boxes.append(box_)
            area.append(abs(box_[0]-box_[2])*abs(box_[1]-box_[3]))
            labels.append(int(box.split(' ').pop()))

        image_id = torch.as_tensor(image_id, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((len(fdata),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return normalize(image.float()), target