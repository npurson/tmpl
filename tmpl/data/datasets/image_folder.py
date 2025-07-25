import logging
import os
import os.path as osp

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class ImageFolderDataset(data.Dataset):
    """
    The root directory should be structured as follows:

    data_root
        ├─train
        │   ├─CLASS_1
        │   ├─CLASS_2
        │   └─...
        ├─val
        └─test
    """

    def __init__(self, data_root, mode='train'):
        assert mode in ('train', 'val', 'test')
        data_root = osp.join(data_root, mode)
        self.classes = tuple(os.listdir(data_root))
        self.transforms = T.Compose([
            T.Resize((448, 448)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.data = []
        for i, cls in enumerate(self.classes):
            cls_dir = osp.join(data_root, cls)
            for img in os.listdir(cls_dir):
                self.data.append((osp.join(cls_dir, img), i))
            logger.info(f'{cls}: {len(os.listdir(cls_dir))}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.long)
        return img, label
