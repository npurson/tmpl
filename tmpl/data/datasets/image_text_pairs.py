import logging
import os
import os.path as osp

import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)


class ImageTextPairsDataset(data.Dataset):
    """
    The root directory should be structured as follows:

    data_root
        ├─train
        │   ├─images
        │   └─labels
        └─test
    """

    def __init__(self, data_root, num_classes, mode='train'):
        assert mode in ('train', 'val', 'test')
        data_root = osp.join(data_root, mode)
        image_root = osp.join(data_root, 'images')
        self.transforms = T.Compose([
            T.Resize((448, 448)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.data = []
        for img_path in os.listdir(image_root):
            label_path = osp.join(data_root, 'labels', img_path.replace('.jpg', '.txt'))
            if osp.exists(label_path):
                with open(label_path) as f:
                    label = [int(line[0]) for line in f.readlines() if line[0] != ' ']
                self.data.append([
                    osp.join(image_root, img_path),
                    sum([1 * (c + 1) if c in label else 0 for c in range(num_classes)])
                ])
        logger.info(f'Loaded {len(self.data)} images in the {mode} split')

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
