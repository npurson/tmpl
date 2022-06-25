import os
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class ImgTxtPairsDataset(data.Dataset):
    """
    The root directory should be structured as follows:

    data_root
        ├─train
        │   ├─images
        │   └─labels
        └─test
    """
    def __init__(self, data_root, mode='train'):
        assert mode in ('train', 'val', 'test')
        data_root = osp.join(data_root, mode)
        image_root = osp.join(data_root, 'images')
        self.mode = mode
        num_classes = 2
        cls_cnt = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

        self.transforms = T.Compose([
            T.Resize((448, 448)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.data = []
        for img_path in os.listdir(image_root):
            lbl_path = osp.join(data_root, 'labels',
                                img_path.replace('.jpg', '.txt'))
            if osp.exists(lbl_path):
                with open(lbl_path) as f:
                    label = [int(line[0]) for line in f.readlines()
                             if line[0] != ' ']
                self.data.append([
                    osp.join(image_root, img_path),
                    # [1 if c in label else 0 for c in range(num_classes)]
                    sum([1 * (c + 1) if c in label else 0
                         for c in range(num_classes)])
                ])
                for c in range(num_classes):
                    cls_cnt[c][1 if c in label else 0] += 1
        print(f'[data] Loaded {len(self.data)} images in {mode} split.')
        print(cls_cnt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transforms(img)
        label = torch.tensor(label, dtype=torch.long)
        # label = [torch.tensor(lbl, dtype=torch.long) for lbl in label]
        return img, label


if __name__ == '__main__':
    ...
