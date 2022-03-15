import os
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image


class ImgDirsDataset(data.Dataset):
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
        self.mode = mode
        self.CLASSES = tuple(os.listdir(data_root))

        self.transforms = T.Compose([
            T.Resize((448, 448)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.data = []
        total_prev = 0
        for i, c in enumerate(self.CLASSES):
            for img in os.listdir(osp.join(data_root, c)):
                self.data.append((osp.join(data_root, c, img), i))
            print(f'{c}: {len(self.data) - total_prev}')
            total_prev = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = self.transforms(Image.open(img_path))
        label = torch.tensor(label, dtype=torch.long)
        return img, label


if __name__ == '__main__':
    data = ImgDirsDataset('...')
    import pdb; pdb.set_trace()
