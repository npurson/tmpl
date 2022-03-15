import os
import os.path as osp
import shutil
import random


def main():
    data_root = '...'
    split_ratio = 0.7
    for dir in os.listdir(data_root):

        if not osp.exists(osp.join(data_root, 'train')):
            os.mkdir(osp.join(data_root, 'train'))
        if not osp.exists(osp.join(data_root, 'train', dir)):
            os.mkdir(osp.join(data_root, 'train', dir))
        if not osp.exists(osp.join(data_root, 'val')):
            os.mkdir(osp.join(data_root, 'val'))
        if not osp.exists(osp.join(data_root, 'val', dir)):
            os.mkdir(osp.join(data_root, 'val', dir))

        files = os.listdir(osp.join(data_root, dir))
        random.shuffle(files)
        s = int(len(files) * split_ratio)

        for f in files[:s]:
            shutil.move(osp.join(data_root, dir, f),
                        osp.join(data_root, 'train', dir, f))
        for f in files[s:]:
            shutil.move(osp.join(data_root, dir, f),
                        osp.join(data_root, 'val', dir, f))
        print(dir)


if __name__ == '__main__':
    main()
