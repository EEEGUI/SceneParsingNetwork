import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image

from torch.utils import data
from utils import recursive_glob
from torchvision.transforms import Compose
from torchvision.transforms import functional as F
from augmentations import RandomHorizonFlip, RandomRotate, RandomCrop


class cityscapesLoader(data.Dataset):

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        max_depth=250,
        augmentations=None,
        img_norm=True,
        mean=[0, 0, 0],
        test_mode=False,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size
        self.max_depth = max_depth
        self.mean = np.array(mean)
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        self.disparity_base = os.path.join(self.root, "disparity", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        depth_path = os.path.join(
            self.disparity_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "disparity.png",
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        depth = Image.open(depth_path)

        sample = {'image': img, 'label': lbl, 'depth': depth}

        if self.augmentations is not None:
            sample = self.augmentations(sample)

        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def transform(self, sample):
        """transform

        :param img:
        :param lbl:
        """
        img, lbl, depth = sample['image'], sample['label'], sample['depth']

        # image
        img = F.resize(img, (self.img_size[0], self.img_size[1]))
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()


        # label
        lbl = F.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=Image.NEAREST)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        classes = np.unique(lbl)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        lbl = torch.from_numpy(lbl).long()


        # depth
        depth = F.resize(depth, (self.img_size[0], self.img_size[1]))
        depth = self.decode_depthmap(np.array(depth, dtype=np.float32))
        depth = torch.from_numpy(depth).float()
        depth = torch.unsqueeze(depth, 0)

        return {'image': img, 'label': lbl, 'depth': depth}

    def img_recover(self, tensor_img):
        img = tensor_img.cpu().numpy()
        # CHW -> HWC
        img = img.transpose(1, 2, 0)
        if self.img_norm:
            img = (img * 255.0)
        img += self.mean
        img = img[:, :, ::-1]
        img = np.array(img, dtype=np.uint8)
        return img

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_depthmap(self, disparity):
        disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256

        disparity[disparity > 0] = (0.209313 * 2262.52) / disparity[disparity > 0]

        disparity[disparity <= 0] = 0
        disparity[disparity > self.max_depth] = 0

        depth = disparity

        return depth

    def encode_depthmap(self, depth):
        depth = (depth * 256 + 1).astype('int16')
        return depth


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    augmentations = Compose([RandomRotate(10), RandomCrop(), RandomHorizonFlip(0.5)])
    # augmentations = None

    local_path = "/home/lin/Documents/dataset/Cityscapes/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels, depth = data_samples['image'], data_samples['label'], data_samples['depth']
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        # depthhh = depth.view(-1).numpy()
        # sns.distplot(depthhh)
        f, axarr = plt.subplots(bs, 3)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            # axarr[j][0].imshow(dst.img_recover(imgs[j]))
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            axarr[j][2].imshow(dst.encode_depthmap(depth.numpy()[j][0]), cmap='gray')
        plt.show()

