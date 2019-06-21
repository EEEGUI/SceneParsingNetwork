import numpy as np
from torchvision.transforms import functional as F


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, depth = sample['image'], sample['label'], sample['depth']

        h, w = image.size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = F.resize(image, (new_h, new_w))
        label = F.resize(label, (new_h, new_w))
        depth = F.resize(depth, (new_h, new_w))

        return {'image': image, 'label': label, 'depth': depth}


class RandomHorizonFlip(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, sample):
        image, label, depth = sample['image'], sample['label'], sample['depth']

        p = np.random.random()
        if p < self.probability:

           image = F.hflip(image)
           label = F.hflip(label)
           depth = F.hflip(depth)

        return {'image': image, 'label': label, 'depth': depth}


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, label, depth = sample['image'], sample['label'], sample['depth']

        angle = np.random.randint(self.angle)

        image = F.rotate(image, angle)
        label = F.rotate(label, angle)
        depth = F.rotate(depth, angle)

        return {'image': image, 'label': label, 'depth': depth}


class RandomCrop(object):
    def __init__(self, input_size):
        self.img_size = input_size

    def __call__(self, sample):
        image, label, depth = sample['image'], sample['label'], sample['depth']
        h = np.random.randint(self.img_size[0], 1024)
        w = np.random.randint(self.img_size[1], 2048)
        i = np.random.randint(0, 1024 - h)
        j = np.random.randint(0, 2048 - w)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        depth = F.crop(depth, i, j, h, w)

        return {'image': image, 'label': label, 'depth': depth}