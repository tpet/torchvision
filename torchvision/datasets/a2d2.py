from __future__ import absolute_import, division, print_function
from glob import glob
import json
import numpy as np
import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

LABEL_IGNORE = 255


def squash_same_class(class_list):
    class_list = class_list.copy()
    for rgb, name in class_list.items():
        # Squash similar class names into one.
        parts = name.split()
        if parts[-1].isdigit():
            name = ' '.join(parts[:-1])
            class_list[rgb] = name
    return class_list


def parse_class_list(class_list):
    class_list = squash_same_class(class_list)
    id_to_name = {i: name for i, name in enumerate(sorted(set(class_list.values())))}
    name_to_id = {name: i for i, name in id_to_name.items()}
    hex_to_rgb = {
        k: tuple(int(k.strip('#')[i: i + 2], 16) for i in (0, 2, 4))
        for k in class_list.keys()
    }
    hex_to_id = {hex_to_rgb[hex]: name_to_id[name] for hex, name in class_list.items()}
    return hex_to_id


def id(path, root):
    """Create ID from path and root.

    Path must be of form '<dir>/<sequence>/<type>/<sensor>/<name>.png'.
    ID is tuple (<sequence>, <sensor>, <name>).

    From ID, we can construct image and label paths:
    '<root>/<sequence>/<camera>/<sensor>/<name>.png',
    '<root>/<sequence>/<label>/<sensor>/<name>.png'.
    """
    path = os.path.relpath(path, root)
    sequence, type, sensor, name = path.split(os.path.sep)
    name = os.path.splitext(name)[0]
    return sequence, sensor, name


def image_path(id, root):
    sequence, sensor, name = id
    name = name.replace('label', 'camera')
    path = os.path.join(root, sequence, 'camera', sensor, name + '.png')
    return path


def label_path(id, root):
    sequence, sensor, name = id
    path = os.path.join(root, sequence, 'label', sensor, name + '.png')
    return path


def label_mono_path(id, root):
    sequence, sensor, name = id
    path = os.path.join(root, sequence, 'label_mono', sensor, name + '.png')
    return path


def segmentation_ids(root):
    labels = sorted(glob(label_path(('*', '*', '*'), root)))
    ids = [id(label, root) for label in labels]
    return ids


def rgb_label_to_mono(label, rgb_to_id):
    mono = np.full(label.shape[:2], LABEL_IGNORE, dtype=np.uint8)
    for rgb, id in rgb_to_id.items():
        mask = (label == np.asarray(rgb).reshape((1, 1, 3))).all(axis=2)
        mono[mask] = id
    if (mono == LABEL_IGNORE).any():
        print('Warning: some labels could not be converted.')
    return mono


class A2D2Segmentation(VisionDataset):

    def __init__(
        self,
        root,
        image_set=None,
        class_list=None,
        transform=None,
        target_transform=None,
        transforms=None,
        # Boundaries for train, val, test (empty by default) image sets.
        split=(0.8, 1.0),
        mono_labels=True,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.ids = segmentation_ids(root)
        if image_set:
            print(f'Found {len(self.ids)} labeled images in {root}.')
            if image_set == 'train':
                self.ids = self.ids[:int(split[0] * len(self.ids))]
            elif image_set == 'trainval':
                self.ids = self.ids[:int(split[1] * len(self.ids))]
            elif image_set == 'val':
                self.ids = self.ids[int(split[0] * len(self.ids)):int(split[1] * len(self.ids))]
            elif image_set == 'test':
                self.ids = self.ids[int(split[1] * len(self.ids)):]
            print(f'Using {len(self.ids)} images for {image_set} set.')

        if not class_list:
            class_list = os.path.join(root, 'class_list.json')
        with open(os.path.join(class_list)) as file:
            self.class_list = json.load(file)
        self.hex_to_id = parse_class_list(self.class_list)
        self.mono_labels = mono_labels

    def convert_labels(self):
        for id in tqdm(self.ids, desc='Converting labels'):
            path = label_path(id, self.root)
            mono_path = label_mono_path(id, self.root)
            os.makedirs(os.path.dirname(mono_path), exist_ok=True)
            label = np.asarray(Image.open(path).convert("RGB"))
            label_mono = rgb_label_to_mono(label, self.hex_to_id)
            Image.fromarray(label_mono).save(mono_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        img = Image.open(image_path(id, self.root)).convert("RGB")

        if self.mono_labels:
            target = Image.open(label_mono_path(id, self.root))
        else:
            # TODO: Use PIL here too. Transforms below work only for PIL images.
            # target = cv.imread(label_path(id, self.root))
            target = np.asarray(Image.open(image_path(id, self.root)).convert("RGB"))
            target = rgb_label_to_mono(target, self.hex_to_id)
            target = Image.fromarray(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def main():
    dataset = A2D2Segmentation('.')
    print(dataset)
    dataset.convert_labels()


if __name__ == '__main__':
    main()
