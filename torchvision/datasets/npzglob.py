from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from glob import glob
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import os
from torchvision.datasets.vision import VisionDataset
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

rng = np.random.default_rng(135)


class GlobDataset(VisionDataset):

    def __init__(
        self,
        root='',
        image_set=None,
        pattern=None,
        transform=None,
        target_transform=None,
        transforms=None,
        # Boundaries for train, val, test image sets.
        randomize=False,
        split=(0.8, 0.9),
        num_outputs=None
    ):
        super().__init__(root, transforms, transform, target_transform)
        if pattern is None:
            # pattern = os.path.join(self.root, '**', '*.npz')
            pattern = os.path.join('**', '*.npz')
        pattern = os.path.join(self.root, pattern)
        self.paths = glob(pattern, recursive=True)
        print(f'Found {len(self.paths)} files in {os.path.abspath(pattern)}.')
        self.paths.sort()
        if randomize:
            rng.shuffle(self.paths)
        if image_set:
            assert image_set in ('train', 'trainval', 'val', 'test')
            if image_set == 'train':
                self.paths = self.paths[:int(split[0] * len(self.paths))]
            elif image_set == 'trainval':
                self.paths = self.paths[:int(split[1] * len(self.paths))]
            elif image_set == 'val':
                self.paths = self.paths[int(split[0] * len(self.paths)):int(split[1] * len(self.paths))]
            elif image_set == 'test':
                self.paths = self.paths[int(split[1] * len(self.paths)):]
            print(f'Using {len(self.paths)} images for {image_set} set.')
        self.num_outputs = num_outputs

    def __len__(self):
        return len(self.paths)

    def load(self, path):
        raise NotImplementedError()

    def stats(self, n=1.0, progress=False):
        """Return the number of classes in the dataset."""
        if isinstance(n, float):
            assert 0.0 < n <= 1.0
            n = int(n * len(self))
        unique_targets = set()
        # Number of examples and pixels for each class.
        # For regression, we distinguish valid and invalid values.
        if self.task == 'regression':
            target_examples = {0.0: 0, float('nan'): 0}
            target_pixels = {0.0: 0, float('nan'): 0}
        else:
            target_examples = {}
            target_pixels = {}
        gen = enumerate(self)
        if progress:
            gen = tqdm(gen, total=n)
        for i, (img, target) in gen:
            if i >= n:
                break

            if self.task == 'regression':
                valid = np.isfinite(target.flatten())
                if valid.any():
                    target_examples[0.0] += 1
                else:
                    target_examples[float('nan')] += 1
                target_pixels[0.0] += valid.sum()
                target_pixels[float('nan')] += (~valid).sum()
                continue

            unique, counts = np.unique(target, return_counts=True)
            unique_targets.update(unique)
            for l, c in zip(unique, counts):
                if l not in target_examples:
                    target_examples[l] = 0
                    target_pixels[l] = 0
                target_examples[l] += 1
                target_pixels[l] += c
        return unique_targets, target_examples, target_pixels

    def __getitem__(self, i):
        img, target = self.load(self.paths[i])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # TODO: Move to transforms? Transforms are separate for img and target.
        # if target.ndim == 3:
        #     img = img.transpose(2, 0, 1)
        #     target = target.transpose(2, 0, 1)

        return img, target


class NpzGlobDataset(GlobDataset):

    def __init__(
        self,
        *args,
        input_array=None,
        input_fields=None,
        target_array=None,
        target_fields=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_array = input_array
        self.input_fields = input_fields
        self.target_array = target_array
        self.target_fields = target_fields

    def load(self, path):
        arr = np.load(path)

        # Use default input and target array names if not specified.
        if self.input_array in arr:
            img = arr[self.input_array]
        elif 'input' in arr:
            img = arr['input']
        else:
            img = list(arr.values())[0]

        if self.target_array in arr:
            target = arr[self.target_array]
        elif 'label' in arr:
            target = arr['label']
        elif 'target' in arr:
            target = arr['target']
        else:
            target = list(arr.values())[0]

        # Use input and target fields only when specified.
        # Extracting fields may be postponed to transforms.
        if self.input_fields:
            img = structured_to_unstructured(img[self.input_fields])
        if self.target_fields:
            target = structured_to_unstructured(target[self.target_fields])

        return img, target


def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--image-set', type=str, default=None)
    parser.add_argument('--pattern', type=str, default=None)
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--split', type=float, nargs=2, default=(0.8, 0.9))
    parser.add_argument('--num-outputs', type=int, default=None)
    parser.add_argument('--input-array', type=str, default=None)
    parser.add_argument('--input-fields', type=str, default=None)
    parser.add_argument('--target-array', type=str, default=None)
    parser.add_argument('--target-fields', type=str, default=None)
    return parser


def main():
    args = get_args_parser().parse_args()
    kwargs = vars(args).copy()
    del kwargs['root']
    ds = NpzGlobDataset(args.root, **kwargs)
    print(ds)
    print(ds[0])
    unique_targets, target_examples, target_pixels = ds.stats(progress=True)
    num_pixels = sum(target_pixels.values())
    for l in sorted(target_examples):
        print(f'Label {l} occurs in '
              f'{target_examples[l]} / {len(ds)} = {target_examples[l] / len(ds):.3g} examples and '
              f'{target_pixels[l]} / {num_pixels} = {target_pixels[l] / num_pixels:.3g} pixels.')


if __name__ == '__main__':
    main()
