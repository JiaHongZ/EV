""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional

import torch
import torch.utils.data as data
from PIL import Image

from .readers import create_reader

_logger = logging.getLogger(__name__)


_ERROR_RETRY = 50

class ImageDataset_NR(data.Dataset):
    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            flexible_rf = True,
            retina_fixed: bool = True,
            retina_sampling_model: str = 'gaussian',
            retina_size: int = 224,
            retina_field=1,
            in_chans: int = 3,
            em=9,  # eye movement
            size_number=32,
            use_random_em = False,            
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform # post trans for augmentation
        self.target_transform = target_transform
        self._consecutive_errors = 0
        self.retina_size = retina_size
        self.size_number = size_number
        self.split = split
        self.flexible_rf = flexible_rf
        self.in_chans = in_chans
        self.retina_fixed = retina_fixed
        self.retina_sampling_model = retina_sampling_model
        self.retina_field = retina_field
        self.em = em
        self.use_random_em = use_random_em
        self.retina_pre = Retina_STN_2_Pre_2flexibleRF2(
            channel=in_chans,
            r_min=0.01,
            r_max=np.sqrt(2),
            image_H=retina_size,
            image_W=retina_size,
            retinal_H=retina_size,
            retinal_W=retina_size,
            em=em,  # eye movement
            sampling_model=retina_sampling_model,
        )
        if use_random_em:
            if em == 4:
                self.l_t = torch.tensor([[ 0.0000,  0.0000],
                    [0.2116, -0.2116],
                    [0.4431, -0.4431],
                    [-0.2116, 0.2431]])
                # self.l_t = torch.tensor([[ 0.0000,  0.0000],
                #     [-0.2116, -0.2116],
                #     [-0.4431, -0.4431],
                #     [-0.2116, -0.4431]])
            elif em == 9:
                self.l_t = torch.tensor([[0.0000, 0.0000],
                                        [0.4670, 0.4670],
                                        [0.4670, -0.4565],
                                        [-0.4565, -0.4565],
                                        [-0.4565, 0.4670],
                                        [0.4670, 0.7882],
                                        [-0.4565, 0.7882],
                                        [-0.4565, 0.1699],
                                        [0.7882, -0.4565]])
            elif em == 16:
                self.l_t = torch.tensor([[0.0000, 0.0000],
                                        [-0.0638, -0.0869],
                                        [-0.0638, 0.4798],
                                        [-0.0638, -0.2772],
                                        [0.0505, -0.0869],
                                        [0.0505, 0.4798],
                                        [0.0505, -0.2772],
                                        [-0.0869, -0.0638],
                                        [-0.0869, -0.2772],
                                        [0.4798, -0.0638],
                                        [0.4798, -0.0869],
                                        [0.4798, 0.4798],
                                        [0.4798, -0.2772],
                                        [-0.2772, -0.0638],
                                        [-0.2772, -0.0869],
                                        [-0.2772, 0.4798]])
            else:
                point_number = self.nearest_square_root(em)
                if sampling_model == 'uniform':
                    # 生成均匀分布的随机数在 [-1, 1] 区间
                    # # 形成二维采样点 (x, y)
                    random_x = 2 * torch.rand(int(np.sqrt(point_number))) - 1
                    # 生成包括(a, b), (b, a), (a, a) 和 (b, b) 形式的所有组合
                    combinations = list(itertools.product(random_x, repeat=2))
                    # 选择前point_number个元素
                    sample_points = combinations[:point_number]
                    sample_points = torch.tensor(sample_points)
                elif sampling_model == 'gaussian':
                    # 创建高斯分布 0, 0.5
                    random_x = torch.randn(int(np.sqrt(point_number))) / 2
                    random_x[random_x > 1] = 1
                    random_x[random_x < -1] = -1
                    # 生成包括(a, b), (b, a), (a, a) 和 (b, b) 形式的所有组合
                    combinations = list(itertools.product(random_x, repeat=2))
                    # 选择前point_number个元素
                    sample_points = combinations[:point_number]
                    sample_points = torch.tensor(sample_points)
                indices = torch.randperm(sample_points.size(0))[:em - 1]
                center = torch.tensor((0, 0)).unsqueeze(0)
                sample_points = torch.cat([center, sample_points[indices, :]], 0)
                self.l_t = sample_points
    def __getitem__(self, index):
        img, target = self.reader[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
            
        h, w = img.size
        if self.transform is not None:
            img = self.transform(img)
        img, l_t, rf, x_residual = self.retina_pre(img)
        img = img.unsqueeze(0)
        
        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, l_t, rf, x_residual 

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)
    
class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            input_img_mode='RGB',
            transform=None,
            target_transform=None,
            **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map,
                **kwargs,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.input_img_mode = input_img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.input_img_mode and not self.load_bytes:
            img = img.convert(self.input_img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=1,
            num_samples=None,
            seed=42,
            repeats=0,
            download=False,
            input_img_mode='RGB',
            input_key=None,
            target_key=None,
            transform=None,
            target_transform=None,
            max_steps=None,
            **kwargs,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                num_samples=num_samples,
                seed=seed,
                repeats=repeats,
                download=download,
                input_img_mode=input_img_mode,
                input_key=input_key,
                target_key=target_key,
                max_steps=max_steps,
                **kwargs,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
