import json
import os
import pathlib
import random

import hydra
import numpy as np
import omegaconf
import torch



class TransformCompose(object):

    def __init__(self, transforms: list[omegaconf.ListConfig]):
        self.transforms = [hydra.utils.instantiate(transform) for transform in transforms]

    def __call__(self, points: np.ndarray):
        for transform in self.transforms:
            points = transform(points)
        return points


class CacheDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        versioning_path: str,
        fold_mapping: dict[str, str],
        limit: int | None,
        num_points: int,
        normals: bool,
        mode: str,
        transforms: omegaconf.ListConfig | None,
        cache_folder_path: str = None,
    ):
        self.versioning_path = pathlib.Path(__file__).parent.parent / versioning_path
        self.cache_folder_path = pathlib.Path(cache_folder_path) if cache_folder_path else None
        self.limit = limit
        self.num_points = num_points
        self.normals = normals

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f'Invalid dataset mode {mode}. Must be train, val or test.')
        self.mode = mode

        self.transforms = None
        if transforms:
            self.transforms = TransformCompose(transforms)

        self.dataset_fold = fold_mapping[mode]
        self.cached_hashes, self.cache_paths = self._load_cached_pointclouds()

    def _load_cached_pointclouds(self):
        """Find sample paths from chosen mode in versioning json.
        
        Filter out samples that do not have a cache path.
        """
        # print(f'Checking cache paths for {self.mode} split.')
        with open(self.versioning_path, 'r', encoding='utf-8') as f:
            versioning = json.load(f)

        all_samples_in_fold = [(hexdigest, sample)
                               for hexdigest, sample in versioning['samples'].items()
                               if sample['split'] in self.dataset_fold]
        if self.limit is not None:
            random.shuffle(all_samples_in_fold)
            all_samples_in_fold = all_samples_in_fold[:self.limit]

        cache_paths = []
        cached_hashes = []
        for hexdigest, sample in all_samples_in_fold:
            if 'cache_path' not in sample:
                continue
            if self.cache_folder_path:
                cache_path = self.cache_folder_path / sample['cache_path']
            else:
                cache_path = pathlib.Path(sample['cache_path'])
            if cache_path.exists():
                cache_paths.append(cache_path)
                cached_hashes.append(hexdigest)

        print(f'Found {len(cache_paths)} out of {len(all_samples_in_fold)} for fold {self.dataset_fold}')
        return cached_hashes, cache_paths

    def __len__(self) -> int:
        return len(self.cache_paths)

    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError('This method should be implemented in a subclass.')


class ClassificationDataset(CacheDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_ids = self._load_label_ids()

    def _load_label_ids(self) -> list[int]:
        """Load label ids from versioning json."""

        with open(self.versioning_path, 'r', encoding='utf-8') as f:
            versioning = json.load(f)

        label_ids = [
            torch.tensor(versioning['samples'][hexdigest]['label_id']).long()
            for hexdigest in self.cached_hashes
        ]
        return label_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        cache_path = self.cache_paths[index]
        label = self.label_ids[index]

        data = np.load(cache_path)
        if data.shape[0] > self.num_points:
            points_idx = np.random.choice(data.shape[0], self.num_points, replace=False)
        else:
            points_idx = np.random.choice(data.shape[0], self.num_points, replace=True)
        points = data[points_idx]
        if self.transforms and self.mode == 'train':
            points = self.transforms(points)
        if self.normals:
            return torch.from_numpy(points[:, :3]), torch.from_numpy(points[:, 3:6]), label
        else:
            return torch.from_numpy(points[:, :3]), label


