import os
import gc
import glob
import shutil
import multiprocessing as mp
from functools import partial
from typing import Tuple

import h5py
import trimesh
import numpy as np
from tqdm import tqdm

import h5py
import trimesh
import torch
<<<<<<< HEAD
=======
import torchvision
import numpy as np
>>>>>>> 7defa3160be6b0fe5f1cdf3a7af15f1c5a19e155
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


URLS = {
    "ModelNet10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "ModelNet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip"
}


def load_mesh(path: str, num_points: int) -> trimesh.Trimesh:
    return trimesh.load(path).sample(num_points)


class ModelNetDataset(torch.utils.data.Dataset):

    path: str = ""
    url: str = ""
    hdf5_path: str = ""
    ext = ".off"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        num_points: int = 2048,
        in_memory: bool = False,
        multiprocessing: bool = True
    ) -> None:

        self.root = root
        self.split = split
        self.num_points = num_points
        self.in_memory = in_memory
        self.path = os.path.join(self.root, self.path)
        self.hdf5_path = os.path.join(
            self.path, f"data_{self.split}_{self.num_points}.h5"
        )

        os.makedirs(self.root, exist_ok=True)

        # Download dataset if necessary
        if not os.path.exists(self.path):
            self.download(root=self.root, url=self.url)

        # Get class labels and mappings
        self.classes = next(os.walk(self.path))[1]
        self.idx2class = dict(enumerate(self.classes))

        if self.in_memory:
            print("Loading dataset into memory...")
            self.X, self.y = self.load(root=self.path, multiprocessing=multiprocessing)

        else:
            print("Loading dataset from hdf5")
            if not os.path.exists(self.hdf5_path):
                print("Loading dataset...")
                X, y = self.load(root=self.path, multiprocessing=multiprocessing)

                print(f"Saving datasets to hdf5 {self.hdf5_path}")
                self.to_hdf5(self.hdf5_path, X, y)

    def download(self, root: str, url: str) -> None:
        download_and_extract_archive(
            url=url,
            download_root=root,
            remove_finished=True
        )

    def load(self, root: str, multiprocessing: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if multiprocessing:
            return self.load_mp(root)
        else:
            return self.load_mp(root)

    def load_sp(self, root: str) -> Tuple[np.ndarray, np.ndarray]:

        points, labels = [], []
        for i, c in enumerate(self.classes):
            print(f"Processing Class: {c}")
            path = os.path.join(root, c, self.split)
            print(path)
            files = glob.glob(os.path.join(path, "*.off"))

            for f in tqdm(files, total=len(files)):
                points.append(trimesh.load(f).sample(self.num_points))
                labels.append(i)

        return np.array(points), np.array(labels)

    def load_mp(self, root: str) -> Tuple[np.ndarray, np.ndarray]:

        load_func = partial(load_mesh, num_points=self.num_points)

        points, labels = [], []
        for i, c in enumerate(self.classes):
            print(f"Processing Class: {c}")
            path = os.path.join(root, c, self.split)
            print(path)
            files = glob.glob(os.path.join(path, "*.off"))

            with mp.Pool(processes=mp.cpu_count()) as pool:
            for mesh in tqdm(pool.map(load_func, files), total=len(files)):
                points.append(mesh)
                labels.append(i)

        return np.array(points), np.array(labels)

    def to_hdf5(self, path: str, X: np.ndarray, y: np.ndarray) -> None:
        with h5py.File(path, "w") as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("y", data=y)

    def __len__(self) -> int:
        if self.in_memory:
            length = len(self.X)
        else:
            with h5py.File(self.hdf5_path, "r") as hf:
                length = len(hf["X"])

        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.in_memory:
            x, y = self.X[idx, ...], self.y[idx, ...]
        else:
            with h5py.File(self.hdf5_path, "r") as hf:
                x, y = hf["X"][idx, ...], hf["y"][idx, ...]

        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y


class ModelNet10(ModelNetDataset):

    path = "ModelNet10"
    url = URLS[path]


class ModelNet40(ModelNetDataset):

    path = "ModelNet40"
    url = URLS[path]
