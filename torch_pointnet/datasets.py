import os
import gc
import glob
import shutil
import multiprocessing as mp
from functools import partial
from typing import List, Tuple

import h5py
import trimesh
import torch
import torchvision
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


URLS = {
    "modelnet10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "modelnet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip"
}


def load_pcl(path: str, num_points: int) -> trimesh.Trimesh:
    return trimesh.load(path).sample(num_points)


class ModelNetDataset(torch.utils.data.Dataset):

    path: str = ""
    url: str = ""
    hdf5_path: str = ""
    ext = ".off"

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_points: int = 2048,
        hdf5: bool = True
    ) -> None:

        self.root = root
        self.split = split
        self.num_points = num_points
        self.hdf5 = hdf5
        self.path = os.path.join(self.root, self.path)

        # Download dataset if necessary
        if not os.path.exists(self.path):
            self.download(root=self.root, url=self.url)

        # Get class labels and mappings
        self.classes = next(os.walk(self.path))[1]
        self.idx2class = {i: c for i, c in enumerate(self.classes)}
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        self.files, self.labels = self.get_files()

        if self.hdf5:
            self.load_data()

    def download(self, root: str, url: str) -> None:
        download_and_extract_archive(
            url=url,
            download_root=root,
            remove_finished=True
        )

        if os.path.exists(os.path.join(self.root, "__MACOSX")):
            shutil.rmtree(os.path.join(self.root, "__MACOSX"))

    def get_files(self):
        files, labels = [], []

        for c in self.classes:
            objects = glob.glob(os.path.join(self.path, c, self.split, "*" + self.ext))
            labels.extend([c] * len(objects))
            files.extend(objects)

        return files, labels

    def load_data(self) -> None:
        
        self.hdf5_path = os.path.join(self.path, self.split + ".h5")

        if not os.path.exists(self.hdf5_path):
            print("Converting meshes to hdf5 for faster loading...")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                X = []
                for pcl in tqdm(
                    pool.map(partial(load_pcl, num_points=self.num_points), self.files),
                    total=len(self.files)
                ):
                    X.append(pcl)

            X = np.array(X)
            y = np.array([self.class2idx[l] for l in self.labels])

            self.to_hdf5(X, y)
            del X, y
            gc.collect()

    def to_hdf5(self, X: np.ndarray, y: np.ndarray) -> None:
        hf = h5py.File(self.hdf5_path, "w")
        hf.create_dataset("X", data=X)
        hf.create_dataset("y", data=y)
        hf.close()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.hdf5:
            with h5py.File(self.hdf5_path, "r") as f:
                x, y = f["X"][idx], f["y"][idx]
        else:
            path, label = self.files[idx], self.labels[idx]
            x = np.array(load_pcl(path, self.num_points))
            y = self.class2idx[label]

        x, y = torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
        x = x.to(torch.float).unsqueeze(0)
        return x, y


class ModelNet10(ModelNetDataset):

    path = "ModelNet10"
    url = URLS["modelnet10"]


class ModelNet40(ModelNetDataset):

    path = "ModelNet40"
    url = URLS["modelnet40"]
