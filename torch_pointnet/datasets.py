import os
import glob
import shutil
from typing import List

import torch
import torchvision
from torchvision.datasets.utils import download_and_extract_archive


URLS = {
    "modelnet10": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    "modelnet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip"
}


class ModelNetDataset(torch.utils.data.Dataset):

    path: str = ""
    url: str = ""

    def __init__(
        self, root: str,
        split: str = "train",
        in_memory: bool = False
    ) -> None:

        self.root = root
        self.in_memory = in_memory
        self.path = os.path.join(self.root, self.path)

        if not os.path.exists(self.path):
            self.download(root=self.root, url=self.url)

            if os.path.exists(os.path.join(self.root, "__MACOSX")):
                shutil.rmtree(os.path.join(self.root, "__MACOSX"))

        self.classes = next(os.walk(self.path))[1]

    def download(self, root: str, url: str) -> None:
        download_and_extract_archive(
            url=url,
            download_root=root,
            remove_finished=True
        )


class ModelNet10(ModelNetDataset):

    path = "ModelNet10"
    url = URLS["modelnet10"]


class ModelNet40(ModelNetDataset):

    path = "ModelNet40"
    url = URLS["modelnet40"]
