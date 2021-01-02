import torch
import trimesh
import numpy as np
import torchvision.transforms as T


class LoadMesh(object):

    def __call__(self, x: str) -> trimesh.Trimesh:
        return trimesh.load(x)


class PointSampler(object):

    def __init__(self, num_points: int):
        self.num_points = num_points

    def __call__(self, x: trimesh.Trimesh):
        return x.sample(self.num_points)


class MeshToArray(object):

    def __call__(self, x: trimesh.Trimesh) -> np.ndarray:
        return np.array(x)


class PointJitter(object):

    def __init__(self, mu: float = 0.0, sigma: float = 0.02):
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: np.ndarray) -> torch.Tensor:
        noise = np.random.normal(loc=self.mu, scale=self.sigma, size=x.shape)
        return x + noise


class PointRotateZ(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        theta = np.random.random() * 2. * np.pi
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return rotation_matrix.dot(x.T).T


class PointShuffle(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        np.random.shuffle(x)
        return x


class NormalizePoints(object):

    def __call__(self, x: np.ndarray) -> np.ndarray:
        norm = x = np.mean(x, axis=0)
        return x / np.max(np.linalg.norm(norm, axis=1))


class PointsToTensor(object):

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x)


def train_transforms(num_points: int, mu: float, sigma: float) -> T.Compose:
    return T.Compose(
        LoadMesh(),
        PointSampler(num_points),
        MeshToArray(),
        PointJitter(mu, sigma),
        PointRotateZ(),
        PointShuffle(),
        NormalizePoints(),
        PointsToTensor()
    )


def test_transforms(num_points: int) -> T.Compose:
    return T.Compose(
        LoadMesh(),
        PointSampler(num_points),
        MeshToArray(),
        NormalizePoints(),
        PointsToTensor()
    )
