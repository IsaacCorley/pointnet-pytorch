import os
import argparse
import multiprocessing as mp
from typing import Dict

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch_pointnet.datasets
from torch_pointnet.losses import PointNetLoss
from torch_pointnet.models import PointNetClassifier
from torch_pointnet.transforms import train_transforms, test_transforms


# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(cfg: DictConfig):

    # Load model
    model = PointNetClassifier(
        num_points=cfg.data.num_points,
        num_classes=cfg.data.num_classes
    )
    model = model.to(cfg.device)

    # Setup optimizer and loss func
    opt = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.train.lr
    )
    loss_func = PointNetLoss(weight_decay=cfg.train.weight_decay)

    # Load datasets
    dataset = getattr(torch_pointnet.datasets, cfg.data.dataset)

    train_dataset = dataset(
        root=cfg.data.root,
        split="train",
        transforms=train_transforms(
            num_points=cfg.data.num_points,
            mu=cfg.data.transforms.jitter_mean,
            sigma=cfg.data.transforms.jitter_std
        )
    )
    test_dataset = dataset(
        root=cfg.data.root,
        split="test",
        transforms=test_transforms(
            num_points=cfg.data.num_points
        )
    )

    # Setup dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )

    writer = SummaryWriter()

    n_iter = 0
    for epoch in range(cfg.train.epochs):

        model.train()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch, (x, y) in pbar:

            opt.zero_grad()
            x, y = x.to(cfg.device), y.to(cfg.device)

            with torch.cuda.amp.autocast():
                x = x.half()    # weird that torch 1.7.0+cu101 doesn't autocast to float
                y_pred, input_matrix, feature_matrix = model(x)
                loss = loss_func(y_pred, y, input_matrix, feature_matrix)

            loss.backward()
            opt.step()

            pbar.set_description("Epoch {}, Loss: {:.4f}".format(epoch, float(loss)))

            if n_iter % cfg.train.log_interval == 0:
                writer.add_scalar(
                    tag="loss", scalar_value=float(loss), global_step=n_iter
                )

            n_iter += 1

        # Evaluate
        model.eval()
        test_loss, num_correct = 0.0, 0
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch, (x, y) in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                x = x.half()
                y_pred, input_matrix, feature_matrix = model(x)
                test_loss += float(loss_func(y_pred, y, input_matrix, feature_matrix))
                pred = y_pred.argmax(dim=1, keepdim=True)
                num_correct += int(pred.eq(y.view_as(pred)).sum().cpu())

        test_loss /= len(test_dataset)
        test_acc = num_correct / len(test_dataset)

        writer.add_scalar(
            tag="test_acc", scalar_value=np.mean(test_acc * 100.0), global_step=n_iter
        )
        writer.add_scalar(
            tag="test_loss", scalar_value=np.mean(test_loss), global_step=n_iter
        )
        writer.add_scalar(
            tag="epoch", scalar_value=epoch, global_step=n_iter
        )


        # save checkpoint
        #torch.save(model.state_dict(), os.path.join(writer.log_dir, "model.pt"))

        # colab
        out_dir = "/content/drive/MyDrive/Github/pointnet-pytorch/models"
        filename = "modelnet10_epoch_{}_loss_{:.4f}_acc_{:.4f}.pt".format(epoch, test_loss, test_acc)
        out_path = os.path.join(out_dir, filename)
        torch.save(model.state_dict(), out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg)
