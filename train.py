import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from siamese import SiameseNetwork
from libs.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        help="Path to directory containing training dataset.",
        default="../dataset/train"
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        default="../dataset/train"
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        default="output"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=1e-4
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=1000
    )
    parser.add_argument(
        '-s',
        '--save_after',
        type=int,
        help="Model checkpoint is saved after each specified number of epochs.",
        default=25
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset   = Dataset(args.train_path, shuffle_pairs=True, augment=True, testing=False)
    val_dataset     = Dataset(args.val_path, shuffle_pairs=False, augment=False, testing=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=8)

    model = SiameseNetwork()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    for epoch in range(args.epochs):
        model.train()

        losses = []
        correct = 0
        total = 0

        pbar = tqdm(train_dataloader)
        for (img1, img2), y in pbar:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('train_acc', correct / total, epoch)

        print("Training: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        writer.add_scalar('val_loss', sum(losses)/len(losses), epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )