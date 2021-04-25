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
        required=True
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        required=True
    )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone from torchvision.models to be used in the siamese network.",
        default="resnet18"
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

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset   = Dataset(args.train_path, shuffle_pairs=True, augment=True)
    val_dataset     = Dataset(args.val_path, shuffle_pairs=False, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, drop_last=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=8)

    model = SiameseNetwork(backbone=args.backbone)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCELoss()

    writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    best_val = 10000000000

    for epoch in range(args.epochs):
        print("[{} / {}]".format(epoch, args.epochs))
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), y, (class1, class2) in train_dataloader:
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

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2) in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        val_loss = sum(losses)/max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )            

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )