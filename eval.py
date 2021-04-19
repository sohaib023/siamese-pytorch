import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from siamese import SiameseNetwork
from libs.dataset import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for saving prediction images.",
        required=True
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset     = Dataset(args.val_path, shuffle_pairs=False, augment=False, testing=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=1)

    criterion = torch.nn.BCELoss()

    checkpoint = torch.load(args.checkpoint)
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0

    inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])
    
    for i, ((img1, img2), y, (class1, class2)) in enumerate(val_dataloader):
        print("[{} / {}]".format(i, len(val_dataloader)))

        img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
        class1 = class1[0]
        class2 = class2[0]

        prob = model(img1, img2)
        loss = criterion(prob, y)

        losses.append(loss.item())
        correct += torch.count_nonzero(y == (prob > 0.5)).item()
        total += len(y)

        fig = plt.figure("class1={}\tclass2={}".format(class1, class2), figsize=(4, 2))
        plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, prob[0][0].item(), class2))

        img1 = inv_transform(img1).cpu().numpy()[0]
        img2 = inv_transform(img2).cpu().numpy()[0]
        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(img1[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(img2[0], cmap=plt.cm.gray)
        plt.axis("off")

        # show the plot
        plt.savefig(os.path.join(args.out_path, '{}.png').format(i))

    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))