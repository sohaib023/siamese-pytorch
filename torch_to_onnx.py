import os
import argparse

import onnx
import torch

from siamese import SiameseNetwork

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
        help="Path of model checkpoint to be used for inference.",
        required=True
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for saving tensorrt model.",
        required=True
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(args.checkpoint)
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    torch.onnx.export(model, (torch.rand(1, 3, 224, 224).to(device), torch.rand(1, 3, 224, 224).to(device)), args.out_path, input_names=['input'],
                      output_names=['output'], export_params=True)
    
    onnx_model = onnx.load(args.out_path)
    onnx.checker.check_model(onnx_model)