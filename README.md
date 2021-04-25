# Siamese Network 

A simple but pragmatic implementation of Siamese Networks in PyTorch using the pre-trained feature extraction networks provided in ```torchvision.models```. 

## Design Choices:
- The siamese network provided in this repository uses a sigmoid at its output, thus making it a binary classification task (positive=same, negative=different) with binary cross entropy loss, as opposed to the triplet loss generally used. 
- I have added dropout to the final classification head network along-with BatchNorm. On online forums there is discussion that dropout with batchnorm is ineffective, however, I found it to improve the results on my specific private dataset. 
- Instead of concatenating the feature vectors of the two images, I opted to multiply them element-wise, which increased the validation accuracy for my specific dataset.

## Setting up environment.
The provided setup instructions assume that anaconda is already installed on the system. To set up the environment for this repository, run the following commands to create and activate an environment named 'pytorch_siamese'. (The command takes a while to run, so please keep patience):
```
conda env create -f environment.yml
conda activate pytorch_siamese
```
The environment contains the required packages for TensorRT as well, hence the arduous task of installing it in global cuda is not required.

## Setting up the dataset.
The expected format for both the training and validation dataset is the same. Image belonging to a single entity/class should be placed in a folder with the name of the class. The folders for every class are then to be placed within a common root directory (which will be passed to the trainined and evaluation scripts). The folder structure is also explained below:
```
|--Train or Validation dataset root directory
  |--Class1
    |-Image1
    |-Image2
    .
    .
    .
    |-ImageN
  |--Class2
  |--Class3
  .
  .
  .
  |--ClassN
```

## Training the model:
To train the model, run the following command along with the required command line arguments:
```
python train.py [-h] --train_path TRAIN_PATH --val_path VAL_PATH -o OUT_PATH
                [-b BACKBONE] [-lr LEARNING_RATE] [-e EPOCHS] [-s SAVE_AFTER]

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        Path to directory containing training dataset.
  --val_path VAL_PATH   Path to directory containing validation dataset.
  -o OUT_PATH, --out_path OUT_PATH
                        Path for outputting model weights and tensorboard
                        summary.
  -b BACKBONE, --backbone BACKBONE
                        Network backbone from torchvision.models to be used in
                        the siamese network.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning Rate
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train
  -s SAVE_AFTER, --save_after SAVE_AFTER
                        Model checkpoint is saved after each specified number
                        of epochs.
```
The backbone can be chosen from any of the networks listed in [torchvision.models](https://pytorch.org/vision/stable/models.html)

## Evaluating the model:
Following command can be used to evaluate the model on a validation set. Output images with containing the pair and their corresponding similarity confidence will be outputted to `{OUT_PATH}`.

Note: During evaluation the pairs are generated with a deterministic seed for the numpy random module, so as to allow comparisons between multiple evaluations.

```
python eval.py [-h] -v VAL_PATH -o OUT_PATH -c CHECKPOINT

optional arguments:
  -h, --help            show this help message and exit
  -v VAL_PATH, --val_path VAL_PATH
                        Path to directory containing validation dataset.
  -o OUT_PATH, --out_path OUT_PATH
                        Path for saving prediction images.
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path of model checkpoint to be used for inference.
```

## Converting from Torch to ONNX
To convert the torch model (.pth extension) outputted by `train.py` into ONNX format, kindly use the file `torch_to_onnx.py`:
```
python torch_to_onnx.py [-h] -c CHECKPOINT -o OUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path of model checkpoint to be used for inference.
  -o OUT_PATH, --out_path OUT_PATH
                        Path for saving tensorrt model.
```

## Converting from ONNX to TensorRT Engine
To generate a TensorRT engine file from the ONNX model outputted by `torch_to_onnx.py`, kindly use the file `onnx_to_trt.py`:
```
python onnx_to_trt.py [-h] --onnx ONNX --engine ENGINE

optional arguments:
  -h, --help       show this help message and exit
  --onnx ONNX      Path of onnx model generated by 'torch_to_onnx.py'.
  --engine ENGINE  Path for saving tensorrt engine.
```

## Inference and Evaluation using TensorRT Engine:
To infer and evaluate the TensorRT engine outputted by `onnx_to_trt.py`, kindly use the files `infer_tensorrt.py` and `eval_tensorrt.py`. Usages for both of these files are provided below:
```
python eval_tensorrt.py [-h] -v VAL_PATH -o OUT_PATH --engine ENGINE

optional arguments:
  -h, --help            show this help message and exit
  -v VAL_PATH, --val_path VAL_PATH
                        Path to directory containing validation dataset.
  -o OUT_PATH, --out_path OUT_PATH
                        Path for saving prediction images.
  --engine ENGINE       Path to tensorrt engine generated by 'onnx_to_trt.py'.
```

```
python infer_tensorrt.py [-h] --image1 IMAGE1 --image2 IMAGE2 --engine ENGINE

optional arguments:
  -h, --help       show this help message and exit
  --image1 IMAGE1  Path to first image of the pair.
  --image2 IMAGE2  Path to second image of the pair.
  --engine ENGINE  Path to tensorrt engine generated by 'onnx_to_trt.py'.
```
