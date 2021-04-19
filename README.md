# Siamese Network 

A simple but pragmatic implementation of Siamese Networks in PyTorch using the pre-trained feature extraction networks provided in ```torchvision.models```. 

## Design Choices:
- The siamese network provided in this repository uses a sigmoid at its output, thus making it a binary classification task (positive=same, negative=different) with binary cross entropy loss, as opposed to the triplet loss generally used. 
- I have added dropout to the final classification head network along-with BatchNorm. On online forums there is discussion that dropout with batchnorm is ineffective, however, I found it to improve the results on my specific private dataset. 
- Instead of concatenating the feature vectors of the two images, I opted to multiply them element-wise, which increased the validation accuracy for my specific dataset.


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
