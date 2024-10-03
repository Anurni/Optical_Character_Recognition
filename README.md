## ﻿ADVANCED MACHINE LEARNING ASSIGNMENT 1: OCR - Anni Nieminen

This repository holds my code for assignment 1 LT2326 (OCR for Thai and English).
Below are instructions on how to run the scripts.

## Script 1: `train_validation_test_split.py`

This script splits the data at the given directory into training, validation, and test sets. If the user wishes to test their model on a different resolution or style than what was used to generate the training data, they can specify different arguments for the testing data. If not, input the same arguments for the testing data.

```bash
positional arguments:
  directory         The directory where the data is located.
  train_language    Specifies the language (subdirectory) that will be used to generate the training data. Options: English, Thai or both.
  train_resolution  DPI resolution. 200, 300, 400 or all.
  train_style       Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.
  test_language     Specifies the language (subdirectory) that will be used to generate the testing data. Options: English, Thai or both.
  test_resolution   DPI resolution. 200, 300, 400 or all.
  test_style        Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.
```

To train on bold font but test on normal, run the following command:

```bash
python train_validation_test_split.py /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet english 400 bold english 400 normal
```
## Script 2: `train.py`

Trains a model for OCR using CNN-architecture. Batch size is set to 32 and learning rate to 0.001. Will print out running loss each epoch.
Saves the model as a pth-file to **my_OCR_model.pth**.

```bash
positional arguments:
  epochs         The number of epochs in the training loop of the model.

```
Simply run by specifying the n of epochs:

```bash
python train.py 10
```
## Script 3: `test.py`

Evaluates the model perfomance with Sklearn metrics (accuracy, precision, recall and F1). Prints out these results.
Takes no arguments. Simply run:

```bash
python test.py
```

## Test runs, evaluation, and discussion

![Results of the model](https://github.com/Anurni/Optical_Character_Recognition/blob/main/OCR_results.png)


