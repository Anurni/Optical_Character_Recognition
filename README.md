## ﻿ADVANCED MACHINE LEARNING ASSIGNMENT 1: OCR - Anni Nieminen

This repository holds my code for assignment 1 LT2326 (OCR for Thai and English).
Below, find instructions on how to run the scripts and also experiments followed by discussion.

## Script 1: `train_validation_test_split.py`

This script splits the data at the given directory into training, validation, and test sets. If the user wishes to test their model on a different resolution or style than what was used to generate the training data, they can specify different arguments for the testing data. If they wish to use the same arguments for testing data, the same arguments need to be given for the testing data.

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

Trains a model for OCR on MLTGPU (cuda:3). The model uses CNN-architecture. Batch size is set to 32 and learning rate to 0.001. Will print out running loss each epoch.
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

Evaluates the model perfomance with Sklearn metrics (accuracy, precision, recall and F1) on GPU (Cuda:3). Prints out these results.
Takes no arguments. Simply run:

```bash
python test.py
```

## Test runs, evaluation, and discussion

![Results of the model](https://github.com/Anurni/Optical_Character_Recognition/blob/main/updated_result_table.png)

Above, the results of some of the experiments with the model. All of these experiments were done with lr 0.001, number of epochs was 10 and batch size 32.
The results of the experiments above were suprisinly good! Based on these runs, the model reached an average accuracy of 88%.
However, experiments 8 and 9 proved that when the model is trained on characters using normal font and tested on italic font, the 
perfomance decreases significantly.
Our model has two convolutional layers, and also uses max pooling after the conv layer and activation function (ReLU). The images in this
dataset are relatively simple, as they are black and white. Furthermore, I suspect that our model's convolutional layers output relatively many channels for 
this type of data (second Conv2D layer produces 64 channels). 

## Challenges

Learning about how to work with data without having access to it on my local was quite time-consuming, but I have learnt a lot about using the GPU during this assignment. 
Splitting the data proved to be one of the most challenging parts of this assignment. I had many frustrations with separating "bold" from "bold_italic" since trying to match the user argument somehow like this

```bash
if user_style in root
```
simply would not work due to 'bold' matching both 'bold' and 'bold_italic' styles.

In the data directory, there were also these mysterical 'Thumbs' files, that I decided to discard from the training, validation and test sets in the data splitting script:

![Results of the model](https://github.com/Anurni/Optical_Character_Recognition/blob/main/thumbs.png)

Another challenge was deciding how to allow the test dataset settings (user arguments) differ from those of the training set, in order to run the required experiments. Granted that there probably could have been a more creative way of doing this,
I simply decided to add more command line arguments to the splitting script. I compensated this by trying to make the help message as crear as possible. Obviously this solution added quite many new lines to the splitting script.

