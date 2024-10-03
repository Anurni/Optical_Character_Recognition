ADVANCED MACHINE LEARNING ASSIGNMENT 1: OCR

This repository holds my code for assignment 1 LT2326 (OCR for Thai and English).
Below are instructions on how to run the scripts.

**1. train_validation_test_split.py**
   
usage: Train and test data splitting program [-h] directory train_language train_resolution train_style test_language test_resolution test_style

Splits the data at the diven directory into train, validation and test data. If user wishes to test their model on different resolution or style than what was used to
generate the training data, it is possible to speficy different arguments for the testing data. If not, simply input the same arguments for the testing data. Example of how
to run (case where we want to train on bold font but test on normal): python train_validation_test_split.py /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet english 400
bold english 400 normal. Note: validation data will use the arguments that you give to the training data.

positional arguments:
  directory         The directory where the data is located.
  train_language    Specifies the language (subdirectory) that will be used to generate the training data. Options: English, Thai or both.
  train_resolution  DPI resolution. 200, 300, 400 or all.
  train_style       Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.
  test_language     Specifies the language (subdirectory) that will be used to generate the testing data. Options: English, Thai or both.
  test_resolution   DPI resolution. 200, 300, 400 or all.
  test_style        Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.

options:
  -h, --help        show this help message and exit

**2. train.py**

**3. test.py**
