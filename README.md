ADVANCED MACHINE LEARNING ASSIGNMENT 1: OCR

This repository holds my code for assignment 1 LT2326 (OCR for Thai and English).
Below are instructions on how to run the scripts.

## Script 1: `train_validation_test_split.py`

This script splits the data at the given directory into training, validation, and test sets. If the user wishes to test their model on a different resolution or style than what was used to generate the training data, they can specify different arguments for the testing data. If not, input the same arguments for the testing data.

### Example:

To train on bold font but test on normal, run the following command:

```bash
python train_validation_test_split.py /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet english 400 bold english 400 normal
