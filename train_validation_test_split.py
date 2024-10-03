import argparse
from sklearn.model_selection import train_test_split 
import os

# COMMAND LINE ARGUMENTS AND PARSING
parser = argparse.ArgumentParser(
                    prog='Train and test data splitting program',
                    description='Splits the data at the diven directory into train, validation and test data. If user wishes to test their model on different resolution or style than what was used to generate the training data, it is possible to speficy different arguments for the testing data. If not, simply input the same arguments for the testing data. Example of how to run (case where we want to train on bold font but test on normal): python train_validation_test_split.py /scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet english 400 bold english 400 normal. Note: validation data will use the arguments that you give to the training data.')

# training data args
parser.add_argument('directory', type=str, help="The directory where the data is located.")
parser.add_argument('train_language', type=str, help="Specifies the language (subdirectory) that will be used to generate the training data. Options: English, Thai or both.")
parser.add_argument('train_resolution',type=str, help="DPI resolution. 200, 300, 400 or all.")
parser.add_argument('train_style', type=str, help="Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.")

# testing data args
parser.add_argument('test_language', type=str, help="Specifies the language (subdirectory) that will be used to generate the testing data. Options: English, Thai or both.")
parser.add_argument('test_resolution',type=str, help="DPI resolution. 200, 300, 400 or all.")
parser.add_argument('test_style', type=str, help="Specifies the style of the font. Options: bold_italic, bold, italic, normal or all.")

arguments = parser.parse_args()

directory = arguments.directory

train_language = arguments.train_language
train_style = arguments.train_style
train_resolution = arguments.train_resolution

test_language = arguments.test_language
test_style = arguments.test_style
test_resolution = arguments.test_resolution


# ********************************************************************************************************************************
    # TRAINING DATA GENERATION
    # RETRIEVING THE DATA FOR TRAINING ACCORDING TO USER'S CMD LINE ARGUMENTS
    # WE NEED TO CONSIDER ALSO THE BOTH-CASE, IF THE USER CHOOSES BOTH ENGLISH AND THAI, AS WELL ASS ALL-CASES WITH STYLE AND DPI

train_wanted_files = []
# language option both
if train_language == "both":
    language_paths = [os.path.join(directory, "Thai"), os.path.join(directory, "English")]
    for language_path in language_paths:
        for root, dirs, files in os.walk(language_path):  # os.walk() yields a tuple!
            for file in files:
                resolution_match = train_resolution in root or train_resolution == "all"
                style_match = os.path.basename(root) == train_style or train_style == "all" #os.path.basename(root) == font style
                
                if resolution_match and style_match:
                    train_wanted_files.append(os.path.join(root, file))

# single language option
else:
    language_path = os.path.join(directory, train_language.capitalize())
    for root, dirs, files in os.walk(language_path):  # os.walk() yields a tuple!
        for file in files:
            resolution_match = train_resolution in root or train_resolution == "all"
            style_match = os.path.basename(root) == train_style or train_style == "all" #os.path.basename(root) == font style
            
            if resolution_match and style_match:
                train_wanted_files.append(os.path.join(root, file))
            
# ********************************************************************************************************************************
    # TESTING DATA GENERATION
    # RETRIEVING THE DATA FOR TRAINING ACCORDING TO USER'S CMD LINE ARGUMENTS
    # WE NEED TO CONSIDER ALSO THE BOTH-CASE, IF THE USER CHOOSES BOTH ENGLISH AND THAI, AS WELL AS ALL-CASES WITH STYLE AND DPI     

test_wanted_files = []
# language option both
if test_language == "both":
    language_paths = [os.path.join(directory, "Thai"), os.path.join(directory, "English")]
    for language_path in language_paths:
        for root, dirs, files in os.walk(language_path):  # os.walk() yields a tuple!
            for file in files:
                resolution_match = test_resolution in root or test_resolution == "all"
                style_match = os.path.basename(root) == test_style or test_style == "all" #os.path.basename(root) == font style
                
                if resolution_match and style_match:
                    test_wanted_files.append(os.path.join(root, file))

# single language option
else:
    language_path = os.path.join(directory, test_language.capitalize())
    for root, dirs, files in os.walk(language_path):  # os.walk() yields a tuple!
        for file in files:
            resolution_match = test_resolution in root or test_resolution == "all"
            style_match = os.path.basename(root) == test_style or test_style == "all" #os.path.basename(root) == font style
            
            if resolution_match and style_match:
                test_wanted_files.append(os.path.join(root, file))
                        
# ********************************************************************************************************************************

# GENERATING THE SPLIT OF THE DATA INTO TRAIN-VALID-TEST, 80%-10%-10%
training_data, validation_data = train_test_split(train_wanted_files,test_size=0.1)
_, testing_data = train_test_split(test_wanted_files,test_size=0.1)

print("this is len of training data:", len(training_data))
print("this is len of validation data:", len(validation_data))
print("this is len of testing data", len(testing_data))

# *********************************************************************************************************************************
# WRITING THE TRAINING, VALIDATION AND TESTING DATA INTO FILES

if train_style == "bold_italic" or test_style == "bold_italic":
    label_index = 5
else:
    label_index = 4

# TRAINING DATA:
with open("training_data.txt", "w") as training_data_file:
    for data_sample in training_data:
        data_sample_split = data_sample.split("_")
        try:
            label = data_sample_split[label_index].split(".")[0] # getting the character label. for instance "080" from this --> ['KTES211', '200', '41', '08', '080.bmp']
            training_data_file.write(data_sample)
            training_data_file.write(' ')
            training_data_file.write(label)
            training_data_file.write('\n')
        except:
            #print(f"problem with{data_sample_split}")
            pass
        


# VALIDATION DATA
with open("validation_data_gold.txt", "w") as validation_data_file_gold:
    for data_sample in validation_data:
        data_sample_split = data_sample.split("_")
        try:
            label = data_sample_split[label_index].split(".")[0] # getting the character label. for instance "080" from this --> ['KTES211', '200', '41', '08', '080.bmp']
            validation_data_file_gold.write(data_sample)
            validation_data_file_gold.write(' ')
            validation_data_file_gold.write(label)
            validation_data_file_gold.write('\n')
        except:
            #print(f"problem with{data_sample_split}")
            pass
        


# TESTING DATA
# gold file and without label file:
with open("testing_data_gold.txt", "w") as testing_data_file_gold:
    for data_sample in testing_data:
        data_sample_split = data_sample.split("_")
        try:
            label = data_sample_split[label_index].split(".")[0] # getting the character label. for instance "080" from this --> ['KTES211', '200', '41', '08', '080.bmp']
            testing_data_file_gold.write(data_sample)
            testing_data_file_gold.write(' ')
            testing_data_file_gold.write(label)
            testing_data_file_gold.write('\n')
        except:
            #print(f"problem with{data_sample_split}")
            pass


