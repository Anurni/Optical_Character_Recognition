# imports
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

# cuda settings
device = torch.device("cuda:3")

# data directories
training_data_dir = './training_data.txt'
validation_data_gold_dir = './validation_data_gold.txt'
testing_data_gold_dir = './testing_data_gold.txt'
img_dir = "/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/"

# creating a custom dataset for our data
class CharacterImageDataset(Dataset):
    def __init__(self, imgnames, img_dir, tensor_target_size=(64,64), transform=None):
        self.data = []
        with open(imgnames, "r") as file:
            for datapoint in file:
                filename, label = datapoint.strip().split(" ")
                self.data.append((filename, label))
                
        self.tensor_target_size = tensor_target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.tensor_target_size),  # resizing to our target tensor size
            transforms.Pad((2, 2, 2, 2), fill=1),  # padding left,top,right,bottom with 2 pixels with value 1 (white)
            transforms.ToTensor()
            ])
        
        # a new mapping for our labels is needed since they are not in order --> [ 81  76 106  71  80 100  97  73  67  90  ...]
        # creating a mapping for the labels
        self.label_mapping = {}
        unique_labels = set()

        # collecting unique labels
        for filename, label in self.data:
            unique_labels.add(label)

        # sorting and mapping labels to integers
        unique_labels = sorted(unique_labels)  
        i = 0
        for label in unique_labels:
            self.label_mapping[label] = i
            i += 1
        self.number_of_classes = len(unique_labels)
        self.labels = unique_labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0] #imagename (filename)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image) #transforming image into a tensor
        
        orig_label = self.data[idx][1]
        label = self.label_mapping[orig_label] #getting the label of the image
        

        return image, label


# creating our datasets
train_dataset = CharacterImageDataset(imgnames=training_data_dir, img_dir=img_dir)
valid_dataset = CharacterImageDataset(imgnames=validation_data_gold_dir, img_dir=img_dir)
test_dataset = CharacterImageDataset(imgnames=testing_data_gold_dir, img_dir=img_dir)

# creating our dataloaders out of the datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  #check later if batch size needs to be given as user varible
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# getting the number of labels
train_labels = train_dataset.number_of_classes
valid_labels = valid_dataset.number_of_classes
test_labels = test_dataset.number_of_classes

# model definition
class CNN_forOCR(nn.Module):
    def __init__(self, num_classes=train_labels):
        super(CNN_forOCR, self).__init__() 

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)   # input channel = 1, since we have b&w photos
        
        self.flatten = nn.Flatten()
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)  # padding is done in dataset
        
        self.poolay = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(64 * 15 * 15, 128) #15x15 will be the height x width after max pooling
        
        self.fc2 = nn.Linear(128, num_classes) 
        
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  #relu for non-linearity
        x = self.poolay(x)    # applying max pooling here
        x = torch.nn.functional.relu(self.conv2(x))  
        x = self.poolay(x)    # applying second max pooling here
        x = x.view(x.size(0), -1)  # flatteniing the tensor for upcoming linear fc layer
        x = self.fc1(x)  
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        
        return self.softmax(x)

# the training loop of the model
def train(epochs, modelfile="my_OCR_model.pth"): #saving the model to --> "my_OCR_model.pth"
    loader = train_loader
    model = CNN_forOCR().to(device)
    loss_function = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting model training...")
    for epoch in range(epochs): 
        running_loss = 0.0
        for batch in loader:
            X,y = batch
            X, y = X.to(device), y.to(device)    # sending the feature tensor and label into the GPU
            optimizer.zero_grad()  
            outputs = model(X)  # getting prediction
            loss_value = loss_function(outputs, y)  
            loss_value.backward()  
            optimizer.step()  

            running_loss += loss_value.item()

        print(f"Epoch {epoch}, Loss: {running_loss/len(loader)}")
        
    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

# finally, training the model
if __name__ == "__main__":
    # argument parsing, as n of training epochs is given as cmd line argument
    parser = argparse.ArgumentParser(
                prog='This is the training script.',
                description='Trains a CNN model for Optical Character Recognition.')
    parser.add_argument('epochs', type=str, help="The number of epochs for the training loop.")
    arguments = parser.parse_args()
    epochs = arguments.epochs
    epochs = int(epochs)
    model = train(epochs)

