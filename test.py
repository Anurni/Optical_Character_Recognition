# imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import CNN_forOCR
from train import test_loader
import torch

# retrieving the trained model 
model = CNN_forOCR()
state_dict = torch.load('my_OCR_model.pth', weights_only=True)
model.load_state_dict(state_dict)

# cuda settings
device = torch.device("cuda:3")
model.to(device)

# lists where we store predictions and true labels
all_labels = []
all_predictions = []

# seting the model in evaluation mode
model.eval()

with torch.no_grad():
    for batch in test_loader:
        X, y = batch
        X, y = X.to(device), y.to(device) # sending the feature tensor and label into the GPU
        outputs = model(X)  # predictions of the model
        _, predicted = torch.max(outputs, 1)  #argmax to get the highest probability out
        all_labels.extend(y.cpu().numpy())  
        all_predictions.extend(predicted.cpu().numpy())  
        
# calculating the metrics with sklearn
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')  #  marco, since we have multi-class and want the weighted average
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

# printing
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')