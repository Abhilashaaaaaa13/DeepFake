import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#1. setup n load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "deepfake_detector.pth"

#model architecture(efficientnet-b0)
from torchvision import models
import torch.nn as nn
model = models.efficientnet_b0()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs,2)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

#2. data loader for validation
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

])
#wahi processed data use kro
dataset = datasets.ImageFolder('./processed_data',transform=data_transforms)
#training wala hi split logic
train_size = int(0.8*len(dataset))
val_size= len(dataset)-train_size
_,val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

#3.predict on validation data
all_preds = []
all_labels=[]

print("Evaluating Model...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#4show metrics
print("\n---Classification Report---")
print(classification_report(all_labels,all_preds,target_names=['Real','Fake']))

#5 plot confusion matrix
cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',xticklabels=['Real','fake'],yticklabels=['Real','Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion_matrix')
plt.show()